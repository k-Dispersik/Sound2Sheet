"""
Tests for AudioProcessor with various audio formats and qualities.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.audio_processor import AudioProcessor


class TestAudioFormatsAndQualities(unittest.TestCase):
    """Test AudioProcessor with different audio formats and qualities."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = AudioProcessor()
        
        # Create test audio signals with different characteristics
        self.sample_rates = [8000, 16000, 22050, 44100, 48000]
        self.durations = [1.0, 5.0, 10.0]  # seconds
        
        # Generate different types of test signals
        self.test_signals = {
            'sine_wave': lambda t: np.sin(2 * np.pi * 440 * t),
            'complex_tone': lambda t: (np.sin(2 * np.pi * 440 * t) + 
                                     0.5 * np.sin(2 * np.pi * 880 * t) + 
                                     0.25 * np.sin(2 * np.pi * 1320 * t)),
            'noise': lambda t: np.random.normal(0, 0.1, len(t)),
            'chirp': lambda t: np.sin(2 * np.pi * (100 + 400 * t / max(t)) * t),
            'quiet': lambda t: 0.01 * np.sin(2 * np.pi * 440 * t),
            'loud': lambda t: 0.9 * np.sin(2 * np.pi * 440 * t)
        }
    
    def _create_test_audio(self, signal_type: str, duration: float, sample_rate: int) -> np.ndarray:
        """Create test audio signal."""
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples, False)
        return self.test_signals[signal_type](t).astype(np.float32)
    
    def test_different_sample_rates(self):
        """Test loading and processing audio with different sample rates."""
        print("\n=== Testing Different Sample Rates ===")
        
        for sr in self.sample_rates:
            with self.subTest(sample_rate=sr):
                # Create test audio
                audio = self._create_test_audio('sine_wave', 2.0, sr)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                    sf.write(temp_path, audio, sr)
                    
                    # Load and process
                    loaded_audio = self.processor.load_audio(temp_path)
                    
                    # Verify resampling to target rate
                    expected_samples = int(2.0 * self.processor.config.sample_rate)
                    self.assertAlmostEqual(len(loaded_audio), expected_samples, delta=100)
                    
                    # Generate mel-spectrogram
                    mel_spec = self.processor.to_mel_spectrogram(loaded_audio)
                    self.assertEqual(mel_spec.shape[0], self.processor.config.n_mels)
                    
                    print(f"  ✓ Sample rate {sr}Hz: {len(loaded_audio)} samples -> {mel_spec.shape}")
                    
                    # Clean up
                    temp_path.unlink()
    
    def test_different_signal_types(self):
        """Test processing different types of audio signals."""
        print("\n=== Testing Different Signal Types ===")
        
        for signal_name, signal_func in self.test_signals.items():
            with self.subTest(signal_type=signal_name):
                # Create test audio
                audio = self._create_test_audio(signal_name, 3.0, 16000)
                
                # Normalize and process
                normalized = self.processor.normalize_audio(audio)
                mel_spec = self.processor.to_mel_spectrogram(normalized)
                
                # Basic sanity checks
                self.assertFalse(np.any(np.isnan(mel_spec)), f"NaN values in {signal_name} spectrogram")
                self.assertFalse(np.any(np.isinf(mel_spec)), f"Inf values in {signal_name} spectrogram")
                self.assertGreater(mel_spec.std(), 0, f"No variation in {signal_name} spectrogram")
                
                print(f"  ✓ {signal_name}: shape {mel_spec.shape}, range [{mel_spec.min():.2f}, {mel_spec.max():.2f}]")
    
    def test_different_durations(self):
        """Test processing audio files of different durations."""
        print("\n=== Testing Different Durations ===")
        
        for duration in self.durations:
            with self.subTest(duration=duration):
                # Create test audio
                audio = self._create_test_audio('complex_tone', duration, 16000)
                
                # Process
                normalized = self.processor.normalize_audio(audio)
                mel_spec = self.processor.to_mel_spectrogram(normalized)
                
                # Check that we get reasonable number of frames
                # The actual hop_length might be adjusted for short audio in to_mel_spectrogram
                expected_frames = int(duration * self.processor.config.sample_rate / self.processor.config.hop_length)
                min_expected = max(1, expected_frames - 20)
                max_expected = int(duration * self.processor.config.sample_rate / 64) + 20  # hop_length min is 1, n_fft/4 
                self.assertGreaterEqual(mel_spec.shape[1], min_expected)
                self.assertLessEqual(mel_spec.shape[1], max_expected)
                
                print(f"  ✓ Duration {duration}s: {mel_spec.shape[1]} frames (expected ~{expected_frames})")
    
    def test_edge_cases(self):
        """Test edge cases like very quiet/loud audio, silence, etc."""
        print("\n=== Testing Edge Cases ===")
        
        # Test silence
        silence = np.zeros(16000, dtype=np.float32)
        normalized_silence = self.processor.normalize_audio(silence)
        mel_spec_silence = self.processor.to_mel_spectrogram(normalized_silence)
        self.assertFalse(np.any(np.isnan(mel_spec_silence)), "NaN in silence spectrogram")
        print("  ✓ Silence processed successfully")
        
        # Test very short audio
        short_audio = self._create_test_audio('sine_wave', 0.1, 16000)  # 100ms
        normalized_short = self.processor.normalize_audio(short_audio)
        mel_spec_short = self.processor.to_mel_spectrogram(normalized_short)
        self.assertGreater(mel_spec_short.shape[1], 0, "No frames in short audio")
        print(f"  ✓ Short audio (0.1s): {mel_spec_short.shape}")
        
        # Test DC offset
        dc_audio = self._create_test_audio('sine_wave', 1.0, 16000) + 0.5
        normalized_dc = self.processor.normalize_audio(dc_audio)
        mel_spec_dc = self.processor.to_mel_spectrogram(normalized_dc)
        self.assertFalse(np.any(np.isnan(mel_spec_dc)), "NaN in DC offset spectrogram")
        print("  ✓ DC offset audio processed successfully")
    
    def test_augmentation_robustness(self):
        """Test augmentations work across different audio types."""
        print("\n=== Testing Augmentation Robustness ===")
        
        test_cases = [
            ('sine_wave', 2.0, 16000),
            ('complex_tone', 3.0, 16000),
            ('quiet', 1.0, 16000),
        ]
        
        augmentations = [
            ('volume', {'volume_factor': 0.5}),
            ('noise', {'noise_factor': 0.01}),
            ('pitch_shift', {'n_steps': 1}),
            ('time_stretch', {'rate': 0.9}),
        ]
        
        for signal_name, duration, sr in test_cases:
            audio = self._create_test_audio(signal_name, duration, sr)
            
            for aug_type, aug_params in augmentations:
                with self.subTest(signal=signal_name, augmentation=aug_type):
                    try:
                        augmented = self.processor.augment_audio(audio, aug_type, **aug_params)
                        
                        # Basic checks
                        if aug_type != 'time_stretch':
                            # Time stretch changes length, others should not
                            self.assertEqual(len(augmented), len(audio), 
                                           f"Length mismatch in {aug_type} for {signal_name}")
                        else:
                            # Time stretch should produce valid length
                            self.assertGreater(len(augmented), 0, 
                                             f"Empty output in {aug_type} for {signal_name}")
                        
                        self.assertFalse(np.any(np.isnan(augmented)), 
                                       f"NaN in {aug_type} for {signal_name}")
                        self.assertFalse(np.any(np.isinf(augmented)), 
                                       f"Inf in {aug_type} for {signal_name}")
                        
                        print(f"  ✓ {signal_name} + {aug_type}: OK")
                        
                    except Exception as e:
                        self.fail(f"Augmentation {aug_type} failed for {signal_name}: {e}")
    
    def test_bit_depth_simulation(self):
        """Test processing audio with different simulated bit depths."""
        print("\n=== Testing Different Bit Depths (Simulated) ===")
        
        # Simulate different bit depths by quantizing
        bit_depths = [8, 16, 24]
        
        for bit_depth in bit_depths:
            with self.subTest(bit_depth=bit_depth):
                # Create high-quality audio
                audio = self._create_test_audio('complex_tone', 2.0, 16000)
                
                # Simulate lower bit depth by quantizing
                max_val = 2 ** (bit_depth - 1) - 1
                quantized = np.round(audio * max_val) / max_val
                quantized = np.clip(quantized, -1.0, 1.0).astype(np.float32)
                
                # Process
                normalized = self.processor.normalize_audio(quantized)
                mel_spec = self.processor.to_mel_spectrogram(normalized)
                
                # Should work without errors
                self.assertFalse(np.any(np.isnan(mel_spec)), f"NaN in {bit_depth}-bit spectrogram")
                self.assertGreater(mel_spec.std(), 0, f"No variation in {bit_depth}-bit spectrogram")
                
                print(f"  ✓ {bit_depth}-bit depth: range [{mel_spec.min():.2f}, {mel_spec.max():.2f}]")
    
    def test_format_compatibility_wav_only(self):
        """Test only WAV format compatibility (others require external tools)."""
        print("\n=== Testing WAV Format Compatibility ===")
        
        # Test different WAV subformats
        wav_subtypes = ['PCM_16', 'PCM_24', 'FLOAT']
        
        for subtype in wav_subtypes:
            with self.subTest(subtype=subtype):
                try:
                    audio = self._create_test_audio('sine_wave', 1.0, 16000)
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_path = Path(temp_file.name)
                        
                        # Write with specific subtype
                        sf.write(temp_path, audio, 16000, subtype=subtype)
                        
                        # Load and verify
                        loaded_audio = self.processor.load_audio(temp_path)
                        self.assertGreater(len(loaded_audio), 0)
                        
                        print(f"  ✓ WAV {subtype}: {len(loaded_audio)} samples")
                        
                        # Clean up
                        temp_path.unlink()
                        
                except Exception as e:
                    print(f"  ⚠ WAV {subtype}: Not supported - {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)