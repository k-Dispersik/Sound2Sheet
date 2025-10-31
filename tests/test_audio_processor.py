"""
Unit tests for AudioProcessor class.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np
import librosa

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.audio_processor import AudioProcessor, AudioConfig


class TestAudioConfig(unittest.TestCase):
    """Test cases for AudioConfig class."""
    
    def test_default_initialization(self):
        """Test AudioConfig initialization with default values."""
        config = AudioConfig()
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.n_fft, 1024)
        self.assertEqual(config.hop_length, 320)
        self.assertEqual(config.n_mels, 128)
        self.assertEqual(config.f_min, 0.0)
        self.assertEqual(config.f_max, 8000.0)
        self.assertTrue(config.normalize)
        self.assertEqual(config.pre_emphasis, 0.97)
    
    def test_config_from_file(self):
        """Test AudioConfig loading from YAML file."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            temp_file.write("""
audio:
  sample_rate: 22050
  n_fft: 2048
  normalize: false
            """)
            temp_file.flush()
            
            config = AudioConfig(temp_file.name)
            self.assertEqual(config.sample_rate, 22050)
            self.assertEqual(config.n_fft, 2048)
            self.assertFalse(config.normalize)
            
            # Clean up
            Path(temp_file.name).unlink()


class TestAudioProcessor(unittest.TestCase):
    """Test cases for AudioProcessor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = AudioProcessor()
        self.test_audio = np.array([0.5, -0.3, 0.8, -0.1, 0.2], dtype=np.float32)
    
    def test_initialization(self):
        """Test AudioProcessor initialization."""
        self.assertIsNotNone(self.processor.config)
        self.assertEqual(self.processor.config.sample_rate, 16000)
    
    def test_initialization_with_config(self):
        """Test AudioProcessor initialization with custom config."""
        custom_config = AudioConfig()
        custom_config.sample_rate = 44100
        processor = AudioProcessor(custom_config)
        self.assertEqual(processor.config.sample_rate, 44100)
    
    def test_validate_audio_file_nonexistent(self):
        """Test validation of non-existent audio file."""
        self.assertFalse(self.processor.validate_audio_file("non_existent.wav"))
    
    def test_validate_audio_file_valid_wav(self):
        """Test validation of valid WAV file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            self.assertTrue(self.processor.validate_audio_file(temp_path))
            temp_path.unlink()
    
    def test_validate_audio_file_valid_mp3(self):
        """Test validation of valid MP3 file."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            self.assertTrue(self.processor.validate_audio_file(temp_path))
            temp_path.unlink()
    
    def test_validate_audio_file_valid_m4a(self):
        """Test validation of valid M4A file."""
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            self.assertTrue(self.processor.validate_audio_file(temp_path))
            temp_path.unlink()
    
    def test_validate_audio_file_unsupported_format(self):
        """Test validation of unsupported audio format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            self.assertFalse(self.processor.validate_audio_file(temp_path))
            temp_path.unlink()
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        normalized = self.processor.normalize_audio(self.test_audio.copy())
        self.assertEqual(len(normalized), len(self.test_audio))
        self.assertTrue(np.all(np.abs(normalized) <= 1.0))
    
    def test_normalize_audio_disabled(self):
        """Test audio normalization when disabled."""
        self.processor.config.normalize = False
        normalized = self.processor.normalize_audio(self.test_audio.copy())
        np.testing.assert_array_equal(normalized, self.test_audio)
    
    def test_apply_pre_emphasis(self):
        """Test pre-emphasis filter application."""
        emphasized = self.processor.apply_pre_emphasis(self.test_audio.copy())
        self.assertEqual(len(emphasized), len(self.test_audio))
        # First sample should remain unchanged
        self.assertEqual(emphasized[0], self.test_audio[0])
    
    def test_apply_pre_emphasis_disabled(self):
        """Test pre-emphasis when disabled."""
        self.processor.config.pre_emphasis = 0.0
        emphasized = self.processor.apply_pre_emphasis(self.test_audio.copy())
        np.testing.assert_array_equal(emphasized, self.test_audio)
    
    def test_load_audio_file_not_found(self):
        """Test loading non-existent audio file."""
        with self.assertRaises(FileNotFoundError):
            self.processor.load_audio("non_existent.wav")
    
    def test_load_audio_valid_file(self):
        """Test loading valid audio file."""
        # Create temporary WAV file with actual audio data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
            # Generate test audio
            duration = 0.5
            sample_rate = 44100
            t = np.linspace(0, duration, int(duration * sample_rate), False)
            audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            # Save as WAV file 
            import soundfile as sf
            sf.write(temp_path, audio_data, sample_rate)
            
            # Load the audio
            loaded_audio = self.processor.load_audio(temp_path)
            
            # Check that we get audio data back
            self.assertIsInstance(loaded_audio, np.ndarray)
            self.assertGreater(len(loaded_audio), 0)
            self.assertEqual(loaded_audio.dtype, np.float32)
            
            # Clean up
            temp_path.unlink()
    
    def test_resample_audio(self):
        """Test audio resampling."""
        # Create test audio at 44100 Hz
        duration = 1.0
        original_sr = 44100
        target_sr = 16000
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(duration * original_sr)))
        
        # Resample
        resampled = self.processor.resample_audio(test_audio, original_sr, target_sr)
        
        # Check new length
        expected_length = int(len(test_audio) * target_sr / original_sr)
        self.assertAlmostEqual(len(resampled), expected_length, delta=10)
    
    def test_resample_audio_same_rate(self):
        """Test resampling when rates are the same."""
        resampled = self.processor.resample_audio(self.test_audio, 16000, 16000)
        np.testing.assert_array_equal(resampled, self.test_audio)
    
    def test_to_mel_spectrogram(self):
        """Test mel-spectrogram generation."""
        # Create longer test audio for spectrogram
        duration = 2.0
        sample_rate = self.processor.config.sample_rate
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(duration * sample_rate)))
        
        mel_spec = self.processor.to_mel_spectrogram(test_audio)
        
        # Check shape
        self.assertEqual(mel_spec.shape[0], self.processor.config.n_mels)
        self.assertGreater(mel_spec.shape[1], 0)  # Should have time frames
    
    def test_augment_audio_pitch_shift(self):
        """Test pitch shifting augmentation."""
        augmented = self.processor.augment_audio(self.test_audio, "pitch_shift", n_steps=2)
        self.assertEqual(len(augmented), len(self.test_audio))
    
    def test_augment_audio_time_stretch(self):
        """Test time stretching augmentation."""
        augmented = self.processor.augment_audio(self.test_audio, "time_stretch", rate=1.2)
        # Time stretching changes length
        self.assertNotEqual(len(augmented), len(self.test_audio))
    
    def test_augment_audio_noise(self):
        """Test noise addition augmentation."""
        augmented = self.processor.augment_audio(self.test_audio, "noise", noise_factor=0.01)
        self.assertEqual(len(augmented), len(self.test_audio))
        # Should be different due to added noise
        self.assertFalse(np.array_equal(augmented, self.test_audio))
    
    def test_augment_audio_volume(self):
        """Test volume adjustment augmentation."""
        volume_factor = 0.5
        augmented = self.processor.augment_audio(self.test_audio, "volume", volume_factor=volume_factor)
        expected = self.test_audio * volume_factor
        np.testing.assert_array_almost_equal(augmented, expected)
    
    def test_augment_audio_unknown_type(self):
        """Test error handling for unknown augmentation type."""
        with self.assertRaises(ValueError):
            self.processor.augment_audio(self.test_audio, "unknown_type")
    
    def test_process_audio_pipeline_with_real_audio(self):
        """Test complete audio processing pipeline with generated audio."""
        # Create temporary WAV file with actual audio data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
            # Generate test audio (1 second of 440Hz sine wave)
            duration = 1.0
            sample_rate = 44100
            t = np.linspace(0, duration, int(duration * sample_rate), False)
            audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            # Save as WAV file
            import soundfile as sf
            sf.write(temp_path, audio_data, sample_rate)
            
            # Process the audio - should return mel-spectrogram by default
            result = self.processor.process_audio(temp_path, return_spectrogram=True)
            
            # Check that we get a 2D mel-spectrogram
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result.shape), 2)
            self.assertEqual(result.shape[0], self.processor.config.n_mels)
            
            # Test returning processed audio instead
            processed_audio = self.processor.process_audio(temp_path, return_spectrogram=False)
            self.assertIsInstance(processed_audio, np.ndarray)
            self.assertEqual(len(processed_audio.shape), 1)  # 1D audio array
            
            # Clean up
            temp_path.unlink()


if __name__ == "__main__":
    unittest.main()