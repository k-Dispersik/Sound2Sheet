"""
Unit tests for advanced noise augmentation in AudioProcessor.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.audio_processor import AudioProcessor
from core.noise_strategies import (
    NoiseStrategyFactory,
    WhiteNoiseStrategy,
    PinkNoiseStrategy,
    BrownNoiseStrategy,
    AmbientNoiseStrategy,
    HumNoiseStrategy
)


class TestNoiseAugmentation(unittest.TestCase):
    """Test cases for various noise augmentation types."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
        
        # Create test audio signal (1 second sine wave at 440Hz)
        duration = 1.0
        t = np.linspace(0, duration, int(self.processor.config.sample_rate * duration))
        self.test_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    def test_white_noise(self):
        """Test white noise augmentation."""
        noisy_audio = self.processor.augment_audio(
            self.test_audio,
            "noise",
            noise_type="white",
            noise_factor=0.01
        )
        
        self.assertEqual(noisy_audio.shape, self.test_audio.shape)
        self.assertIsInstance(noisy_audio, np.ndarray)
        
        # Audio should be different after adding noise
        self.assertFalse(np.array_equal(noisy_audio, self.test_audio))
        
        # But not too different (noise should be relatively small)
        correlation = np.corrcoef(self.test_audio, noisy_audio)[0, 1]
        self.assertGreater(correlation, 0.9)
    
    def test_pink_noise(self):
        """Test pink noise augmentation."""
        noisy_audio = self.processor.augment_audio(
            self.test_audio,
            "noise",
            noise_type="pink",
            noise_factor=0.01
        )
        
        self.assertEqual(noisy_audio.shape, self.test_audio.shape)
        self.assertIsInstance(noisy_audio, np.ndarray)
        self.assertFalse(np.array_equal(noisy_audio, self.test_audio))
    
    def test_brown_noise(self):
        """Test brown noise augmentation."""
        noisy_audio = self.processor.augment_audio(
            self.test_audio,
            "noise",
            noise_type="brown",
            noise_factor=0.01
        )
        
        self.assertEqual(noisy_audio.shape, self.test_audio.shape)
        self.assertIsInstance(noisy_audio, np.ndarray)
        self.assertFalse(np.array_equal(noisy_audio, self.test_audio))
    
    def test_ambient_noise(self):
        """Test ambient noise augmentation."""
        noisy_audio = self.processor.augment_audio(
            self.test_audio,
            "noise",
            noise_type="ambient",
            noise_factor=0.01
        )
        
        self.assertEqual(noisy_audio.shape, self.test_audio.shape)
        self.assertIsInstance(noisy_audio, np.ndarray)
        self.assertFalse(np.array_equal(noisy_audio, self.test_audio))
    
    def test_hum_noise_50hz(self):
        """Test power line hum noise at 50Hz (Europe)."""
        noisy_audio = self.processor.augment_audio(
            self.test_audio,
            "noise",
            noise_type="hum",
            hum_frequency=50,
            noise_factor=0.05
        )
        
        self.assertEqual(noisy_audio.shape, self.test_audio.shape)
        self.assertIsInstance(noisy_audio, np.ndarray)
        self.assertFalse(np.array_equal(noisy_audio, self.test_audio))
    
    def test_hum_noise_60hz(self):
        """Test power line hum noise at 60Hz (US)."""
        noisy_audio = self.processor.augment_audio(
            self.test_audio,
            "noise",
            noise_type="hum",
            hum_frequency=60,
            noise_factor=0.05
        )
        
        self.assertEqual(noisy_audio.shape, self.test_audio.shape)
        self.assertIsInstance(noisy_audio, np.ndarray)
        self.assertFalse(np.array_equal(noisy_audio, self.test_audio))
    
    def test_snr_white_noise(self):
        """Test SNR-based white noise."""
        # Test with high SNR (less noise)
        noisy_high_snr = self.processor.augment_audio(
            self.test_audio,
            "noise",
            noise_type="white",
            snr_db=30
        )
        
        # Test with low SNR (more noise)
        noisy_low_snr = self.processor.augment_audio(
            self.test_audio,
            "noise",
            noise_type="white",
            snr_db=10
        )
        
        # High SNR should be closer to original
        diff_high = np.mean(np.abs(noisy_high_snr - self.test_audio))
        diff_low = np.mean(np.abs(noisy_low_snr - self.test_audio))
        
        self.assertLess(diff_high, diff_low)
    
    def test_snr_pink_noise(self):
        """Test SNR-based pink noise."""
        noisy_audio = self.processor.augment_audio(
            self.test_audio,
            "noise",
            noise_type="pink",
            snr_db=20
        )
        
        self.assertEqual(noisy_audio.shape, self.test_audio.shape)
        
        # Calculate actual SNR
        noise = noisy_audio - self.test_audio
        signal_power = np.mean(self.test_audio ** 2)
        noise_power = np.mean(noise ** 2)
        actual_snr = 10 * np.log10(signal_power / noise_power)
        
        # Should be close to target SNR (within 3 dB)
        self.assertAlmostEqual(actual_snr, 20, delta=3)
    
    def test_invalid_noise_type(self):
        """Test that invalid noise type raises error."""
        with self.assertRaises(ValueError):
            self.processor.augment_audio(
                self.test_audio,
                "noise",
                noise_type="invalid_type"
            )
    
    def test_noise_preservation_of_length(self):
        """Test that all noise types preserve audio length."""
        noise_types = ['white', 'pink', 'brown', 'ambient', 'hum']
        
        for noise_type in noise_types:
            with self.subTest(noise_type=noise_type):
                noisy = self.processor.augment_audio(
                    self.test_audio,
                    "noise",
                    noise_type=noise_type,
                    noise_factor=0.01
                )
                self.assertEqual(len(noisy), len(self.test_audio))
    
    def test_noise_with_real_audio_file(self):
        """Test noise augmentation with real audio file."""
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
            # Write test audio to file
            sf.write(str(temp_path), self.test_audio, self.processor.config.sample_rate)
            
            try:
                # Load audio
                audio = self.processor.load_audio(temp_path)
                
                # Apply various noise types
                for noise_type in ['white', 'pink', 'brown', 'ambient']:
                    with self.subTest(noise_type=noise_type):
                        noisy = self.processor.augment_audio(
                            audio,
                            "noise",
                            noise_type=noise_type,
                            snr_db=25
                        )
                        
                        self.assertEqual(len(noisy), len(audio))
                        self.assertFalse(np.array_equal(noisy, audio))
            
            finally:
                temp_path.unlink()
    
    def test_pink_noise_frequency_characteristics(self):
        """Test that pink noise has correct frequency characteristics."""
        strategy = PinkNoiseStrategy()
        pink = strategy.generate(len(self.test_audio), self.processor.config.sample_rate)
        
        # Compute power spectrum
        fft = np.fft.rfft(pink)
        power = np.abs(fft) ** 2
        frequencies = np.fft.rfftfreq(len(pink))
        
        # Pink noise should have decreasing power with frequency
        # Check that low frequencies have more power than high frequencies
        low_freq_power = np.mean(power[:len(power)//4])
        high_freq_power = np.mean(power[3*len(power)//4:])
        
        self.assertGreater(low_freq_power, high_freq_power)
    
    def test_brown_noise_frequency_characteristics(self):
        """Test that brown noise has correct frequency characteristics."""
        strategy = BrownNoiseStrategy()
        brown = strategy.generate(len(self.test_audio), self.processor.config.sample_rate)
        
        # Compute power spectrum
        fft = np.fft.rfft(brown)
        power = np.abs(fft) ** 2
        
        # Brown noise should have even stronger low-frequency emphasis than pink
        low_freq_power = np.mean(power[:len(power)//4])
        high_freq_power = np.mean(power[3*len(power)//4:])
        
        self.assertGreater(low_freq_power, high_freq_power * 10)
    
    def test_hum_noise_frequency_peak(self):
        """Test that hum noise has peak at specified frequency."""
        strategy = HumNoiseStrategy()
        hum = strategy.generate(len(self.test_audio), self.processor.config.sample_rate, frequency=50)
        
        # Compute power spectrum
        fft = np.fft.rfft(hum)
        power = np.abs(fft) ** 2
        frequencies = np.fft.rfftfreq(len(hum), 1/self.processor.config.sample_rate)
        
        # Find peak frequency
        peak_idx = np.argmax(power)
        peak_freq = frequencies[peak_idx]
        
        # Should be close to 50 Hz (within 5 Hz)
        self.assertAlmostEqual(peak_freq, 50, delta=5)


class TestNoiseGenerationMethods(unittest.TestCase):
    """Test individual noise generation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
        self.factory = NoiseStrategyFactory()
        self.length = 16000  # 1 second at 16kHz
    
    def test_generate_pink_noise(self):
        """Test pink noise generation."""
        strategy = self.factory.get_strategy('pink')
        pink = strategy.generate(self.length, self.processor.config.sample_rate)
        
        self.assertEqual(len(pink), self.length)
        self.assertEqual(pink.dtype, np.float32)
        
        # Should be normalized
        self.assertLessEqual(np.max(np.abs(pink)), 1.0)
    
    def test_generate_brown_noise(self):
        """Test brown noise generation."""
        strategy = self.factory.get_strategy('brown')
        brown = strategy.generate(self.length, self.processor.config.sample_rate)
        
        self.assertEqual(len(brown), self.length)
        self.assertEqual(brown.dtype, np.float32)
        self.assertLessEqual(np.max(np.abs(brown)), 1.0)
    
    def test_generate_ambient_noise(self):
        """Test ambient noise generation."""
        strategy = self.factory.get_strategy('ambient')
        ambient = strategy.generate(self.length, self.processor.config.sample_rate)
        
        self.assertEqual(len(ambient), self.length)
        self.assertEqual(ambient.dtype, np.float32)
        self.assertLessEqual(np.max(np.abs(ambient)), 1.0)
    
    def test_generate_hum_noise(self):
        """Test hum noise generation."""
        strategy = self.factory.get_strategy('hum')
        hum = strategy.generate(self.length, self.processor.config.sample_rate, frequency=50)
        
        self.assertEqual(len(hum), self.length)
        self.assertEqual(hum.dtype, np.float32)
        self.assertLessEqual(np.max(np.abs(hum)), 1.0)
    
    def test_apply_snr(self):
        """Test SNR application."""
        signal = np.random.randn(self.length).astype(np.float32)
        noise = np.random.randn(self.length).astype(np.float32)
        
        # Apply 20 dB SNR
        scaled_noise = self.processor._apply_snr(signal, noise, 20)
        
        # Calculate actual SNR
        noisy_signal = signal + scaled_noise
        actual_noise = noisy_signal - signal
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(actual_noise ** 2)
        actual_snr = 10 * np.log10(signal_power / noise_power)
        
        # Should be close to target (within 1 dB)
        self.assertAlmostEqual(actual_snr, 20, delta=1)
    
    def test_factory_get_all_strategies(self):
        """Test that factory can create all noise strategies."""
        for noise_type in ['white', 'pink', 'brown', 'ambient', 'hum']:
            with self.subTest(noise_type=noise_type):
                strategy = self.factory.get_strategy(noise_type)
                self.assertIsNotNone(strategy)
                self.assertEqual(strategy.get_name(), noise_type)
    
    def test_factory_invalid_strategy(self):
        """Test that factory raises error for invalid strategy."""
        with self.assertRaises(ValueError) as context:
            self.factory.get_strategy('invalid_type')
        
        self.assertIn('Unknown noise type', str(context.exception))
    
    def test_factory_available_types(self):
        """Test getting available noise types from factory."""
        available = self.factory.get_available_types()
        
        self.assertIsInstance(available, list)
        self.assertIn('white', available)
        self.assertIn('pink', available)
        self.assertIn('brown', available)
        self.assertIn('ambient', available)
        self.assertIn('hum', available)


if __name__ == '__main__':
    unittest.main()
