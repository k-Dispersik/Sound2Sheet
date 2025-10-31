"""
Unit tests for AudioProcessor class.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np

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
    
    def test_process_audio_pipeline(self):
        """Test complete audio processing pipeline with dummy file."""
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
            # Process the audio (will use dummy data since we don't have real audio loading yet)
            processed = self.processor.process_audio(temp_path)
            
            # Check that we get an array back
            self.assertIsInstance(processed, np.ndarray)
            self.assertGreater(len(processed), 0)
            
            # Clean up
            temp_path.unlink()


if __name__ == "__main__":
    unittest.main()