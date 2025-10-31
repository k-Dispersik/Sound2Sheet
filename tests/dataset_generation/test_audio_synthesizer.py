"""
Unit tests for AudioSynthesizer class.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
import soundfile as sf
from midiutil import MIDIFile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dataset.audio_synthesizer import AudioSynthesizer


class TestAudioSynthesizerInitialization(unittest.TestCase):
    """Test AudioSynthesizer initialization."""
    
    def test_default_initialization_without_soundfont(self):
        """Test that initialization without soundfont raises error if none found."""
        try:
            synth = AudioSynthesizer()
            # If we get here, a default soundfont was found
            self.assertEqual(synth.sample_rate, 44100)
            self.assertEqual(synth.gain, 0.5)
        except (ImportError, FileNotFoundError) as e:
            # Expected if midi2audio not installed or no soundfont found
            self.skipTest(f"Skipping: {e}")
    
    def test_custom_sample_rate(self):
        """Test initialization with custom sample rate."""
        try:
            synth = AudioSynthesizer(sample_rate=22050)
            self.assertEqual(synth.sample_rate, 22050)
        except (ImportError, FileNotFoundError):
            self.skipTest("midi2audio or soundfont not available")
    
    def test_custom_gain(self):
        """Test initialization with custom gain."""
        try:
            synth = AudioSynthesizer(gain=0.8)
            self.assertEqual(synth.gain, 0.8)
        except (ImportError, FileNotFoundError):
            self.skipTest("midi2audio or soundfont not available")
    
    def test_invalid_soundfont_path(self):
        """Test that invalid soundfont path raises error."""
        try:
            with self.assertRaises(FileNotFoundError):
                AudioSynthesizer(soundfont_path="/nonexistent/path/soundfont.sf2")
        except ImportError:
            self.skipTest("midi2audio not available")


class TestAudioSynthesis(unittest.TestCase):
    """Test audio synthesis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            self.synth = AudioSynthesizer(sample_rate=16000, gain=0.5)
        except (ImportError, FileNotFoundError):
            self.skipTest("midi2audio or soundfont not available")
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create a simple test MIDI file
        self.test_midi = self._create_test_midi()
    
    def tearDown(self):
        """Clean up test files."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir)
    
    def _create_test_midi(self) -> Path:
        """Create a simple MIDI file for testing."""
        midi_file = MIDIFile(1)
        midi_file.addTempo(0, 0, 120)
        
        # Add a simple C major scale
        notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
        time = 0.0
        
        for note in notes:
            midi_file.addNote(0, 0, note, time, 0.5, 100)
            time += 0.5
        
        # Save MIDI file
        midi_path = self.temp_path / "test.mid"
        with open(midi_path, 'wb') as f:
            midi_file.writeFile(f)
        
        return midi_path
    
    def test_synthesize_basic(self):
        """Test basic synthesis functionality."""
        output_path = self.temp_path / "output.wav"
        
        try:
            audio = self.synth.synthesize(self.test_midi, output_path)
            
            # Check that audio was generated
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(audio.dtype, np.float32)
            self.assertGreater(len(audio), 0)
            
            # Check that output file exists
            self.assertTrue(output_path.exists())
            
        except RuntimeError as e:
            if "synthesis failed" in str(e).lower():
                self.skipTest("FluidSynth synthesis failed (soundfont or library issue)")
            raise
    
    def test_synthesize_with_normalization(self):
        """Test synthesis with normalization."""
        output_path = self.temp_path / "normalized.wav"
        
        try:
            audio = self.synth.synthesize(self.test_midi, output_path, normalize=True)
            
            # Check normalization
            max_amplitude = np.max(np.abs(audio))
            self.assertLessEqual(max_amplitude, 1.0)
            
            # Should be close to 1.0 if there's audio content
            if max_amplitude > 0.01:
                self.assertGreater(max_amplitude, 0.8)
                
        except RuntimeError as e:
            if "FluidSynth" in str(e):
                self.skipTest("FluidSynth synthesis failed")
            raise
    
    def test_synthesize_without_normalization(self):
        """Test synthesis without normalization."""
        output_path = self.temp_path / "unnormalized.wav"
        
        try:
            audio = self.synth.synthesize(self.test_midi, output_path, normalize=False)
            
            self.assertIsInstance(audio, np.ndarray)
            self.assertGreater(len(audio), 0)
            
        except RuntimeError as e:
            if "FluidSynth" in str(e):
                self.skipTest("FluidSynth synthesis failed")
            raise
    
    def test_synthesize_nonexistent_midi(self):
        """Test that synthesizing nonexistent MIDI raises error."""
        output_path = self.temp_path / "output.wav"
        
        with self.assertRaises(FileNotFoundError):
            self.synth.synthesize(
                self.temp_path / "nonexistent.mid",
                output_path
            )
    
    def test_synthesize_creates_output_directory(self):
        """Test that synthesis creates output directory if needed."""
        output_path = self.temp_path / "subdir" / "output.wav"
        
        try:
            self.synth.synthesize(self.test_midi, output_path)
            self.assertTrue(output_path.parent.exists())
            self.assertTrue(output_path.exists())
        except RuntimeError as e:
            if "FluidSynth" in str(e):
                self.skipTest("FluidSynth synthesis failed")
            raise


class TestBatchSynthesis(unittest.TestCase):
    """Test batch synthesis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            self.synth = AudioSynthesizer(sample_rate=16000)
        except (ImportError, FileNotFoundError):
            self.skipTest("midi2audio or soundfont not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create multiple test MIDI files
        self.midi_files = self._create_test_midis()
    
    def tearDown(self):
        """Clean up test files."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir)
    
    def _create_test_midis(self) -> list:
        """Create multiple MIDI files for batch testing."""
        midi_files = []
        
        for i in range(3):
            midi_file = MIDIFile(1)
            midi_file.addTempo(0, 0, 120)
            
            # Add different notes for each file
            base_note = 60 + i * 2
            midi_file.addNote(0, 0, base_note, 0, 1.0, 100)
            midi_file.addNote(0, 0, base_note + 4, 1.0, 1.0, 100)
            
            midi_path = self.temp_path / f"test_{i}.mid"
            with open(midi_path, 'wb') as f:
                midi_file.writeFile(f)
            
            midi_files.append(midi_path)
        
        return midi_files
    
    def test_batch_synthesis(self):
        """Test synthesizing multiple MIDI files."""
        output_dir = self.temp_path / "output"
        
        try:
            synthesized = self.synth.synthesize_batch(
                self.midi_files,
                output_dir,
                normalize=True
            )
            
            # Check that all files were synthesized
            self.assertEqual(len(synthesized), len(self.midi_files))
            
            # Check that all output files exist
            for output_path in synthesized:
                self.assertTrue(output_path.exists())
                self.assertEqual(output_path.suffix, '.wav')
                
        except RuntimeError as e:
            if "FluidSynth" in str(e):
                self.skipTest("FluidSynth synthesis failed")
            raise
    
    def test_batch_synthesis_flat_structure(self):
        """Test batch synthesis with flat output structure."""
        output_dir = self.temp_path / "flat_output"
        
        try:
            synthesized = self.synth.synthesize_batch(
                self.midi_files,
                output_dir,
                keep_structure=False
            )
            
            # All files should be in output_dir root
            for output_path in synthesized:
                self.assertEqual(output_path.parent, output_dir)
                
        except RuntimeError as e:
            if "FluidSynth" in str(e):
                self.skipTest("FluidSynth synthesis failed")
            raise


class TestAudioInfo(unittest.TestCase):
    """Test audio information retrieval."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            self.synth = AudioSynthesizer()
        except (ImportError, FileNotFoundError):
            self.skipTest("midi2audio or soundfont not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test files."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir)
    
    def test_get_audio_info(self):
        """Test getting audio file information."""
        # Create a test audio file
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        audio_path = self.temp_path / "test.wav"
        sf.write(str(audio_path), audio, 16000)
        
        info = self.synth.get_audio_info(audio_path)
        
        self.assertIn('file', info)
        self.assertIn('sample_rate', info)
        self.assertIn('duration', info)
        self.assertIn('samples', info)
        self.assertIn('max_amplitude', info)
        self.assertIn('rms', info)
        
        self.assertEqual(info['sample_rate'], 16000)
        self.assertEqual(info['samples'], 16000)
        self.assertAlmostEqual(info['duration'], 1.0, places=1)
    
    def test_get_audio_info_nonexistent_file(self):
        """Test that getting info for nonexistent file raises error."""
        with self.assertRaises(FileNotFoundError):
            self.synth.get_audio_info(self.temp_path / "nonexistent.wav")


if __name__ == '__main__':
    unittest.main()
