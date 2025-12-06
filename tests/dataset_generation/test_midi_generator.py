"""
Unit tests for MIDIGenerator class.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dataset.midi_generator import (
    MIDIGenerator,
    MIDIConfig,
    ComplexityLevel,
    TimeSignature,
    KeySignature
)


class TestMIDIConfig(unittest.TestCase):
    """Test cases for MIDIConfig class."""
    
    def test_default_initialization(self):
        """Test default configuration values."""
        config = MIDIConfig()
        
        self.assertEqual(config.tempo, 120)
        self.assertEqual(config.time_signature, TimeSignature.FOUR_FOUR)
        self.assertEqual(config.key_signature, KeySignature.C_MAJOR)
        self.assertEqual(config.num_measures, 8)
        self.assertEqual(config.complexity, ComplexityLevel.INTERMEDIATE)
        self.assertTrue(config.include_chords)
        self.assertTrue(config.include_melody)
        self.assertEqual(config.velocity_variation, 0.2)
    
    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = MIDIConfig(
            tempo=140,
            time_signature=TimeSignature.THREE_FOUR,
            key_signature=KeySignature.A_MINOR,
            num_measures=16,
            complexity=ComplexityLevel.ADVANCED
        )
        
        self.assertEqual(config.tempo, 140)
        self.assertEqual(config.time_signature, TimeSignature.THREE_FOUR)
        self.assertEqual(config.key_signature, KeySignature.A_MINOR)
        self.assertEqual(config.num_measures, 16)
        self.assertEqual(config.complexity, ComplexityLevel.ADVANCED)


class TestMIDIGenerator(unittest.TestCase):
    """Test cases for MIDIGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = MIDIGenerator()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator.config)
            
    def test_initialization_with_config(self):
        """Test generator initialization with custom config."""
        config = MIDIConfig(tempo=100, complexity=ComplexityLevel.BEGINNER)
        generator = MIDIGenerator(config)
        
        self.assertEqual(generator.config.tempo, 100)
        self.assertEqual(generator.config.complexity, ComplexityLevel.BEGINNER)
    
    def test_generate_midi_file(self):
        """Test generating a single MIDI file."""
        output_path = self.temp_dir / "test.mid"
        
        midi_file = self.generator.generate(output_path)
        
        self.assertIsNotNone(midi_file)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
    
    def test_generate_without_save(self):
        """Test generating MIDI without saving to file."""
        midi_file = self.generator.generate()
        
        self.assertIsNotNone(midi_file)
    
    def test_generate_with_different_complexities(self):
        """Test generation with different complexity levels."""
        for complexity in ComplexityLevel:
            with self.subTest(complexity=complexity):
                config = MIDIConfig(complexity=complexity)
                generator = MIDIGenerator(config)
                
                output_path = self.temp_dir / f"test_{complexity.value}.mid"
                midi_file = generator.generate(output_path)
                
                self.assertTrue(output_path.exists())
    
    def test_generate_with_different_time_signatures(self):
        """Test generation with different time signatures."""
        for time_sig in [TimeSignature.FOUR_FOUR, TimeSignature.THREE_FOUR, TimeSignature.SIX_EIGHT]:
            with self.subTest(time_signature=time_sig):
                config = MIDIConfig(time_signature=time_sig)
                generator = MIDIGenerator(config)
                
                midi_file = generator.generate()
                self.assertIsNotNone(midi_file)
    
    def test_generate_with_different_keys(self):
        """Test generation with different key signatures."""
        for key in [KeySignature.C_MAJOR, KeySignature.A_MINOR, KeySignature.G_MAJOR]:
            with self.subTest(key=key):
                config = MIDIConfig(key_signature=key)
                generator = MIDIGenerator(config)
                
                midi_file = generator.generate()
                self.assertIsNotNone(midi_file)
    
    def test_generate_chords_only(self):
        """Test generation with only chords."""
        config = MIDIConfig(include_melody=False)
        generator = MIDIGenerator(config)
        
        output_path = self.temp_dir / "chords_only.mid"
        midi_file = generator.generate(output_path)
        
        self.assertTrue(output_path.exists())
    
    def test_generate_melody_only(self):
        """Test generation with only melody."""
        config = MIDIConfig(include_chords=False)
        generator = MIDIGenerator(config)
        
        output_path = self.temp_dir / "melody_only.mid"
        midi_file = generator.generate(output_path)
        
        self.assertTrue(output_path.exists())
    
    def test_generate_batch(self):
        """Test batch generation of MIDI files."""
        num_files = 5
        output_dir = self.temp_dir / "batch"
        
        generated_files = self.generator.generate_batch(
            num_files=num_files,
            output_dir=output_dir,
            prefix="test"
        )
        
        self.assertEqual(len(generated_files), num_files)
        
        for file_path in generated_files:
            self.assertTrue(file_path.exists())
            self.assertGreater(file_path.stat().st_size, 0)
    
    def test_chord_progression_generation(self):
        """Test chord progression generation."""
        chords = self.generator._generate_chord_progression()
        
        self.assertIsInstance(chords, list)
        self.assertGreater(len(chords), 0)
        
        # Each chord should have notes
        for chord in chords:
            self.assertIsInstance(chord, list)
            self.assertGreater(len(chord), 0)
            # MIDI note numbers should be in valid range
            for note in chord:
                self.assertGreaterEqual(note, 0)
                self.assertLessEqual(note, 127)
    
    def test_melody_generation(self):
        """Test melody generation."""
        melody = self.generator._generate_melody()
        
        self.assertIsInstance(melody, list)
        
        # Melody should have notes (tuples of note, duration, start_time)
        for note_data in melody:
            self.assertIsInstance(note_data, tuple)
            self.assertEqual(len(note_data), 3)
            
            note, duration, start_time = note_data
            # MIDI note number should be valid
            self.assertGreaterEqual(note, 0)
            self.assertLessEqual(note, 127)
            # Duration should be positive
            self.assertGreater(duration, 0)
            # Start time should be non-negative
            self.assertGreaterEqual(start_time, 0)
    
    def test_config_randomization(self):
        """Test configuration randomization for diversity."""
        original_tempo = self.generator.config.tempo
        original_key = self.generator.config.key_signature
        
        # Randomize multiple times to check variability
        configs_changed = False
        for _ in range(10):
            self.generator._randomize_config()
            if (self.generator.config.tempo != original_tempo or 
                self.generator.config.key_signature != original_key):
                configs_changed = True
                break
        
        self.assertTrue(configs_changed, "Configuration should vary after randomization")


class TestEnums(unittest.TestCase):
    """Test cases for enum classes."""
    
    def test_complexity_levels(self):
        """Test ComplexityLevel enum."""
        self.assertEqual(len(ComplexityLevel), 3)
        self.assertIn(ComplexityLevel.BEGINNER, ComplexityLevel)
        self.assertIn(ComplexityLevel.INTERMEDIATE, ComplexityLevel)
        self.assertIn(ComplexityLevel.ADVANCED, ComplexityLevel)
    
    def test_time_signatures(self):
        """Test TimeSignature enum."""
        self.assertIsInstance(TimeSignature.FOUR_FOUR.value, tuple)
        self.assertEqual(TimeSignature.FOUR_FOUR.value, (4, 4))
        self.assertEqual(TimeSignature.THREE_FOUR.value, (3, 4))
    
    def test_key_signatures(self):
        """Test KeySignature enum."""
        self.assertIsInstance(KeySignature.C_MAJOR.value, tuple)
        # Each key should have (tonic, mode)
        tonic, mode = KeySignature.C_MAJOR.value
        self.assertIsInstance(tonic, int)
        self.assertIn(mode, ["major", "minor"])


if __name__ == '__main__':
    unittest.main()
