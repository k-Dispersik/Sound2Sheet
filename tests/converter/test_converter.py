"""
Tests for converter classes.
"""

import json
import pytest
from pathlib import Path
import mido
from src.converter.note import Note
from src.converter.note_sequence import NoteSequence
from src.converter.converter import JSONConverter, MIDIConverter, MusicXMLConverter


class TestJSONConverter:
    """Test JSON converter."""
    
    def test_convert_to_json(self, tmp_path):
        """Test converting sequence to JSON."""
        notes = [
            Note(pitch=60, start_time=0.0, duration=0.5, velocity=80),
            Note(pitch=64, start_time=0.5, duration=0.5, velocity=75)
        ]
        sequence = NoteSequence(notes=notes, tempo=120)
        
        output_path = tmp_path / "output.json"
        converter = JSONConverter()
        converter.convert(sequence, output_path)
        
        # Verify file exists
        assert output_path.exists()
        
        # Load and verify content
        with open(output_path) as f:
            data = json.load(f)
        
        assert data['metadata']['tempo'] == 120
        assert len(data['notes']) == 2
        assert data['notes'][0]['pitch'] == 60
    
    def test_load_json(self, tmp_path):
        """Test loading JSON file."""
        data = {
            'metadata': {'tempo': 120},
            'notes': [{'pitch': 60, 'start_time': 0.0}]
        }
        
        json_path = tmp_path / "test.json"
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        loaded = JSONConverter.load(json_path)
        
        assert loaded['metadata']['tempo'] == 120
        assert len(loaded['notes']) == 1


class TestMIDIConverter:
    """Test MIDI converter."""
    
    def test_convert_to_midi(self, tmp_path):
        """Test converting sequence to MIDI."""
        notes = [
            Note(pitch=60, start_time=0.0, duration=0.5, velocity=80),
            Note(pitch=64, start_time=0.5, duration=0.5, velocity=75),
            Note(pitch=67, start_time=1.0, duration=0.5, velocity=70)
        ]
        sequence = NoteSequence(
            notes=notes,
            tempo=120,
            time_signature=(4, 4)
        )
        
        output_path = tmp_path / "output.mid"
        converter = MIDIConverter()
        converter.convert(sequence, output_path)
        
        # Verify file exists
        assert output_path.exists()
        
        # Load and verify MIDI content
        midi = mido.MidiFile(output_path)
        
        # Check we have tracks
        assert len(midi.tracks) > 0
        
        # Check for tempo and time signature messages
        track = midi.tracks[0]
        has_tempo = any(msg.type == 'set_tempo' for msg in track)
        has_time_sig = any(msg.type == 'time_signature' for msg in track)
        
        assert has_tempo
        assert has_time_sig
        
        # Count note on/off messages
        note_messages = [msg for msg in track if msg.type in ['note_on', 'note_off']]
        # Should have 3 note_on + 3 note_off = 6 messages
        assert len(note_messages) == 6
    
    def test_midi_note_values(self, tmp_path):
        """Test MIDI contains correct note values."""
        notes = [Note(pitch=60, start_time=0.0, duration=0.5, velocity=80)]
        sequence = NoteSequence(notes=notes, tempo=120)
        
        output_path = tmp_path / "output.mid"
        converter = MIDIConverter()
        converter.convert(sequence, output_path)
        
        midi = mido.MidiFile(output_path)
        track = midi.tracks[0]
        
        # Find note_on message
        note_on = next(msg for msg in track if msg.type == 'note_on')
        
        assert note_on.note == 60
        assert note_on.velocity == 80


class TestMusicXMLConverter:
    """Test MusicXML converter."""
    
    def test_musicxml_requires_music21(self, tmp_path):
        """Test MusicXML converter requires music21 library."""
        sequence = NoteSequence()
        output_path = tmp_path / "output.xml"
        
        converter = MusicXMLConverter()
        
        # Should work if music21 installed, else raise ImportError
        try:
            import music21
            # If music21 available, test export
            notes = [Note(pitch=60, start_time=0.0, duration=0.5)]
            sequence = NoteSequence(notes=notes)
            converter.convert(sequence, output_path)
            assert output_path.exists()
        except ImportError:
            # If music21 not available, expect ImportError
            with pytest.raises(ImportError, match="music21 library required"):
                converter.convert(sequence, output_path)
