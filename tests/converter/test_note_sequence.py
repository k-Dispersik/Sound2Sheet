"""
Tests for NoteSequence class.
"""

import pytest
from src.converter.note import Note
from src.converter.note_sequence import NoteSequence


class TestNoteSequenceCreation:
    """Test note sequence creation."""
    
    def test_create_empty_sequence(self):
        """Test creating empty sequence with defaults."""
        seq = NoteSequence()
        assert seq.note_count == 0
        assert seq.tempo == 120
        assert seq.time_signature == (4, 4)
    
    def test_create_with_notes(self):
        """Test creating sequence with notes."""
        notes = [
            Note(pitch=60, start_time=0.0, duration=0.5),
            Note(pitch=64, start_time=0.5, duration=0.5)
        ]
        seq = NoteSequence(notes=notes, tempo=140)
        
        assert seq.note_count == 2
        assert seq.tempo == 140
    
    def test_invalid_tempo(self):
        """Test invalid tempo raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tempo"):
            NoteSequence(tempo=0)
    
    def test_invalid_time_signature(self):
        """Test invalid time signature raises ValueError."""
        with pytest.raises(ValueError, match="Invalid time_signature"):
            NoteSequence(time_signature=(0, 4))


class TestNoteSequenceProperties:
    """Test sequence properties."""
    
    def test_total_duration_empty(self):
        """Test total duration of empty sequence."""
        seq = NoteSequence()
        assert seq.total_duration == 0.0
    
    def test_total_duration_with_notes(self):
        """Test total duration calculation."""
        notes = [
            Note(pitch=60, start_time=0.0, duration=0.5),
            Note(pitch=64, start_time=0.5, duration=1.0)
        ]
        seq = NoteSequence(notes=notes)
        assert seq.total_duration == 1.5


class TestNoteSequenceOperations:
    """Test sequence operations."""
    
    def test_add_note(self):
        """Test adding notes to sequence."""
        seq = NoteSequence()
        note = Note(pitch=60, start_time=0.0, duration=0.5)
        seq.add_note(note)
        
        assert seq.note_count == 1
        assert seq.notes[0] == note
    
    def test_sort_by_time(self):
        """Test sorting notes by time."""
        notes = [
            Note(pitch=64, start_time=1.0, duration=0.5),
            Note(pitch=60, start_time=0.0, duration=0.5),
            Note(pitch=67, start_time=0.5, duration=0.5)
        ]
        seq = NoteSequence(notes=notes)
        seq.sort_by_time()
        
        assert seq.notes[0].start_time == 0.0
        assert seq.notes[1].start_time == 0.5
        assert seq.notes[2].start_time == 1.0
    
    def test_get_notes_in_range(self):
        """Test getting notes in time range."""
        notes = [
            Note(pitch=60, start_time=0.0, duration=0.5),
            Note(pitch=62, start_time=0.5, duration=0.5),
            Note(pitch=64, start_time=1.0, duration=0.5)
        ]
        seq = NoteSequence(notes=notes)
        
        range_notes = seq.get_notes_in_range(0.3, 0.8)
        assert len(range_notes) == 1
        assert range_notes[0].pitch == 62


class TestNoteSequenceValidation:
    """Test sequence validation."""
    
    def test_validate_valid_sequence(self):
        """Test validation of valid sequence."""
        notes = [
            Note(pitch=60, start_time=0.0, duration=0.5),
            Note(pitch=64, start_time=0.5, duration=0.5)
        ]
        seq = NoteSequence(notes=notes)
        errors = seq.validate()
        
        assert len(errors) == 0
    
    def test_validate_overlapping_notes(self):
        """Test validation detects overlapping notes."""
        notes = [
            Note(pitch=60, start_time=0.0, duration=1.0),
            Note(pitch=60, start_time=0.5, duration=1.0)
        ]
        seq = NoteSequence(notes=notes)
        errors = seq.validate()
        
        assert len(errors) > 0
        assert "Overlapping" in errors[0]


class TestNoteSequenceToDict:
    """Test sequence serialization."""
    
    def test_to_dict_empty(self):
        """Test empty sequence to dict."""
        seq = NoteSequence(tempo=140, time_signature=(3, 4))
        data = seq.to_dict()
        
        assert data['metadata']['tempo'] == 140
        assert data['metadata']['time_signature'] == [3, 4]
        assert data['metadata']['note_count'] == 0
        assert len(data['notes']) == 0
    
    def test_to_dict_with_notes(self):
        """Test sequence with notes to dict."""
        notes = [
            Note(pitch=60, start_time=0.0, duration=0.5),
            Note(pitch=64, start_time=0.5, duration=0.5)
        ]
        seq = NoteSequence(notes=notes, tempo=120)
        data = seq.to_dict()
        
        assert data['metadata']['note_count'] == 2
        assert len(data['notes']) == 2
        assert data['notes'][0]['pitch'] == 60
