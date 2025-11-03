"""
Tests for Note class.
"""

import pytest
from src.converter.note import Note


class TestNoteCreation:
    """Test note creation and validation."""
    
    def test_create_valid_note(self):
        """Test creating a valid note."""
        note = Note(pitch=60, start_time=0.0, duration=0.5, velocity=80)
        assert note.pitch == 60
        assert note.start_time == 0.0
        assert note.duration == 0.5
        assert note.velocity == 80
    
    def test_invalid_pitch(self):
        """Test invalid pitch raises ValueError."""
        with pytest.raises(ValueError, match="Invalid pitch"):
            Note(pitch=200, start_time=0.0, duration=0.5)
    
    def test_invalid_negative_start_time(self):
        """Test negative start time raises ValueError."""
        with pytest.raises(ValueError, match="Invalid start_time"):
            Note(pitch=60, start_time=-1.0, duration=0.5)
    
    def test_invalid_duration(self):
        """Test invalid duration raises ValueError."""
        with pytest.raises(ValueError, match="Invalid duration"):
            Note(pitch=60, start_time=0.0, duration=0.0)
    
    def test_invalid_velocity(self):
        """Test invalid velocity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid velocity"):
            Note(pitch=60, start_time=0.0, duration=0.5, velocity=200)
    
    def test_invalid_confidence(self):
        """Test invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="Invalid confidence"):
            Note(pitch=60, start_time=0.0, duration=0.5, confidence=1.5)


class TestNoteProperties:
    """Test note properties and methods."""
    
    def test_end_time(self):
        """Test end_time calculation."""
        note = Note(pitch=60, start_time=1.0, duration=0.5)
        assert note.end_time == 1.5
    
    def test_pitch_name_middle_c(self):
        """Test pitch name for middle C."""
        note = Note(pitch=60, start_time=0.0, duration=0.5)
        assert note.pitch_name == "C4"
    
    def test_pitch_name_a0(self):
        """Test pitch name for A0."""
        note = Note(pitch=21, start_time=0.0, duration=0.5)
        assert note.pitch_name == "A0"
    
    def test_pitch_name_c8(self):
        """Test pitch name for C8."""
        note = Note(pitch=108, start_time=0.0, duration=0.5)
        assert note.pitch_name == "C8"


class TestNoteOverlap:
    """Test note overlap detection."""
    
    def test_overlapping_notes(self):
        """Test overlapping notes are detected."""
        note1 = Note(pitch=60, start_time=0.0, duration=1.0)
        note2 = Note(pitch=60, start_time=0.5, duration=1.0)
        assert note1.overlaps_with(note2)
        assert note2.overlaps_with(note1)
    
    def test_non_overlapping_notes(self):
        """Test non-overlapping notes."""
        note1 = Note(pitch=60, start_time=0.0, duration=0.5)
        note2 = Note(pitch=60, start_time=0.5, duration=0.5)
        assert not note1.overlaps_with(note2)
    
    def test_different_pitch_no_overlap(self):
        """Test different pitches don't overlap."""
        note1 = Note(pitch=60, start_time=0.0, duration=1.0)
        note2 = Note(pitch=62, start_time=0.5, duration=1.0)
        assert not note1.overlaps_with(note2)


class TestNoteToDict:
    """Test note serialization."""
    
    def test_to_dict_basic(self):
        """Test basic note to dict conversion."""
        note = Note(pitch=60, start_time=0.0, duration=0.5, velocity=80)
        data = note.to_dict()
        
        assert data['pitch'] == 60
        assert data['pitch_name'] == "C4"
        assert data['start_time'] == 0.0
        assert data['duration'] == 0.5
        assert data['velocity'] == 80
    
    def test_to_dict_with_beats(self):
        """Test note to dict with beat information."""
        note = Note(
            pitch=60,
            start_time=0.0,
            duration=0.5,
            velocity=80,
            start_beat=0.0,
            duration_beats=1.0
        )
        data = note.to_dict()
        
        assert data['start_beat'] == 0.0
        assert data['duration_beats'] == 1.0
    
    def test_to_dict_with_confidence(self):
        """Test note to dict with confidence."""
        note = Note(
            pitch=60,
            start_time=0.0,
            duration=0.5,
            velocity=80,
            confidence=0.95
        )
        data = note.to_dict()
        
        assert data['confidence'] == 0.95
