"""
Tests for tied notes, measures, and expression markers.
"""

import pytest
from src.converter import Note, NoteSequence, Dynamic, Articulation, Measure


class TestTiedNotes:
    """Tests for tied notes functionality."""
    
    def test_note_tie_attributes(self):
        """Test that Note supports tie attributes."""
        note = Note(
            pitch=60,
            start_time=0.0,
            duration=1.0,
            is_tied_start=True,
            tie_group_id=1
        )
        assert note.is_tied_start is True
        assert note.is_tied_end is False
        assert note.tie_group_id == 1
    
    def test_tied_notes_in_dict(self):
        """Test that tie information is exported to dict."""
        note = Note(
            pitch=60,
            start_time=0.0,
            duration=1.0,
            is_tied_start=True,
            is_tied_end=False,
            tie_group_id=1
        )
        data = note.to_dict()
        assert data['is_tied_start'] is True
        assert 'is_tied_end' not in data  # Only export True values
        assert data['tie_group_id'] == 1
    
    def test_apply_tied_notes_single_measure(self):
        """Test that notes within one measure are not tied."""
        sequence = NoteSequence(tempo=120, time_signature=(4, 4))
        
        # Note that fits in one measure (4 beats = 2 seconds at 120 BPM)
        sequence.add_note(Note(pitch=60, start_time=0.0, duration=1.0))
        sequence.add_note(Note(pitch=62, start_time=1.0, duration=0.5))
        
        sequence.apply_tied_notes()
        
        # Notes should not be tied
        assert all(not note.is_tied_start for note in sequence.notes)
        assert all(not note.is_tied_end for note in sequence.notes)
    
    def test_apply_tied_notes_multiple_measures(self):
        """Test that notes spanning measures are tied."""
        sequence = NoteSequence(tempo=120, time_signature=(4, 4))
        
        # Measure duration = 4 beats * (60s / 120 BPM) = 2.0 seconds
        # Note spanning from measure 1 to measure 2
        sequence.add_note(Note(pitch=60, start_time=1.5, duration=1.5))
        
        sequence.apply_tied_notes()
        
        # Should create 2 tied notes
        assert len(sequence.notes) == 2
        
        # First note (in measure 1)
        note1 = sequence.notes[0]
        assert note1.is_tied_start is True
        assert note1.is_tied_end is False
        assert note1.start_time == 1.5
        assert note1.duration == pytest.approx(0.5)  # Until measure boundary
        
        # Second note (in measure 2)
        note2 = sequence.notes[1]
        assert note2.is_tied_start is False
        assert note2.is_tied_end is True
        assert note2.start_time == 2.0
        assert note2.duration == pytest.approx(1.0)
        
        # Both should have same tie_group_id
        assert note1.tie_group_id == note2.tie_group_id
    
    def test_apply_tied_notes_three_measures(self):
        """Test note spanning three measures."""
        sequence = NoteSequence(tempo=60, time_signature=(4, 4))
        
        # Measure duration = 4 beats * (60s / 60 BPM) = 4.0 seconds
        # Note from 1.0 to 9.0 (spans 3 measures)
        sequence.add_note(Note(pitch=64, start_time=1.0, duration=8.0))
        
        sequence.apply_tied_notes()
        
        # Should create 3 tied notes
        assert len(sequence.notes) == 3
        assert sequence.notes[0].is_tied_start is True
        assert sequence.notes[1].is_tied_start is False
        assert sequence.notes[1].is_tied_end is False
        assert sequence.notes[2].is_tied_end is True


class TestMeasures:
    """Tests for measure organization."""
    
    def test_measure_creation(self):
        """Test Measure class creation."""
        measure = Measure(
            number=1,
            start_time=0.0,
            duration=2.0,
            time_signature=(4, 4)
        )
        assert measure.number == 1
        assert measure.start_time == 0.0
        assert measure.duration == 2.0
        assert measure.end_time == 2.0
    
    def test_organize_into_measures(self):
        """Test organizing notes into measures."""
        sequence = NoteSequence(tempo=120, time_signature=(4, 4))
        
        # Add notes across 2 measures
        sequence.add_note(Note(pitch=60, start_time=0.0, duration=1.0))
        sequence.add_note(Note(pitch=62, start_time=1.5, duration=0.5))
        sequence.add_note(Note(pitch=64, start_time=2.5, duration=1.0))
        
        sequence.organize_into_measures()
        
        # Should have 2 measures (measure duration = 2.0s at 120 BPM, 4/4)
        assert len(sequence.measures) >= 2
        
        # First measure should have 2 notes
        assert len(sequence.measures[0].notes) == 2
        
        # Second measure should have 1 note
        assert len(sequence.measures[1].notes) == 1
    
    def test_measure_to_dict(self):
        """Test measure serialization to dict."""
        measure = Measure(number=1, start_time=0.0, duration=2.0)
        measure.add_note(Note(pitch=60, start_time=0.0, duration=1.0))
        
        data = measure.to_dict()
        assert data['number'] == 1
        assert data['start_time'] == 0.0
        assert data['duration'] == 2.0
        assert data['note_count'] == 1
        assert 'notes' in data
    
    def test_sequence_with_measures_to_dict(self):
        """Test that measures are included in sequence dict."""
        sequence = NoteSequence()
        sequence.add_note(Note(pitch=60, start_time=0.0, duration=1.0))
        sequence.organize_into_measures()
        
        data = sequence.to_dict()
        assert 'measures' in data
        assert len(data['measures']) > 0


class TestExpressionMarkers:
    """Tests for dynamics and articulation."""
    
    def test_dynamic_enum(self):
        """Test Dynamic enum values."""
        assert Dynamic.P.value == "p"
        assert Dynamic.F.value == "f"
        assert Dynamic.PP.value == "pp"
        assert Dynamic.FF.value == "ff"
    
    def test_articulation_enum(self):
        """Test Articulation enum values."""
        assert Articulation.STACCATO.value == "staccato"
        assert Articulation.LEGATO.value == "legato"
        assert Articulation.ACCENT.value == "accent"
    
    def test_note_with_dynamic(self):
        """Test note with dynamic marking."""
        note = Note(
            pitch=60,
            start_time=0.0,
            duration=1.0,
            dynamic=Dynamic.F
        )
        assert note.dynamic == Dynamic.F
        
        data = note.to_dict()
        assert data['dynamic'] == 'f'
    
    def test_note_with_articulation(self):
        """Test note with articulation marking."""
        note = Note(
            pitch=60,
            start_time=0.0,
            duration=1.0,
            articulation=Articulation.STACCATO
        )
        assert note.articulation == Articulation.STACCATO
        
        data = note.to_dict()
        assert data['articulation'] == 'staccato'
    
    def test_infer_dynamic_from_velocity(self):
        """Test automatic dynamic inference from velocity."""
        # Test various velocity ranges
        note_ppp = Note(pitch=60, start_time=0.0, duration=1.0, velocity=10)
        assert note_ppp.infer_dynamic_from_velocity() == Dynamic.PPP
        
        note_p = Note(pitch=60, start_time=0.0, duration=1.0, velocity=40)
        assert note_p.infer_dynamic_from_velocity() == Dynamic.P
        
        note_mf = Note(pitch=60, start_time=0.0, duration=1.0, velocity=70)
        assert note_mf.infer_dynamic_from_velocity() == Dynamic.MF
        
        note_f = Note(pitch=60, start_time=0.0, duration=1.0, velocity=85)
        assert note_f.infer_dynamic_from_velocity() == Dynamic.F
        
        note_fff = Note(pitch=60, start_time=0.0, duration=1.0, velocity=120)
        assert note_fff.infer_dynamic_from_velocity() == Dynamic.FFF
    
    def test_infer_expression_marks(self):
        """Test automatic expression marking inference."""
        sequence = NoteSequence(tempo=120)
        
        # Add notes with different characteristics
        sequence.add_note(Note(
            pitch=60, start_time=0.0, duration=0.5,
            velocity=40, duration_beats=0.2  # Short, soft -> staccato + p
        ))
        sequence.add_note(Note(
            pitch=62, start_time=1.0, duration=1.0,
            velocity=110, duration_beats=0.8  # Loud, medium -> accent
        ))
        sequence.add_note(Note(
            pitch=64, start_time=2.0, duration=1.5,
            velocity=70, duration_beats=1.2  # Long -> tenuto
        ))
        
        sequence.infer_expression_marks()
        
        # Check that dynamics were inferred
        assert sequence.notes[0].dynamic == Dynamic.P
        assert sequence.notes[1].dynamic == Dynamic.FF
        assert sequence.notes[2].dynamic == Dynamic.MF
        
        # Check that articulation was inferred
        assert sequence.notes[0].articulation == Articulation.STACCATO
        assert sequence.notes[1].articulation == Articulation.ACCENT
        assert sequence.notes[2].articulation == Articulation.TENUTO
