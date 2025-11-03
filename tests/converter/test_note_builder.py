"""
Tests for NoteBuilder class.
"""

import pytest
from src.converter.note_builder import NoteBuilder
from src.converter.quantizer import QuantizationConfig


class TestNoteBuilderFromPredictions:
    """Test building from model predictions."""
    
    def test_build_from_predictions_basic(self):
        """Test basic prediction parsing."""
        builder = NoteBuilder()
        
        # Tokens: 1=MIDI 21 (A0), 41=MIDI 61 (C#4)
        predictions = [1, 41, 21, 41]
        
        sequence = builder.build_from_predictions(
            predictions,
            tempo=120,
            default_duration=0.5
        )
        
        assert sequence.note_count == 4
        assert sequence.tempo == 120
        assert sequence.notes[0].pitch == 21  # Token 1 -> MIDI 21
        assert sequence.notes[1].pitch == 61  # Token 41 -> MIDI 61
    
    def test_build_filters_special_tokens(self):
        """Test that special tokens are filtered out."""
        builder = NoteBuilder()
        
        # 0=pad, 89=sos, 90=eos, 91=unk - should be filtered
        predictions = [89, 1, 0, 41, 90, 21, 91]
        
        sequence = builder.build_from_predictions(predictions, tempo=120)
        
        # Only tokens 1, 41, 21 should create notes
        assert sequence.note_count == 3
    
    def test_build_empty_predictions(self):
        """Test building from empty predictions."""
        builder = NoteBuilder()
        
        sequence = builder.build_from_predictions([], tempo=120)
        
        assert sequence.note_count == 0
        assert sequence.tempo == 120
    
    def test_build_auto_tempo_detection(self):
        """Test automatic tempo detection."""
        config = QuantizationConfig(auto_tempo_detection=True)
        builder = NoteBuilder(config)
        
        # Regular pattern of notes
        predictions = [1, 2, 3, 4, 5, 6, 7, 8]
        
        sequence = builder.build_from_predictions(
            predictions,
            tempo=None,  # Should auto-detect
            default_duration=0.5
        )
        
        # Should have detected some tempo
        assert 40 <= sequence.tempo <= 240


class TestNoteBuilderFromTimestamps:
    """Test building from timestamped events."""
    
    def test_build_from_timestamps(self):
        """Test building from timestamp events."""
        builder = NoteBuilder()
        
        events = [
            {'pitch': 60, 'start_time': 0.0, 'duration': 0.5, 'velocity': 80},
            {'pitch': 64, 'start_time': 0.5, 'duration': 0.5, 'velocity': 75},
            {'pitch': 67, 'start_time': 1.0, 'duration': 0.5, 'velocity': 70}
        ]
        
        sequence = builder.build_from_timestamps(events, tempo=120)
        
        assert sequence.note_count == 3
        assert sequence.notes[0].pitch == 60
        assert sequence.notes[0].velocity == 80
    
    def test_build_from_timestamps_with_confidence(self):
        """Test building with confidence values."""
        builder = NoteBuilder()
        
        events = [
            {
                'pitch': 60,
                'start_time': 0.0,
                'duration': 0.5,
                'confidence': 0.95
            }
        ]
        
        sequence = builder.build_from_timestamps(events, tempo=120)
        
        assert sequence.notes[0].confidence == 0.95


class TestNoteBuilderRests:
    """Test rest detection and insertion."""
    
    def test_add_rests(self):
        """Test adding rest markers."""
        builder = NoteBuilder()
        
        events = [
            {'pitch': 60, 'start_time': 0.0, 'duration': 0.5},
            {'pitch': 64, 'start_time': 1.0, 'duration': 0.5}  # 0.5s gap
        ]
        
        sequence = builder.build_from_timestamps(events, tempo=120)
        sequence = builder.add_rests(sequence, min_rest_duration=0.25)
        
        # Should have rest metadata
        assert 'rests' in sequence.metadata
        assert len(sequence.metadata['rests']) == 1
        assert sequence.metadata['rests'][0]['duration'] == pytest.approx(0.5)
    
    def test_add_rests_no_gaps(self):
        """Test no rests when notes are continuous."""
        builder = NoteBuilder()
        
        events = [
            {'pitch': 60, 'start_time': 0.0, 'duration': 0.5},
            {'pitch': 64, 'start_time': 0.5, 'duration': 0.5}  # No gap
        ]
        
        sequence = builder.build_from_timestamps(events, tempo=120)
        sequence = builder.add_rests(sequence, min_rest_duration=0.25)
        
        # Should have no rests
        assert 'rests' not in sequence.metadata


class TestNoteBuilderValidation:
    """Test sequence validation."""
    
    def test_validate_valid_sequence(self):
        """Test validation of valid sequence."""
        builder = NoteBuilder()
        
        events = [
            {'pitch': 60, 'start_time': 0.0, 'duration': 0.5},
            {'pitch': 64, 'start_time': 0.5, 'duration': 0.5}
        ]
        
        sequence = builder.build_from_timestamps(events, tempo=120)
        
        assert builder.validate_sequence(sequence) is True
    
    def test_validate_invalid_sequence(self):
        """Test validation of invalid sequence (overlapping notes)."""
        builder = NoteBuilder()
        
        events = [
            {'pitch': 60, 'start_time': 0.0, 'duration': 1.0},
            {'pitch': 60, 'start_time': 0.5, 'duration': 1.0}  # Overlaps
        ]
        
        sequence = builder.build_from_timestamps(events, tempo=120)
        
        assert builder.validate_sequence(sequence) is False


class TestNoteBuilderQuantization:
    """Test quantization integration."""
    
    def test_quantization_applied(self):
        """Test that quantization is applied to notes."""
        config = QuantizationConfig(beat_resolution=16)
        builder = NoteBuilder(config)
        
        # Slightly off-beat events
        events = [
            {'pitch': 60, 'start_time': 0.02, 'duration': 0.48},
            {'pitch': 64, 'start_time': 0.52, 'duration': 0.47}
        ]
        
        sequence = builder.build_from_timestamps(events, tempo=120)
        
        # Should be quantized (120 BPM = 0.5s per beat)
        assert sequence.notes[0].start_time == pytest.approx(0.0, abs=0.05)
        assert sequence.notes[1].start_time == pytest.approx(0.5, abs=0.05)
