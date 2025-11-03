"""
Tests for Quantizer class.
"""

import pytest
from src.converter.note import Note
from src.converter.quantizer import Quantizer, QuantizationConfig


class TestQuantizer:
    """Test note quantization."""
    
    def test_quantize_notes(self):
        """Test basic note quantization."""
        config = QuantizationConfig(beat_resolution=16)
        quantizer = Quantizer(config)
        
        # Slightly off-beat notes
        notes = [
            Note(pitch=60, start_time=0.02, duration=0.48),  # ~0.0
            Note(pitch=64, start_time=0.52, duration=0.47)   # ~0.5
        ]
        
        quantized = quantizer.quantize(notes, tempo=120)
        
        # Should snap to grid (120 BPM = 0.5s per beat)
        assert quantized[0].start_time == pytest.approx(0.0, abs=0.05)
        assert quantized[1].start_time == pytest.approx(0.5, abs=0.05)
    
    def test_detect_tempo(self):
        """Test tempo detection from note onsets."""
        config = QuantizationConfig()
        quantizer = Quantizer(config)
        
        # 120 BPM = 0.5s per beat
        notes = [
            Note(pitch=60, start_time=0.0, duration=0.1),
            Note(pitch=62, start_time=0.5, duration=0.1),
            Note(pitch=64, start_time=1.0, duration=0.1),
            Note(pitch=65, start_time=1.5, duration=0.1)
        ]
        
        tempo = quantizer.detect_tempo(notes)
        
        # Should be close to 120 BPM
        assert 110 <= tempo <= 130
    
    def test_detect_tempo_few_notes(self):
        """Test tempo detection with few notes returns default."""
        config = QuantizationConfig()
        quantizer = Quantizer(config)
        
        notes = [Note(pitch=60, start_time=0.0, duration=0.5)]
        tempo = quantizer.detect_tempo(notes)
        
        assert tempo == 120  # Default
    
    def test_align_to_beats(self):
        """Test beat alignment."""
        config = QuantizationConfig(timing_tolerance=0.1)
        quantizer = Quantizer(config)
        
        # Notes close to beat positions
        notes = [
            Note(pitch=60, start_time=0.05, duration=0.5),   # Close to beat 0
            Note(pitch=64, start_time=0.48, duration=0.5),   # Close to beat 1
            Note(pitch=67, start_time=1.2, duration=0.5)     # Far from beat, no align
        ]
        
        aligned = quantizer.align_to_beats(notes, tempo=120)
        
        # First two should be aligned, third not
        assert aligned[0].start_time == pytest.approx(0.0, abs=0.01)
        assert aligned[1].start_time == pytest.approx(0.5, abs=0.01)
        assert aligned[2].start_time == 1.2  # Unchanged


class TestQuantizationConfig:
    """Test quantization configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = QuantizationConfig()
        
        assert config.beat_resolution == 16
        assert config.min_note_duration == 0.05
        assert config.timing_tolerance == 0.1
        assert config.auto_tempo_detection is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = QuantizationConfig(
            beat_resolution=8,
            min_note_duration=0.1,
            auto_tempo_detection=False
        )
        
        assert config.beat_resolution == 8
        assert config.min_note_duration == 0.1
        assert config.auto_tempo_detection is False
