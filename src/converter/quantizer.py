"""
Timing quantization for musical notes.

Handles timing alignment of notes to musical grid.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from .note import Note


@dataclass
class QuantizationConfig:
    """Configuration for timing quantization."""
    beat_resolution: int = 16  # Subdivisions per beat (16th notes)
    min_note_duration: float = 0.05  # Minimum note length in seconds
    timing_tolerance: float = 0.1  # Tolerance for beat alignment (seconds)
    auto_tempo_detection: bool = True


class Quantizer:
    """
    Quantizes note timing to musical grid.
    
    Applies timing quantization based on detected or specified tempo.
    """
    
    def __init__(self, config: QuantizationConfig):
        """
        Initialize quantizer.
        
        Args:
            config: Quantization configuration
        """
        self.config = config
    
    def quantize(self, notes: List[Note], tempo: int) -> List[Note]:
        """
        Quantize note timing to musical grid.
        
        Args:
            notes: List of notes to quantize
            tempo: Tempo in BPM
            
        Returns:
            List of quantized notes
        """
        if not notes:
            return []
        
        beat_duration = 60.0 / tempo
        grid_size = beat_duration / self.config.beat_resolution
        
        quantized = []
        for note in notes:
            # Quantize start time
            start_beat = note.start_time / beat_duration
            start_grid = round(start_beat * self.config.beat_resolution)
            quantized_start = (start_grid / self.config.beat_resolution) * beat_duration
            
            # Quantize duration
            duration_grid = max(1, round(note.duration / grid_size))
            quantized_duration = duration_grid * grid_size
            
            # Apply minimum duration
            quantized_duration = max(quantized_duration, self.config.min_note_duration)
            
            # Create quantized note
            quantized_note = Note(
                pitch=note.pitch,
                start_time=quantized_start,
                duration=quantized_duration,
                velocity=note.velocity,
                start_beat=start_grid / self.config.beat_resolution,
                duration_beats=duration_grid / self.config.beat_resolution,
                confidence=note.confidence
            )
            quantized.append(quantized_note)
        
        return quantized
    
    def detect_tempo(self, notes: List[Note]) -> int:
        """
        Detect tempo from note onset intervals.
        
        Args:
            notes: List of notes
            
        Returns:
            Estimated tempo in BPM
        """
        if len(notes) < 2:
            return 120  # Default tempo
        
        # Calculate inter-onset intervals
        onsets = sorted([note.start_time for note in notes])
        intervals = np.diff(onsets)
        
        # Filter very short intervals (likely ornaments/grace notes)
        intervals = intervals[intervals > 0.1]
        
        if len(intervals) == 0:
            return 120
        
        # Find most common interval (likely beat duration)
        hist, bins = np.histogram(intervals, bins=50)
        peak_bin = np.argmax(hist)
        beat_duration = (bins[peak_bin] + bins[peak_bin + 1]) / 2
        
        # Convert to BPM
        tempo = int(round(60.0 / beat_duration))
        
        # Clamp to reasonable range
        return max(40, min(240, tempo))
    
    def align_to_beats(self, notes: List[Note], tempo: int) -> List[Note]:
        """
        Align notes to beat positions with tolerance.
        
        Args:
            notes: List of notes
            tempo: Tempo in BPM
            
        Returns:
            List of beat-aligned notes
        """
        beat_duration = 60.0 / tempo
        aligned = []
        
        for note in notes:
            # Calculate nearest beat
            beat_position = note.start_time / beat_duration
            nearest_beat = round(beat_position)
            beat_start = nearest_beat * beat_duration
            
            # Only align if within tolerance
            if abs(note.start_time - beat_start) <= self.config.timing_tolerance:
                aligned_note = Note(
                    pitch=note.pitch,
                    start_time=beat_start,
                    duration=note.duration,
                    velocity=note.velocity,
                    start_beat=float(nearest_beat),
                    duration_beats=note.duration / beat_duration,
                    confidence=note.confidence
                )
                aligned.append(aligned_note)
            else:
                aligned.append(note)
        
        return aligned
