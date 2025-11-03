"""
Note data structure for musical notation.

Represents a single musical note with timing and pitch information.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Note:
    """
    Represents a single musical note.
    
    Attributes:
        pitch: MIDI note number (21-108 for piano, A0 to C8)
        start_time: Note onset in seconds
        duration: Note duration in seconds
        velocity: MIDI velocity (0-127, default 80)
        start_beat: Quantized beat position (optional)
        duration_beats: Duration in beats (optional)
        confidence: Model prediction confidence (0.0-1.0, optional)
    """
    pitch: int
    start_time: float
    duration: float
    velocity: int = 80
    start_beat: Optional[float] = None
    duration_beats: Optional[float] = None
    confidence: Optional[float] = None
    
    def __post_init__(self):
        """Validate note parameters."""
        if not 0 <= self.pitch <= 127:
            raise ValueError(f"Invalid pitch: {self.pitch}. Must be 0-127.")
        if self.start_time < 0:
            raise ValueError(f"Invalid start_time: {self.start_time}. Must be >= 0.")
        if self.duration <= 0:
            raise ValueError(f"Invalid duration: {self.duration}. Must be > 0.")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"Invalid velocity: {self.velocity}. Must be 0-127.")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError(f"Invalid confidence: {self.confidence}. Must be 0-1.")
    
    @property
    def end_time(self) -> float:
        """Calculate note end time."""
        return self.start_time + self.duration
    
    @property
    def pitch_name(self) -> str:
        """Convert MIDI pitch to note name (e.g., 60 -> C4)."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (self.pitch // 12) - 1
        note = note_names[self.pitch % 12]
        return f"{note}{octave}"
    
    def overlaps_with(self, other: 'Note') -> bool:
        """Check if this note overlaps with another note."""
        return (self.pitch == other.pitch and 
                self.start_time < other.end_time and 
                other.start_time < self.end_time)
    
    def to_dict(self) -> dict:
        """Convert note to dictionary representation."""
        result = {
            'pitch': self.pitch,
            'pitch_name': self.pitch_name,
            'start_time': self.start_time,
            'duration': self.duration,
            'velocity': self.velocity,
        }
        if self.start_beat is not None:
            result['start_beat'] = self.start_beat
        if self.duration_beats is not None:
            result['duration_beats'] = self.duration_beats
        if self.confidence is not None:
            result['confidence'] = self.confidence
        return result
