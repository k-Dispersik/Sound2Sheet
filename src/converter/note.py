"""
Note data structure for musical notation.

Represents a single musical note with timing and pitch information.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Dynamic(Enum):
    """Musical dynamics (volume/intensity markings)."""
    PPP = "ppp"  # pianississimo
    PP = "pp"    # pianissimo
    P = "p"      # piano
    MP = "mp"    # mezzo-piano
    MF = "mf"    # mezzo-forte
    F = "f"      # forte
    FF = "ff"    # fortissimo
    FFF = "fff"  # fortississimo


class Articulation(Enum):
    """Musical articulation markings."""
    STACCATO = "staccato"      # Short, detached
    STACCATISSIMO = "staccatissimo"  # Very short
    TENUTO = "tenuto"          # Held for full value
    ACCENT = "accent"          # Emphasized
    MARCATO = "marcato"        # Strongly accented
    LEGATO = "legato"          # Smooth, connected


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
        is_tied_start: True if this note is the start of a tie (optional)
        is_tied_end: True if this note is the end of a tie (optional)
        tie_group_id: ID linking tied notes together (optional)
        dynamic: Dynamic marking (ppp, pp, p, mp, mf, f, ff, fff) (optional)
        articulation: Articulation marking (staccato, legato, etc.) (optional)
    """
    pitch: int
    start_time: float
    duration: float
    velocity: int = 80
    start_beat: Optional[float] = None
    duration_beats: Optional[float] = None
    confidence: Optional[float] = None
    is_tied_start: bool = False
    is_tied_end: bool = False
    tie_group_id: Optional[int] = None
    dynamic: Optional[Dynamic] = None
    articulation: Optional[Articulation] = None
    
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
    
    def infer_dynamic_from_velocity(self) -> Dynamic:
        """
        Infer dynamic marking from MIDI velocity.
        
        Velocity ranges (approximate):
        0-15: ppp, 16-31: pp, 32-47: p, 48-63: mp,
        64-79: mf, 80-95: f, 96-111: ff, 112-127: fff
        """
        if self.velocity <= 15:
            return Dynamic.PPP
        elif self.velocity <= 31:
            return Dynamic.PP
        elif self.velocity <= 47:
            return Dynamic.P
        elif self.velocity <= 63:
            return Dynamic.MP
        elif self.velocity <= 79:
            return Dynamic.MF
        elif self.velocity <= 95:
            return Dynamic.F
        elif self.velocity <= 111:
            return Dynamic.FF
        else:
            return Dynamic.FFF
    
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
        if self.is_tied_start:
            result['is_tied_start'] = True
        if self.is_tied_end:
            result['is_tied_end'] = True
        if self.tie_group_id is not None:
            result['tie_group_id'] = self.tie_group_id
        if self.dynamic is not None:
            result['dynamic'] = self.dynamic.value
        if self.articulation is not None:
            result['articulation'] = self.articulation.value
        return result
