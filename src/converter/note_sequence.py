"""
Note sequence data structure for musical notation.

Manages a collection of notes with musical metadata.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from .note import Note


@dataclass
class NoteSequence:
    """
    Represents a sequence of musical notes with metadata.
    
    Attributes:
        notes: List of Note objects
        tempo: Tempo in BPM (beats per minute)
        time_signature: Time signature as (numerator, denominator)
        key_signature: Key signature (e.g., "C major", "A minor")
        metadata: Additional metadata dictionary
    """
    notes: List[Note] = field(default_factory=list)
    tempo: int = 120
    time_signature: Tuple[int, int] = (4, 4)
    key_signature: str = "C major"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate sequence parameters."""
        if self.tempo <= 0:
            raise ValueError(f"Invalid tempo: {self.tempo}. Must be > 0.")
        if len(self.time_signature) != 2 or any(x <= 0 for x in self.time_signature):
            raise ValueError(f"Invalid time_signature: {self.time_signature}")
    
    @property
    def total_duration(self) -> float:
        """Calculate total duration of the sequence in seconds."""
        if not self.notes:
            return 0.0
        return max(note.end_time for note in self.notes)
    
    @property
    def note_count(self) -> int:
        """Get total number of notes."""
        return len(self.notes)
    
    def add_note(self, note: Note) -> None:
        """Add a note to the sequence."""
        self.notes.append(note)
    
    def sort_by_time(self) -> None:
        """Sort notes by start time."""
        self.notes.sort(key=lambda n: n.start_time)
    
    def get_notes_in_range(self, start: float, end: float) -> List[Note]:
        """Get notes that start within time range [start, end)."""
        return [n for n in self.notes if start <= n.start_time < end]
    
    def validate(self) -> List[str]:
        """
        Validate musical correctness.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check for overlapping notes with same pitch
        for i, note1 in enumerate(self.notes):
            for note2 in self.notes[i + 1:]:
                if note1.overlaps_with(note2):
                    errors.append(
                        f"Overlapping notes at pitch {note1.pitch}: "
                        f"t1={note1.start_time:.3f}-{note1.end_time:.3f}, "
                        f"t2={note2.start_time:.3f}-{note2.end_time:.3f}"
                    )
        
        # Check for notes with invalid timing
        for note in self.notes:
            if note.start_time < 0:
                errors.append(f"Negative start time: {note.start_time}")
            if note.duration <= 0:
                errors.append(f"Invalid duration: {note.duration}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sequence to dictionary representation."""
        return {
            'metadata': {
                'tempo': self.tempo,
                'time_signature': list(self.time_signature),
                'key_signature': self.key_signature,
                'duration': self.total_duration,
                'note_count': self.note_count,
                **self.metadata
            },
            'notes': [note.to_dict() for note in self.notes]
        }
