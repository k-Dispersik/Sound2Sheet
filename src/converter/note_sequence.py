"""
Note sequence data structure for musical notation.

Manages a collection of notes with musical metadata.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from .note import Note


@dataclass
class Measure:
    """
    Represents a single measure (bar) in musical notation.
    
    Attributes:
        number: Measure number (1-indexed)
        start_time: Start time in seconds
        duration: Duration in seconds
        time_signature: Time signature for this measure
        notes: List of notes in this measure
    """
    number: int
    start_time: float
    duration: float
    time_signature: Tuple[int, int] = (4, 4)
    notes: List[Note] = field(default_factory=list)
    
    @property
    def end_time(self) -> float:
        """Calculate measure end time."""
        return self.start_time + self.duration
    
    def add_note(self, note: Note) -> None:
        """Add a note to this measure."""
        self.notes.append(note)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert measure to dictionary representation."""
        return {
            'number': self.number,
            'start_time': self.start_time,
            'duration': self.duration,
            'time_signature': list(self.time_signature),
            'note_count': len(self.notes),
            'notes': [note.to_dict() for note in self.notes]
        }


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
        measures: List of Measure objects (optional)
    """
    notes: List[Note] = field(default_factory=list)
    tempo: int = 120
    time_signature: Tuple[int, int] = (4, 4)
    key_signature: str = "C major"
    metadata: Dict[str, Any] = field(default_factory=dict)
    measures: List[Measure] = field(default_factory=list)
    
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
    
    def organize_into_measures(self) -> None:
        """
        Organize notes into measures based on time signature and tempo.
        Creates Measure objects and assigns notes to them.
        """
        if not self.notes:
            return
        
        self.measures.clear()
        self.sort_by_time()
        
        # Calculate measure duration in seconds
        beats_per_measure = self.time_signature[0]
        seconds_per_beat = 60.0 / self.tempo
        measure_duration = beats_per_measure * seconds_per_beat
        
        # Create measures covering the entire sequence
        total_duration = self.total_duration
        measure_count = int(total_duration / measure_duration) + 1
        
        for i in range(measure_count):
            measure = Measure(
                number=i + 1,
                start_time=i * measure_duration,
                duration=measure_duration,
                time_signature=self.time_signature
            )
            self.measures.append(measure)
        
        # Assign notes to measures
        for note in self.notes:
            measure_idx = int(note.start_time / measure_duration)
            if 0 <= measure_idx < len(self.measures):
                self.measures[measure_idx].add_note(note)
    
    def apply_tied_notes(self) -> None:
        """
        Detect and mark notes that should be tied across measure boundaries.
        A note is split into tied notes if it spans multiple measures.
        """
        if not self.measures:
            self.organize_into_measures()
        
        new_notes = []
        tie_group_counter = 0
        
        for note in self.notes:
            note_end = note.end_time
            
            # Find which measures this note spans
            measure_duration = self.measures[0].duration if self.measures else 0
            start_measure_idx = int(note.start_time / measure_duration)
            end_measure_idx = int(note_end / measure_duration)
            
            # Note fits within one measure - keep as is
            if start_measure_idx == end_measure_idx:
                new_notes.append(note)
                continue
            
            # Note spans multiple measures - split it
            tie_group_counter += 1
            current_time = note.start_time
            
            for measure_idx in range(start_measure_idx, end_measure_idx + 1):
                if measure_idx >= len(self.measures):
                    break
                
                measure = self.measures[measure_idx]
                segment_start = max(current_time, measure.start_time)
                segment_end = min(note_end, measure.end_time)
                segment_duration = segment_end - segment_start
                
                if segment_duration > 0:
                    tied_note = Note(
                        pitch=note.pitch,
                        start_time=segment_start,
                        duration=segment_duration,
                        velocity=note.velocity,
                        start_beat=note.start_beat,
                        confidence=note.confidence,
                        is_tied_start=(measure_idx == start_measure_idx),
                        is_tied_end=(measure_idx == end_measure_idx),
                        tie_group_id=tie_group_counter
                    )
                    new_notes.append(tied_note)
        
        self.notes = new_notes
        
        # Re-organize measures with tied notes
        self.organize_into_measures()
    
    def infer_expression_marks(self) -> None:
        """
        Automatically infer expression markings (dynamics, articulation) from note data.
        Uses velocity for dynamics and duration patterns for articulation.
        """
        if not self.notes:
            return
        
        for note in self.notes:
            # Infer dynamic from velocity if not already set
            if note.dynamic is None:
                note.dynamic = note.infer_dynamic_from_velocity()
            
            # Infer articulation from duration if not already set
            if note.articulation is None and note.duration_beats is not None:
                # Import here to avoid circular dependency
                from .note import Articulation
                
                # Very short notes -> staccato
                if note.duration_beats < 0.3:
                    note.articulation = Articulation.STACCATO
                # Medium notes with high velocity -> accent
                elif note.velocity > 100 and note.duration_beats < 1.0:
                    note.articulation = Articulation.ACCENT
                # Longer notes -> tenuto/legato
                elif note.duration_beats > 0.9:
                    note.articulation = Articulation.TENUTO
    
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
        result = {
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
        
        if self.measures:
            result['measures'] = [measure.to_dict() for measure in self.measures]
        
        return result
