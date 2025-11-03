"""
Note builder for converting model predictions to musical notation.

Orchestrates note sequence construction from model predictions.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .note import Note
from .note_sequence import NoteSequence
from .quantizer import Quantizer, QuantizationConfig


class NoteBuilder:
    """
    Builds structured note sequences from model predictions.
    
    Orchestrates the conversion pipeline:
    1. Parse predictions to notes
    2. Detect tempo (optional)
    3. Quantize timing
    4. Validate musical correctness
    5. Build final sequence
    """
    
    def __init__(self, quantization_config: Optional[QuantizationConfig] = None):
        """
        Initialize note builder.
        
        Args:
            quantization_config: Configuration for timing quantization
        """
        self.quantization_config = quantization_config or QuantizationConfig()
        self.quantizer = Quantizer(self.quantization_config)
    
    def build_from_predictions(
        self,
        predictions: List[int],
        tempo: Optional[int] = None,
        time_signature: tuple = (4, 4),
        key_signature: str = "C major",
        default_duration: float = 0.5,
        default_velocity: int = 80
    ) -> NoteSequence:
        """
        Build note sequence from model predictions.
        
        Args:
            predictions: List of predicted note tokens (MIDI numbers)
            tempo: Tempo in BPM (auto-detected if None)
            time_signature: Time signature as (numerator, denominator)
            key_signature: Key signature string
            default_duration: Default note duration in seconds
            default_velocity: Default MIDI velocity
            
        Returns:
            NoteSequence with quantized notes
        """
        # Parse predictions to notes
        notes = self._parse_predictions(predictions, default_duration, default_velocity)
        
        if not notes:
            return NoteSequence(
                tempo=tempo or 120,
                time_signature=time_signature,
                key_signature=key_signature
            )
        
        # Detect tempo if not provided
        if tempo is None and self.quantization_config.auto_tempo_detection:
            tempo = self.quantizer.detect_tempo(notes)
        else:
            tempo = tempo or 120
        
        # Quantize timing
        quantized_notes = self.quantizer.quantize(notes, tempo)
        
        # Create sequence
        sequence = NoteSequence(
            notes=quantized_notes,
            tempo=tempo,
            time_signature=time_signature,
            key_signature=key_signature
        )
        
        # Sort notes by time
        sequence.sort_by_time()
        
        return sequence
    
    def _parse_predictions(
        self,
        predictions: List[int],
        default_duration: float,
        default_velocity: int
    ) -> List[Note]:
        """
        Parse model predictions to Note objects.
        
        Args:
            predictions: List of MIDI note numbers
            default_duration: Default note duration
            default_velocity: Default velocity
            
        Returns:
            List of Note objects
        """
        notes = []
        current_time = 0.0
        
        for token in predictions:
            # Skip special tokens (pad, sos, eos, unk)
            if token == 0 or token >= 89:  # pad, sos, eos, unk
                continue
            
            # Convert token to MIDI note (tokens 1-88 map to MIDI 21-108)
            midi_note = token + 20  # token 1 = MIDI 21 (A0)
            
            # Create note
            note = Note(
                pitch=midi_note,
                start_time=current_time,
                duration=default_duration,
                velocity=default_velocity
            )
            notes.append(note)
            
            # Advance time
            current_time += default_duration
        
        return notes
    
    def build_from_timestamps(
        self,
        note_events: List[Dict[str, Any]],
        tempo: Optional[int] = None,
        time_signature: tuple = (4, 4),
        key_signature: str = "C major"
    ) -> NoteSequence:
        """
        Build note sequence from timestamped note events.
        
        Args:
            note_events: List of dicts with 'pitch', 'start_time', 'duration', etc.
            tempo: Tempo in BPM (auto-detected if None)
            time_signature: Time signature
            key_signature: Key signature
            
        Returns:
            NoteSequence with quantized notes
        """
        # Parse events to notes
        notes = [
            Note(
                pitch=event['pitch'],
                start_time=event['start_time'],
                duration=event['duration'],
                velocity=event.get('velocity', 80),
                confidence=event.get('confidence')
            )
            for event in note_events
        ]
        
        if not notes:
            return NoteSequence(
                tempo=tempo or 120,
                time_signature=time_signature,
                key_signature=key_signature
            )
        
        # Detect tempo if needed
        if tempo is None and self.quantization_config.auto_tempo_detection:
            tempo = self.quantizer.detect_tempo(notes)
        else:
            tempo = tempo or 120
        
        # Quantize timing
        quantized_notes = self.quantizer.quantize(notes, tempo)
        
        # Create sequence
        sequence = NoteSequence(
            notes=quantized_notes,
            tempo=tempo,
            time_signature=time_signature,
            key_signature=key_signature
        )
        
        sequence.sort_by_time()
        
        return sequence
    
    def add_rests(self, sequence: NoteSequence, min_rest_duration: float = 0.25) -> NoteSequence:
        """
        Add rest markers between notes (metadata only, not actual notes).
        
        Args:
            sequence: Input note sequence
            min_rest_duration: Minimum duration to consider as rest
            
        Returns:
            Sequence with rest metadata
        """
        if not sequence.notes:
            return sequence
        
        rests = []
        sorted_notes = sorted(sequence.notes, key=lambda n: n.start_time)
        
        for i in range(len(sorted_notes) - 1):
            current_end = sorted_notes[i].end_time
            next_start = sorted_notes[i + 1].start_time
            gap = next_start - current_end
            
            if gap >= min_rest_duration:
                rests.append({
                    'start_time': current_end,
                    'duration': gap
                })
        
        if rests:
            sequence.metadata['rests'] = rests
        
        return sequence
    
    def validate_sequence(self, sequence: NoteSequence) -> bool:
        """
        Validate musical correctness of sequence.
        
        Args:
            sequence: Note sequence to validate
            
        Returns:
            True if valid, False otherwise
        """
        errors = sequence.validate()
        return len(errors) == 0
