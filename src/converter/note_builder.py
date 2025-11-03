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
    
    def detect_time_signature(self, notes: List[Note], tempo: int) -> tuple:
        """
        Detect time signature from note patterns.
        
        Args:
            notes: List of notes
            tempo: Tempo in BPM
            
        Returns:
            Time signature as (numerator, denominator)
        """
        if not notes or len(notes) < 4:
            return (4, 4)  # Default
        
        beat_duration = 60.0 / tempo
        
        # Group notes by measures (try different measure lengths)
        for measure_beats in [3, 4, 6, 2]:
            measure_duration = measure_beats * beat_duration
            
            # Count notes per measure
            measure_counts = []
            current_measure = 0
            count = 0
            
            for note in sorted(notes, key=lambda n: n.start_time):
                note_measure = int(note.start_time / measure_duration)
                
                if note_measure > current_measure:
                    if count > 0:
                        measure_counts.append(count)
                    current_measure = note_measure
                    count = 1
                else:
                    count += 1
            
            if count > 0:
                measure_counts.append(count)
            
            # Check consistency
            if len(measure_counts) >= 2:
                avg = np.mean(measure_counts)
                std = np.std(measure_counts)
                
                # If notes are evenly distributed, this is likely the time signature
                if std / avg < 0.3:  # Low variation
                    # Determine denominator (usually 4 for most music)
                    return (measure_beats, 4)
        
        return (4, 4)  # Default
    
    def detect_key_signature(self, notes: List[Note]) -> str:
        """
        Detect key signature from pitch content.
        
        Uses Krumhansl-Schmuckler key-finding algorithm.
        
        Args:
            notes: List of notes
            
        Returns:
            Key signature string (e.g., "C major", "A minor")
        """
        if not notes:
            return "C major"
        
        # Major key profiles (Krumhansl-Kessler)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        
        # Minor key profiles
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Count pitch class occurrences
        pitch_class_counts = np.zeros(12)
        for note in notes:
            pitch_class = note.pitch % 12
            pitch_class_counts[pitch_class] += 1
        
        # Normalize
        if pitch_class_counts.sum() > 0:
            pitch_class_counts = pitch_class_counts / pitch_class_counts.sum()
        
        # Calculate correlation with each key
        best_correlation = -1
        best_key = "C major"
        
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for tonic in range(12):
            # Rotate profiles to match tonic
            major_rotated = np.roll(major_profile, tonic)
            minor_rotated = np.roll(minor_profile, tonic)
            
            # Calculate correlations
            major_corr = np.corrcoef(pitch_class_counts, major_rotated)[0, 1]
            minor_corr = np.corrcoef(pitch_class_counts, minor_rotated)[0, 1]
            
            # Check major
            if major_corr > best_correlation:
                best_correlation = major_corr
                best_key = f"{key_names[tonic]} major"
            
            # Check minor
            if minor_corr > best_correlation:
                best_correlation = minor_corr
                best_key = f"{key_names[tonic]} minor"
        
        return best_key
    
    def build_with_auto_detection(
        self,
        note_events: List[Dict[str, Any]],
        detect_tempo: bool = True,
        detect_time_sig: bool = True,
        detect_key: bool = True
    ) -> NoteSequence:
        """
        Build sequence with automatic detection of musical parameters.
        
        Args:
            note_events: List of note events
            detect_tempo: Auto-detect tempo
            detect_time_sig: Auto-detect time signature
            detect_key: Auto-detect key signature
            
        Returns:
            NoteSequence with detected parameters
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
            return NoteSequence()
        
        # Detect tempo
        tempo = self.quantizer.detect_tempo(notes) if detect_tempo else 120
        
        # Detect time signature
        time_signature = self.detect_time_signature(notes, tempo) if detect_time_sig else (4, 4)
        
        # Detect key signature
        key_signature = self.detect_key_signature(notes) if detect_key else "C major"
        
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
