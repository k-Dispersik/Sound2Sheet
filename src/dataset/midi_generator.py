"""
MIDI generation module for creating synthetic musical datasets.

This module provides functionality for generating realistic MIDI sequences
with various musical characteristics including chord progressions, melodies,
rhythmic patterns, and different complexity levels.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
from enum import Enum
import random
import logging
from dataclasses import dataclass
from midiutil import MIDIFile
import numpy as np


class ComplexityLevel(Enum):
    """Complexity levels for generated music."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class TimeSignature(Enum):
    """Supported time signatures."""
    FOUR_FOUR = (4, 4)
    THREE_FOUR = (3, 4)
    SIX_EIGHT = (6, 8)
    TWO_FOUR = (2, 4)


class KeySignature(Enum):
    """Supported key signatures with their tonic notes."""
    # Major keys with sharps
    C_MAJOR = (0, "major")
    G_MAJOR = (7, "major")
    D_MAJOR = (2, "major")
    A_MAJOR = (9, "major")
    E_MAJOR = (4, "major")
    B_MAJOR = (11, "major")
    F_SHARP_MAJOR = (6, "major")
    C_SHARP_MAJOR = (1, "major")
    
    # Major keys with flats
    F_MAJOR = (5, "major")
    B_FLAT_MAJOR = (10, "major")
    E_FLAT_MAJOR = (3, "major")
    A_FLAT_MAJOR = (8, "major")
    D_FLAT_MAJOR = (1, "major")
    G_FLAT_MAJOR = (6, "major")
    
    # Minor keys (natural minor)
    A_MINOR = (9, "minor")
    E_MINOR = (4, "minor")
    B_MINOR = (11, "minor")
    F_SHARP_MINOR = (6, "minor")
    C_SHARP_MINOR = (1, "minor")
    G_SHARP_MINOR = (8, "minor")
    D_MINOR = (2, "minor")
    G_MINOR = (7, "minor")
    C_MINOR = (0, "minor")
    F_MINOR = (5, "minor")
    B_FLAT_MINOR = (10, "minor")
    E_FLAT_MINOR = (3, "minor")


@dataclass
class MIDIConfig:
    """Configuration for MIDI generation."""
    tempo: int = 120  # BPM
    time_signature: TimeSignature = TimeSignature.FOUR_FOUR
    key_signature: KeySignature = KeySignature.C_MAJOR
    num_measures: int = 8
    complexity: ComplexityLevel = ComplexityLevel.INTERMEDIATE
    include_chords: bool = True
    include_melody: bool = True
    velocity_variation: float = 0.2  # 0-1, amount of velocity randomization
    
    # Octave ranges (MIDI note numbers)
    chord_octave_min: int = 1  # Contra octave (C1 = 24)
    chord_octave_max: int = 3  # Third octave (C3 = 48)
    melody_octave_min: int = 3  # Third octave (C3 = 48)
    melody_octave_max: int = 6  # Sixth octave (C6 = 84)


class MIDIGenerator:
    """
    Generator for creating realistic MIDI sequences.
    
    Creates musical MIDI files with realistic chord progressions, melodies,
    and rhythmic patterns suitable for piano transcription training.
    """
    
    # Major scale intervals (semitones from root)
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
    
    # Minor scale intervals (natural minor)
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
    
    # Common chord progressions (as scale degrees)
    CHORD_PROGRESSIONS = {
        ComplexityLevel.BEGINNER: [
            [1, 4, 5, 1],  # I-IV-V-I (classic)
            [1, 5, 6, 4],  # I-V-vi-IV (very popular)
            [1, 4, 1, 5],  # I-IV-I-V
            [1, 6, 4, 5],  # I-vi-IV-V (50s progression)
            [6, 4, 1, 5],  # vi-IV-I-V
        ],
        ComplexityLevel.INTERMEDIATE: [
            [1, 5, 6, 4],  # I-V-vi-IV
            [1, 6, 4, 5],  # I-vi-IV-V
            [2, 5, 1, 1],  # ii-V-I (jazz)
            [1, 4, 2, 5],  # I-IV-ii-V
            [6, 2, 5, 1],  # vi-ii-V-I
            [1, 3, 4, 4],  # I-iii-IV-IV
            [1, 4, 6, 5],  # I-IV-vi-V
            [4, 1, 5, 6],  # IV-I-V-vi (alternative)
        ],
        ComplexityLevel.ADVANCED: [
            [1, 3, 6, 2, 5],  # I-iii-vi-ii-V
            [1, 6, 2, 5, 1],  # I-vi-ii-V-I
            [4, 5, 3, 6],     # IV-V-iii-vi (modal)
            [1, 7, 3, 6, 2, 5, 1],  # I-vii°-iii-vi-ii-V-I
            [2, 5, 1, 6, 2, 5, 1],  # ii-V-I-vi-ii-V-I (extended)
            [1, 4, 7, 3, 6, 2, 5, 1],  # I-IV-vii°-iii-vi-ii-V-I (chromatic)
            [6, 4, 1, 5, 6, 4, 2, 5],  # vi-IV-I-V-vi-IV-ii-V (complex)
            [1, 3, 4, 6, 2, 5, 1],  # I-iii-IV-vi-ii-V-I
        ]
    }
    
    # Rhythmic patterns (as fractions of a beat)
    RHYTHMIC_PATTERNS = {
        ComplexityLevel.BEGINNER: [
            [1.0, 1.0, 1.0, 1.0],  # Quarter notes
            [2.0, 2.0],  # Half notes
            [0.5, 0.5, 1.0, 1.0],  # Mixed eighths and quarters
            [1.0, 1.0, 2.0],  # Quarter + half
            [2.0, 1.0, 1.0],  # Half + quarters
            [4.0],  # Whole note
        ],
        ComplexityLevel.INTERMEDIATE: [
            [0.5, 0.5, 0.5, 0.5, 1.0, 1.0],  # Eighth note pattern
            [1.0, 0.5, 0.5, 1.0, 1.0],  # Syncopation
            [0.75, 0.25, 1.0, 1.0],  # Dotted rhythm
            [0.5, 0.5, 1.0, 0.5, 0.5, 1.0],  # Mixed pattern
            [1.5, 0.5, 1.0, 1.0],  # Dotted quarter
            [0.5, 1.0, 0.5, 1.0, 1.0],  # Syncopated
            [1.0, 0.5, 1.0, 0.5, 1.0],  # Complex syncopation
        ],
        ComplexityLevel.ADVANCED: [
            [0.5, 0.5, 0.25, 0.25, 0.5, 1.0, 1.0],  # Complex rhythm
            [0.33, 0.33, 0.34, 1.0, 1.0],  # Triplets
            [1.0, 0.25, 0.25, 0.5, 1.0, 0.5, 0.5],  # Syncopated pattern
            [0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 1.0, 1.0],  # Sixteenth notes
            [0.33, 0.33, 0.34, 0.5, 0.5, 1.0, 1.0],  # Triplet + eighths
            [0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 1.0, 1.0],  # Mixed complex
            [1.5, 0.25, 0.25, 1.0, 1.0],  # Dotted + sixteenths
            [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0],  # Running sixteenths
        ]
    }
    
    def __init__(self, config: Optional[MIDIConfig] = None):
        """
        Initialize MIDI generator with configuration.
        
        Args:
            config: MIDI generation configuration. If None, uses defaults.
        """
        self.config = config or MIDIConfig()
        self.logger = logging.getLogger(__name__)
        
        # Configure logging if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Set random seed for reproducibility if needed
        self._rng = np.random.default_rng()
    
    def generate(self, output_path: Optional[Path] = None) -> MIDIFile:
        """
        Generate a complete MIDI file with chords and melody.
        
        Args:
            output_path: Path to save MIDI file. If None, returns MIDIFile object only.
            
        Returns:
            MIDIFile object containing the generated music
        """
        self.logger.info(f"Generating MIDI with {self.config.complexity.value} complexity")
        
        # Create MIDI file with 2 tracks (melody and chords)
        num_tracks = 2 if self.config.include_chords and self.config.include_melody else 1
        midi_file = MIDIFile(num_tracks)
        
        # Set tempo and time signature
        time_num, time_den = self.config.time_signature.value
        midi_file.addTempo(0, 0, self.config.tempo)
        midi_file.addTimeSignature(0, 0, time_num, time_den, 24)
        
        track_num = 0
        
        # Generate chord progression
        if self.config.include_chords:
            self.logger.debug("Generating chord progression")
            chord_progression = self._generate_chord_progression()
            self._add_chords_to_midi(midi_file, track_num, chord_progression)
            track_num += 1
        
        # Generate melody
        if self.config.include_melody:
            self.logger.debug("Generating melody")
            melody = self._generate_melody()
            self._add_melody_to_midi(midi_file, track_num, melody)
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                midi_file.writeFile(f)
            
            self.logger.info(f"MIDI file saved to {output_path}")
        
        return midi_file
    
    def _generate_chord_progression(self) -> List[List[int]]:
        """
        Generate a chord progression based on complexity level.
        
        Returns:
            List of chords, where each chord is a list of MIDI note numbers
        """
        # Select a random progression for the complexity level
        progressions = self.CHORD_PROGRESSIONS[self.config.complexity]
        progression_degrees = random.choice(progressions)
        
        # Repeat progression to fill the number of measures
        time_num, _ = self.config.time_signature.value
        beats_per_measure = time_num
        total_beats = self.config.num_measures * beats_per_measure
        
        # Get scale for the key
        tonic, mode = self.config.key_signature.value
        scale = self.MAJOR_SCALE if mode == "major" else self.MINOR_SCALE
        
        # Convert scale degrees to actual MIDI notes
        chords = []
        
        # Calculate base note from configured octave range
        # Use random octave within configured range for variety
        min_midi_note = 12 * (self.config.chord_octave_min + 1)  # +1 because C1=24
        max_midi_note = 12 * (self.config.chord_octave_max + 1)
        
        # Ensure we have enough range for a chord (at least 2 octaves)
        if max_midi_note - min_midi_note < 24:
            max_midi_note = min_midi_note + 24
        
        base_note = random.randint(min_midi_note, max_midi_note - 24)  # Leave room for chord
        
        for degree in progression_degrees:
            # Get root note of chord (scale degree - 1 for 0-indexing)
            root_index = (degree - 1) % len(scale)
            root_note = base_note + tonic + scale[root_index]
            
            # Ensure root is within octave range
            while root_note < min_midi_note:
                root_note += 12
            while root_note > max_midi_note - 12:
                root_note -= 12
            
            # Build triad (root, third, fifth)
            third_index = (root_index + 2) % len(scale)
            fifth_index = (root_index + 4) % len(scale)
            
            third_note = root_note + scale[third_index] - scale[root_index]
            if third_note <= root_note:
                third_note += 12
                
            fifth_note = root_note + scale[fifth_index] - scale[root_index]
            if fifth_note <= third_note:
                fifth_note += 12
            
            chord = [root_note, third_note, fifth_note]
            chords.append(chord)
        
        self.logger.debug(f"Generated {len(chords)} chords in octave range {self.config.chord_octave_min}-{self.config.chord_octave_max}")
        return chords
    
    def _generate_melody(self) -> List[Tuple[int, float, float]]:
        """
        Generate a melody based on complexity level.
        
        Returns:
            List of tuples (note, duration, start_time) in beats
        """
        # Get scale for the key
        tonic, mode = self.config.key_signature.value
        scale = self.MAJOR_SCALE if mode == "major" else self.MINOR_SCALE
        
        # Build scale notes across configured octave range
        scale_notes = []
        min_midi_note = 12 * (self.config.melody_octave_min + 1)  # +1 because C1=24
        max_midi_note = 12 * (self.config.melody_octave_max + 2)  # +2 to include full octave
        
        # Generate all scale notes in the range
        for octave_offset in range(0, (max_midi_note - min_midi_note) // 12 + 1):
            for interval in scale:
                note = min_midi_note + tonic + interval + (octave_offset * 12)
                if min_midi_note <= note <= max_midi_note:
                    scale_notes.append(note)
        
        # Get rhythmic pattern for complexity
        patterns = self.RHYTHMIC_PATTERNS[self.config.complexity]
        
        # Generate melody
        melody = []
        current_time = 0.0
        time_num, _ = self.config.time_signature.value
        total_beats = self.config.num_measures * time_num
        
        # Keep track of previous note for melodic contour
        prev_note = random.choice(scale_notes)
        
        while current_time < total_beats:
            # Select random rhythmic pattern
            pattern = random.choice(patterns)
            
            for duration in pattern:
                if current_time >= total_beats:
                    break
                
                # Select note from scale with melodic contour preference
                # Prefer stepwise motion (nearby notes) 60% of the time
                if random.random() < 0.6 and len(scale_notes) > 1:
                    # Find nearby notes (within 5 semitones)
                    nearby = [n for n in scale_notes if abs(n - prev_note) <= 5]
                    if nearby:
                        note = random.choice(nearby)
                    else:
                        note = random.choice(scale_notes)
                else:
                    # Jump to any note in the range
                    note = random.choice(scale_notes)
                
                # Add some rests (adjusted by complexity)
                rest_chance = {
                    ComplexityLevel.BEGINNER: 0.15,
                    ComplexityLevel.INTERMEDIATE: 0.1,
                    ComplexityLevel.ADVANCED: 0.05
                }[self.config.complexity]
                
                if random.random() > rest_chance:
                    melody.append((note, duration, current_time))
                    prev_note = note
                
                current_time += duration
        
        self.logger.debug(f"Generated melody with {len(melody)} notes in octave range {self.config.melody_octave_min}-{self.config.melody_octave_max}")
        return melody
    
    def _add_chords_to_midi(
        self, 
        midi_file: MIDIFile, 
        track: int, 
        chords: List[List[int]]
    ) -> None:
        """Add chord progression to MIDI file."""
        time_num, _ = self.config.time_signature.value
        beats_per_chord = time_num  # One chord per measure for now
        
        current_time = 0.0
        base_velocity = 80
        
        for chord in chords:
            for note in chord:
                # Add velocity variation
                velocity = int(base_velocity * (1 + self._rng.normal(0, self.config.velocity_variation)))
                velocity = np.clip(velocity, 40, 120)
                
                midi_file.addNote(
                    track=track,
                    channel=0,
                    pitch=note,
                    time=current_time,
                    duration=beats_per_chord,
                    volume=velocity
                )
            
            current_time += beats_per_chord
    
    def _add_melody_to_midi(
        self,
        midi_file: MIDIFile,
        track: int,
        melody: List[Tuple[int, float, float]]
    ) -> None:
        """Add melody to MIDI file."""
        base_velocity = 100
        
        for note, duration, start_time in melody:
            # Add velocity variation
            velocity = int(base_velocity * (1 + self._rng.normal(0, self.config.velocity_variation)))
            velocity = np.clip(velocity, 60, 127)
            
            midi_file.addNote(
                track=track,
                channel=0,
                pitch=note,
                time=start_time,
                duration=duration,
                volume=velocity
            )
    
    def generate_batch(
        self,
        num_files: int,
        output_dir: Path,
        prefix: str = "midi"
    ) -> List[Path]:
        """
        Generate multiple MIDI files with varied configurations.
        
        Args:
            num_files: Number of MIDI files to generate
            output_dir: Directory to save generated files
            prefix: Prefix for generated filenames
            
        Returns:
            List of paths to generated MIDI files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        for i in range(num_files):
            # Vary configuration for diversity
            self._randomize_config()
            
            output_path = output_dir / f"{prefix}_{i:04d}.mid"
            self.generate(output_path)
            generated_files.append(output_path)
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Generated {i + 1}/{num_files} MIDI files")
        
        self.logger.info(f"Batch generation complete: {len(generated_files)} files")
        return generated_files
    
    def _randomize_config(self) -> None:
        """Randomize configuration parameters for diversity."""
        # Random tempo (60-180 BPM)
        self.config.tempo = random.randint(60, 180)
        
        # Random time signature
        self.config.time_signature = random.choice(list(TimeSignature))
        
        # Random key signature (all 24 major and minor keys)
        self.config.key_signature = random.choice(list(KeySignature))
        
        # Random number of measures (4-16)
        self.config.num_measures = random.randint(4, 16)
        
        # Occasionally change complexity
        if random.random() < 0.2:
            self.config.complexity = random.choice(list(ComplexityLevel))
        
        # Vary octave ranges for diversity
        if random.random() < 0.3:
            # Randomly adjust chord octave range
            self.config.chord_octave_min = random.randint(1, 2)
            self.config.chord_octave_max = random.randint(self.config.chord_octave_min + 1, 4)
            
            # Randomly adjust melody octave range
            self.config.melody_octave_min = random.randint(3, 5)
            self.config.melody_octave_max = random.randint(self.config.melody_octave_min + 1, 7)
