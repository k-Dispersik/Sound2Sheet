"""
Format converters for note sequences.

Exports note sequences to different formats (JSON, MIDI, MusicXML).
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import mido
from .note_sequence import NoteSequence


class Converter(ABC):
    """Abstract base class for format converters."""
    
    @abstractmethod
    def convert(self, sequence: NoteSequence, output_path: Union[str, Path]) -> None:
        """
        Convert note sequence to target format.
        
        Args:
            sequence: NoteSequence to convert
            output_path: Output file path
        """
        pass


class JSONConverter(Converter):
    """Converts note sequence to JSON format."""
    
    def convert(self, sequence: NoteSequence, output_path: Union[str, Path]) -> None:
        """
        Export note sequence to JSON file.
        
        Args:
            sequence: NoteSequence to export
            output_path: Output JSON file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = sequence.to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load(file_path: Union[str, Path]) -> dict:
        """
        Load JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary with note data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


class MIDIConverter(Converter):
    """Converts note sequence to MIDI format."""
    
    def convert(self, sequence: NoteSequence, output_path: Union[str, Path]) -> None:
        """
        Export note sequence to MIDI file.
        
        Args:
            sequence: NoteSequence to export
            output_path: Output MIDI file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create MIDI file
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        
        # Set tempo
        tempo = mido.bpm2tempo(sequence.tempo)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Set time signature
        numerator, denominator = sequence.time_signature
        track.append(mido.MetaMessage(
            'time_signature',
            numerator=numerator,
            denominator=denominator
        ))
        
        # Convert notes to MIDI messages
        events = self._create_midi_events(sequence.notes)
        
        # Sort by time and add to track
        events.sort(key=lambda x: x[0])
        
        current_time = 0
        for time, msg in events:
            delta = int(midi.ticks_per_beat * (time - current_time) * (sequence.tempo / 60))
            msg.time = max(0, delta)
            track.append(msg)
            current_time = time
        
        # Save MIDI file
        midi.save(output_path)
    
    def _create_midi_events(self, notes):
        """Create MIDI note on/off events from notes."""
        events = []
        for note in notes:
            # Note on
            events.append((
                note.start_time,
                mido.Message('note_on', note=note.pitch, velocity=note.velocity, time=0)
            ))
            # Note off
            events.append((
                note.end_time,
                mido.Message('note_off', note=note.pitch, velocity=0, time=0)
            ))
        return events


class MusicXMLConverter(Converter):
    """Converts note sequence to MusicXML format."""
    
    def convert(self, sequence: NoteSequence, output_path: Union[str, Path]) -> None:
        """
        Export note sequence to MusicXML file.
        
        Args:
            sequence: NoteSequence to export
            output_path: Output MusicXML file path
        """
        try:
            from music21 import stream, note, tempo, meter, key
        except ImportError:
            raise ImportError(
                "music21 library required for MusicXML export. "
                "Install with: pip install music21"
            )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create music21 stream
        score = stream.Score()
        part = stream.Part()
        
        # Add tempo marking
        mm = tempo.MetronomeMark(number=sequence.tempo)
        part.append(mm)
        
        # Add time signature
        time_sig = meter.TimeSignature(f'{sequence.time_signature[0]}/{sequence.time_signature[1]}')
        part.append(time_sig)
        
        # Add key signature
        # Parse key signature (e.g., "C major" -> "C", mode="major")
        key_parts = sequence.key_signature.split()
        if len(key_parts) >= 2:
            tonic = key_parts[0]
            mode_str = key_parts[1]
            key_sig = key.Key(tonic, mode_str)
        else:
            key_sig = key.Key('C')  # Default
        part.append(key_sig)
        
        # Sort notes by start time
        sorted_notes = sorted(sequence.notes, key=lambda n: n.start_time)
        
        # Convert notes
        current_offset = 0.0
        for seq_note in sorted_notes:
            # Create music21 note
            m21_note = note.Note(seq_note.pitch)
            m21_note.quarterLength = seq_note.duration * (sequence.tempo / 60)
            m21_note.volume.velocity = seq_note.velocity
            
            # Set offset
            m21_note.offset = seq_note.start_time * (sequence.tempo / 60)
            
            part.append(m21_note)
        
        score.append(part)
        
        # Write to file
        score.write('musicxml', fp=str(output_path))
