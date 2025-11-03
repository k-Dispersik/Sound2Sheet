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
    """Converts note sequence to MusicXML format (placeholder)."""
    
    def convert(self, sequence: NoteSequence, output_path: Union[str, Path]) -> None:
        """
        Export note sequence to MusicXML file.
        
        Args:
            sequence: NoteSequence to export
            output_path: Output MusicXML file path
            
        Note:
            This is a placeholder implementation. Full MusicXML support
            requires music21 library integration.
        """
        raise NotImplementedError(
            "MusicXML export not yet implemented. "
            "Use JSONConverter or MIDIConverter instead."
        )
