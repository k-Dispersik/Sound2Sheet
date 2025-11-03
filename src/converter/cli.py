"""
CLI tool for note transcription and conversion.

Usage:
    # Convert model predictions to MIDI
    python -m src.converter.cli predict-to-midi \
        --predictions predictions.json \
        --output output.mid \
        --tempo 120
    
    # Convert model predictions to JSON
    python -m src.converter.cli predict-to-json \
        --predictions predictions.json \
        --output output.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .note_builder import NoteBuilder
from .quantizer import QuantizationConfig
from .converter import JSONConverter, MIDIConverter


def load_predictions(file_path: Path) -> List[int]:
    """Load predictions from JSON file."""
    with open(file_path) as f:
        data = json.load(f)
    
    # Support different formats
    if isinstance(data, list):
        return data
    elif 'predictions' in data:
        return data['predictions']
    else:
        raise ValueError("Invalid predictions format. Expected list or dict with 'predictions' key.")


def predict_to_midi(args):
    """Convert predictions to MIDI file."""
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_predictions(Path(args.predictions))
    
    print(f"Building note sequence (tempo={args.tempo}, resolution={args.resolution})...")
    config = QuantizationConfig(
        beat_resolution=args.resolution,
        auto_tempo_detection=args.auto_tempo
    )
    builder = NoteBuilder(config)
    
    sequence = builder.build_from_predictions(
        predictions=predictions,
        tempo=args.tempo if not args.auto_tempo else None,
        time_signature=tuple(map(int, args.time_signature.split('/'))),
        default_duration=args.default_duration,
        default_velocity=args.velocity
    )
    
    print(f"Generated {sequence.note_count} notes, duration: {sequence.total_duration:.2f}s")
    
    # Validate
    if not builder.validate_sequence(sequence):
        print("Warning: Sequence validation failed (overlapping notes detected)")
    
    print(f"Exporting to {args.output}...")
    converter = MIDIConverter()
    converter.convert(sequence, Path(args.output))
    
    print("Done!")


def predict_to_json(args):
    """Convert predictions to JSON file."""
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_predictions(Path(args.predictions))
    
    print(f"Building note sequence (tempo={args.tempo})...")
    config = QuantizationConfig(
        beat_resolution=args.resolution,
        auto_tempo_detection=args.auto_tempo
    )
    builder = NoteBuilder(config)
    
    sequence = builder.build_from_predictions(
        predictions=predictions,
        tempo=args.tempo if not args.auto_tempo else None,
        time_signature=tuple(map(int, args.time_signature.split('/'))),
        default_duration=args.default_duration,
        default_velocity=args.velocity
    )
    
    print(f"Generated {sequence.note_count} notes, duration: {sequence.total_duration:.2f}s")
    
    print(f"Exporting to {args.output}...")
    converter = JSONConverter()
    converter.convert(sequence, Path(args.output))
    
    print("Done!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert model predictions to musical notation formats"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # predict-to-midi command
    midi_parser = subparsers.add_parser(
        'predict-to-midi',
        help='Convert predictions to MIDI file'
    )
    midi_parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions JSON file'
    )
    midi_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output MIDI file path'
    )
    midi_parser.add_argument(
        '--tempo',
        type=int,
        default=120,
        help='Tempo in BPM (default: 120)'
    )
    midi_parser.add_argument(
        '--auto-tempo',
        action='store_true',
        help='Auto-detect tempo from note intervals'
    )
    midi_parser.add_argument(
        '--time-signature',
        type=str,
        default='4/4',
        help='Time signature (default: 4/4)'
    )
    midi_parser.add_argument(
        '--resolution',
        type=int,
        default=16,
        help='Beat resolution (subdivisions per beat, default: 16)'
    )
    midi_parser.add_argument(
        '--default-duration',
        type=float,
        default=0.5,
        help='Default note duration in seconds (default: 0.5)'
    )
    midi_parser.add_argument(
        '--velocity',
        type=int,
        default=80,
        help='MIDI velocity (0-127, default: 80)'
    )
    midi_parser.set_defaults(func=predict_to_midi)
    
    # predict-to-json command
    json_parser = subparsers.add_parser(
        'predict-to-json',
        help='Convert predictions to JSON file'
    )
    json_parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions JSON file'
    )
    json_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path'
    )
    json_parser.add_argument(
        '--tempo',
        type=int,
        default=120,
        help='Tempo in BPM (default: 120)'
    )
    json_parser.add_argument(
        '--auto-tempo',
        action='store_true',
        help='Auto-detect tempo from note intervals'
    )
    json_parser.add_argument(
        '--time-signature',
        type=str,
        default='4/4',
        help='Time signature (default: 4/4)'
    )
    json_parser.add_argument(
        '--resolution',
        type=int,
        default=16,
        help='Beat resolution (default: 16)'
    )
    json_parser.add_argument(
        '--default-duration',
        type=float,
        default=0.5,
        help='Default note duration in seconds (default: 0.5)'
    )
    json_parser.add_argument(
        '--velocity',
        type=int,
        default=80,
        help='MIDI velocity (default: 80)'
    )
    json_parser.set_defaults(func=predict_to_json)
    
    # Parse and run
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
