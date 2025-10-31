#!/usr/bin/env python3
"""
Command-line interface for dataset generation.

Usage:
    python -m src.dataset.cli generate --samples 100 --name my_dataset
    python -m src.dataset.cli info /path/to/dataset
"""

import argparse
import sys
from pathlib import Path
import json

from .dataset_generator import DatasetGenerator, DatasetConfig
from .midi_generator import ComplexityLevel


def generate_command(args):
    """Generate a new dataset."""
    # Create configuration
    config = DatasetConfig(
        name=args.name,
        version=args.version,
        total_samples=args.samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        sample_rate=args.sample_rate,
        audio_format=args.audio_format,
        output_dir=Path(args.output_dir)
    )
    
    # Custom complexity distribution if provided
    if args.complexity_dist:
        try:
            dist = {}
            for pair in args.complexity_dist.split(','):
                level, ratio = pair.split(':')
                dist[level.strip()] = float(ratio.strip())
            config.complexity_distribution = dist
        except Exception as e:
            print(f"Error parsing complexity distribution: {e}")
            print("Expected format: beginner:0.5,intermediate:0.3,advanced:0.2")
            return 1
    
    # Initialize generator
    print(f"Initializing dataset generator...")
    generator = DatasetGenerator(config)
    
    # Check if audio synthesis is available
    if not generator.synthesis_available and not args.midi_only:
        print("\nWarning: Audio synthesis not available (missing midi2audio or FluidSynth)")
        print("Generate MIDI files only? (y/n): ", end='')
        
        if not args.yes:
            response = input().lower()
            if response != 'y':
                print("Aborted.")
                return 1
        
        args.midi_only = True
    
    # Generate dataset
    try:
        dataset_dir = generator.generate(generate_audio=not args.midi_only)
        
        print(f"\n✓ Dataset generated successfully!")
        print(f"Location: {dataset_dir}")
        print(f"Total samples: {len(generator.samples)}")
        
        # Print split information
        train_count = sum(1 for s in generator.samples if s.split == 'train')
        val_count = sum(1 for s in generator.samples if s.split == 'val')
        test_count = sum(1 for s in generator.samples if s.split == 'test')
        
        print(f"\nSplits:")
        print(f"  Train: {train_count} samples")
        print(f"  Val:   {val_count} samples")
        print(f"  Test:  {test_count} samples")
        
        if not args.midi_only:
            print(f"\nAudio files: {config.audio_format} @ {config.sample_rate}Hz")
        
        return 0
        
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


def info_command(args):
    """Display information about an existing dataset."""
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        return 1
    
    # Load dataset info
    info_file = dataset_path / 'metadata' / 'dataset_info.json'
    
    if not info_file.exists():
        print(f"Error: Not a valid dataset (missing dataset_info.json)")
        return 1
    
    with open(info_file) as f:
        info = json.load(f)
    
    # Display information
    print(f"Dataset: {info['name']} v{info['version']}")
    print(f"Generated: {info['generated_at']}")
    print(f"Location: {dataset_path}")
    print(f"\nTotal samples: {info['total_samples']}")
    
    print(f"\nSplits:")
    for split, count in info['splits'].items():
        print(f"  {split:5s}: {count:4d} samples")
    
    print(f"\nConfiguration:")
    print(f"  Sample rate: {info['config']['sample_rate']}Hz")
    print(f"  Audio format: {info['config']['audio_format']}")
    
    # Statistics
    if 'statistics' in info:
        stats = info['statistics']
        
        print(f"\nStatistics:")
        
        # Complexity distribution
        if 'complexity' in stats:
            print(f"  Complexity:")
            for level, count in stats['complexity'].items():
                percentage = (count / info['total_samples']) * 100
                print(f"    {level:12s}: {count:3d} ({percentage:5.1f}%)")
        
        # Tempo range
        if 'tempo' in stats:
            tempo = stats['tempo']
            print(f"  Tempo: {tempo['min']}-{tempo['max']} BPM (avg: {tempo['avg']:.1f})")
        
        # Measures
        if 'measures' in stats:
            measures = stats['measures']
            print(f"  Measures: {measures['min']}-{measures['max']} (avg: {measures['avg']:.1f})")
        
        # Duration
        if 'duration' in stats and stats['duration']['total'] > 0:
            duration = stats['duration']
            total_minutes = duration['total'] / 60
            avg_seconds = duration['avg']
            print(f"  Duration: {total_minutes:.1f} min total (avg: {avg_seconds:.1f}s per sample)")
    
    return 0


def validate_command(args):
    """Validate dataset integrity."""
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        return 1
    
    print(f"Validating dataset: {dataset_path}")
    
    errors = []
    warnings = []
    
    # Check metadata files
    metadata_dir = dataset_path / 'metadata'
    required_files = [
        'dataset_info.json',
        'samples.json',
        'train_manifest.json',
        'val_manifest.json',
        'test_manifest.json'
    ]
    
    for filename in required_files:
        filepath = metadata_dir / filename
        if not filepath.exists():
            errors.append(f"Missing metadata file: {filename}")
    
    if errors:
        print("\n✗ Validation failed!")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1
    
    # Load and validate manifests
    generator = DatasetGenerator()
    
    for split in ['train', 'val', 'test']:
        manifest_path = metadata_dir / f'{split}_manifest.json'
        try:
            samples = generator.load_manifest(manifest_path)
            
            # Check files exist
            for sample in samples:
                midi_path = dataset_path / sample.midi_path
                if not midi_path.exists():
                    errors.append(f"Missing MIDI file: {sample.midi_path}")
                
                if sample.audio_path:
                    audio_path = dataset_path / sample.audio_path
                    if not audio_path.exists():
                        warnings.append(f"Missing audio file: {sample.audio_path}")
            
        except Exception as e:
            errors.append(f"Error loading {split} manifest: {e}")
    
    # Print results
    if errors:
        print("\n✗ Validation failed!")
        for error in errors:
            print(f"  ERROR: {error}")
    else:
        print("\n✓ Validation passed!")
    
    if warnings:
        print(f"\nWarnings:")
        for warning in warnings:
            print(f"  WARN: {warning}")
    
    return 1 if errors else 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sound2Sheet Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate small dataset with MIDI only
  %(prog)s generate --samples 100 --name test_dataset --midi-only

  # Generate full dataset with audio
  %(prog)s generate --samples 1000 --name piano_v1

  # Custom split ratios
  %(prog)s generate --samples 500 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

  # Custom complexity distribution
  %(prog)s generate --samples 200 --complexity-dist "beginner:0.5,intermediate:0.3,advanced:0.2"

  # Show dataset information
  %(prog)s info data/datasets/piano_dataset_v1.0.0_20250101_120000

  # Validate dataset
  %(prog)s validate data/datasets/piano_dataset_v1.0.0_20250101_120000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate a new dataset')
    gen_parser.add_argument('--name', default='piano_dataset', help='Dataset name')
    gen_parser.add_argument('--version', default='1.0.0', help='Dataset version')
    gen_parser.add_argument('--samples', type=int, default=1000, help='Total number of samples')
    gen_parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio')
    gen_parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio')
    gen_parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio')
    gen_parser.add_argument('--sample-rate', type=int, default=16000, help='Audio sample rate')
    gen_parser.add_argument('--audio-format', default='wav', help='Audio file format')
    gen_parser.add_argument('--output-dir', default='data/datasets', help='Output directory')
    gen_parser.add_argument('--complexity-dist', help='Complexity distribution (e.g., beginner:0.5,intermediate:0.3,advanced:0.2)')
    gen_parser.add_argument('--midi-only', action='store_true', help='Generate MIDI files only (no audio synthesis)')
    gen_parser.add_argument('-y', '--yes', action='store_true', help='Answer yes to all prompts')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display dataset information')
    info_parser.add_argument('dataset_path', help='Path to dataset directory')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate dataset integrity')
    val_parser.add_argument('dataset_path', help='Path to dataset directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'generate':
        return generate_command(args)
    elif args.command == 'info':
        return info_command(args)
    elif args.command == 'validate':
        return validate_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
