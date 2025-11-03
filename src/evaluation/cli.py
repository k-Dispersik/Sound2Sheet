"""
Command-line interface for evaluation system.

Provides commands for evaluating model predictions and generating reports.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.evaluation import (
    Evaluator,
    EvaluationConfig,
    ReportGenerator,
    ReportFormat,
    Visualizer
)
from src.evaluation.metrics import Note


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_notes_from_json(json_path: str) -> List[Note]:
    """
    Load notes from JSON file.
    
    Expected format:
    {
        "notes": [
            {"pitch": 60, "onset": 0.0, "offset": 1.0, "velocity": 64},
            ...
        ]
    }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    notes = []
    for note_dict in data.get('notes', []):
        notes.append(Note(
            pitch=note_dict['pitch'],
            onset=note_dict['onset'],
            offset=note_dict['offset'],
            velocity=note_dict.get('velocity', 64)
        ))
    
    return notes


def load_evaluation_samples(manifest_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation samples from manifest file.
    
    Expected format:
    {
        "samples": [
            {
                "sample_id": "test_001",
                "predicted_path": "path/to/predicted.json",
                "ground_truth_path": "path/to/ground_truth.json",
                "predicted_tempo": 120.0,
                "ground_truth_tempo": 120.0,
                "metadata": {"duration": 3.0}
            },
            ...
        ]
    }
    """
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    samples = []
    for sample_dict in data.get('samples', []):
        # Load notes from files
        predicted_notes = load_notes_from_json(sample_dict['predicted_path'])
        ground_truth_notes = load_notes_from_json(sample_dict['ground_truth_path'])
        
        samples.append({
            'sample_id': sample_dict['sample_id'],
            'predicted_notes': predicted_notes,
            'ground_truth_notes': ground_truth_notes,
            'predicted_tempo': sample_dict.get('predicted_tempo'),
            'ground_truth_tempo': sample_dict.get('ground_truth_tempo'),
            'metadata': sample_dict.get('metadata', {})
        })
    
    return samples


def cmd_evaluate(args: argparse.Namespace) -> None:
    """
    Evaluate predictions against ground truth.
    
    Args:
        args: Command line arguments
    """
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading samples from {args.manifest}")
    
    # Load samples
    samples = load_evaluation_samples(args.manifest)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Create evaluator
    config = EvaluationConfig(
        onset_tolerance=args.onset_tolerance,
        offset_tolerance=args.offset_tolerance,
        pitch_tolerance=args.pitch_tolerance
    )
    evaluator = Evaluator(config)
    
    # Progress callback
    def progress(current, total):
        logger.info(f"Progress: {current}/{total} samples evaluated")
    
    # Evaluate
    logger.info("Starting evaluation...")
    evaluator.evaluate_batch(samples, progress_callback=progress)
    
    # Print summary
    print("\n" + evaluator.get_summary())
    
    # Save results
    if args.output:
        evaluator.save_results(args.output)
        logger.info(f"Saved results to {args.output}")


def cmd_report(args: argparse.Namespace) -> None:
    """
    Generate evaluation report.
    
    Args:
        args: Command line arguments
    """
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load results
    logger.info(f"Loading results from {args.results}")
    evaluator = Evaluator()
    evaluator.load_results(args.results)
    
    # Determine format
    format_map = {
        'csv': ReportFormat.CSV,
        'json': ReportFormat.JSON,
    }
    report_format = format_map.get(args.format, ReportFormat.JSON)
    
    # Generate report
    logger.info(f"Generating {args.format} report...")
    reporter = ReportGenerator(evaluator)
    reporter.generate_report(
        output_path=args.output,
        format=report_format
    )
    
    logger.info(f"Report saved to {args.output}")


def cmd_visualize(args: argparse.Namespace) -> None:
    """
    Generate visualizations from results.
    
    Args:
        args: Command line arguments
    """
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load results
    logger.info(f"Loading results from {args.results}")
    evaluator = Evaluator()
    evaluator.load_results(args.results)
    
    visualizer = Visualizer(evaluator)
    
    # Generate visualizations
    if args.dashboard:
        logger.info("Generating dashboard...")
        visualizer.create_dashboard(args.dashboard)
        logger.info(f"Dashboard saved to {args.dashboard}")
    
    if args.confusion_matrix:
        logger.info("Generating confusion matrix...")
        visualizer.plot_confusion_matrix(args.confusion_matrix)
        logger.info(f"Confusion matrix saved to {args.confusion_matrix}")
    
    if args.metrics_over_time:
        logger.info("Generating metrics over time plot...")
        visualizer.plot_metrics_over_time(args.metrics_over_time)
        logger.info(f"Metrics plot saved to {args.metrics_over_time}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Sound2Sheet Evaluation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate predictions
  python -m src.evaluation.cli evaluate \\
      --manifest samples.json \\
      --output results.json

  # Generate HTML report
  python -m src.evaluation.cli report \\
      --results results.json \\
      --output report.html \\
      --format html

  # Create visualizations
  python -m src.evaluation.cli visualize \\
      --results results.json \\
      --dashboard dashboard.png \\
      --confusion-matrix confusion.png
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate predictions against ground truth'
    )
    evaluate_parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help='Path to evaluation manifest JSON file'
    )
    evaluate_parser.add_argument(
        '--output',
        type=str,
        help='Path to save evaluation results JSON'
    )
    evaluate_parser.add_argument(
        '--onset-tolerance',
        type=float,
        default=0.05,
        help='Onset tolerance in seconds (default: 0.05)'
    )
    evaluate_parser.add_argument(
        '--offset-tolerance',
        type=float,
        default=0.05,
        help='Offset tolerance in seconds (default: 0.05)'
    )
    evaluate_parser.add_argument(
        '--pitch-tolerance',
        type=int,
        default=0,
        help='Pitch tolerance in semitones (default: 0)'
    )
    
    # Report command
    report_parser = subparsers.add_parser(
        'report',
        help='Generate evaluation report'
    )
    report_parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to evaluation results JSON'
    )
    report_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save report'
    )
    report_parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'json'],
        default='json',
        help='Report format (default: json)'
    )
    
    # Visualize command
    visualize_parser = subparsers.add_parser(
        'visualize',
        help='Generate visualizations'
    )
    visualize_parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to evaluation results JSON'
    )
    visualize_parser.add_argument(
        '--dashboard',
        type=str,
        help='Path to save dashboard image'
    )
    visualize_parser.add_argument(
        '--confusion-matrix',
        type=str,
        help='Path to save confusion matrix image'
    )
    visualize_parser.add_argument(
        '--metrics-over-time',
        type=str,
        help='Path to save metrics over time plot'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'report':
        cmd_report(args)
    elif args.command == 'visualize':
        cmd_visualize(args)


if __name__ == '__main__':
    main()
