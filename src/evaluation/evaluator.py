"""
Evaluator for batch evaluation and statistical analysis.

Orchestrates the evaluation process for multiple samples and provides
statistical summaries and aggregated metrics.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import numpy as np
from collections import defaultdict

from .metrics import MetricCalculator, EvaluationMetrics, Note


@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation process.
    
    Attributes:
        onset_tolerance: Tolerance for onset matching (seconds)
        offset_tolerance: Tolerance for offset matching (seconds)
        pitch_tolerance: Tolerance for pitch matching (semitones)
        min_note_duration: Minimum note duration to consider (seconds)
        max_note_duration: Maximum note duration to consider (seconds)
        tempo_tolerance: Tolerance for tempo matching (BPM)
    """
    onset_tolerance: float = 0.05
    offset_tolerance: float = 0.05
    pitch_tolerance: int = 0
    min_note_duration: float = 0.01
    max_note_duration: float = 10.0
    tempo_tolerance: float = 5.0


@dataclass
class SampleEvaluation:
    """
    Evaluation results for a single sample.
    
    Attributes:
        sample_id: Identifier for the sample
        metrics: Evaluation metrics for this sample
        predicted_notes: List of predicted notes
        ground_truth_notes: List of ground truth notes
        metadata: Additional metadata (e.g., audio path, duration)
    """
    sample_id: str
    metrics: EvaluationMetrics
    predicted_notes: List[Note]
    ground_truth_notes: List[Note]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sample_id': self.sample_id,
            'metrics': self.metrics.to_dict(),
            'num_predicted': len(self.predicted_notes),
            'num_ground_truth': len(self.ground_truth_notes),
            'metadata': self.metadata,
        }


@dataclass
class AggregatedMetrics:
    """
    Aggregated metrics across multiple samples.
    
    Contains mean, std, min, max for all metrics.
    """
    mean_metrics: EvaluationMetrics
    std_metrics: EvaluationMetrics
    min_metrics: EvaluationMetrics
    max_metrics: EvaluationMetrics
    num_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'num_samples': self.num_samples,
            'mean': self.mean_metrics.to_dict(),
            'std': self.std_metrics.to_dict(),
            'min': self.min_metrics.to_dict(),
            'max': self.max_metrics.to_dict(),
        }


class Evaluator:
    """
    Orchestrates evaluation process for transcription model.
    
    Handles batch evaluation, statistical analysis, and results aggregation.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.metric_calculator = MetricCalculator(
            onset_tolerance=self.config.onset_tolerance,
            offset_tolerance=self.config.offset_tolerance,
            pitch_tolerance=self.config.pitch_tolerance
        )
        self.logger = logging.getLogger(__name__)
        self.sample_results: List[SampleEvaluation] = []
    
    def evaluate_sample(
        self,
        sample_id: str,
        predicted_notes: List[Note],
        ground_truth_notes: List[Note],
        predicted_tempo: Optional[float] = None,
        ground_truth_tempo: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SampleEvaluation:
        """
        Evaluate a single sample.
        
        Args:
            sample_id: Unique identifier for the sample
            predicted_notes: List of predicted notes
            ground_truth_notes: List of ground truth notes
            predicted_tempo: Predicted tempo in BPM
            ground_truth_tempo: Ground truth tempo in BPM
            metadata: Additional metadata
            
        Returns:
            SampleEvaluation with metrics and results
        """
        # Filter notes by duration if needed
        predicted_notes = self._filter_notes_by_duration(predicted_notes)
        ground_truth_notes = self._filter_notes_by_duration(ground_truth_notes)
        
        # Calculate metrics
        metrics = self.metric_calculator.calculate_metrics(
            predicted_notes=predicted_notes,
            ground_truth_notes=ground_truth_notes,
            predicted_tempo=predicted_tempo,
            ground_truth_tempo=ground_truth_tempo
        )
        
        # Create evaluation result
        result = SampleEvaluation(
            sample_id=sample_id,
            metrics=metrics,
            predicted_notes=predicted_notes,
            ground_truth_notes=ground_truth_notes,
            metadata=metadata or {}
        )
        
        # Store result
        self.sample_results.append(result)
        
        self.logger.info(
            f"Evaluated {sample_id}: "
            f"Accuracy={metrics.note_accuracy:.3f}, "
            f"F1={metrics.onset_f1:.3f}"
        )
        
        return result
    
    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[SampleEvaluation]:
        """
        Evaluate multiple samples in batch.
        
        Args:
            samples: List of sample dictionaries with keys:
                - sample_id: str
                - predicted_notes: List[Note]
                - ground_truth_notes: List[Note]
                - predicted_tempo: Optional[float]
                - ground_truth_tempo: Optional[float]
                - metadata: Optional[Dict]
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of SampleEvaluation results
        """
        results = []
        total = len(samples)
        
        for i, sample in enumerate(samples):
            try:
                result = self.evaluate_sample(
                    sample_id=sample['sample_id'],
                    predicted_notes=sample['predicted_notes'],
                    ground_truth_notes=sample['ground_truth_notes'],
                    predicted_tempo=sample.get('predicted_tempo'),
                    ground_truth_tempo=sample.get('ground_truth_tempo'),
                    metadata=sample.get('metadata')
                )
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, total)
                    
            except Exception as e:
                self.logger.error(
                    f"Error evaluating sample {sample.get('sample_id', 'unknown')}: {e}"
                )
        
        return results
    
    def get_aggregated_metrics(self) -> AggregatedMetrics:
        """
        Calculate aggregated statistics across all evaluated samples.
        
        Returns:
            AggregatedMetrics with mean, std, min, max
        """
        if not self.sample_results:
            self.logger.warning("No samples evaluated yet")
            return AggregatedMetrics(
                mean_metrics=EvaluationMetrics(),
                std_metrics=EvaluationMetrics(),
                min_metrics=EvaluationMetrics(),
                max_metrics=EvaluationMetrics(),
                num_samples=0
            )
        
        # Collect all metric values
        metric_values = defaultdict(list)
        
        for result in self.sample_results:
            metrics_dict = result.metrics.to_dict()
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)) and key != 'pitch_errors':
                    metric_values[key].append(value)
        
        # Calculate statistics
        mean_metrics = EvaluationMetrics()
        std_metrics = EvaluationMetrics()
        min_metrics = EvaluationMetrics()
        max_metrics = EvaluationMetrics()
        
        for key, values in metric_values.items():
            if values:
                setattr(mean_metrics, key, float(np.mean(values)))
                setattr(std_metrics, key, float(np.std(values)))
                setattr(min_metrics, key, float(np.min(values)))
                setattr(max_metrics, key, float(np.max(values)))
        
        return AggregatedMetrics(
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            min_metrics=min_metrics,
            max_metrics=max_metrics,
            num_samples=len(self.sample_results)
        )
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """
        Perform detailed error analysis across all samples.
        
        Returns:
            Dictionary with error statistics and distributions
        """
        if not self.sample_results:
            return {}
        
        analysis = {
            'total_samples': len(self.sample_results),
            'total_predicted_notes': sum(
                len(r.predicted_notes) for r in self.sample_results
            ),
            'total_ground_truth_notes': sum(
                len(r.ground_truth_notes) for r in self.sample_results
            ),
            'total_true_positives': sum(
                r.metrics.true_positives for r in self.sample_results
            ),
            'total_false_positives': sum(
                r.metrics.false_positives for r in self.sample_results
            ),
            'total_false_negatives': sum(
                r.metrics.false_negatives for r in self.sample_results
            ),
        }
        
        # Pitch error distribution
        pitch_error_dist = defaultdict(int)
        for result in self.sample_results:
            for pitch, count in result.metrics.pitch_errors.items():
                pitch_error_dist[pitch] += count
        
        analysis['pitch_error_distribution'] = dict(pitch_error_dist)
        
        # Timing error distribution
        timing_errors = [
            r.metrics.timing_error_mean
            for r in self.sample_results
            if r.metrics.timing_error_mean > 0
        ]
        
        if timing_errors:
            analysis['timing_error_distribution'] = {
                'mean': float(np.mean(timing_errors)),
                'std': float(np.std(timing_errors)),
                'min': float(np.min(timing_errors)),
                'max': float(np.max(timing_errors)),
                'median': float(np.median(timing_errors)),
            }
        
        return analysis
    
    def save_results(self, output_path: str) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        results_data = {
            'config': {
                'onset_tolerance': self.config.onset_tolerance,
                'offset_tolerance': self.config.offset_tolerance,
                'pitch_tolerance': self.config.pitch_tolerance,
            },
            'aggregated_metrics': self.get_aggregated_metrics().to_dict(),
            'error_analysis': self.get_error_analysis(),
            'samples': [r.to_dict() for r in self.sample_results],
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Saved evaluation results to {output_path}")
    
    def load_results(self, input_path: str) -> None:
        """
        Load evaluation results from JSON file.
        
        Args:
            input_path: Path to input JSON file
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load configuration
        config_data = data.get('config', {})
        self.config = EvaluationConfig(**config_data)
        
        # Note: Cannot fully reconstruct sample results without Note objects
        # This method is mainly for loading aggregated metrics
        
        self.logger.info(f"Loaded evaluation results from {input_path}")
    
    def clear_results(self) -> None:
        """Clear all evaluation results."""
        self.sample_results.clear()
        self.logger.info("Cleared all evaluation results")
    
    def _filter_notes_by_duration(self, notes: List[Note]) -> List[Note]:
        """Filter notes by duration constraints."""
        filtered = []
        for note in notes:
            duration = note.offset - note.onset
            if self.config.min_note_duration <= duration <= self.config.max_note_duration:
                filtered.append(note)
        return filtered
    
    def get_summary(self) -> str:
        """
        Get human-readable summary of evaluation results.
        
        Returns:
            Formatted summary string
        """
        if not self.sample_results:
            return "No evaluation results available"
        
        agg = self.get_aggregated_metrics()
        error_analysis = self.get_error_analysis()
        
        summary = [
            "=" * 60,
            "EVALUATION SUMMARY",
            "=" * 60,
            f"Number of samples: {agg.num_samples}",
            f"Total predicted notes: {error_analysis['total_predicted_notes']}",
            f"Total ground truth notes: {error_analysis['total_ground_truth_notes']}",
            "",
            "AGGREGATED METRICS (Mean ± Std):",
            "-" * 60,
            f"Note Accuracy:     {agg.mean_metrics.note_accuracy:.3f} ± {agg.std_metrics.note_accuracy:.3f}",
            f"Onset F1:          {agg.mean_metrics.onset_f1:.3f} ± {agg.std_metrics.onset_f1:.3f}",
            f"Offset F1:         {agg.mean_metrics.offset_f1:.3f} ± {agg.std_metrics.offset_f1:.3f}",
            f"Pitch F1:          {agg.mean_metrics.pitch_f1:.3f} ± {agg.std_metrics.pitch_f1:.3f}",
            f"Timing Error (ms): {agg.mean_metrics.timing_error_mean*1000:.1f} ± {agg.std_metrics.timing_error_mean*1000:.1f}",
            "",
            "ERROR ANALYSIS:",
            "-" * 60,
            f"True Positives:    {error_analysis['total_true_positives']}",
            f"False Positives:   {error_analysis['total_false_positives']}",
            f"False Negatives:   {error_analysis['total_false_negatives']}",
            "=" * 60,
        ]
        
        return "\n".join(summary)
