"""
Visualization tools for evaluation results.

Creates plots and charts for error analysis, confusion matrices,
and performance dashboards.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

from .evaluator import Evaluator, SampleEvaluation
from .metrics import EvaluationMetrics


class Visualizer:
    """
    Create visualizations for evaluation results.
    
    Generates plots including:
    - Confusion matrices
    - Error distribution charts
    - Performance comparison plots
    - Timing error distributions
    - Per-sample metrics
    """
    
    def __init__(self, evaluator: Evaluator):
        """
        Initialize visualizer.
        
        Args:
            evaluator: Evaluator instance with results
        """
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    def create_dashboard(
        self,
        output_path: str,
        figsize: Tuple[int, int] = (16, 12)
    ) -> None:
        """
        Create comprehensive dashboard with multiple plots.
        
        Args:
            output_path: Path to save the dashboard image
            figsize: Figure size in inches
        """
        if not self.evaluator.sample_results:
            self.logger.warning("No results to visualize")
            return
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Metrics comparison bar chart (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_metrics_comparison(ax1)
        
        # 2. Error distribution (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_error_distribution(ax2)
        
        # 3. Per-sample accuracy (middle left, wide)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_per_sample_metrics(ax3)
        
        # 4. Timing error histogram (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_timing_error_histogram(ax4)
        
        # 5. Pitch error distribution (bottom middle)
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_pitch_error_distribution(ax5)
        
        # 6. F1-scores comparison (bottom right)
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_f1_comparison(ax6)
        
        plt.suptitle('Evaluation Dashboard', fontsize=16, fontweight='bold')
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved dashboard to {output_path}")
    
    def _plot_metrics_comparison(self, ax: plt.Axes) -> None:
        """Plot comparison of different metrics."""
        agg = self.evaluator.get_aggregated_metrics()
        
        metrics_names = ['Note\nAccuracy', 'Onset\nF1', 'Offset\nF1', 'Pitch\nF1']
        metrics_values = [
            agg.mean_metrics.note_accuracy,
            agg.mean_metrics.onset_f1,
            agg.mean_metrics.offset_f1,
            agg.mean_metrics.pitch_f1
        ]
        metrics_std = [
            agg.std_metrics.note_accuracy,
            agg.std_metrics.onset_f1,
            agg.std_metrics.offset_f1,
            agg.std_metrics.pitch_f1
        ]
        
        x = np.arange(len(metrics_names))
        bars = ax.bar(x, metrics_values, yerr=metrics_std, capsize=5,
                     color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
                     alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Aggregated Metrics Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    def _plot_error_distribution(self, ax: plt.Axes) -> None:
        """Plot error type distribution (TP, FP, FN)."""
        error_analysis = self.evaluator.get_error_analysis()
        
        labels = ['True\nPositives', 'False\nPositives', 'False\nNegatives']
        values = [
            error_analysis.get('total_true_positives', 0),
            error_analysis.get('total_false_positives', 0),
            error_analysis.get('total_false_negatives', 0)
        ]
        
        colors_pie = ['#2ecc71', '#e74c3c', '#f39c12']
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%',
            colors=colors_pie, startangle=90,
            textprops={'fontsize': 9, 'fontweight': 'bold'}
        )
        
        ax.set_title('Error Distribution', fontweight='bold')
    
    def _plot_per_sample_metrics(self, ax: plt.Axes) -> None:
        """Plot metrics for each sample."""
        samples = self.evaluator.sample_results[:30]  # Limit to 30 for readability
        
        if not samples:
            return
        
        x = np.arange(len(samples))
        width = 0.25
        
        accuracies = [s.metrics.note_accuracy for s in samples]
        onset_f1s = [s.metrics.onset_f1 for s in samples]
        pitch_f1s = [s.metrics.pitch_f1 for s in samples]
        
        ax.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x, onset_f1s, width, label='Onset F1', alpha=0.8)
        ax.bar(x + width, pitch_f1s, width, label='Pitch F1', alpha=0.8)
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Per-Sample Metrics', fontweight='bold')
        ax.set_xticks(x[::5])  # Show every 5th label
        ax.set_xticklabels([s.sample_id[:10] for s in samples[::5]], rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_timing_error_histogram(self, ax: plt.Axes) -> None:
        """Plot distribution of timing errors."""
        timing_errors = [
            s.metrics.timing_error_mean * 1000  # Convert to milliseconds
            for s in self.evaluator.sample_results
            if s.metrics.timing_error_mean > 0
        ]
        
        if not timing_errors:
            ax.text(0.5, 0.5, 'No timing data', ha='center', va='center',
                   transform=ax.transAxes)
            return
        
        ax.hist(timing_errors, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Timing Error (ms)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Timing Error Distribution', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean line
        mean_error = np.mean(timing_errors)
        ax.axvline(mean_error, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_error:.1f}ms')
        ax.legend()
    
    def _plot_pitch_error_distribution(self, ax: plt.Axes) -> None:
        """Plot distribution of pitch errors."""
        error_analysis = self.evaluator.get_error_analysis()
        pitch_errors = error_analysis.get('pitch_error_distribution', {})
        
        if not pitch_errors:
            ax.text(0.5, 0.5, 'No pitch errors', ha='center', va='center',
                   transform=ax.transAxes)
            return
        
        # Get top 10 pitches with most errors
        sorted_errors = sorted(pitch_errors.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if sorted_errors:
            pitches = [f'MIDI {p}' for p, _ in sorted_errors]
            counts = [c for _, c in sorted_errors]
            
            ax.barh(pitches, counts, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Error Count', fontweight='bold')
            ax.set_title('Top 10 Pitch Errors', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
    
    def _plot_f1_comparison(self, ax: plt.Axes) -> None:
        """Plot comparison of precision, recall, and F1 for different aspects."""
        agg = self.evaluator.get_aggregated_metrics()
        
        categories = ['Onset', 'Offset', 'Pitch']
        precision = [
            agg.mean_metrics.onset_precision,
            agg.mean_metrics.offset_precision,
            agg.mean_metrics.pitch_precision
        ]
        recall = [
            agg.mean_metrics.onset_recall,
            agg.mean_metrics.offset_recall,
            agg.mean_metrics.pitch_recall
        ]
        f1 = [
            agg.mean_metrics.onset_f1,
            agg.mean_metrics.offset_f1,
            agg.mean_metrics.pitch_f1
        ]
        
        x = np.arange(len(categories))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1', alpha=0.8)
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Precision/Recall/F1', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
    
    def plot_confusion_matrix(
        self,
        output_path: str,
        pitch_range: Optional[Tuple[int, int]] = None,
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """
        Plot confusion matrix for pitch predictions.
        
        Args:
            output_path: Path to save the plot
            pitch_range: Optional (min_pitch, max_pitch) to limit display
            figsize: Figure size in inches
        """
        if not self.evaluator.sample_results:
            self.logger.warning("No results to visualize")
            return
        
        # Collect all predicted and ground truth pitches
        all_predicted = []
        all_ground_truth = []
        
        for result in self.evaluator.sample_results:
            # Simple matching by index (for visualization purposes)
            min_len = min(len(result.predicted_notes), len(result.ground_truth_notes))
            for i in range(min_len):
                all_predicted.append(result.predicted_notes[i].pitch)
                all_ground_truth.append(result.ground_truth_notes[i].pitch)
        
        if not all_predicted:
            self.logger.warning("No matched notes for confusion matrix")
            return
        
        # Determine pitch range
        if pitch_range is None:
            min_pitch = min(min(all_predicted, default=0), min(all_ground_truth, default=0))
            max_pitch = max(max(all_predicted, default=127), max(all_ground_truth, default=127))
            pitch_range = (min_pitch, max_pitch)
        
        # Create confusion matrix
        pitch_range_size = pitch_range[1] - pitch_range[0] + 1
        confusion = np.zeros((pitch_range_size, pitch_range_size))
        
        for pred, gt in zip(all_predicted, all_ground_truth):
            if pitch_range[0] <= pred <= pitch_range[1] and pitch_range[0] <= gt <= pitch_range[1]:
                confusion[gt - pitch_range[0], pred - pitch_range[0]] += 1
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(confusion, cmap='YlOrRd', aspect='auto')
        
        ax.set_xlabel('Predicted Pitch', fontweight='bold')
        ax.set_ylabel('Ground Truth Pitch', fontweight='bold')
        ax.set_title('Pitch Confusion Matrix', fontweight='bold', fontsize=14)
        
        # Set ticks
        tick_positions = np.linspace(0, pitch_range_size - 1, min(10, pitch_range_size))
        tick_labels = [str(int(pitch_range[0] + p)) for p in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count', fontweight='bold')
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved confusion matrix to {output_path}")
    
    def plot_metrics_over_time(
        self,
        output_path: str,
        metric_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot how metrics change across samples.
        
        Args:
            output_path: Path to save the plot
            metric_names: List of metric names to plot
            figsize: Figure size in inches
        """
        if not self.evaluator.sample_results:
            self.logger.warning("No results to visualize")
            return
        
        if metric_names is None:
            metric_names = ['note_accuracy', 'onset_f1', 'pitch_f1']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(self.evaluator.sample_results))
        
        for metric_name in metric_names:
            values = [
                getattr(result.metrics, metric_name)
                for result in self.evaluator.sample_results
            ]
            ax.plot(x, values, marker='o', label=metric_name, alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Sample Index', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Metrics Over Samples', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved metrics over time plot to {output_path}")
