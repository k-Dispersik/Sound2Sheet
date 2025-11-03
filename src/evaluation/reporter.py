"""
Report generator for evaluation results.

Generates reports in CSV and JSON formats for evaluation metrics.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum
import json
import csv
import logging

from .evaluator import Evaluator


class ReportFormat(Enum):
    """Supported report formats."""
    CSV = "csv"
    JSON = "json"


class ReportGenerator:
    """
    Generate evaluation reports in CSV and JSON formats.
    
    Provides simple export of metrics for further analysis.
    """
    
    def __init__(self, evaluator: Evaluator):
        """
        Initialize report generator.
        
        Args:
            evaluator: Evaluator instance with results
        """
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__)
    
    def generate_report(
        self,
        output_path: str,
        format: ReportFormat = ReportFormat.JSON
    ) -> None:
        """
        Generate evaluation report.
        
        Args:
            output_path: Path to output file
            format: Report format (CSV or JSON)
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == ReportFormat.CSV:
            self._generate_csv_report(output_file)
        elif format == ReportFormat.JSON:
            self._generate_json_report(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Generated {format.value} report: {output_path}")
    
    def _generate_csv_report(self, output_path: Path) -> None:
        """Generate CSV report with sample-level metrics."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'sample_id',
                'note_accuracy',
                'onset_f1',
                'onset_precision',
                'onset_recall',
                'offset_f1',
                'pitch_f1',
                'pitch_precision',
                'pitch_recall',
                'timing_error_mean',
                'num_predicted',
                'num_ground_truth',
                'true_positives',
                'false_positives',
                'false_negatives'
            ])
            
            # Data rows
            for result in self.evaluator.sample_results:
                m = result.metrics
                writer.writerow([
                    result.sample_id,
                    m.note_accuracy,
                    m.onset_f1,
                    m.onset_precision,
                    m.onset_recall,
                    m.offset_f1,
                    m.pitch_f1,
                    m.pitch_precision,
                    m.pitch_recall,
                    m.timing_error_mean,
                    len(result.predicted_notes),
                    len(result.ground_truth_notes),
                    m.true_positives,
                    m.false_positives,
                    m.false_negatives
                ])
    
    def _generate_json_report(self, output_path: Path) -> None:
        """Generate JSON report."""
        self.evaluator.save_results(str(output_path))
