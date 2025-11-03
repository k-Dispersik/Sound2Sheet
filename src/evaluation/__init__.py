"""
Evaluation system for Sound2Sheet model performance assessment.

This module provides comprehensive evaluation capabilities including:
- Metrics calculation (accuracy, F1-scores, precision/recall)
- Batch evaluation and statistical analysis
- Report generation (HTML, PDF, CSV)
- Visualization (confusion matrices, error distributions)
"""

from .metrics import MetricCalculator, EvaluationMetrics
from .evaluator import Evaluator, EvaluationConfig
from .reporter import ReportGenerator, ReportFormat
from .visualizer import Visualizer

__all__ = [
    'MetricCalculator',
    'EvaluationMetrics',
    'Evaluator',
    'EvaluationConfig',
    'ReportGenerator',
    'ReportFormat',
    'Visualizer',
]
