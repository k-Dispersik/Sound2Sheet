"""Tests for evaluation package initialization."""

import pytest


def test_imports():
    """Test that all main classes can be imported."""
    from src.evaluation import (
        MetricCalculator,
        EvaluationMetrics,
        Evaluator,
        EvaluationConfig,
        ReportGenerator,
        ReportFormat,
        Visualizer
    )
    
    # Basic checks
    assert MetricCalculator is not None
    assert EvaluationMetrics is not None
    assert Evaluator is not None
    assert EvaluationConfig is not None
    assert ReportGenerator is not None
    assert ReportFormat is not None
    assert Visualizer is not None
