"""
Tests for evaluation metrics calculator.
"""

import pytest
from src.evaluation.metrics import (
    MetricCalculator,
    EvaluationMetrics,
    Note
)


class TestNote:
    """Test Note class."""
    
    def test_note_creation(self):
        """Test creating a note."""
        note = Note(pitch=60, onset=0.0, offset=1.0, velocity=64)
        assert note.pitch == 60
        assert note.onset == 0.0
        assert note.offset == 1.0
        assert note.velocity == 64
    
    def test_note_default_velocity(self):
        """Test default velocity."""
        note = Note(pitch=60, onset=0.0, offset=1.0)
        assert note.velocity == 64


class TestEvaluationMetrics:
    """Test EvaluationMetrics dataclass."""
    
    def test_metrics_defaults(self):
        """Test default metric values."""
        metrics = EvaluationMetrics()
        assert metrics.note_accuracy == 0.0
        assert metrics.onset_f1 == 0.0
        assert metrics.pitch_f1 == 0.0
        assert metrics.total_notes_predicted == 0
        assert metrics.total_notes_ground_truth == 0
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = EvaluationMetrics(
            note_accuracy=0.95,
            onset_f1=0.90,
            pitch_f1=0.88,
            total_notes_predicted=10,
            total_notes_ground_truth=12
        )
        
        metrics_dict = metrics.to_dict()
        assert metrics_dict['note_accuracy'] == 0.95
        assert metrics_dict['onset_f1'] == 0.90
        assert metrics_dict['pitch_f1'] == 0.88
        assert metrics_dict['total_notes_predicted'] == 10
        assert metrics_dict['total_notes_ground_truth'] == 12


class TestMetricCalculator:
    """Test MetricCalculator class."""
    
    def test_initialization(self):
        """Test calculator initialization."""
        calc = MetricCalculator(
            onset_tolerance=0.05,
            offset_tolerance=0.05,
            pitch_tolerance=0
        )
        assert calc.onset_tolerance == 0.05
        assert calc.offset_tolerance == 0.05
        assert calc.pitch_tolerance == 0
    
    def test_perfect_match(self):
        """Test metrics with perfect predictions."""
        calc = MetricCalculator()
        
        predicted = [
            Note(pitch=60, onset=0.0, offset=1.0),
            Note(pitch=62, onset=1.0, offset=2.0),
            Note(pitch=64, onset=2.0, offset=3.0),
        ]
        ground_truth = [
            Note(pitch=60, onset=0.0, offset=1.0),
            Note(pitch=62, onset=1.0, offset=2.0),
            Note(pitch=64, onset=2.0, offset=3.0),
        ]
        
        metrics = calc.calculate_metrics(predicted, ground_truth)
        
        assert metrics.note_accuracy == 1.0
        assert metrics.onset_f1 == 1.0
        assert metrics.pitch_f1 == 1.0
        assert metrics.true_positives == 3
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
    
    def test_no_predictions(self):
        """Test metrics with no predictions."""
        calc = MetricCalculator()
        
        predicted = []
        ground_truth = [
            Note(pitch=60, onset=0.0, offset=1.0),
            Note(pitch=62, onset=1.0, offset=2.0),
        ]
        
        metrics = calc.calculate_metrics(predicted, ground_truth)
        
        assert metrics.note_accuracy == 0.0
        assert metrics.onset_precision == 0.0
        assert metrics.onset_recall == 0.0
        assert metrics.onset_f1 == 0.0
        assert metrics.false_negatives == 2
    
    def test_no_ground_truth(self):
        """Test metrics with no ground truth."""
        calc = MetricCalculator()
        
        predicted = [
            Note(pitch=60, onset=0.0, offset=1.0),
            Note(pitch=62, onset=1.0, offset=2.0),
        ]
        ground_truth = []
        
        metrics = calc.calculate_metrics(predicted, ground_truth)
        
        assert metrics.note_accuracy == 0.0
        assert metrics.total_notes_predicted == 2
        assert metrics.total_notes_ground_truth == 0
    
    def test_partial_match(self):
        """Test metrics with partial matches."""
        calc = MetricCalculator()
        
        predicted = [
            Note(pitch=60, onset=0.0, offset=1.0),
            Note(pitch=62, onset=1.0, offset=2.0),
            Note(pitch=65, onset=2.5, offset=3.5),  # Wrong pitch
        ]
        ground_truth = [
            Note(pitch=60, onset=0.0, offset=1.0),
            Note(pitch=62, onset=1.0, offset=2.0),
            Note(pitch=64, onset=2.0, offset=3.0),
        ]
        
        metrics = calc.calculate_metrics(predicted, ground_truth)
        
        assert metrics.true_positives == 2  # First two notes match
        assert metrics.false_positives == 1  # Third note doesn't match
        assert metrics.false_negatives == 1  # Missing MIDI 64
        assert 0 < metrics.note_accuracy < 1.0
    
    def test_onset_tolerance(self):
        """Test onset tolerance matching."""
        calc = MetricCalculator(onset_tolerance=0.1)
        
        predicted = [
            Note(pitch=60, onset=0.05, offset=1.0),  # Slightly off onset
        ]
        ground_truth = [
            Note(pitch=60, onset=0.0, offset=1.0),
        ]
        
        metrics = calc.calculate_metrics(predicted, ground_truth)
        
        # Should match within tolerance
        assert metrics.true_positives == 1
        assert metrics.onset_f1 == 1.0
    
    def test_pitch_tolerance(self):
        """Test pitch tolerance matching."""
        calc = MetricCalculator(pitch_tolerance=1)
        
        predicted = [
            Note(pitch=61, onset=0.0, offset=1.0),  # Off by 1 semitone
        ]
        ground_truth = [
            Note(pitch=60, onset=0.0, offset=1.0),
        ]
        
        metrics = calc.calculate_metrics(predicted, ground_truth)
        
        # Should match within tolerance
        assert metrics.true_positives == 1
        assert metrics.pitch_f1 == 1.0
    
    def test_timing_error_calculation(self):
        """Test timing error metrics."""
        calc = MetricCalculator()
        
        predicted = [
            Note(pitch=60, onset=0.02, offset=1.0),  # 20ms late
            Note(pitch=62, onset=1.03, offset=2.0),  # 30ms late
        ]
        ground_truth = [
            Note(pitch=60, onset=0.0, offset=1.0),
            Note(pitch=62, onset=1.0, offset=2.0),
        ]
        
        metrics = calc.calculate_metrics(predicted, ground_truth)
        
        assert metrics.timing_error_mean > 0
        assert abs(metrics.timing_error_mean - 0.025) < 0.001  # ~25ms average
    
    def test_tempo_error_calculation(self):
        """Test tempo error calculation."""
        calc = MetricCalculator()
        
        predicted = [Note(pitch=60, onset=0.0, offset=1.0)]
        ground_truth = [Note(pitch=60, onset=0.0, offset=1.0)]
        
        metrics = calc.calculate_metrics(
            predicted, ground_truth,
            predicted_tempo=120.0,
            ground_truth_tempo=100.0
        )
        
        # Relative error: |120-100|/100 = 0.20
        assert abs(metrics.tempo_error - 0.20) < 0.001
    
    def test_pitch_error_distribution(self):
        """Test pitch error distribution."""
        calc = MetricCalculator()
        
        predicted = [
            Note(pitch=60, onset=0.0, offset=1.0),
            Note(pitch=62, onset=1.0, offset=2.0),
        ]
        ground_truth = [
            Note(pitch=60, onset=0.0, offset=1.0),
            Note(pitch=62, onset=1.0, offset=2.0),
            Note(pitch=64, onset=2.0, offset=3.0),  # Missing in predictions
            Note(pitch=67, onset=3.0, offset=4.0),  # Missing in predictions
        ]
        
        metrics = calc.calculate_metrics(predicted, ground_truth)
        
        # Should have errors for pitches 64 and 67
        assert 64 in metrics.pitch_errors
        assert 67 in metrics.pitch_errors
        assert metrics.pitch_errors[64] == 1
        assert metrics.pitch_errors[67] == 1
    
    def test_f1_score_calculation(self):
        """Test F1-score calculation."""
        calc = MetricCalculator()
        
        # Manually test F1 calculation
        precision = 0.75
        recall = 0.60
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        
        calculated_f1 = calc._calculate_f1(precision, recall)
        assert abs(calculated_f1 - expected_f1) < 0.001
    
    def test_f1_score_edge_cases(self):
        """Test F1-score edge cases."""
        calc = MetricCalculator()
        
        # Both zero
        assert calc._calculate_f1(0.0, 0.0) == 0.0
        
        # One zero
        assert calc._calculate_f1(1.0, 0.0) == 0.0
        assert calc._calculate_f1(0.0, 1.0) == 0.0
    
    def test_safe_divide(self):
        """Test safe division."""
        calc = MetricCalculator()
        
        assert calc._safe_divide(10, 2) == 5.0
        assert calc._safe_divide(10, 0) == 0.0
        assert calc._safe_divide(0, 0) == 0.0
    
    def test_offset_metrics(self):
        """Test offset-based metrics."""
        calc = MetricCalculator(offset_tolerance=0.1)
        
        predicted = [
            Note(pitch=60, onset=0.0, offset=1.05),  # Slightly long
        ]
        ground_truth = [
            Note(pitch=60, onset=0.0, offset=1.0),
        ]
        
        metrics = calc.calculate_metrics(predicted, ground_truth)
        
        # Should match within offset tolerance
        assert metrics.offset_f1 == 1.0
    
    def test_multiple_notes_same_pitch(self):
        """Test handling multiple notes with same pitch."""
        calc = MetricCalculator()
        
        predicted = [
            Note(pitch=60, onset=0.0, offset=0.5),
            Note(pitch=60, onset=1.0, offset=1.5),
        ]
        ground_truth = [
            Note(pitch=60, onset=0.0, offset=0.5),
            Note(pitch=60, onset=1.0, offset=1.5),
        ]
        
        metrics = calc.calculate_metrics(predicted, ground_truth)
        
        assert metrics.true_positives == 2
        assert metrics.note_accuracy == 1.0
