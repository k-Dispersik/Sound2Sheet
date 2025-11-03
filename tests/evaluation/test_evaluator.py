"""
Tests for evaluator.
"""

import pytest
import tempfile
from pathlib import Path

from src.evaluation.evaluator import (
    Evaluator,
    EvaluationConfig,
    SampleEvaluation,
    AggregatedMetrics
)
from src.evaluation.metrics import Note, EvaluationMetrics


class TestEvaluationConfig:
    """Test EvaluationConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = EvaluationConfig()
        assert config.onset_tolerance == 0.05
        assert config.offset_tolerance == 0.05
        assert config.pitch_tolerance == 0
        assert config.min_note_duration == 0.01
        assert config.max_note_duration == 10.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EvaluationConfig(
            onset_tolerance=0.1,
            pitch_tolerance=1,
            min_note_duration=0.05
        )
        assert config.onset_tolerance == 0.1
        assert config.pitch_tolerance == 1
        assert config.min_note_duration == 0.05


class TestSampleEvaluation:
    """Test SampleEvaluation."""
    
    def test_sample_evaluation_creation(self):
        """Test creating sample evaluation."""
        metrics = EvaluationMetrics(note_accuracy=0.95)
        notes = [Note(pitch=60, onset=0.0, offset=1.0)]
        
        sample = SampleEvaluation(
            sample_id="test_001",
            metrics=metrics,
            predicted_notes=notes,
            ground_truth_notes=notes,
            metadata={'duration': 3.0}
        )
        
        assert sample.sample_id == "test_001"
        assert sample.metrics.note_accuracy == 0.95
        assert len(sample.predicted_notes) == 1
        assert sample.metadata['duration'] == 3.0
    
    def test_sample_to_dict(self):
        """Test converting sample to dictionary."""
        metrics = EvaluationMetrics(note_accuracy=0.95)
        notes = [Note(pitch=60, onset=0.0, offset=1.0)]
        
        sample = SampleEvaluation(
            sample_id="test_001",
            metrics=metrics,
            predicted_notes=notes,
            ground_truth_notes=notes
        )
        
        sample_dict = sample.to_dict()
        assert sample_dict['sample_id'] == "test_001"
        assert sample_dict['num_predicted'] == 1
        assert sample_dict['num_ground_truth'] == 1
        assert 'metrics' in sample_dict


class TestEvaluator:
    """Test Evaluator class."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = Evaluator()
        assert evaluator.config is not None
        assert evaluator.metric_calculator is not None
        assert len(evaluator.sample_results) == 0
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = EvaluationConfig(onset_tolerance=0.1)
        evaluator = Evaluator(config)
        assert evaluator.config.onset_tolerance == 0.1
    
    def test_evaluate_single_sample(self):
        """Test evaluating a single sample."""
        evaluator = Evaluator()
        
        predicted = [
            Note(pitch=60, onset=0.0, offset=1.0),
            Note(pitch=62, onset=1.0, offset=2.0),
        ]
        ground_truth = [
            Note(pitch=60, onset=0.0, offset=1.0),
            Note(pitch=62, onset=1.0, offset=2.0),
        ]
        
        result = evaluator.evaluate_sample(
            sample_id="test_001",
            predicted_notes=predicted,
            ground_truth_notes=ground_truth
        )
        
        assert result.sample_id == "test_001"
        assert result.metrics.note_accuracy == 1.0
        assert len(evaluator.sample_results) == 1
    
    def test_evaluate_with_tempo(self):
        """Test evaluation with tempo information."""
        evaluator = Evaluator()
        
        predicted = [Note(pitch=60, onset=0.0, offset=1.0)]
        ground_truth = [Note(pitch=60, onset=0.0, offset=1.0)]
        
        result = evaluator.evaluate_sample(
            sample_id="test_001",
            predicted_notes=predicted,
            ground_truth_notes=ground_truth,
            predicted_tempo=120.0,
            ground_truth_tempo=120.0
        )
        
        assert result.metrics.tempo_error == 0.0
    
    def test_evaluate_with_metadata(self):
        """Test evaluation with metadata."""
        evaluator = Evaluator()
        
        predicted = [Note(pitch=60, onset=0.0, offset=1.0)]
        ground_truth = [Note(pitch=60, onset=0.0, offset=1.0)]
        
        metadata = {'audio_path': '/path/to/audio.wav', 'duration': 3.0}
        
        result = evaluator.evaluate_sample(
            sample_id="test_001",
            predicted_notes=predicted,
            ground_truth_notes=ground_truth,
            metadata=metadata
        )
        
        assert result.metadata['audio_path'] == '/path/to/audio.wav'
        assert result.metadata['duration'] == 3.0
    
    def test_evaluate_batch(self):
        """Test batch evaluation."""
        evaluator = Evaluator()
        
        samples = [
            {
                'sample_id': f'test_{i:03d}',
                'predicted_notes': [Note(pitch=60, onset=0.0, offset=1.0)],
                'ground_truth_notes': [Note(pitch=60, onset=0.0, offset=1.0)],
            }
            for i in range(5)
        ]
        
        results = evaluator.evaluate_batch(samples)
        
        assert len(results) == 5
        assert len(evaluator.sample_results) == 5
        assert all(r.metrics.note_accuracy == 1.0 for r in results)
    
    def test_evaluate_batch_with_progress_callback(self):
        """Test batch evaluation with progress callback."""
        evaluator = Evaluator()
        
        progress_updates = []
        
        def progress_callback(current, total):
            progress_updates.append((current, total))
        
        samples = [
            {
                'sample_id': f'test_{i:03d}',
                'predicted_notes': [Note(pitch=60, onset=0.0, offset=1.0)],
                'ground_truth_notes': [Note(pitch=60, onset=0.0, offset=1.0)],
            }
            for i in range(3)
        ]
        
        evaluator.evaluate_batch(samples, progress_callback=progress_callback)
        
        assert len(progress_updates) == 3
        assert progress_updates[-1] == (3, 3)
    
    def test_get_aggregated_metrics(self):
        """Test aggregated metrics calculation."""
        evaluator = Evaluator()
        
        # Evaluate multiple samples with different accuracies
        for i in range(3):
            predicted = [Note(pitch=60, onset=0.0, offset=1.0) for _ in range(i + 1)]
            ground_truth = [Note(pitch=60, onset=0.0, offset=1.0) for _ in range(3)]
            
            evaluator.evaluate_sample(
                sample_id=f'test_{i:03d}',
                predicted_notes=predicted,
                ground_truth_notes=ground_truth
            )
        
        agg = evaluator.get_aggregated_metrics()
        
        assert agg.num_samples == 3
        assert 0 < agg.mean_metrics.note_accuracy < 1.0
        assert agg.std_metrics.note_accuracy >= 0
    
    def test_get_aggregated_metrics_empty(self):
        """Test aggregated metrics with no results."""
        evaluator = Evaluator()
        
        agg = evaluator.get_aggregated_metrics()
        
        assert agg.num_samples == 0
        assert agg.mean_metrics.note_accuracy == 0.0
    
    def test_get_error_analysis(self):
        """Test error analysis."""
        evaluator = Evaluator()
        
        predicted = [
            Note(pitch=60, onset=0.0, offset=1.0),
            Note(pitch=62, onset=1.0, offset=2.0),
        ]
        ground_truth = [
            Note(pitch=60, onset=0.0, offset=1.0),
            Note(pitch=64, onset=2.0, offset=3.0),
        ]
        
        evaluator.evaluate_sample(
            sample_id="test_001",
            predicted_notes=predicted,
            ground_truth_notes=ground_truth
        )
        
        analysis = evaluator.get_error_analysis()
        
        assert analysis['total_samples'] == 1
        assert analysis['total_predicted_notes'] == 2
        assert analysis['total_ground_truth_notes'] == 2
        assert 'pitch_error_distribution' in analysis
    
    def test_save_and_load_results(self):
        """Test saving and loading results."""
        evaluator = Evaluator()
        
        predicted = [Note(pitch=60, onset=0.0, offset=1.0)]
        ground_truth = [Note(pitch=60, onset=0.0, offset=1.0)]
        
        evaluator.evaluate_sample(
            sample_id="test_001",
            predicted_notes=predicted,
            ground_truth_notes=ground_truth
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            
            # Save
            evaluator.save_results(str(output_path))
            assert output_path.exists()
            
            # Load
            evaluator2 = Evaluator()
            evaluator2.load_results(str(output_path))
            assert evaluator2.config.onset_tolerance == evaluator.config.onset_tolerance
    
    def test_clear_results(self):
        """Test clearing results."""
        evaluator = Evaluator()
        
        predicted = [Note(pitch=60, onset=0.0, offset=1.0)]
        ground_truth = [Note(pitch=60, onset=0.0, offset=1.0)]
        
        evaluator.evaluate_sample(
            sample_id="test_001",
            predicted_notes=predicted,
            ground_truth_notes=ground_truth
        )
        
        assert len(evaluator.sample_results) == 1
        
        evaluator.clear_results()
        
        assert len(evaluator.sample_results) == 0
    
    def test_filter_notes_by_duration(self):
        """Test filtering notes by duration."""
        config = EvaluationConfig(min_note_duration=0.5, max_note_duration=2.0)
        evaluator = Evaluator(config)
        
        notes = [
            Note(pitch=60, onset=0.0, offset=0.3),   # Too short
            Note(pitch=62, onset=1.0, offset=2.0),   # OK
            Note(pitch=64, onset=3.0, offset=6.0),   # Too long
            Note(pitch=67, onset=7.0, offset=8.5),   # OK
        ]
        
        filtered = evaluator._filter_notes_by_duration(notes)
        
        assert len(filtered) == 2
        assert filtered[0].pitch == 62
        assert filtered[1].pitch == 67
    
    def test_get_summary(self):
        """Test getting summary string."""
        evaluator = Evaluator()
        
        predicted = [Note(pitch=60, onset=0.0, offset=1.0)]
        ground_truth = [Note(pitch=60, onset=0.0, offset=1.0)]
        
        evaluator.evaluate_sample(
            sample_id="test_001",
            predicted_notes=predicted,
            ground_truth_notes=ground_truth
        )
        
        summary = evaluator.get_summary()
        
        assert "EVALUATION SUMMARY" in summary
        assert "Number of samples: 1" in summary
        assert "Note Accuracy" in summary
    
    def test_get_summary_empty(self):
        """Test summary with no results."""
        evaluator = Evaluator()
        
        summary = evaluator.get_summary()
        
        assert "No evaluation results available" in summary
    
    def test_batch_evaluation_with_errors(self):
        """Test batch evaluation handles errors gracefully."""
        evaluator = Evaluator()
        
        samples = [
            {
                'sample_id': 'test_001',
                'predicted_notes': [Note(pitch=60, onset=0.0, offset=1.0)],
                'ground_truth_notes': [Note(pitch=60, onset=0.0, offset=1.0)],
            },
            {
                'sample_id': 'test_002',
                # Missing required keys - should cause error
            },
            {
                'sample_id': 'test_003',
                'predicted_notes': [Note(pitch=62, onset=0.0, offset=1.0)],
                'ground_truth_notes': [Note(pitch=62, onset=0.0, offset=1.0)],
            },
        ]
        
        results = evaluator.evaluate_batch(samples)
        
        # Should successfully process valid samples despite errors
        assert len(results) == 2
        assert results[0].sample_id == 'test_001'
        assert results[1].sample_id == 'test_003'
