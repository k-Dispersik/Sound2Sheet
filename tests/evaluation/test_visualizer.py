"""
Tests for visualizer.
"""

import pytest
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from src.evaluation.evaluator import Evaluator
from src.evaluation.visualizer import Visualizer
from src.evaluation.metrics import Note


class TestVisualizer:
    """Test Visualizer class."""
    
    @pytest.fixture
    def evaluator_with_results(self):
        """Create evaluator with sample results."""
        evaluator = Evaluator()
        
        for i in range(10):
            predicted = [
                Note(pitch=60 + i, onset=0.0 + i*0.01, offset=1.0),
                Note(pitch=62, onset=1.0, offset=2.0),
            ]
            ground_truth = [
                Note(pitch=60, onset=0.0, offset=1.0),
                Note(pitch=62, onset=1.0, offset=2.0),
            ]
            
            evaluator.evaluate_sample(
                sample_id=f'test_{i:03d}',
                predicted_notes=predicted,
                ground_truth_notes=ground_truth
            )
        
        return evaluator
    
    def test_initialization(self, evaluator_with_results):
        """Test visualizer initialization."""
        visualizer = Visualizer(evaluator_with_results)
        assert visualizer.evaluator is evaluator_with_results
    
    def test_create_dashboard(self, evaluator_with_results):
        """Test dashboard creation."""
        visualizer = Visualizer(evaluator_with_results)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dashboard.png"
            
            visualizer.create_dashboard(str(output_path))
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_dashboard_with_empty_results(self):
        """Test dashboard with no results."""
        evaluator = Evaluator()
        visualizer = Visualizer(evaluator)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dashboard.png"
            
            # Should not crash, just log warning
            visualizer.create_dashboard(str(output_path))
            
            # File should not be created
            assert not output_path.exists()
    
    def test_plot_confusion_matrix(self, evaluator_with_results):
        """Test confusion matrix plotting."""
        visualizer = Visualizer(evaluator_with_results)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "confusion_matrix.png"
            
            visualizer.plot_confusion_matrix(str(output_path))
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_plot_confusion_matrix_with_pitch_range(self, evaluator_with_results):
        """Test confusion matrix with specific pitch range."""
        visualizer = Visualizer(evaluator_with_results)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "confusion_matrix.png"
            
            visualizer.plot_confusion_matrix(
                str(output_path),
                pitch_range=(60, 70)
            )
            
            assert output_path.exists()
    
    def test_plot_confusion_matrix_empty_results(self):
        """Test confusion matrix with no results."""
        evaluator = Evaluator()
        visualizer = Visualizer(evaluator)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "confusion_matrix.png"
            
            # Should not crash
            visualizer.plot_confusion_matrix(str(output_path))
            
            # File should not be created
            assert not output_path.exists()
    
    def test_plot_metrics_over_time(self, evaluator_with_results):
        """Test metrics over time plotting."""
        visualizer = Visualizer(evaluator_with_results)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics_over_time.png"
            
            visualizer.plot_metrics_over_time(str(output_path))
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_plot_metrics_over_time_custom_metrics(self, evaluator_with_results):
        """Test metrics over time with custom metric names."""
        visualizer = Visualizer(evaluator_with_results)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics_over_time.png"
            
            visualizer.plot_metrics_over_time(
                str(output_path),
                metric_names=['onset_f1', 'pitch_f1']
            )
            
            assert output_path.exists()
    
    def test_plot_metrics_over_time_empty_results(self):
        """Test metrics over time with no results."""
        evaluator = Evaluator()
        visualizer = Visualizer(evaluator)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics_over_time.png"
            
            # Should not crash
            visualizer.plot_metrics_over_time(str(output_path))
            
            # File should not be created
            assert not output_path.exists()
    
    def test_creates_output_directory(self, evaluator_with_results):
        """Test that plotting creates output directory."""
        visualizer = Visualizer(evaluator_with_results)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "plot.png"
            
            visualizer.create_dashboard(str(output_path))
            
            assert output_path.exists()
            assert output_path.parent.exists()
