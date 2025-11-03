"""
Tests for report generator.
"""

import pytest
import tempfile
from pathlib import Path

from src.evaluation.evaluator import Evaluator
from src.evaluation.reporter import ReportGenerator, ReportFormat
from src.evaluation.metrics import Note


class TestReportGenerator:
    """Test ReportGenerator class."""
    
    @pytest.fixture
    def evaluator_with_results(self):
        """Create evaluator with some sample results."""
        evaluator = Evaluator()
        
        for i in range(3):
            predicted = [
                Note(pitch=60, onset=0.0, offset=1.0),
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
        """Test reporter initialization."""
        reporter = ReportGenerator(evaluator_with_results)
        assert reporter.evaluator is evaluator_with_results
    
    def test_generate_json_report(self, evaluator_with_results):
        """Test JSON report generation."""
        reporter = ReportGenerator(evaluator_with_results)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            
            reporter.generate_report(
                output_path=str(output_path),
                format=ReportFormat.JSON
            )
            
            assert output_path.exists()
            
            # Check it's valid JSON
            import json
            with open(output_path) as f:
                data = json.load(f)
            
            assert 'config' in data
            assert 'aggregated_metrics' in data
            assert 'samples' in data
    
    def test_generate_csv_report(self, evaluator_with_results):
        """Test CSV report generation."""
        reporter = ReportGenerator(evaluator_with_results)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.csv"
            
            reporter.generate_report(
                output_path=str(output_path),
                format=ReportFormat.CSV
            )
            
            assert output_path.exists()
            
            # Check content
            content = output_path.read_text(encoding='utf-8')
            assert "sample_id" in content
            assert "note_accuracy" in content
            assert "test_000" in content
    
    def test_generate_json_report(self, evaluator_with_results):
        """Test JSON report generation."""
        reporter = ReportGenerator(evaluator_with_results)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            
            reporter.generate_report(
                output_path=str(output_path),
                format=ReportFormat.JSON
            )
            
            assert output_path.exists()
            
            # Check it's valid JSON
            import json
            with open(output_path) as f:
                data = json.load(f)
            
            assert 'config' in data
            assert 'aggregated_metrics' in data
            assert 'samples' in data
    
    def test_report_creates_directory(self, evaluator_with_results):
        """Test that report generation creates output directory."""
        reporter = ReportGenerator(evaluator_with_results)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "report.json"
            
            reporter.generate_report(
                output_path=str(output_path),
                format=ReportFormat.JSON
            )
            
            assert output_path.exists()
            assert output_path.parent.exists()
