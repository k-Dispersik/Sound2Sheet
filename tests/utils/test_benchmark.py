"""
Tests for performance benchmarking utilities.
"""

import time
import pytest
from src.utils import (
    benchmark,
    Timer,
    BenchmarkCollector,
    BenchmarkResult,
    get_global_collector,
    clear_benchmarks,
)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""
    
    def test_benchmark_result_creation(self):
        """Test creating a benchmark result."""
        result = BenchmarkResult(
            name="test_operation",
            duration=1.234,
            metadata={'param': 'value'}
        )
        assert result.name == "test_operation"
        assert result.duration == 1.234
        assert result.metadata == {'param': 'value'}
    
    def test_benchmark_result_str(self):
        """Test string representation of benchmark result."""
        result = BenchmarkResult(
            name="test_op",
            duration=0.123,
            metadata={'size': 100}
        )
        result_str = str(result)
        assert "test_op" in result_str
        assert "0.123" in result_str
        assert "size=100" in result_str


class TestBenchmarkCollector:
    """Tests for BenchmarkCollector."""
    
    def test_collector_initialization(self):
        """Test creating a new collector."""
        collector = BenchmarkCollector()
        assert len(collector.results) == 0
    
    def test_add_result(self):
        """Test adding benchmark results."""
        collector = BenchmarkCollector()
        result = BenchmarkResult(name="op1", duration=1.0)
        
        collector.add_result(result)
        assert len(collector.results) == 1
        assert "op1" in collector.results
    
    def test_get_statistics(self):
        """Test calculating statistics for benchmarks."""
        collector = BenchmarkCollector()
        
        # Add multiple results for same operation
        for duration in [1.0, 2.0, 3.0, 4.0, 5.0]:
            collector.add_result(BenchmarkResult(name="test_op", duration=duration))
        
        stats = collector.get_statistics("test_op")
        assert stats['count'] == 5
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['mean'] == 3.0
        assert stats['total'] == 15.0
    
    def test_get_all_statistics(self):
        """Test getting statistics for all operations."""
        collector = BenchmarkCollector()
        
        collector.add_result(BenchmarkResult(name="op1", duration=1.0))
        collector.add_result(BenchmarkResult(name="op2", duration=2.0))
        collector.add_result(BenchmarkResult(name="op1", duration=1.5))
        
        all_stats = collector.get_all_statistics()
        assert "op1" in all_stats
        assert "op2" in all_stats
        assert all_stats["op1"]["count"] == 2
        assert all_stats["op2"]["count"] == 1
    
    def test_report_generation(self):
        """Test generating a formatted report."""
        collector = BenchmarkCollector()
        
        collector.add_result(BenchmarkResult(name="operation_a", duration=1.5))
        collector.add_result(BenchmarkResult(name="operation_b", duration=0.5))
        
        report = collector.report()
        assert "Performance Benchmark Report" in report
        assert "operation_a" in report
        assert "operation_b" in report
        assert "1.5" in report
        assert "0.5" in report
    
    def test_clear(self):
        """Test clearing all results."""
        collector = BenchmarkCollector()
        collector.add_result(BenchmarkResult(name="op", duration=1.0))
        
        assert len(collector.results) == 1
        collector.clear()
        assert len(collector.results) == 0


class TestBenchmarkDecorator:
    """Tests for @benchmark decorator."""
    
    def test_benchmark_decorator_basic(self):
        """Test basic decorator functionality."""
        collector = BenchmarkCollector()
        
        @benchmark(collector=collector)
        def test_function():
            time.sleep(0.01)
            return 42
        
        result = test_function()
        assert result == 42
        assert len(collector.results) == 1
        assert collector.results["test_function"][0].duration >= 0.01
    
    def test_benchmark_decorator_custom_name(self):
        """Test decorator with custom name."""
        collector = BenchmarkCollector()
        
        @benchmark(name="custom_operation", collector=collector)
        def my_function():
            return "done"
        
        my_function()
        assert "custom_operation" in collector.results
    
    def test_benchmark_decorator_metadata(self):
        """Test decorator with metadata."""
        collector = BenchmarkCollector()
        
        @benchmark(collector=collector, dataset="test", size=100)
        def process_data():
            return True
        
        process_data()
        result = collector.results["process_data"][0]
        assert result.metadata == {'dataset': 'test', 'size': 100}
    
    def test_benchmark_decorator_multiple_calls(self):
        """Test multiple calls to decorated function."""
        collector = BenchmarkCollector()
        
        @benchmark(collector=collector)
        def fast_operation():
            return 1
        
        for _ in range(5):
            fast_operation()
        
        stats = collector.get_statistics("fast_operation")
        assert stats['count'] == 5
    
    def test_benchmark_decorator_with_exception(self):
        """Test that decorator records time even with exception."""
        collector = BenchmarkCollector()
        
        @benchmark(collector=collector)
        def failing_function():
            time.sleep(0.01)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        # Should still record the benchmark
        assert len(collector.results["failing_function"]) == 1


class TestTimerContextManager:
    """Tests for Timer context manager."""
    
    def test_timer_basic(self):
        """Test basic timer usage."""
        collector = BenchmarkCollector()
        
        with Timer("test_block", collector=collector) as timer:
            time.sleep(0.01)
        
        assert timer.duration >= 0.01
        assert len(collector.results["test_block"]) == 1
    
    def test_timer_metadata(self):
        """Test timer with metadata."""
        collector = BenchmarkCollector()
        
        with Timer("data_load", collector=collector, source="file", size=1024):
            pass
        
        result = collector.results["data_load"][0]
        assert result.metadata == {'source': 'file', 'size': 1024}
    
    def test_timer_duration_access(self):
        """Test accessing duration after context."""
        with Timer("test") as t:
            time.sleep(0.01)
        
        assert t.duration is not None
        assert t.duration >= 0.01
    
    def test_timer_multiple_blocks(self):
        """Test timing multiple blocks."""
        collector = BenchmarkCollector()
        
        with Timer("block1", collector=collector):
            time.sleep(0.01)
        
        with Timer("block2", collector=collector):
            time.sleep(0.02)
        
        stats1 = collector.get_statistics("block1")
        stats2 = collector.get_statistics("block2")
        
        assert stats1['count'] == 1
        assert stats2['count'] == 1
        assert stats2['total'] > stats1['total']


class TestGlobalCollector:
    """Tests for global collector functions."""
    
    def test_get_global_collector(self):
        """Test getting global collector instance."""
        collector = get_global_collector()
        assert isinstance(collector, BenchmarkCollector)
    
    def test_global_collector_persistence(self):
        """Test that global collector persists across calls."""
        clear_benchmarks()
        
        @benchmark()
        def operation():
            pass
        
        operation()
        
        collector = get_global_collector()
        assert len(collector.results) == 1
        
        clear_benchmarks()
        assert len(collector.results) == 0
    
    def test_clear_benchmarks(self):
        """Test clearing global benchmarks."""
        @benchmark()
        def test_op():
            pass
        
        test_op()
        collector = get_global_collector()
        assert len(collector.results) > 0
        
        clear_benchmarks()
        assert len(collector.results) == 0


class TestBenchmarkIntegration:
    """Integration tests for benchmark system."""
    
    def test_mixed_decorators_and_timers(self):
        """Test using decorators and timers together."""
        collector = BenchmarkCollector()
        
        @benchmark(name="func_a", collector=collector)
        def function_a():
            time.sleep(0.01)
        
        function_a()
        
        with Timer("block_b", collector=collector):
            time.sleep(0.01)
        
        all_stats = collector.get_all_statistics()
        assert len(all_stats) == 2
        assert "func_a" in all_stats
        assert "block_b" in all_stats
    
    def test_nested_timing(self):
        """Test nested benchmark operations."""
        collector = BenchmarkCollector()
        
        @benchmark(name="outer", collector=collector)
        def outer_function():
            with Timer("inner", collector=collector):
                time.sleep(0.01)
        
        outer_function()
        
        stats = collector.get_all_statistics()
        assert "outer" in stats
        assert "inner" in stats
        # Outer should take longer than inner
        assert stats["outer"]["total"] >= stats["inner"]["total"]
