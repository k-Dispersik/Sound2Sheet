"""
Performance benchmarking utilities for Sound2Sheet.

Provides decorators and tools for measuring execution time and performance.
"""

import time
import functools
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class BenchmarkResult:
    """
    Results from a performance benchmark.
    
    Attributes:
        name: Name of the benchmarked operation
        duration: Execution time in seconds
        timestamp: When the benchmark was run
        metadata: Additional metadata about the operation
    """
    name: str
    duration: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Format benchmark result as string."""
        meta_str = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
        return f"{self.name}: {self.duration:.4f}s ({meta_str})"


class BenchmarkCollector:
    """
    Collects and aggregates benchmark results.
    """
    
    def __init__(self):
        """Initialize benchmark collector."""
        self.results: Dict[str, list] = defaultdict(list)
        self._current_section: Optional[str] = None
    
    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results[result.name].append(result)
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a specific benchmark.
        
        Args:
            name: Name of the benchmark
            
        Returns:
            Dictionary with min, max, mean, total execution times
        """
        if name not in self.results:
            return {}
        
        durations = [r.duration for r in self.results[name]]
        return {
            'count': len(durations),
            'min': min(durations),
            'max': max(durations),
            'mean': sum(durations) / len(durations),
            'total': sum(durations)
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all benchmarks."""
        return {name: self.get_statistics(name) for name in self.results.keys()}
    
    def report(self) -> str:
        """
        Generate a formatted benchmark report.
        
        Returns:
            Multi-line string with benchmark statistics
        """
        lines = ["=" * 70, "Performance Benchmark Report", "=" * 70, ""]
        
        stats = self.get_all_statistics()
        
        if not stats:
            lines.append("No benchmarks recorded.")
            return "\n".join(lines)
        
        # Sort by total time (descending)
        sorted_stats = sorted(stats.items(), key=lambda x: x[1].get('total', 0), reverse=True)
        
        for name, stat in sorted_stats:
            lines.append(f"{name}:")
            lines.append(f"  Runs:  {stat['count']}")
            lines.append(f"  Total: {stat['total']:.4f}s")
            lines.append(f"  Mean:  {stat['mean']:.4f}s")
            lines.append(f"  Min:   {stat['min']:.4f}s")
            lines.append(f"  Max:   {stat['max']:.4f}s")
            lines.append("")
        
        total_time = sum(s['total'] for s in stats.values())
        lines.append("=" * 70)
        lines.append(f"Total execution time: {total_time:.4f}s")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all benchmark results."""
        self.results.clear()


# Global benchmark collector
_global_collector = BenchmarkCollector()


def get_global_collector() -> BenchmarkCollector:
    """Get the global benchmark collector."""
    return _global_collector


def benchmark(name: Optional[str] = None, collector: Optional[BenchmarkCollector] = None, **metadata):
    """
    Decorator to benchmark function execution time.
    
    Args:
        name: Optional name for the benchmark (defaults to function name)
        collector: Optional BenchmarkCollector (defaults to global)
        **metadata: Additional metadata to store with the benchmark
        
    Example:
        @benchmark(name="audio_processing")
        def process_audio(audio_data):
            # ... processing code ...
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            benchmark_name = name or func.__name__
            benchmark_collector = collector or _global_collector
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                benchmark_result = BenchmarkResult(
                    name=benchmark_name,
                    duration=duration,
                    metadata=metadata
                )
                benchmark_collector.add_result(benchmark_result)
        
        return wrapper
    return decorator


class Timer:
    """
    Context manager for timing code blocks.
    
    Example:
        with Timer("data_loading") as t:
            data = load_data()
        print(f"Loading took {t.duration:.4f}s")
    """
    
    def __init__(self, name: str, collector: Optional[BenchmarkCollector] = None, **metadata):
        """
        Initialize timer.
        
        Args:
            name: Name for this timing measurement
            collector: Optional BenchmarkCollector (defaults to global)
            **metadata: Additional metadata to store
        """
        self.name = name
        self.collector = collector or _global_collector
        self.metadata = metadata
        self.start_time: Optional[float] = None
        self.duration: Optional[float] = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record result."""
        self.duration = time.perf_counter() - self.start_time
        result = BenchmarkResult(
            name=self.name,
            duration=self.duration,
            metadata=self.metadata
        )
        self.collector.add_result(result)
        return False


def print_benchmark_report():
    """Print the global benchmark report."""
    print(_global_collector.report())


def clear_benchmarks():
    """Clear all global benchmarks."""
    _global_collector.clear()
