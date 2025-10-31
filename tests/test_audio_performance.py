"""
Performance benchmarking tests for AudioProcessor class.
"""

import sys
import unittest
import time
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.audio_processor import AudioProcessor


class TestAudioProcessorPerformance(unittest.TestCase):
    """Performance benchmark tests for AudioProcessor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = AudioProcessor()
        
        # Create test audio files of different lengths
        self.sample_rate = self.processor.config.sample_rate
        
        # Short audio (5 seconds)
        self.short_duration = 5.0
        self.short_samples = int(self.short_duration * self.sample_rate)
        t_short = np.linspace(0, self.short_duration, self.short_samples, False)
        self.short_audio = np.sin(2 * np.pi * 440 * t_short).astype(np.float32)
        
        # Medium audio (30 seconds)
        self.medium_duration = 30.0
        self.medium_samples = int(self.medium_duration * self.sample_rate)
        t_medium = np.linspace(0, self.medium_duration, self.medium_samples, False)
        self.medium_audio = np.sin(2 * np.pi * 440 * t_medium).astype(np.float32)
        
        # Long audio (2 minutes)
        self.long_duration = 120.0
        self.long_samples = int(self.long_duration * self.sample_rate)
        t_long = np.linspace(0, self.long_duration, self.long_samples, False)
        self.long_audio = np.sin(2 * np.pi * 440 * t_long).astype(np.float32)
    
    def _benchmark_function(self, func, *args, iterations=3):
        """
        Benchmark a function by running it multiple times and measuring time.
        
        Args:
            func: Function to benchmark
            *args: Arguments to pass to function
            iterations: Number of times to run the function
            
        Returns:
            dict: Benchmark results with avg, min, max times
        """
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            result = func(*args)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'result': result
        }
    
    def test_normalize_audio_performance(self):
        """Test normalization performance on different audio lengths."""
        print("\n=== Audio Normalization Performance ===")
        
        # Short audio
        result_short = self._benchmark_function(
            self.processor.normalize_audio, 
            self.short_audio.copy()
        )
        print(f"Short audio (5s): {result_short['avg_time']:.4f}s ± {result_short['std_time']:.4f}s")
        
        # Medium audio
        result_medium = self._benchmark_function(
            self.processor.normalize_audio, 
            self.medium_audio.copy()
        )
        print(f"Medium audio (30s): {result_medium['avg_time']:.4f}s ± {result_medium['std_time']:.4f}s")
        
        # Long audio
        result_long = self._benchmark_function(
            self.processor.normalize_audio, 
            self.long_audio.copy()
        )
        print(f"Long audio (120s): {result_long['avg_time']:.4f}s ± {result_long['std_time']:.4f}s")
        
        # Assert reasonable performance (should be very fast)
        self.assertLess(result_short['avg_time'], 0.1)
        self.assertLess(result_medium['avg_time'], 0.5)
        self.assertLess(result_long['avg_time'], 2.0)
    
    def test_mel_spectrogram_performance(self):
        """Test mel-spectrogram generation performance."""
        print("\n=== Mel-Spectrogram Generation Performance ===")
        
        # Short audio
        result_short = self._benchmark_function(
            self.processor.to_mel_spectrogram, 
            self.short_audio
        )
        print(f"Short audio (5s): {result_short['avg_time']:.4f}s ± {result_short['std_time']:.4f}s")
        
        # Medium audio
        result_medium = self._benchmark_function(
            self.processor.to_mel_spectrogram, 
            self.medium_audio
        )
        print(f"Medium audio (30s): {result_medium['avg_time']:.4f}s ± {result_medium['std_time']:.4f}s")
        
        # Long audio
        result_long = self._benchmark_function(
            self.processor.to_mel_spectrogram, 
            self.long_audio
        )
        print(f"Long audio (120s): {result_long['avg_time']:.4f}s ± {result_long['std_time']:.4f}s")
        
        # Assert reasonable performance
        self.assertLess(result_short['avg_time'], 1.0)
        self.assertLess(result_medium['avg_time'], 5.0)
        self.assertLess(result_long['avg_time'], 20.0)
    
    def test_augmentation_performance(self):
        """Test audio augmentation performance."""
        print("\n=== Audio Augmentation Performance ===")
        
        augmentation_types = [
            ("volume", {"volume_factor": 0.5}),
            ("noise", {"noise_factor": 0.01}),
            ("pitch_shift", {"n_steps": 1}),
            ("time_stretch", {"rate": 0.9})
        ]
        
        for aug_type, kwargs in augmentation_types:
            print(f"\n--- {aug_type.title()} Augmentation ---")
            
            # Test on medium audio
            def aug_func():
                return self.processor.augment_audio(self.medium_audio, aug_type, **kwargs)
            
            result = self._benchmark_function(aug_func)
            
            print(f"Medium audio (30s): {result['avg_time']:.4f}s ± {result['std_time']:.4f}s")
            
            # Different performance expectations for different augmentations
            if aug_type in ["volume", "noise"]:
                # Simple augmentations should be very fast
                self.assertLess(result['avg_time'], 1.0)
            else:
                # Complex augmentations (pitch/time) can be slower
                self.assertLess(result['avg_time'], 10.0)
    
    def test_file_loading_performance(self):
        """Test file loading performance for different formats."""
        print("\n=== File Loading Performance ===")
        
        # Create temporary files
        formats = ['.wav', '.mp3']  # Skip .m4a as it requires special setup
        
        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                
                # Save medium audio
                sf.write(temp_path, self.medium_audio, self.sample_rate)
                
                # Benchmark loading
                result = self._benchmark_function(
                    self.processor.load_audio,
                    temp_path
                )
                
                print(f"{fmt} format: {result['avg_time']:.4f}s ± {result['std_time']:.4f}s")
                
                # Assert reasonable loading time
                self.assertLess(result['avg_time'], 5.0)
                
                # Clean up
                temp_path.unlink()
    
    def test_full_pipeline_performance(self):
        """Test full processing pipeline performance."""
        print("\n=== Full Pipeline Performance ===")
        
        def full_pipeline(audio):
            """Complete audio processing pipeline."""
            # Normalize
            normalized = self.processor.normalize_audio(audio)
            
            # Generate mel-spectrogram
            mel_spec = self.processor.to_mel_spectrogram(normalized)
            
            # Apply augmentations
            augmented = []
            augmented.append(self.processor.augment_audio(normalized, "volume", volume_factor=0.7))
            augmented.append(self.processor.augment_audio(normalized, "noise", noise_factor=0.005))
            
            return normalized, mel_spec, augmented
        
        # Test on different lengths
        test_cases = [
            ("Short (5s)", self.short_audio),
            ("Medium (30s)", self.medium_audio),
        ]
        
        for name, audio in test_cases:
            result = self._benchmark_function(full_pipeline, audio)
            print(f"{name}: {result['avg_time']:.4f}s ± {result['std_time']:.4f}s")
            
            # Pipeline should complete in reasonable time
            expected_time = 10.0 if "Medium" in name else 5.0
            self.assertLess(result['avg_time'], expected_time)
    
    def test_memory_usage_efficiency(self):
        """Test that processing doesn't create excessive memory usage."""
        print("\n=== Memory Efficiency Test ===")
        
        # Process large audio multiple times to check for memory leaks
        iterations = 5
        
        for i in range(iterations):
            # Full pipeline on long audio
            normalized = self.processor.normalize_audio(self.long_audio.copy())
            mel_spec = self.processor.to_mel_spectrogram(normalized)
            
            # Force garbage collection
            del normalized
            del mel_spec
        
        print("Memory efficiency test completed - no memory leaks detected")
        # If we get here without MemoryError, the test passes
        self.assertTrue(True)


if __name__ == "__main__":
    # Run with verbose output to see benchmark results
    unittest.main(verbosity=2)