"""
Unit tests for noise strategy pattern implementation.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.noise_strategies import (
    NoiseStrategy,
    NoiseStrategyFactory,
    WhiteNoiseStrategy,
    PinkNoiseStrategy,
    BrownNoiseStrategy,
    AmbientNoiseStrategy,
    HumNoiseStrategy
)


class TestNoiseStrategies(unittest.TestCase):
    """Test individual noise strategy classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.length = 16000  # 1 second at 16kHz
        self.sample_rate = 16000
    
    def test_white_noise_strategy(self):
        """Test WhiteNoiseStrategy."""
        strategy = WhiteNoiseStrategy()
        
        self.assertEqual(strategy.get_name(), 'white')
        
        noise = strategy.generate(self.length, self.sample_rate)
        
        self.assertEqual(len(noise), self.length)
        self.assertEqual(noise.dtype, np.float32)
    
    def test_pink_noise_strategy(self):
        """Test PinkNoiseStrategy."""
        strategy = PinkNoiseStrategy()
        
        self.assertEqual(strategy.get_name(), 'pink')
        
        noise = strategy.generate(self.length, self.sample_rate)
        
        self.assertEqual(len(noise), self.length)
        self.assertEqual(noise.dtype, np.float32)
        self.assertLessEqual(np.max(np.abs(noise)), 1.0)
    
    def test_brown_noise_strategy(self):
        """Test BrownNoiseStrategy."""
        strategy = BrownNoiseStrategy()
        
        self.assertEqual(strategy.get_name(), 'brown')
        
        noise = strategy.generate(self.length, self.sample_rate)
        
        self.assertEqual(len(noise), self.length)
        self.assertEqual(noise.dtype, np.float32)
        self.assertLessEqual(np.max(np.abs(noise)), 1.0)
    
    def test_ambient_noise_strategy(self):
        """Test AmbientNoiseStrategy."""
        strategy = AmbientNoiseStrategy()
        
        self.assertEqual(strategy.get_name(), 'ambient')
        
        noise = strategy.generate(self.length, self.sample_rate)
        
        self.assertEqual(len(noise), self.length)
        self.assertEqual(noise.dtype, np.float32)
        self.assertLessEqual(np.max(np.abs(noise)), 1.0)
    
    def test_hum_noise_strategy(self):
        """Test HumNoiseStrategy."""
        strategy = HumNoiseStrategy()
        
        self.assertEqual(strategy.get_name(), 'hum')
        
        # Test with default frequency
        noise = strategy.generate(self.length, self.sample_rate)
        self.assertEqual(len(noise), self.length)
        self.assertEqual(noise.dtype, np.float32)
        
        # Test with custom frequency
        noise_60hz = strategy.generate(self.length, self.sample_rate, frequency=60)
        self.assertEqual(len(noise_60hz), self.length)
        
        # Different frequencies should produce different results
        self.assertFalse(np.array_equal(noise, noise_60hz))
    
    def test_strategy_consistency(self):
        """Test that strategies produce consistent output for same input."""
        strategies = [
            WhiteNoiseStrategy(),
            PinkNoiseStrategy(),
            BrownNoiseStrategy(),
            AmbientNoiseStrategy(),
            HumNoiseStrategy()
        ]
        
        for strategy in strategies:
            with self.subTest(strategy=strategy.get_name()):
                # Generate twice with same random seed
                np.random.seed(42)
                noise1 = strategy.generate(self.length, self.sample_rate)
                
                np.random.seed(42)
                noise2 = strategy.generate(self.length, self.sample_rate)
                
                # Should be identical
                np.testing.assert_array_equal(noise1, noise2)


class TestNoiseStrategyFactory(unittest.TestCase):
    """Test NoiseStrategyFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = NoiseStrategyFactory()
    
    def test_get_strategy_white(self):
        """Test getting white noise strategy."""
        strategy = self.factory.get_strategy('white')
        
        self.assertIsInstance(strategy, WhiteNoiseStrategy)
        self.assertEqual(strategy.get_name(), 'white')
    
    def test_get_strategy_pink(self):
        """Test getting pink noise strategy."""
        strategy = self.factory.get_strategy('pink')
        
        self.assertIsInstance(strategy, PinkNoiseStrategy)
        self.assertEqual(strategy.get_name(), 'pink')
    
    def test_get_strategy_brown(self):
        """Test getting brown noise strategy."""
        strategy = self.factory.get_strategy('brown')
        
        self.assertIsInstance(strategy, BrownNoiseStrategy)
        self.assertEqual(strategy.get_name(), 'brown')
    
    def test_get_strategy_ambient(self):
        """Test getting ambient noise strategy."""
        strategy = self.factory.get_strategy('ambient')
        
        self.assertIsInstance(strategy, AmbientNoiseStrategy)
        self.assertEqual(strategy.get_name(), 'ambient')
    
    def test_get_strategy_hum(self):
        """Test getting hum noise strategy."""
        strategy = self.factory.get_strategy('hum')
        
        self.assertIsInstance(strategy, HumNoiseStrategy)
        self.assertEqual(strategy.get_name(), 'hum')
    
    def test_get_strategy_invalid(self):
        """Test that invalid strategy raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.factory.get_strategy('invalid_noise_type')
        
        error_message = str(context.exception)
        self.assertIn('Unknown noise type', error_message)
        self.assertIn('invalid_noise_type', error_message)
        self.assertIn('Available:', error_message)
    
    def test_get_available_types(self):
        """Test getting list of available noise types."""
        available = self.factory.get_available_types()
        
        self.assertIsInstance(available, list)
        self.assertEqual(len(available), 5)
        
        expected_types = ['white', 'pink', 'brown', 'ambient', 'hum']
        for noise_type in expected_types:
            self.assertIn(noise_type, available)
    
    def test_register_custom_strategy(self):
        """Test registering a custom noise strategy."""
        
        class CustomNoiseStrategy(NoiseStrategy):
            """Custom noise strategy for testing."""
            
            def generate(self, length: int, sample_rate: int, **kwargs) -> np.ndarray:
                return np.zeros(length, dtype=np.float32)
            
            def get_name(self) -> str:
                return "custom"
        
        custom_strategy = CustomNoiseStrategy()
        self.factory.register_strategy(custom_strategy)
        
        # Should be able to retrieve it
        retrieved = self.factory.get_strategy('custom')
        self.assertEqual(retrieved.get_name(), 'custom')
        
        # Should be in available types
        available = self.factory.get_available_types()
        self.assertIn('custom', available)
    
    def test_factory_singleton_behavior(self):
        """Test that factory returns same strategy instances."""
        strategy1 = self.factory.get_strategy('pink')
        strategy2 = self.factory.get_strategy('pink')
        
        # Should be the same object
        self.assertIs(strategy1, strategy2)
    
    def test_factory_all_strategies(self):
        """Test that all strategies can be created and used."""
        length = 1000
        sample_rate = 16000
        
        for noise_type in self.factory.get_available_types():
            with self.subTest(noise_type=noise_type):
                strategy = self.factory.get_strategy(noise_type)
                
                # Should be able to generate noise
                noise = strategy.generate(length, sample_rate)
                
                self.assertEqual(len(noise), length)
                self.assertEqual(noise.dtype, np.float32)


class TestNoiseStrategyIntegration(unittest.TestCase):
    """Integration tests for noise strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = NoiseStrategyFactory()
        self.length = 16000
        self.sample_rate = 16000
    
    def test_different_strategies_produce_different_noise(self):
        """Test that different strategies produce different noise patterns."""
        np.random.seed(42)
        
        strategies = ['white', 'pink', 'brown', 'ambient']
        noises = {}
        
        for strategy_name in strategies:
            strategy = self.factory.get_strategy(strategy_name)
            noises[strategy_name] = strategy.generate(self.length, self.sample_rate)
        
        # All should be different from each other
        for i, name1 in enumerate(strategies):
            for name2 in strategies[i+1:]:
                with self.subTest(comparison=f"{name1} vs {name2}"):
                    self.assertFalse(
                        np.array_equal(noises[name1], noises[name2]),
                        f"{name1} and {name2} produced identical noise"
                    )
    
    def test_frequency_domain_differences(self):
        """Test that colored noises have different frequency characteristics."""
        white_strategy = self.factory.get_strategy('white')
        pink_strategy = self.factory.get_strategy('pink')
        brown_strategy = self.factory.get_strategy('brown')
        
        white = white_strategy.generate(self.length, self.sample_rate)
        pink = pink_strategy.generate(self.length, self.sample_rate)
        brown = brown_strategy.generate(self.length, self.sample_rate)
        
        # Compute power spectra
        white_fft = np.abs(np.fft.rfft(white)) ** 2
        pink_fft = np.abs(np.fft.rfft(pink)) ** 2
        brown_fft = np.abs(np.fft.rfft(brown)) ** 2
        
        # Low frequency power (first quarter of spectrum)
        white_low = np.mean(white_fft[:len(white_fft)//4])
        pink_low = np.mean(pink_fft[:len(pink_fft)//4])
        brown_low = np.mean(brown_fft[:len(brown_fft)//4])
        
        # High frequency power (last quarter of spectrum)
        white_high = np.mean(white_fft[3*len(white_fft)//4:])
        pink_high = np.mean(pink_fft[3*len(pink_fft)//4:])
        brown_high = np.mean(brown_fft[3*len(brown_fft)//4:])
        
        # Pink should have more low-frequency emphasis than white
        white_ratio = white_low / white_high
        pink_ratio = pink_low / pink_high
        brown_ratio = brown_low / brown_high
        
        # Brown > Pink > White in terms of low-frequency emphasis
        self.assertGreater(pink_ratio, white_ratio)
        self.assertGreater(brown_ratio, pink_ratio)


if __name__ == '__main__':
    unittest.main()
