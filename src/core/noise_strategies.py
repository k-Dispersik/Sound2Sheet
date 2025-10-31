"""
Noise augmentation strategies for audio processing.

This module implements the Strategy pattern for different types of noise augmentation,
allowing flexible and extensible noise generation without bloating the AudioProcessor class.
"""

from abc import ABC, abstractmethod
import numpy as np


class NoiseStrategy(ABC):
    """Abstract base class for noise generation strategies."""
    
    @abstractmethod
    def generate(self, length: int, sample_rate: int, **kwargs) -> np.ndarray:
        """
        Generate noise of specified length.
        
        Args:
            length: Number of samples to generate
            sample_rate: Audio sample rate in Hz
            **kwargs: Additional parameters specific to noise type
            
        Returns:
            Normalized noise array as float32
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this noise strategy."""
        pass


class WhiteNoiseStrategy(NoiseStrategy):
    """
    White noise (Gaussian) strategy.
    
    Equal energy at all frequencies. Most basic type of noise.
    """
    
    def generate(self, length: int, sample_rate: int, **kwargs) -> np.ndarray:
        """Generate white Gaussian noise."""
        return np.random.normal(0, 1, length).astype(np.float32)
    
    def get_name(self) -> str:
        return "white"


class PinkNoiseStrategy(NoiseStrategy):
    """
    Pink noise (1/f noise) strategy.
    
    Equal energy per octave. More natural than white noise,
    with emphasis on lower frequencies. Common in natural phenomena.
    """
    
    def generate(self, length: int, sample_rate: int, **kwargs) -> np.ndarray:
        """Generate pink noise using FFT filtering."""
        # Generate white noise
        white = np.random.randn(length).astype(np.float32)
        
        # Apply 1/f filter using FFT
        fft = np.fft.rfft(white)
        frequencies = np.fft.rfftfreq(length)
        
        # Avoid division by zero
        frequencies[0] = 1
        
        # Apply 1/sqrt(f) filter (pink noise characteristic)
        fft = fft / np.sqrt(frequencies)
        
        # Convert back to time domain
        pink = np.fft.irfft(fft, n=length).astype(np.float32)
        
        # Normalize
        pink = pink / np.max(np.abs(pink))
        
        return pink
    
    def get_name(self) -> str:
        return "pink"


class BrownNoiseStrategy(NoiseStrategy):
    """
    Brown noise (Brownian noise, 1/f² noise) strategy.
    
    Even stronger emphasis on low frequencies than pink noise.
    Sounds like a deep rumble or waterfall.
    """
    
    def generate(self, length: int, sample_rate: int, **kwargs) -> np.ndarray:
        """Generate brown noise using FFT filtering."""
        # Generate white noise
        white = np.random.randn(length).astype(np.float32)
        
        # Apply 1/f² filter using FFT
        fft = np.fft.rfft(white)
        frequencies = np.fft.rfftfreq(length)
        
        # Avoid division by zero
        frequencies[0] = 1
        
        # Apply 1/f filter (brown noise characteristic)
        fft = fft / frequencies
        
        # Convert back to time domain
        brown = np.fft.irfft(fft, n=length).astype(np.float32)
        
        # Normalize
        brown = brown / np.max(np.abs(brown))
        
        return brown
    
    def get_name(self) -> str:
        return "brown"


class AmbientNoiseStrategy(NoiseStrategy):
    """
    Ambient/background noise strategy.
    
    Simulates typical background noise in recording environments:
    combination of low-frequency rumble and mid-frequency hiss.
    Mix of brown noise (70%) and white noise (30%).
    """
    
    def __init__(self):
        """Initialize ambient noise with component strategies."""
        self.brown_strategy = BrownNoiseStrategy()
        self.white_strategy = WhiteNoiseStrategy()
    
    def generate(self, length: int, sample_rate: int, **kwargs) -> np.ndarray:
        """Generate ambient noise as a mix of brown and white noise."""
        # Mix of brown noise (70%) and white noise (30%)
        brown = self.brown_strategy.generate(length, sample_rate) * 0.7
        white = self.white_strategy.generate(length, sample_rate) * 0.3
        
        ambient = brown + white
        
        # Normalize
        ambient = ambient / np.max(np.abs(ambient))
        
        return ambient
    
    def get_name(self) -> str:
        return "ambient"


class HumNoiseStrategy(NoiseStrategy):
    """
    Power line hum noise strategy.
    
    Simulates the characteristic hum from electrical interference,
    common in recordings made near power lines or electronic devices.
    """
    
    def generate(self, length: int, sample_rate: int, **kwargs) -> np.ndarray:
        """
        Generate power line hum noise.
        
        Args:
            length: Number of samples to generate
            sample_rate: Audio sample rate in Hz
            **kwargs: Should contain 'frequency' (default 50 Hz)
            
        Returns:
            Hum noise array
        """
        frequency = kwargs.get('frequency', 50)  # 50Hz for Europe, 60Hz for US
        
        t = np.arange(length, dtype=np.float32) / sample_rate
        
        # Fundamental frequency
        hum = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Add harmonics (2nd and 3rd) for more realistic sound
        hum += 0.3 * np.sin(2 * np.pi * 2 * frequency * t).astype(np.float32)
        hum += 0.1 * np.sin(2 * np.pi * 3 * frequency * t).astype(np.float32)
        
        # Add slight amplitude modulation
        modulation = (1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)).astype(np.float32)
        hum = hum * modulation
        
        # Normalize
        hum = hum / np.max(np.abs(hum))
        
        return hum.astype(np.float32)
    
    def get_name(self) -> str:
        return "hum"


class NoiseStrategyFactory:
    """
    Factory for creating noise strategies.
    
    Centralizes noise strategy creation and registration,
    making it easy to add new noise types.
    """
    
    def __init__(self):
        """Initialize factory with default strategies."""
        self._strategies = {
            'white': WhiteNoiseStrategy(),
            'pink': PinkNoiseStrategy(),
            'brown': BrownNoiseStrategy(),
            'ambient': AmbientNoiseStrategy(),
            'hum': HumNoiseStrategy(),
        }
    
    def get_strategy(self, noise_type: str) -> NoiseStrategy:
        """
        Get noise strategy by type.
        
        Args:
            noise_type: Type of noise ('white', 'pink', 'brown', 'ambient', 'hum')
            
        Returns:
            NoiseStrategy instance
            
        Raises:
            ValueError: If noise type is not supported
        """
        if noise_type not in self._strategies:
            available = ', '.join(self._strategies.keys())
            raise ValueError(f"Unknown noise type: {noise_type}. Available: {available}")
        
        return self._strategies[noise_type]
    
    def register_strategy(self, strategy: NoiseStrategy) -> None:
        """
        Register a custom noise strategy.
        
        Args:
            strategy: NoiseStrategy instance to register
        """
        self._strategies[strategy.get_name()] = strategy
    
    def get_available_types(self) -> list:
        """Get list of available noise types."""
        return list(self._strategies.keys())
