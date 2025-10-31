"""
Audio processing module for Sound2Sheet.

This module provides functionality for loading, preprocessing, and converting
audio files into mel-spectrograms suitable for machine learning model input.
"""

from typing import Optional, Tuple, Union
from pathlib import Path
import numpy as np
import yaml


class AudioConfig:
    """Configuration class for audio processing parameters."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AudioConfig with default values or from config file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Default values
        self.sample_rate: int = 16000
        self.n_fft: int = 1024  
        self.hop_length: int = 320
        self.n_mels: int = 128
        self.f_min: float = 0.0
        self.f_max: float = 8000.0
        self.normalize: bool = True
        self.pre_emphasis: float = 0.97
        
        # Load from config file if provided
        if config_path:
            self._load_from_config(config_path)
    
    def _load_from_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            audio_config = config.get('audio', {})
            for key, value in audio_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)


class AudioProcessor:
    """
    Audio processing pipeline for music transcription.
    
    Handles loading, preprocessing, and converting audio files to mel-spectrograms
    with support for various audio formats.
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize AudioProcessor with configuration.
        
        Args:
            config: Audio processing configuration. If None, uses defaults.
        """
        self.config = config or AudioConfig()
    
    def validate_audio_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate if the audio file exists and has supported format.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if file is valid, False otherwise
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False
            
        # Check file extension
        supported_formats = {'.wav', '.mp3'}
        if file_path.suffix.lower() not in supported_formats:
            return False
            
        return True
    
    def load_audio(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load audio file and convert to numpy array.
        
        Args:
            file_path: Path to audio file (.wav, .mp3)
            
        Returns:
            Audio data as numpy array
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is not supported
        """
        if not self.validate_audio_file(file_path):
            raise FileNotFoundError(f"Audio file not found or unsupported format: {file_path}")
        
        # Placeholder implementation - will be completed with librosa
        # For now, return a dummy audio array
        duration = 5.0  # 5 seconds
        num_samples = int(duration * self.config.sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32)
        
        return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        if not self.config.normalize:
            return audio
            
        # RMS normalization
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            audio = audio / (rms * 10)  # Scale factor for comfortable listening
            
        # Clip to [-1, 1] range
        audio = np.clip(audio, -1.0, 1.0)
        return audio
    
    def apply_pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply pre-emphasis filter to audio.
        
        Args:
            audio: Input audio array
            
        Returns:
            Pre-emphasized audio array
        """
        if self.config.pre_emphasis <= 0:
            return audio
            
        # Apply pre-emphasis: y[n] = x[n] - α * x[n-1]
        emphasized = np.zeros_like(audio)
        emphasized[0] = audio[0]
        emphasized[1:] = audio[1:] - self.config.pre_emphasis * audio[:-1]
        
        return emphasized
    
    def process_audio(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Complete audio processing pipeline.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Processed audio array
        """
        # Load audio
        audio = self.load_audio(file_path)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Apply pre-emphasis
        audio = self.apply_pre_emphasis(audio)
        
        return audio