"""
Audio processing module for Sound2Sheet.

This module provides functionality for loading, preprocessing, and converting
audio files into mel-spectrograms suitable for machine learning model input.
"""

from typing import Optional, Tuple, Union
from pathlib import Path
import logging
import numpy as np
import yaml
import librosa
import soundfile as sf


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
        self.logger = logging.getLogger(__name__)
        
        # Configure logging if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
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
        supported_formats = {'.wav', '.mp3', '.m4a'}
        if file_path.suffix.lower() not in supported_formats:
            return False
            
        return True
    
    def load_audio(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load audio file and convert to numpy array.
        
        Args:
            file_path: Path to audio file (.wav, .mp3, .m4a)
            
        Returns:
            Audio data as numpy array
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is not supported
        """
        if not self.validate_audio_file(file_path):
            self.logger.error(f"Audio file validation failed: {file_path}")
            raise FileNotFoundError(f"Audio file not found or unsupported format: {file_path}")
        
        self.logger.info(f"Loading audio file: {file_path}")
        
        try:
            # Load audio using librosa with target sample rate and mono conversion
            audio, sr = librosa.load(
                str(file_path), 
                sr=self.config.sample_rate, 
                mono=True,
                dtype=np.float32
            )
            self.logger.info(f"Successfully loaded audio: {len(audio)} samples, {len(audio)/self.config.sample_rate:.2f}s")
            return audio
            
        except Exception as e:
            self.logger.error(f"Failed to load audio file {file_path}: {str(e)}")
            raise ValueError(f"Failed to load audio file {file_path}: {str(e)}")
    
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
    
    def resample_audio(self, audio: np.ndarray, current_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Input audio array
            current_sr: Current sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio array
        """
        if current_sr == target_sr:
            return audio
            
        return librosa.resample(audio, orig_sr=current_sr, target_sr=target_sr)
    
    def to_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to mel-spectrogram.
        
        Args:
            audio: Input audio array
            
        Returns:
            Mel-spectrogram with shape (n_mels, time_frames)
        """
        self.logger.debug(f"Generating mel-spectrogram for audio: {len(audio)} samples")
        # Apply pre-emphasis if configured
        if self.config.pre_emphasis > 0:
            audio = self.apply_pre_emphasis(audio)
        
        # Convert to mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.f_min,
            fmax=self.config.f_max
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        self.logger.debug(f"Generated mel-spectrogram shape: {mel_spec_db.shape}")
        return mel_spec_db
    
    def augment_audio(self, audio: np.ndarray, augment_type: str, **kwargs) -> np.ndarray:
        """
        Apply audio augmentation.
        
        Args:
            audio: Input audio array
            augment_type: Type of augmentation ('pitch_shift', 'time_stretch', 'noise', 'volume')
            **kwargs: Additional parameters for specific augmentation types
            
        Returns:
            Augmented audio array
        """
        self.logger.debug(f"Applying {augment_type} augmentation with params: {kwargs}")
        
        if augment_type == "pitch_shift":
            # Pitch shifting in semitones
            n_steps = kwargs.get('n_steps', 0)  # Default: no shift
            return librosa.effects.pitch_shift(
                audio, 
                sr=self.config.sample_rate, 
                n_steps=n_steps
            )
            
        elif augment_type == "time_stretch":
            # Time stretching
            rate = kwargs.get('rate', 1.0)  # Default: no stretch
            return librosa.effects.time_stretch(audio, rate=rate)
            
        elif augment_type == "noise":
            # Add Gaussian noise
            noise_factor = kwargs.get('noise_factor', 0.005)
            noise = np.random.normal(0, noise_factor, audio.shape).astype(np.float32)
            return audio + noise
            
        elif augment_type == "volume":
            # Volume adjustment
            volume_factor = kwargs.get('volume_factor', 1.0)
            return audio * volume_factor
            
        else:
            raise ValueError(f"Unknown augmentation type: {augment_type}")
    
    def process_audio(self, file_path: Union[str, Path], return_spectrogram: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Complete audio processing pipeline.
        
        Args:
            file_path: Path to audio file
            return_spectrogram: If True, returns mel-spectrogram, else returns processed audio
            
        Returns:
            Processed audio array or mel-spectrogram, or both as tuple
        """
        # Load audio
        audio = self.load_audio(file_path)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        if return_spectrogram:
            # Convert to mel-spectrogram
            mel_spec = self.to_mel_spectrogram(audio)
            return mel_spec
        else:
            # Return processed audio
            audio = self.apply_pre_emphasis(audio)
            return audio