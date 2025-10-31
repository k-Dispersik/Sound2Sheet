"""
Audio visualization module for Sound2Sheet.

This module provides functionality for visualizing audio waveforms,
spectrograms, and augmentation effects.
"""

from typing import Optional, Tuple, List, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import librosa
import librosa.display

from .audio_processor import AudioProcessor, AudioConfig


class AudioVisualizer:
    """
    Audio visualization utilities for Sound2Sheet.
    
    Provides methods for plotting waveforms, spectrograms, and comparing
    different audio processing effects.
    """
    
    def __init__(self, audio_processor: Optional[AudioProcessor] = None):
        """
        Initialize AudioVisualizer.
        
        Args:
            audio_processor: AudioProcessor instance. If None, creates default.
        """
        self.processor = audio_processor or AudioProcessor()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_waveform(self, 
                     audio: np.ndarray, 
                     sr: int = None,
                     title: str = "Audio Waveform",
                     figsize: Tuple[int, int] = (12, 4),
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot audio waveform.
        
        Args:
            audio: Audio data array
            sr: Sample rate (uses processor config if None)
            title: Plot title
            figsize: Figure size tuple
            save_path: Path to save plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        if sr is None:
            sr = self.processor.config.sample_rate
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create time axis
        time = np.linspace(0, len(audio) / sr, len(audio))
        
        # Plot waveform
        ax.plot(time, audio, linewidth=0.5, alpha=0.8)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add some statistics as text
        stats_text = f'Duration: {len(audio)/sr:.2f}s\nMax: {np.max(audio):.3f}\nMin: {np.min(audio):.3f}\nRMS: {np.sqrt(np.mean(audio**2)):.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_spectrogram(self, 
                        audio: np.ndarray,
                        sr: int = None,
                        title: str = "Mel-Spectrogram", 
                        figsize: Tuple[int, int] = (12, 6),
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot mel-spectrogram.
        
        Args:
            audio: Audio data array
            sr: Sample rate (uses processor config if None)
            title: Plot title
            figsize: Figure size tuple
            save_path: Path to save plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        if sr is None:
            sr = self.processor.config.sample_rate
            
        # Generate mel-spectrogram
        mel_spec = self.processor.to_mel_spectrogram(audio)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot spectrogram
        img = librosa.display.specshow(
            mel_spec,
            sr=sr,
            hop_length=self.processor.config.hop_length,
            x_axis='time',
            y_axis='mel',
            ax=ax,
            cmap='viridis'
        )
        
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_augmentation_comparison(self,
                                   original_audio: np.ndarray,
                                   augmented_audio: np.ndarray, 
                                   augmentation_type: str,
                                   sr: int = None,
                                   figsize: Tuple[int, int] = (15, 8),
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison between original and augmented audio.
        
        Args:
            original_audio: Original audio data
            augmented_audio: Augmented audio data
            augmentation_type: Type of augmentation applied
            sr: Sample rate (uses processor config if None)
            figsize: Figure size tuple
            save_path: Path to save plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        if sr is None:
            sr = self.processor.config.sample_rate
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Audio Augmentation Comparison: {augmentation_type.title()}', fontsize=16)
        
        # Original waveform
        time_orig = np.linspace(0, len(original_audio) / sr, len(original_audio))
        axes[0, 0].plot(time_orig, original_audio, linewidth=0.5, alpha=0.8, color='blue')
        axes[0, 0].set_title('Original Waveform')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Augmented waveform
        time_aug = np.linspace(0, len(augmented_audio) / sr, len(augmented_audio))
        axes[0, 1].plot(time_aug, augmented_audio, linewidth=0.5, alpha=0.8, color='red')
        axes[0, 1].set_title(f'{augmentation_type.title()} Waveform')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Original spectrogram
        mel_spec_orig = self.processor.to_mel_spectrogram(original_audio)
        img1 = librosa.display.specshow(
            mel_spec_orig,
            sr=sr,
            hop_length=self.processor.config.hop_length,
            x_axis='time',
            y_axis='mel',
            ax=axes[1, 0],
            cmap='viridis'
        )
        axes[1, 0].set_title('Original Spectrogram')
        
        # Augmented spectrogram
        mel_spec_aug = self.processor.to_mel_spectrogram(augmented_audio)
        img2 = librosa.display.specshow(
            mel_spec_aug,
            sr=sr,
            hop_length=self.processor.config.hop_length,
            x_axis='time',
            y_axis='mel',
            ax=axes[1, 1],
            cmap='viridis'
        )
        axes[1, 1].set_title(f'{augmentation_type.title()} Spectrogram')
        
        # Add colorbars
        fig.colorbar(img1, ax=axes[1, 0], format='%+2.0f dB')
        fig.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_multiple_augmentations(self,
                                   audio: np.ndarray,
                                   augmentations: List[Tuple[str, dict]],
                                   sr: int = None,
                                   figsize: Tuple[int, int] = (16, 10),
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple augmentation effects on the same audio.
        
        Args:
            audio: Original audio data
            augmentations: List of (augmentation_type, kwargs) tuples
            sr: Sample rate (uses processor config if None)
            figsize: Figure size tuple
            save_path: Path to save plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        if sr is None:
            sr = self.processor.config.sample_rate
            
        n_augs = len(augmentations)
        fig, axes = plt.subplots(n_augs + 1, 2, figsize=figsize)
        fig.suptitle('Multiple Audio Augmentations Comparison', fontsize=16)
        
        # Original audio plots
        time_orig = np.linspace(0, len(audio) / sr, len(audio))
        axes[0, 0].plot(time_orig, audio, linewidth=0.5, alpha=0.8, color='blue')
        axes[0, 0].set_title('Original Waveform')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        mel_spec_orig = self.processor.to_mel_spectrogram(audio)
        img_orig = librosa.display.specshow(
            mel_spec_orig,
            sr=sr,
            hop_length=self.processor.config.hop_length,
            x_axis='time',
            y_axis='mel',
            ax=axes[0, 1],
            cmap='viridis'
        )
        axes[0, 1].set_title('Original Spectrogram')
        fig.colorbar(img_orig, ax=axes[0, 1], format='%+2.0f dB')
        
        # Augmented audio plots
        colors = plt.cm.tab10(np.linspace(0, 1, n_augs))
        
        for i, (aug_type, aug_kwargs) in enumerate(augmentations):
            # Apply augmentation
            aug_audio = self.processor.augment_audio(audio, aug_type, **aug_kwargs)
            
            # Plot waveform
            time_aug = np.linspace(0, len(aug_audio) / sr, len(aug_audio))
            axes[i + 1, 0].plot(time_aug, aug_audio, linewidth=0.5, alpha=0.8, color=colors[i])
            axes[i + 1, 0].set_title(f'{aug_type.title()} Waveform')
            axes[i + 1, 0].set_ylabel('Amplitude')
            axes[i + 1, 0].grid(True, alpha=0.3)
            
            # Plot spectrogram
            mel_spec_aug = self.processor.to_mel_spectrogram(aug_audio)
            img_aug = librosa.display.specshow(
                mel_spec_aug,
                sr=sr,
                hop_length=self.processor.config.hop_length,
                x_axis='time',
                y_axis='mel',
                ax=axes[i + 1, 1],
                cmap='viridis'
            )
            axes[i + 1, 1].set_title(f'{aug_type.title()} Spectrogram')
            fig.colorbar(img_aug, ax=axes[i + 1, 1], format='%+2.0f dB')
        
        # Set x-label for bottom plots
        axes[-1, 0].set_xlabel('Time (s)')
        axes[-1, 1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_audio_from_file(self,
                           file_path: Union[str, Path],
                           plot_type: str = "both",
                           figsize: Tuple[int, int] = (15, 8),
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Load and plot audio from file.
        
        Args:
            file_path: Path to audio file
            plot_type: Type of plot ("waveform", "spectrogram", or "both")
            figsize: Figure size tuple
            save_path: Path to save plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        # Load audio
        audio = self.processor.load_audio(file_path)
        file_name = Path(file_path).name
        
        if plot_type == "waveform":
            return self.plot_waveform(audio, title=f"Waveform: {file_name}", 
                                    figsize=figsize, save_path=save_path)
        elif plot_type == "spectrogram":
            return self.plot_spectrogram(audio, title=f"Spectrogram: {file_name}",
                                       figsize=figsize, save_path=save_path)
        elif plot_type == "both":
            fig, axes = plt.subplots(2, 1, figsize=figsize)
            fig.suptitle(f'Audio Analysis: {file_name}', fontsize=16)
            
            # Waveform
            sr = self.processor.config.sample_rate
            time = np.linspace(0, len(audio) / sr, len(audio))
            axes[0].plot(time, audio, linewidth=0.5, alpha=0.8)
            axes[0].set_ylabel('Amplitude')
            axes[0].set_title('Waveform')
            axes[0].grid(True, alpha=0.3)
            
            # Spectrogram
            mel_spec = self.processor.to_mel_spectrogram(audio)
            img = librosa.display.specshow(
                mel_spec,
                sr=sr,
                hop_length=self.processor.config.hop_length,
                x_axis='time',
                y_axis='mel',
                ax=axes[1],
                cmap='viridis'
            )
            axes[1].set_title('Mel-Spectrogram')
            fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")