"""
Unit tests for AudioVisualizer class.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.audio_visualizer import AudioVisualizer
from core.audio_processor import AudioProcessor


class TestAudioVisualizer(unittest.TestCase):
    """Test cases for AudioVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = AudioProcessor()
        self.visualizer = AudioVisualizer(self.processor)
        
        # Create test audio (1 second of 440Hz sine wave)
        self.duration = 1.0
        self.sample_rate = self.processor.config.sample_rate
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), False)
        self.test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    def tearDown(self):
        """Clean up after each test."""
        plt.close('all')  # Close all matplotlib figures
    
    def test_initialization(self):
        """Test AudioVisualizer initialization."""
        # Test with processor
        visualizer = AudioVisualizer(self.processor)
        self.assertIsNotNone(visualizer.processor)
        self.assertEqual(visualizer.processor, self.processor)
        
        # Test without processor (should create default)
        visualizer_default = AudioVisualizer()
        self.assertIsNotNone(visualizer_default.processor)
        self.assertIsInstance(visualizer_default.processor, AudioProcessor)
    
    def test_plot_waveform(self):
        """Test waveform plotting."""
        fig = self.visualizer.plot_waveform(self.test_audio, title="Test Waveform")
        
        # Check that figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has expected elements
        axes = fig.get_axes()
        self.assertEqual(len(axes), 1)
        
        ax = axes[0]
        self.assertEqual(ax.get_xlabel(), 'Time (seconds)')
        self.assertEqual(ax.get_ylabel(), 'Amplitude')
        self.assertEqual(ax.get_title(), 'Test Waveform')
    
    def test_plot_waveform_with_save(self):
        """Test waveform plotting with save functionality."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
            fig = self.visualizer.plot_waveform(
                self.test_audio, 
                title="Test Save",
                save_path=str(temp_path)
            )
            
            # Check that file was created
            self.assertTrue(temp_path.exists())
            self.assertGreater(temp_path.stat().st_size, 0)
            
            # Clean up
            temp_path.unlink()
    
    def test_plot_spectrogram(self):
        """Test spectrogram plotting."""
        fig = self.visualizer.plot_spectrogram(self.test_audio, title="Test Spectrogram")
        
        # Check that figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has expected elements
        axes = fig.get_axes()
        self.assertGreaterEqual(len(axes), 1)  # At least main axis (colorbar might add another)
        
        # Find the main axis (not colorbar)
        main_ax = None
        for ax in axes:
            if ax.get_title() == "Test Spectrogram":
                main_ax = ax
                break
        
        self.assertIsNotNone(main_ax)
    
    def test_plot_augmentation_comparison(self):
        """Test augmentation comparison plotting."""
        # Create augmented audio (volume change)
        augmented_audio = self.processor.augment_audio(
            self.test_audio, 
            "volume", 
            volume_factor=0.5
        )
        
        fig = self.visualizer.plot_augmentation_comparison(
            self.test_audio,
            augmented_audio,
            "volume"
        )
        
        # Check that figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that we have 4 subplots (2x2)
        axes = fig.get_axes()
        self.assertGreaterEqual(len(axes), 4)  # At least 4 (might have more due to colorbars)
    
    def test_plot_multiple_augmentations(self):
        """Test multiple augmentations plotting."""
        augmentations = [
            ("volume", {"volume_factor": 0.5}),
            ("noise", {"noise_factor": 0.01})
        ]
        
        fig = self.visualizer.plot_multiple_augmentations(
            self.test_audio,
            augmentations
        )
        
        # Check that figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that we have the right number of subplot rows
        axes = fig.get_axes()
        # Should have 3 rows (original + 2 augmentations) x 2 columns = 6 main plots
        # Plus colorbars makes it more
        self.assertGreaterEqual(len(axes), 6)
    
    def test_plot_audio_from_file_waveform(self):
        """Test plotting audio from file - waveform only."""
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
            # Save test audio
            sf.write(temp_path, self.test_audio, self.sample_rate)
            
            # Plot waveform
            fig = self.visualizer.plot_audio_from_file(
                temp_path, 
                plot_type="waveform"
            )
            
            # Check that figure was created
            self.assertIsInstance(fig, plt.Figure)
            
            # Clean up
            temp_path.unlink()
    
    def test_plot_audio_from_file_spectrogram(self):
        """Test plotting audio from file - spectrogram only."""
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
            # Save test audio
            sf.write(temp_path, self.test_audio, self.sample_rate)
            
            # Plot spectrogram
            fig = self.visualizer.plot_audio_from_file(
                temp_path, 
                plot_type="spectrogram"
            )
            
            # Check that figure was created
            self.assertIsInstance(fig, plt.Figure)
            
            # Clean up
            temp_path.unlink()
    
    def test_plot_audio_from_file_both(self):
        """Test plotting audio from file - both waveform and spectrogram."""
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
            # Save test audio
            sf.write(temp_path, self.test_audio, self.sample_rate)
            
            # Plot both
            fig = self.visualizer.plot_audio_from_file(
                temp_path, 
                plot_type="both"
            )
            
            # Check that figure was created
            self.assertIsInstance(fig, plt.Figure)
            
            # Should have 2 subplots
            axes = fig.get_axes()
            self.assertGreaterEqual(len(axes), 2)
            
            # Clean up
            temp_path.unlink()
    
    def test_plot_audio_from_file_invalid_type(self):
        """Test error handling for invalid plot type."""
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
            # Save test audio
            sf.write(temp_path, self.test_audio, self.sample_rate)
            
            # Test invalid plot type
            with self.assertRaises(ValueError):
                self.visualizer.plot_audio_from_file(
                    temp_path, 
                    plot_type="invalid"
                )
            
            # Clean up
            temp_path.unlink()
    
    def test_custom_figure_size(self):
        """Test custom figure size setting."""
        custom_size = (10, 6)
        fig = self.visualizer.plot_waveform(
            self.test_audio, 
            figsize=custom_size
        )
        
        # Check figure size (allowing for small differences due to DPI)
        actual_size = fig.get_size_inches()
        self.assertAlmostEqual(actual_size[0], custom_size[0], places=1)
        self.assertAlmostEqual(actual_size[1], custom_size[1], places=1)


if __name__ == "__main__":
    unittest.main()