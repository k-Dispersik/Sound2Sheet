"""
Audio synthesis module for converting MIDI to audio.

This module provides functionality for synthesizing realistic piano audio
from MIDI files using FluidSynth Python bindings.
"""

from typing import Optional, Union, List
from pathlib import Path
import logging
import subprocess
import numpy as np
import soundfile as sf

try:
    from midi2audio import FluidSynth
    MIDI2AUDIO_AVAILABLE = True
except ImportError:
    MIDI2AUDIO_AVAILABLE = False
    FluidSynth = None


class AudioSynthesizer:
    """
    Synthesizer for converting MIDI files to realistic piano audio.
    
    Uses FluidSynth Python binding (midi2audio) for high-quality synthesis
    with customizable sample rate and soundfont selection.
    """
    
    def __init__(
        self,
        soundfont_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 44100,
        gain: float = 0.5
    ):
        """
        Initialize AudioSynthesizer.
        
        Args:
            soundfont_path: Path to SF2 soundfont file. If None, uses system default.
            sample_rate: Output audio sample rate in Hz
            gain: Overall gain/volume (0.0-1.0)
            
        Raises:
            ImportError: If midi2audio is not installed
            FileNotFoundError: If soundfont file not found
        """
        if not MIDI2AUDIO_AVAILABLE:
            raise ImportError(
                "midi2audio is required but not installed. Install it with:\n"
                "  pip install midi2audio"
            )
        
        self.sample_rate = sample_rate
        self.gain = gain
        
        # Find soundfont
        if soundfont_path:
            self.soundfont_path = Path(soundfont_path)
            if not self.soundfont_path.exists():
                raise FileNotFoundError(f"Soundfont file not found: {self.soundfont_path}")
        else:
            self.soundfont_path = self._find_default_soundfont()
            if not self.soundfont_path:
                raise FileNotFoundError(
                    "No soundfont found. Please provide a soundfont path or install one:\n"
                    "  Ubuntu/Debian: sudo apt-get install fluid-soundfont-gm\n"
                    "  Location: /usr/share/sounds/sf2/FluidR3_GM.sf2"
                )
        
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
        
        # Initialize FluidSynth
        self.fs = FluidSynth(sound_font=str(self.soundfont_path), sample_rate=self.sample_rate)
    
    def synthesize(
        self,
        midi_path: Union[str, Path],
        output_path: Union[str, Path],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Synthesize audio from MIDI file.
        
        Args:
            midi_path: Path to input MIDI file
            output_path: Path to output audio file (WAV)
            normalize: Whether to normalize audio to [-1, 1] range
            
        Returns:
            Audio data as numpy array
            
        Raises:
            FileNotFoundError: If MIDI file not found
            RuntimeError: If synthesis fails
        """
        midi_path = Path(midi_path)
        output_path = Path(output_path)
        
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Synthesize MIDI to WAV using FluidSynth (suppress output)
            result = subprocess.run(
                ['fluidsynth', '-ni', str(self.soundfont_path), str(midi_path), '-F', str(output_path), '-r', str(self.sample_rate)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            
        except Exception as e:
            raise RuntimeError(f"FluidSynth synthesis failed: {e}")
        
        # Load synthesized audio
        audio, sr = sf.read(str(output_path))
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Apply gain
        if self.gain != 1.0:
            audio = audio * self.gain
        
        # Normalize if requested
        if normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
        
        # Clip to valid range
        audio = np.clip(audio, -1.0, 1.0)
        
        self.logger.debug(f"Synthesized audio: {len(audio)} samples, {len(audio)/sr:.2f}s")
        
        # Re-save with processed audio
        sf.write(str(output_path), audio, sr)
        
        return audio.astype(np.float32)
    
    def _find_default_soundfont(self) -> Optional[Path]:
        """Try to find a default soundfont on the system."""
        # Common soundfont locations
        common_paths = [
            '/usr/share/sounds/sf2/FluidR3_GM.sf2',
            '/usr/share/sounds/sf2/default.sf2',
            '/usr/local/share/soundfonts/default.sf2',
            '/usr/share/soundfonts/default.sf2',
            '/usr/share/sounds/sf2/FluidR3_GS.sf2',
            Path.home() / '.soundfonts' / 'default.sf2',
        ]
        
        for path in common_paths:
            sf_path = Path(path)
            if sf_path.exists():
                return sf_path
        
        return None
    
    def synthesize_batch(
        self,
        midi_files: List[Path],
        output_dir: Path,
        normalize: bool = True,
        keep_structure: bool = True
    ) -> List[Path]:
        """
        Synthesize multiple MIDI files to audio.
        
        Args:
            midi_files: List of MIDI file paths
            output_dir: Directory to save synthesized audio files
            normalize: Whether to normalize audio
            keep_structure: Keep relative directory structure from input
            
        Returns:
            List of paths to synthesized audio files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        synthesized_files = []
        
        for i, midi_path in enumerate(midi_files):
            midi_path = Path(midi_path)
            
            # Determine output path
            if keep_structure:
                # Keep relative path structure
                try:
                    rel_path = midi_path.relative_to(midi_path.parent.parent)
                except ValueError:
                    rel_path = midi_path.name
                output_path = output_dir / rel_path.with_suffix('.wav')
            else:
                # Flat structure
                output_path = output_dir / midi_path.with_suffix('.wav').name
            
            try:
                self.synthesize(midi_path, output_path, normalize)
                synthesized_files.append(output_path)
                    
            except Exception as e:
                self.logger.error(f"Failed to synthesize {midi_path.name}: {e}")
                continue
        
        return synthesized_files
    
    def get_audio_info(self, audio_path: Union[str, Path]) -> dict:
        """
        Get information about synthesized audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio to get info
        audio, sr = sf.read(str(audio_path))
        
        info = {
            'file': str(audio_path),
            'sample_rate': sr,
            'channels': audio.ndim,
            'samples': len(audio) if audio.ndim == 1 else audio.shape[0],
            'duration': len(audio) / sr if audio.ndim == 1 else audio.shape[0] / sr,
            'dtype': str(audio.dtype),
            'max_amplitude': float(np.max(np.abs(audio))),
            'rms': float(np.sqrt(np.mean(audio ** 2))),
        }
        
        return info
