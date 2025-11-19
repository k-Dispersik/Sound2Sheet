"""
Dataset loader for Sound2Sheet model training.

Provides PyTorch Dataset classes for loading audio and MIDI pairs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

from ..core.audio_processor import AudioProcessor
from ..core.noise_strategies import NoiseStrategyFactory
from .config import DataConfig, ModelConfig


class PianoDataset(Dataset):
    """
    PyTorch Dataset for piano audio and MIDI transcription.
    
    Loads audio files and corresponding MIDI ground truth from manifests.
    """
    
    def __init__(
        self,
        manifest_path: Path,
        data_config: DataConfig,
        model_config: ModelConfig,
        is_training: bool = True
    ):
        """
        Initialize PianoDataset.
        
        Args:
            manifest_path: Path to JSON manifest file
            data_config: Data configuration
            model_config: Model configuration
            is_training: Whether dataset is for training (enables augmentation)
        """
        self.manifest_path = Path(manifest_path)
        self.data_config = data_config
        self.model_config = model_config
        self.is_training = is_training
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize audio processor with compatible config
        from src.core.audio_processor import AudioProcessor, AudioConfig
        audio_config = AudioConfig()
        audio_config.sample_rate = data_config.sample_rate
        audio_config.n_mels = data_config.n_mels
        audio_config.hop_length = data_config.hop_length
        audio_config.n_fft = data_config.n_fft
        self.audio_processor = AudioProcessor(config=audio_config)
        
        # Initialize noise strategy factory for augmentation
        self.noise_factory = NoiseStrategyFactory()
        
        # Load manifest
        self.samples = self._load_manifest()
        
        # Get dataset base directory from data_config (not manifest parent, as manifests may be in metadata/ subdirectory)
        self.dataset_dir = Path(data_config.dataset_dir)
        
        self.logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")
    
    def _load_manifest(self) -> List[Dict]:
        """Load samples from manifest JSON file."""
        with open(self.manifest_path, 'r') as f:
            samples = json.load(f)
        return samples
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - mel: Mel-spectrogram tensor [n_mels, time]
                - piano_roll: Piano roll ground truth [time_frames, num_piano_keys]
                - audio_path: Path to audio file (for debugging)
        """
        sample = self.samples[idx]
        
        # Load audio
        audio_path = self.dataset_dir / sample['audio_path']
        audio = self._load_audio(audio_path)
        
        # Apply augmentation if training
        if self.is_training and self.data_config.use_augmentation:
            audio = self._augment_audio(audio)
        
        # Generate mel-spectrogram
        mel_spec = self.audio_processor.to_mel_spectrogram(audio)
        mel_spec = torch.from_numpy(mel_spec).float()
        
        # Load MIDI and convert to piano roll
        midi_path = self.dataset_dir / sample['midi_path']
        notes = self._load_midi_notes(midi_path)
        
        # Generate piano roll from notes
        # Calculate number of time frames based on mel-spectrogram
        num_time_frames = mel_spec.shape[1]  # Mel-spec is [n_mels, time]
        piano_roll = self._notes_to_piano_roll(notes, num_time_frames, audio.shape[0])
        
        return {
            'mel': mel_spec,  # [n_mels, time]
            'piano_roll': piano_roll,  # [time_frames, num_piano_keys]
            'audio_path': str(audio_path)  # For debugging/tracking
        }
    
    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """Load audio file."""
        audio = self.audio_processor.load_audio(str(audio_path))
        return audio
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply random augmentation to audio using noise strategies."""
        # Get noise type from config
        noise_type = getattr(self.data_config, 'noise_type', 'white')
        
        # If noise_type is 'random', randomly choose a type
        if noise_type == 'random':
            # Check if custom pool is defined
            noise_types_pool = getattr(self.data_config, 'noise_types_pool', None)
            if noise_types_pool and len(noise_types_pool) > 0:
                # Use custom pool
                noise_type = np.random.choice(noise_types_pool)
            else:
                # Use all available types
                available_types = self.noise_factory.get_available_types()
                noise_type = np.random.choice(available_types)
        
        # Get noise strategy
        try:
            noise_strategy = self.noise_factory.get_strategy(noise_type)
        except ValueError as e:
            self.logger.warning(f"Unknown noise type '{noise_type}', falling back to white noise: {e}")
            noise_strategy = self.noise_factory.get_strategy('white')
        
        # Generate noise
        noise_level = self.data_config.noise_scale
        sample_rate = self.data_config.sample_rate
        noise = noise_strategy.generate(len(audio), sample_rate) * noise_level
        
        # Add noise to audio
        augmented = audio + noise
        
        # Clip to valid range
        augmented = np.clip(augmented, -1.0, 1.0)
        
        return augmented
    
    def _load_midi_notes(self, midi_path: Path) -> List[Dict]:
        """
        Load MIDI notes from file.
        
        Returns list of note dictionaries with:
            - pitch: MIDI note number (21-108)
            - onset: Start time in seconds
            - offset: End time in seconds
            - velocity: Note velocity (0-127)
        """
        from mido import MidiFile
        
        midi_file = MidiFile(midi_path)
        notes = []
        
        # Get tempo (microseconds per beat)
        tempo = 500000  # Default: 120 BPM
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break
        
        # Convert ticks to seconds
        ticks_per_beat = midi_file.ticks_per_beat
        seconds_per_tick = (tempo / 1000000) / ticks_per_beat
        
        # Extract notes
        current_time = 0
        active_notes = {}  # pitch -> (onset_time, velocity)
        
        for track in midi_file.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time * seconds_per_tick
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Note starts
                    active_notes[msg.note] = (current_time, msg.velocity)
                
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # Note ends
                    if msg.note in active_notes:
                        onset, velocity = active_notes.pop(msg.note)
                        notes.append({
                            'pitch': msg.note,
                            'onset': onset,
                            'offset': current_time,
                            'velocity': velocity
                        })
        
        # Sort by onset time
        notes.sort(key=lambda x: x['onset'])
        
        return notes
    
    def _notes_to_piano_roll(
        self,
        notes: List[Dict],
        num_time_frames: int,
        audio_length_samples: int
    ) -> torch.Tensor:
        """
        Convert MIDI notes to piano roll representation.
        
        Args:
            notes: List of note dictionaries with onset/offset times in seconds
            num_time_frames: Number of time frames (from mel-spectrogram)
            audio_length_samples: Length of audio in samples
            
        Returns:
            Piano roll tensor [time_frames, num_piano_keys]
            Binary values: 1 if key is active at that frame, 0 otherwise
        """
        # Initialize piano roll
        piano_roll = np.zeros((num_time_frames, self.model_config.num_piano_keys), dtype=np.float32)
        
        # Calculate audio duration in seconds
        audio_duration_sec = audio_length_samples / self.data_config.sample_rate
        
        # Calculate frame duration in seconds
        # hop_length samples per frame
        frame_duration_sec = self.data_config.hop_length / self.data_config.sample_rate
        
        # Fill piano roll
        for note in notes:
            pitch = note['pitch']
            onset_sec = note['onset']
            offset_sec = note['offset']
            
            # Convert MIDI pitch to key index (0-87)
            if pitch < self.model_config.min_midi_note or pitch > self.model_config.max_midi_note:
                continue  # Skip notes outside piano range
            
            key_idx = pitch - self.model_config.min_midi_note
            
            # Convert onset/offset times to frame indices
            onset_frame = int(onset_sec / frame_duration_sec)
            offset_frame = int(offset_sec / frame_duration_sec)
            
            # Clip to valid frame range
            onset_frame = max(0, min(onset_frame, num_time_frames - 1))
            offset_frame = max(0, min(offset_frame, num_time_frames - 1))
            
            # Set piano roll values to 1 for active frames
            if offset_frame > onset_frame:
                piano_roll[onset_frame:offset_frame, key_idx] = 1.0
        
        return torch.from_numpy(piano_roll).float()
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for DataLoader.
        
        Handles variable-length mel-spectrograms and piano rolls by padding.
        """
        # Get max time dimension
        max_time = max(item['mel'].shape[1] for item in batch)
        
        # Pad mel spectrograms and piano rolls
        padded_mel = []
        padded_piano_roll = []
        
        for item in batch:
            mel = item['mel']
            piano_roll = item['piano_roll']
            time_steps = mel.shape[1]
            
            # Pad mel to max_time
            if time_steps < max_time:
                mel_padding = torch.zeros(mel.shape[0], max_time - time_steps)
                mel = torch.cat([mel, mel_padding], dim=1)
            
            # Pad piano roll to max_time
            if piano_roll.shape[0] < max_time:
                pr_padding = torch.zeros(max_time - piano_roll.shape[0], piano_roll.shape[1])
                piano_roll = torch.cat([piano_roll, pr_padding], dim=0)
            
            padded_mel.append(mel)
            padded_piano_roll.append(piano_roll)
        
        # Stack tensors
        batch_dict = {
            'mel': torch.stack(padded_mel),  # [batch, n_mels, time]
            'piano_roll': torch.stack(padded_piano_roll),  # [batch, time, num_piano_keys]
            'audio_path': [item['audio_path'] for item in batch]  # List of paths
        }
        
        return batch_dict


def create_dataloaders(
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_config: Data configuration
        model_config: Model configuration
        training_config: Training configuration
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get manifest paths from dataset directory
    # Try metadata subdirectory first (new format), then root (legacy)
    metadata_dir = data_config.dataset_dir / 'metadata'
    if metadata_dir.exists():
        train_manifest = metadata_dir / 'train_manifest.json'
        val_manifest = metadata_dir / 'val_manifest.json'
        test_manifest = metadata_dir / 'test_manifest.json'
    else:
        train_manifest = data_config.dataset_dir / 'train_manifest.json'
        val_manifest = data_config.dataset_dir / 'val_manifest.json'
        test_manifest = data_config.dataset_dir / 'test_manifest.json'
    
    # Create datasets
    train_dataset = PianoDataset(
        manifest_path=train_manifest,
        data_config=data_config,
        model_config=model_config,
        is_training=True
    )
    
    val_dataset = PianoDataset(
        manifest_path=val_manifest,
        data_config=data_config,
        model_config=model_config,
        is_training=False
    )
    
    test_dataset = PianoDataset(
        manifest_path=test_manifest,
        data_config=data_config,
        model_config=model_config,
        is_training=False
    )
    
    # Create dataloader kwargs (prefetch_factor only works with num_workers > 0)
    loader_kwargs = {
        'num_workers': training_config.num_workers,
        'pin_memory': training_config.pin_memory,
    }
    if training_config.num_workers > 0:
        loader_kwargs['prefetch_factor'] = training_config.prefetch_factor
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        **loader_kwargs
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config.val_batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        **loader_kwargs
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=training_config.val_batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        **loader_kwargs
    )
    
    return train_loader, val_loader, test_loader
