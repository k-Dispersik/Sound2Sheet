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
                - audio: Mel-spectrogram tensor [n_mels, time]
                - notes: MIDI note sequence tensor [max_notes]
                - note_mask: Mask for valid notes [max_notes]
                - metadata: Sample metadata dictionary
        """
        sample = self.samples[idx]
        
        # Load audio
        audio_path = self.dataset_dir / sample['audio_path']
        audio = self._load_audio(audio_path)
        
        # Apply augmentation if training (always apply when enabled)
        if self.is_training and self.data_config.use_augmentation:
            audio = self._augment_audio(audio)
        
        # Generate mel-spectrogram
        mel_spec = self.audio_processor.to_mel_spectrogram(audio)
        mel_spec = torch.from_numpy(mel_spec).float()
        
        # Load MIDI ground truth
        midi_path = self.dataset_dir / sample['midi_path']
        notes = self._load_midi_notes(midi_path)
        
        # Convert to tensor with padding
        notes_tensor, note_mask = self._prepare_note_sequence(notes)
        
        return {
            'mel': mel_spec,  # Changed from 'audio' to 'mel'
            'notes': notes_tensor,
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
    
    def _prepare_note_sequence(
        self,
        notes: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare note sequence for model input.
        
        Returns:
            - notes_tensor: [max_notes] tensor with SOS, note IDs, EOS, and padding
            - note_mask: [max_notes] boolean mask (True for valid notes)
        """
        max_notes = self.model_config.max_notes_per_sample
        
        # Extract pitches and convert to token IDs
        # Token ID = MIDI note - min_midi_note + 1 (reserve 0 for padding)
        note_ids = [self.model_config.sos_token_id]  # Start with SOS token
        
        for note in notes[:max_notes-2]:  # Leave space for SOS and EOS
            pitch = note['pitch']
            # Clip to valid range
            pitch = max(self.model_config.min_midi_note, 
                       min(pitch, self.model_config.max_midi_note))
            token_id = pitch - self.model_config.min_midi_note + 1
            note_ids.append(token_id)
        
        note_ids.append(self.model_config.eos_token_id)  # End with EOS token
        
        # Create tensor with padding
        notes_tensor = torch.full((max_notes,), self.model_config.pad_token_id, dtype=torch.long)
        notes_tensor[:len(note_ids)] = torch.tensor(note_ids, dtype=torch.long)
        
        # Create mask
        note_mask = torch.zeros(max_notes, dtype=torch.bool)
        note_mask[:len(note_ids)] = True
        
        return notes_tensor, note_mask
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for DataLoader.
        
        Handles variable-length mel-spectrograms by padding.
        """
        # Get max time dimension
        max_time = max(item['mel'].shape[1] for item in batch)
        
        # Pad mel spectrograms
        padded_mel = []
        mel_lengths = []
        
        for item in batch:
            mel = item['mel']
            time_steps = mel.shape[1]
            mel_lengths.append(time_steps)
            
            # Pad to max_time
            if time_steps < max_time:
                padding = torch.zeros(mel.shape[0], max_time - time_steps)
                mel = torch.cat([mel, padding], dim=1)
            
            padded_mel.append(mel)
        
        # Stack tensors
        batch_dict = {
            'mel': torch.stack(padded_mel),  # [batch, n_mels, time]
            'notes': torch.stack([item['notes'] for item in batch]),  # [batch, max_notes]
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
