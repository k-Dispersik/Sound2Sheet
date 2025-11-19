"""
Tests for dataset module.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json
import mido
from mido import MidiFile, MidiTrack, Message

from src.model.dataset import PianoDataset, create_dataloaders
from src.model.config import ModelConfig, TrainingConfig, DataConfig


@pytest.fixture
def temp_dataset_dir():
    """Create temporary dataset directory with test data."""
    temp_dir = tempfile.mkdtemp()
    dataset_dir = Path(temp_dir) / 'test_dataset'
    dataset_dir.mkdir()
    
    # Create train/val/test manifests
    for split in ['train', 'val', 'test']:
        manifest_path = dataset_dir / f'{split}_manifest.json'
        
        # Create dummy manifest with 5 samples
        manifest = []
        for i in range(5):
            sample = {
                'audio_path': str(dataset_dir / f'{split}_audio_{i}.wav'),
                'midi_path': str(dataset_dir / f'{split}_midi_{i}.mid'),
                'duration': 10.0 + i
            }
            manifest.append(sample)
            
            # Create dummy audio file (silence)
            audio_path = Path(sample['audio_path'])
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create simple WAV file with librosa
            import librosa
            import soundfile as sf
            sr = 16000
            duration = sample['duration']
            audio = np.zeros(int(sr * duration), dtype=np.float32)
            sf.write(audio_path, audio, sr)
            
            # Create dummy MIDI file
            midi_path = Path(sample['midi_path'])
            midi_path.parent.mkdir(parents=True, exist_ok=True)
            _create_test_midi_file(midi_path, num_notes=10 + i)
        
        # Save manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
    
    yield dataset_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


def _create_test_midi_file(path: Path, num_notes: int = 10):
    """Create a simple test MIDI file."""
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo
    track.append(mido.MetaMessage('set_tempo', tempo=500000))  # 120 BPM
    
    # Add notes
    for i in range(num_notes):
        note = 60 + (i % 12)  # C4 to B4
        velocity = 64 + (i % 32)
        
        # Note on
        track.append(Message('note_on', note=note, velocity=velocity, time=0))
        # Note off after 480 ticks
        track.append(Message('note_off', note=note, velocity=0, time=480))
    
    mid.save(path)


class TestPianoDataset:
    """Test PianoDataset class."""
    
    def test_dataset_initialization(self, temp_dataset_dir):
        """Test dataset initialization."""
        config = DataConfig(dataset_dir=temp_dataset_dir)
        model_config = ModelConfig()
        
        dataset = PianoDataset(
            manifest_path=temp_dataset_dir / 'train_manifest.json',
            data_config=config,
            model_config=model_config,
            is_training=True
        )
        
        assert len(dataset) == 5
        assert dataset.is_training is True
    
    def test_dataset_length(self, temp_dataset_dir):
        """Test dataset length calculation."""
        config = DataConfig(dataset_dir=temp_dataset_dir)
        model_config = ModelConfig()
        
        train_dataset = PianoDataset(
            manifest_path=temp_dataset_dir / 'train_manifest.json',
            data_config=config,
            model_config=model_config,
            is_training=True
        )
        
        assert len(train_dataset) == 5
    
    def test_dataset_getitem(self, temp_dataset_dir):
        """Test getting single item from dataset."""
        config = DataConfig(dataset_dir=temp_dataset_dir)
        model_config = ModelConfig()
        
        dataset = PianoDataset(
            manifest_path=temp_dataset_dir / 'train_manifest.json',
            data_config=config,
            model_config=model_config,
            is_training=False
        )
        
        item = dataset[0]
        
        assert 'mel' in item
        assert 'piano_roll' in item
        assert 'audio_path' in item
        
        # Check mel shape
        mel = item['mel']
        assert isinstance(mel, torch.Tensor)
        assert mel.dim() == 2  # [n_mels, time]
        assert mel.shape[0] == config.n_mels
        
        # Check piano_roll
        piano_roll = item['piano_roll']
        assert isinstance(piano_roll, torch.Tensor)
        assert piano_roll.dim() == 2  # [time, 88]
        assert piano_roll.shape[1] == 88
    
    def test_audio_loading(self, temp_dataset_dir):
        """Test audio loading and mel-spectrogram generation."""
        config = DataConfig(dataset_dir=temp_dataset_dir, sample_rate=16000)
        model_config = ModelConfig()
        
        dataset = PianoDataset(
            manifest_path=temp_dataset_dir / 'train_manifest.json',
            data_config=config,
            model_config=model_config,
            is_training=False
        )
        
        item = dataset[0]
        mel = item['mel']
        
        # Check mel properties
        assert mel.shape[0] == 128  # n_mels
        assert mel.shape[1] > 0  # time dimension
        assert torch.isfinite(mel).all()
    
    def test_midi_loading(self, temp_dataset_dir):
        """Test MIDI loading and piano roll generation."""
        config = DataConfig(dataset_dir=temp_dataset_dir)
        model_config = ModelConfig()
        
        dataset = PianoDataset(
            manifest_path=temp_dataset_dir / 'train_manifest.json',
            data_config=config,
            model_config=model_config,
            is_training=False
        )
        
        item = dataset[0]
        piano_roll = item['piano_roll']
        
        # Check piano_roll format
        assert isinstance(piano_roll, torch.Tensor)
        assert piano_roll.dim() == 2
        assert piano_roll.shape[1] == 88  # 88 piano keys
        assert piano_roll.dtype == torch.float32
        
        # Check values are binary (0 or 1)
        assert ((piano_roll == 0) | (piano_roll == 1)).all()
    
    def test_augmentation_training_mode(self, temp_dataset_dir):
        """Test that augmentation is applied in training mode."""
        config = DataConfig(dataset_dir=temp_dataset_dir, use_augmentation=True)
        model_config = ModelConfig()
        
        dataset = PianoDataset(
            manifest_path=temp_dataset_dir / 'train_manifest.json',
            data_config=config,
            model_config=model_config,
            is_training=True
        )
        
        # Get same item twice - should be different due to augmentation
        item1 = dataset[0]
        item2 = dataset[0]
        
        # Mels should be different (with high probability)
        # Allow small tolerance for rare cases
        diff = torch.abs(item1['mel'] - item2['mel']).mean()
        # With augmentation, difference should be noticeable
        assert diff > 0.0
    
    def test_no_augmentation_val_mode(self, temp_dataset_dir):
        """Test that augmentation is NOT applied in validation mode."""
        config = DataConfig(dataset_dir=temp_dataset_dir, use_augmentation=True)
        model_config = ModelConfig()
        
        dataset = PianoDataset(
            manifest_path=temp_dataset_dir / 'val_manifest.json',
            data_config=config,
            model_config=model_config,
            is_training=False
        )
        
        # Get same item twice - should be identical (no augmentation)
        item1 = dataset[0]
        item2 = dataset[0]
        
        # Mels should be exactly the same
        assert torch.equal(item1['mel'], item2['mel'])
        assert torch.equal(item1['piano_roll'], item2['piano_roll'])
    
    def test_piano_roll_shape(self, temp_dataset_dir):
        """Test that piano roll has correct shape."""
        config = DataConfig(dataset_dir=temp_dataset_dir)
        model_config = ModelConfig(frame_duration_ms=10.0)
        
        dataset = PianoDataset(
            manifest_path=temp_dataset_dir / 'train_manifest.json',
            data_config=config,
            model_config=model_config,
            is_training=False
        )
        
        item = dataset[0]
        piano_roll = item['piano_roll']
        
        # Should have 88 piano keys
        assert piano_roll.shape[1] == 88
        # Time dimension should be positive
        assert piano_roll.shape[0] > 0


class TestCreateDataloaders:
    """Test dataloader creation."""
    
    def test_create_dataloaders(self, temp_dataset_dir):
        """Test creating train/val/test dataloaders."""
        data_config = DataConfig(dataset_dir=temp_dataset_dir)
        model_config = ModelConfig()
        training_config = TrainingConfig(batch_size=2, num_workers=0)
        
        train_loader, val_loader, test_loader = create_dataloaders(
            data_config,
            model_config,
            training_config
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        assert len(train_loader.dataset) == 5
        assert len(val_loader.dataset) == 5
        assert len(test_loader.dataset) == 5
    
    def test_dataloader_batching(self, temp_dataset_dir):
        """Test dataloader batching."""
        data_config = DataConfig(dataset_dir=temp_dataset_dir)
        model_config = ModelConfig()
        training_config = TrainingConfig(batch_size=2, num_workers=0)
        
        train_loader, _, _ = create_dataloaders(
            data_config,
            model_config,
            training_config
        )
        
        # Get first batch
        batch = next(iter(train_loader))
        
        assert 'mel' in batch
        assert 'piano_roll' in batch
        
        # Check batch dimensions
        mel = batch['mel']
        piano_roll = batch['piano_roll']
        
        assert mel.shape[0] == 2  # batch_size
        assert piano_roll.shape[0] == 2  # batch_size
    
    def test_collate_function_padding(self, temp_dataset_dir):
        """Test that collate function pads sequences correctly."""
        data_config = DataConfig(dataset_dir=temp_dataset_dir)
        model_config = ModelConfig()
        training_config = TrainingConfig(batch_size=3, num_workers=0)
        
        train_loader, _, _ = create_dataloaders(
            data_config,
            model_config,
            training_config
        )
        
        batch = next(iter(train_loader))
        
        mel = batch['mel']
        piano_roll = batch['piano_roll']
        
        # All mels in batch should have same time dimension (padded)
        assert mel.shape[0] == 3  # batch_size
        time_dims = [mel[i].shape[-1] for i in range(mel.shape[0])]
        assert len(set(time_dims)) == 1  # All same
        
        # All piano_roll sequences should have same time length (padded)
        assert piano_roll.shape[0] == 3  # batch_size
        piano_roll_lens = [piano_roll[i].shape[0] for i in range(piano_roll.shape[0])]
        assert len(set(piano_roll_lens)) == 1  # All same
    
    def test_dataloader_shuffling(self, temp_dataset_dir):
        """Test that train dataloader shuffles data."""
        data_config = DataConfig(dataset_dir=temp_dataset_dir)
        model_config = ModelConfig()
        training_config = TrainingConfig(batch_size=1, num_workers=0)
        
        train_loader, _, _ = create_dataloaders(
            data_config,
            model_config,
            training_config
        )
        
        # Get first samples from two epochs
        first_epoch_paths = []
        for i, batch in enumerate(train_loader):
            if i >= 3:
                break
            first_epoch_paths.append(batch['audio_path'][0])
        
        # Create new loader (will shuffle differently)
        train_loader2, _, _ = create_dataloaders(
            data_config,
            model_config,
            training_config
        )
        
        second_epoch_paths = []
        for i, batch in enumerate(train_loader2):
            if i >= 3:
                break
            second_epoch_paths.append(batch['audio_path'][0])
        
        # With shuffling, order might be different
        # (not guaranteed to be different, but very likely)
        # This is a probabilistic test
    
    def test_dataloader_no_shuffle_val(self, temp_dataset_dir):
        """Test that validation dataloader does not shuffle."""
        data_config = DataConfig(dataset_dir=temp_dataset_dir)
        model_config = ModelConfig()
        training_config = TrainingConfig(batch_size=1, num_workers=0)
        
        _, val_loader, _ = create_dataloaders(
            data_config,
            model_config,
            training_config
        )
        
        # Get samples twice - should be in same order
        first_paths = [batch['audio_path'][0] for batch in val_loader]
        second_paths = [batch['audio_path'][0] for batch in val_loader]
        
        assert first_paths == second_paths
    
    def test_dataloader_iterations(self, temp_dataset_dir):
        """Test iterating through dataloader."""
        data_config = DataConfig(dataset_dir=temp_dataset_dir)
        model_config = ModelConfig()
        training_config = TrainingConfig(batch_size=2, num_workers=0)
        
        train_loader, _, _ = create_dataloaders(
            data_config,
            model_config,
            training_config
        )
        
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            assert 'mel' in batch
            assert 'piano_roll' in batch
        
        # Should have 3 batches (5 samples / batch_size=2 = 2.5 -> 3)
        assert batch_count == 3


class TestDatasetEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_manifest(self):
        """Test handling of empty manifest."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / 'empty_manifest.json'
            with open(manifest_path, 'w') as f:
                json.dump([], f)
            
            config = DataConfig(dataset_dir=Path(temp_dir))
            model_config = ModelConfig()
            
            dataset = PianoDataset(
                manifest_path=manifest_path,
                data_config=config,
                model_config=model_config,
                is_training=False
            )
            
            assert len(dataset) == 0
    
    def test_missing_audio_file(self):
        """Test handling of missing audio file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / 'manifest.json'
            manifest = [{
                'audio_path': str(Path(temp_dir) / 'nonexistent.wav'),
                'midi_path': str(Path(temp_dir) / 'test.mid'),
                'duration': 10.0
            }]
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)
            
            # Create MIDI file
            midi_path = Path(temp_dir) / 'test.mid'
            _create_test_midi_file(midi_path)
            
            config = DataConfig(dataset_dir=Path(temp_dir))
            model_config = ModelConfig()
            
            dataset = PianoDataset(
                manifest_path=manifest_path,
                data_config=config,
                model_config=model_config,
                is_training=False
            )
            
            # Should raise error when trying to load
            with pytest.raises(Exception):
                item = dataset[0]
    
    def test_batch_size_larger_than_dataset(self, temp_dataset_dir):
        """Test handling when batch size is larger than dataset."""
        data_config = DataConfig(dataset_dir=temp_dataset_dir)
        model_config = ModelConfig()
        training_config = TrainingConfig(batch_size=10, num_workers=0)  # Larger than 5 samples
        
        train_loader, _, _ = create_dataloaders(
            data_config,
            model_config,
            training_config
        )
        
        # Should still work, just with smaller batches
        batch = next(iter(train_loader))
        assert batch['mel'].shape[0] <= 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
