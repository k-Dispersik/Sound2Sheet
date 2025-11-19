"""
Tests for model configuration classes.
"""

import pytest
from pathlib import Path
from src.model.config import (
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    DataConfig
)


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_default_initialization(self):
        """Test default ModelConfig initialization."""
        config = ModelConfig()
        
        assert config.ast_model_name == 'MIT/ast-finetuned-audioset-10-10-0.4593'
        assert config.num_piano_keys == 88
        assert config.hidden_size == 768
        assert config.dropout == 0.1
        assert config.frame_duration_ms == 10.0
        assert config.classification_threshold == 0.5
        assert config.use_temporal_conv is True
        assert config.classifier_hidden_dim == 512
        assert config.device == 'cuda'
    
    def test_custom_initialization(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            hidden_size=512,
            num_piano_keys=88,
            frame_duration_ms=20.0,
            use_temporal_conv=False,
            device='cpu'
        )
        
        assert config.hidden_size == 512
        assert config.num_piano_keys == 88
        assert config.frame_duration_ms == 20.0
        assert config.use_temporal_conv is False
        assert config.device == 'cpu'
    
    def test_num_piano_keys_validation(self):
        """Test that num_piano_keys must be positive."""
        with pytest.raises(Exception):
            config = ModelConfig(num_piano_keys=-1)
    
    def test_hidden_size_validation(self):
        """Test that hidden_size must be positive."""
        with pytest.raises(Exception):
            config = ModelConfig(hidden_size=0)
    
    def test_dropout_range(self):
        """Test that dropout is in valid range."""
        config = ModelConfig(dropout=0.5)
        assert 0.0 <= config.dropout <= 1.0
        
        with pytest.raises(Exception):
            ModelConfig(dropout=1.5)
    
    def test_classifier_layers_validation(self):
        """Test that num_classifier_layers must be positive."""
        # Valid configuration
        config = ModelConfig(num_classifier_layers=3)
        assert config.num_classifier_layers == 3
        
        # Invalid configuration
        with pytest.raises(ValueError):
            ModelConfig(num_classifier_layers=0)


class TestTrainingConfig:
    """Test TrainingConfig class."""
    
    def test_default_initialization(self):
        """Test default TrainingConfig initialization."""
        config = TrainingConfig()
        
        assert config.batch_size == 16
        assert config.num_epochs == 50
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.01
        assert config.max_grad_norm == 1.0
        assert config.warmup_steps == 1000
        assert config.optimizer == 'adamw'
        assert config.scheduler == 'linear'
        assert config.use_mixed_precision is True
        assert config.gradient_accumulation_steps == 1
    
    def test_custom_initialization(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            batch_size=32,
            num_epochs=100,
            learning_rate=5e-5,
            use_mixed_precision=False
        )
        
        assert config.batch_size == 32
        assert config.num_epochs == 100
        assert config.learning_rate == 5e-5
        assert config.use_mixed_precision is False
    
    def test_learning_rate_validation(self):
        """Test learning rate must be positive."""
        with pytest.raises(Exception):
            TrainingConfig(learning_rate=-0.001)
    
    def test_batch_size_validation(self):
        """Test batch size must be positive."""
        with pytest.raises(Exception):
            TrainingConfig(batch_size=0)
    
    def test_gradient_accumulation_validation(self):
        """Test gradient accumulation steps must be positive."""
        with pytest.raises(Exception):
            TrainingConfig(gradient_accumulation_steps=-1)
    
    def test_checkpoint_dir_creation(self):
        """Test checkpoint directory path handling."""
        config = TrainingConfig(checkpoint_dir=Path('test_checkpoints'))
        assert isinstance(config.checkpoint_dir, Path)
        assert config.checkpoint_dir == Path('test_checkpoints')
    
    def test_early_stopping_config(self):
        """Test early stopping configuration."""
        config = TrainingConfig(
            early_stopping_patience=10,
            early_stopping_threshold=0.001
        )
        
        assert config.early_stopping_patience == 10
        assert config.early_stopping_threshold == 0.001
    
    def test_optimizer_choices(self):
        """Test valid optimizer choices."""
        for optimizer in ['adam', 'adamw', 'sgd']:
            config = TrainingConfig(optimizer=optimizer)
            assert config.optimizer == optimizer
    
    def test_scheduler_choices(self):
        """Test valid scheduler choices."""
        for scheduler in ['linear', 'cosine', 'constant']:
            config = TrainingConfig(scheduler=scheduler)
            assert config.scheduler == scheduler


class TestInferenceConfig:
    """Test InferenceConfig class."""
    
    def test_default_initialization(self):
        """Test default InferenceConfig initialization."""
        config = InferenceConfig()
        
        assert config.median_filter_size == 3
        assert config.min_note_duration_ms == 30.0
        assert config.output_format == 'events'
    
    def test_custom_initialization(self):
        """Test InferenceConfig with custom values."""
        config = InferenceConfig(
            median_filter_size=7,
            min_note_duration_ms=100.0,
            output_format='piano_roll'
        )
        
        assert config.median_filter_size == 7
        assert config.min_note_duration_ms == 100.0
        assert config.output_format == 'piano_roll'
    
    def test_median_filter_validation(self):
        """Test median filter size must be positive odd number."""
        config = InferenceConfig(median_filter_size=7)
        assert config.median_filter_size == 7
        
        # Should accept positive odd numbers
        config = InferenceConfig(median_filter_size=3)
        assert config.median_filter_size == 3
    
    def test_min_note_duration_validation(self):
        """Test min_note_duration_ms must be positive."""
        config = InferenceConfig(min_note_duration_ms=100.0)
        assert config.min_note_duration_ms == 100.0
        
        with pytest.raises(Exception):
            InferenceConfig(min_note_duration_ms=-1.0)
    
    def test_output_format_validation(self):
        """Test output format choices."""
        # Valid formats
        for fmt in ['piano_roll', 'events']:
            config = InferenceConfig(output_format=fmt)
            assert config.output_format == fmt


class TestDataConfig:
    """Test DataConfig class."""
    
    def test_default_initialization(self):
        """Test default DataConfig initialization."""
        config = DataConfig(dataset_dir=Path('data/datasets/test'))
        
        assert config.dataset_dir == Path('data/datasets/test')
        assert config.train_split == 0.8
        assert config.val_split == 0.1
        assert config.test_split == 0.1
        assert config.sample_rate == 16000
        assert config.n_mels == 128
        assert config.use_augmentation is True
    
    def test_custom_initialization(self):
        """Test DataConfig with custom values."""
        config = DataConfig(
            dataset_dir=Path('custom/path'),
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            sample_rate=22050,
            n_mels=256
        )
        
        assert config.dataset_dir == Path('custom/path')
        assert config.train_split == 0.7
        assert config.val_split == 0.15
        assert config.test_split == 0.15
        assert config.sample_rate == 22050
        assert config.n_mels == 256
    
    def test_splits_sum_to_one(self):
        """Test that train/val/test splits sum to 1.0."""
        config = DataConfig(
            dataset_dir=Path('test'),
            train_split=0.8,
            val_split=0.1,
            test_split=0.1
        )
        
        total = config.train_split + config.val_split + config.test_split
        assert abs(total - 1.0) < 1e-6
    
    def test_invalid_splits(self):
        """Test that invalid splits raise errors."""
        # Splits don't sum to 1.0
        with pytest.raises(Exception):
            DataConfig(
                dataset_dir=Path('test'),
                train_split=0.5,
                val_split=0.3,
                test_split=0.1
            )
    
    def test_sample_rate_validation(self):
        """Test sample rate must be positive."""
        with pytest.raises(Exception):
            DataConfig(
                dataset_dir=Path('test'),
                sample_rate=0
            )
    
    def test_n_mels_validation(self):
        """Test n_mels must be positive."""
        with pytest.raises(Exception):
            DataConfig(
                dataset_dir=Path('test'),
                n_mels=0
            )
    
    def test_augmentation_parameters(self):
        """Test augmentation configuration."""
        config = DataConfig(
            dataset_dir=Path('test'),
            use_augmentation=True,
            noise_scale=0.01,  # Fixed: was noise_scale
            pitch_shift_range=2
        )
        
        assert config.use_augmentation is True
        assert config.noise_scale == 0.01
        assert config.pitch_shift_range == 2
    
    def test_path_handling(self):
        """Test that paths are handled correctly."""
        # String path
        config1 = DataConfig(dataset_dir='data/datasets/test')
        assert isinstance(config1.dataset_dir, Path)
        
        # Path object
        config2 = DataConfig(dataset_dir=Path('data/datasets/test'))
        assert isinstance(config2.dataset_dir, Path)


class TestConfigIntegration:
    """Test configuration classes working together."""
    
    def test_configs_compatibility(self):
        """Test that all configs work together."""
        model_config = ModelConfig(hidden_size=768)
        training_config = TrainingConfig(batch_size=16)
        inference_config = InferenceConfig(median_filter_size=5)
        data_config = DataConfig(dataset_dir=Path('test'))
        
        # Model and data configs should be compatible
        assert model_config.num_piano_keys == 88
        assert data_config.sample_rate == 16000
        
        # Training and model configs
        assert training_config.batch_size > 0
        assert model_config.hidden_size > 0
    
    def test_device_consistency(self):
        """Test device setting across configs."""
        model_config = ModelConfig(device='cpu')
        assert model_config.device == 'cpu'
        
        model_config_gpu = ModelConfig(device='cuda')
        assert model_config_gpu.device == 'cuda'
    
    def test_config_serialization(self):
        """Test that configs can be converted to/from dict."""
        from dataclasses import asdict
        
        config = ModelConfig()
        config_dict = asdict(config)
        
        assert isinstance(config_dict, dict)
        assert 'hidden_size' in config_dict
        assert 'num_piano_keys' in config_dict


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
