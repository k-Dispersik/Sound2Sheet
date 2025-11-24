"""
Tests for model architecture components.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from src.model.config import ModelConfig, InferenceConfig
from src.model.ast_model import ASTWrapper, PianoRollClassifier
from src.model.sound2sheet_model import Sound2SheetModel


class TestPianoRollClassifier:
    """Test PianoRollClassifier module."""
    
    def test_initialization(self):
        """Test piano roll classifier initialization."""
        config = ModelConfig()
        classifier = PianoRollClassifier(config)
        assert classifier.num_piano_keys == config.num_piano_keys
        assert classifier.hidden_size == config.hidden_size
    
    def test_forward_pass(self):
        """Test forward pass through classifier."""
        batch_size = 4
        seq_len = 100
        hidden_size = 768
        
        config = ModelConfig(hidden_size=hidden_size, device='cpu')
        classifier = PianoRollClassifier(config)
        classifier.eval()
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        with torch.no_grad():
            output = classifier(x)
        
        # Output should be [batch, seq_len, num_piano_keys]
        assert output.shape == (batch_size, seq_len, 88)
    
    def test_temporal_conv(self):
        """Test classifier with temporal convolution."""
        config = ModelConfig(use_temporal_conv=True, device='cpu')
        classifier = PianoRollClassifier(config)
        classifier.eval()
        
        batch_size = 2
        seq_len = 50
        hidden_size = config.hidden_size
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        with torch.no_grad():
            output = classifier(x)
        
        assert output.shape == (batch_size, seq_len, 88)
    
    def test_no_temporal_conv(self):
        """Test classifier without temporal convolution."""
        config = ModelConfig(use_temporal_conv=False, device='cpu')
        classifier = PianoRollClassifier(config)
        classifier.eval()
        
        batch_size = 2
        seq_len = 50
        hidden_size = config.hidden_size
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        with torch.no_grad():
            output = classifier(x)
        
        assert output.shape == (batch_size, seq_len, 88)


class TestASTWrapper:
    """Test AST encoder wrapper."""
    
    @pytest.mark.slow
    def test_initialization(self):
        """Test AST wrapper initialization."""
        config = ModelConfig()
        ast_wrapper = ASTWrapper(config)
        
        assert ast_wrapper.hidden_size == config.hidden_size
        assert ast_wrapper.projection is not None
    
    @pytest.mark.slow
    def test_encoder_freezing(self):
        """Test that encoder can be frozen."""
        config = ModelConfig()
        
        # Unfrozen encoder
        ast_unfrozen = ASTWrapper(config, freeze_encoder=False)
        for param in ast_unfrozen.ast.parameters():
            assert param.requires_grad is True
        
        # Frozen encoder
        ast_frozen = ASTWrapper(config, freeze_encoder=True)
        for param in ast_frozen.ast.parameters():
            assert param.requires_grad is False
    
    @pytest.mark.slow
    def test_forward_pass(self):
        """Test forward pass through AST encoder."""
        config = ModelConfig(device='cpu')
        ast_wrapper = ASTWrapper(config)
        ast_wrapper.eval()
        
        batch_size = 2
        n_mels = 128
        time_frames = 1024  # AST expects max_length=1024
        
        # Create dummy mel-spectrogram
        mel = torch.randn(batch_size, n_mels, time_frames)
        
        with torch.no_grad():
            output = ast_wrapper(mel)
        
        # Output should be [batch, seq_len, hidden_size]
        assert output.dim() == 3
        assert output.shape[0] == batch_size
        assert output.shape[2] == config.hidden_size
    
    @pytest.mark.slow
    def test_output_shape_variability(self):
        """Test that output shape is consistent due to padding to max_length."""
        config = ModelConfig(device='cpu')
        ast_wrapper = ASTWrapper(config)
        ast_wrapper.eval()
        
        batch_size = 1
        n_mels = 128
        
        with torch.no_grad():
            # Short sequence (will be padded to max_length)
            mel_short = torch.randn(batch_size, n_mels, 50)
            output_short = ast_wrapper(mel_short)
            
            # Long sequence (will also be padded/truncated to max_length)
            mel_long = torch.randn(batch_size, n_mels, 200)
            output_long = ast_wrapper(mel_long)
        
        # Both should have same sequence length after padding to max_length
        assert output_short.shape[1] == output_long.shape[1]


class TestSound2SheetModel:
    """Test complete Sound2Sheet model."""
    
    @pytest.mark.slow
    def test_initialization(self):
        """Test model initialization."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        
        assert model.encoder is not None
        assert model.classifier is not None
        assert model.config == config
    
    @pytest.mark.slow
    def test_forward_pass_training(self):
        """Test forward pass in training mode."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        model.eval()  # Use eval for deterministic behavior
        
        batch_size = 2
        n_mels = 128
        time_frames = 100
        
        # Create dummy mel-spectrogram
        mel = torch.randn(batch_size, n_mels, time_frames)
        
        with torch.no_grad():
            output = model(mel)
        
        # Output should be [batch, time, num_piano_keys]
        assert output.dim() == 3
        assert output.shape[0] == batch_size
        assert output.shape[2] == 88
    
    @pytest.mark.slow
    def test_predict_with_events(self):
        """Test prediction with event extraction."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        model.eval()
        
        batch_size = 1
        n_mels = 128
        time_frames = 100
        
        mel = torch.randn(batch_size, n_mels, time_frames)
        
        inference_config = InferenceConfig(output_format='events')
        
        with torch.no_grad():
            piano_roll, events = model.predict(mel, inference_config)
        
        # Should return piano roll and events
        assert piano_roll is not None
        assert events is not None
        assert piano_roll.shape[-1] == 88  # num_piano_keys
    
    @pytest.mark.slow
    def test_predict_with_piano_roll_only(self):
        """Test prediction with piano roll output only."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        model.eval()
        
        mel = torch.randn(1, 128, 100)
        
        inference_config = InferenceConfig(output_format='piano_roll')
        
        with torch.no_grad():
            piano_roll, events = model.predict(mel, inference_config)
        
        # Should return piano roll, events might be None
        assert piano_roll is not None
        assert piano_roll.shape[-1] == 88
    
    @pytest.mark.slow
    def test_count_parameters(self):
        """Test parameter counting."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        
        params = model.count_parameters()
        
        assert 'total' in params
        assert 'trainable' in params
        assert 'frozen' in params
        
        assert params['total'] > 0
        assert params['trainable'] >= 0
        assert params['frozen'] >= 0
        assert params['total'] == params['trainable'] + params['frozen']
    
    @pytest.mark.slow
    def test_save_and_load_checkpoint(self):
        """Test saving and loading model checkpoints."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / 'test_checkpoint.pt'
            model.save_checkpoint(str(checkpoint_path))
            
            assert checkpoint_path.exists()
            
            # Load checkpoint
            loaded_model = Sound2SheetModel.from_pretrained(str(checkpoint_path))
            
            assert loaded_model.config.num_piano_keys == config.num_piano_keys
            assert loaded_model.config.hidden_size == config.hidden_size
    
    @pytest.mark.slow
    def test_freeze_encoder(self):
        """Test encoder freezing functionality."""
        config = ModelConfig(device='cpu')
        
        # Model with frozen encoder
        model_frozen = Sound2SheetModel(config, freeze_encoder=True)
        
        # Check encoder parameters are frozen
        encoder_params_frozen = sum(p.requires_grad for p in model_frozen.encoder.parameters())
        assert encoder_params_frozen == 0  # Encoder should have all frozen params
        
        # Check classifier parameters are trainable
        classifier_params_trainable = sum(p.requires_grad for p in model_frozen.classifier.parameters())
        assert classifier_params_trainable > 0
    
    @pytest.mark.slow
    def test_model_modes(self):
        """Test switching between train and eval modes."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        
        # Train mode
        model.train()
        assert model.training is True
        assert model.encoder.training is True
        assert model.classifier.training is True
        
        # Eval mode
        model.eval()
        assert model.training is False
        assert model.encoder.training is False
        assert model.classifier.training is False


class TestModelIntegration:
    """Integration tests for complete model pipeline."""
    
    @pytest.mark.slow
    def test_end_to_end_forward(self):
        """Test end-to-end forward pass."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        model.eval()
        
        # Simulate realistic batch
        batch_size = 4
        mel = torch.randn(batch_size, 128, 100)
        
        with torch.no_grad():
            output = model(mel)
        
        # Output should be [batch, time, 88]
        assert output.dim() == 3
        assert output.shape[0] == batch_size
        assert output.shape[2] == 88
        assert torch.isfinite(output).all()
    
    @pytest.mark.slow
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction with event extraction."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        model.eval()
        
        mel = torch.randn(1, 128, 100)
        inference_config = InferenceConfig(output_format='events')
        
        with torch.no_grad():
            piano_roll, events = model.predict(mel, inference_config)

        # Remove batch dimension for test
        piano_roll = piano_roll.squeeze(0)

        print(piano_roll)

        # Check piano roll
        assert piano_roll is not None
        assert piano_roll.dim() == 2
        assert piano_roll.shape[-1] == 88

        # Check events
        assert events is not None
        assert isinstance(events, list)
    
    @pytest.mark.slow
    def test_gradient_flow(self):
        """Test that gradients flow through model."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config, freeze_encoder=False)
        model.train()
        
        mel = torch.randn(2, 128, 100, requires_grad=True)
        
        # Forward pass
        output = model(mel)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Invalid gradient for {name}"


class TestModelEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.slow
    def test_very_short_audio(self):
        """Test handling of very short audio input."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        model.eval()
        
        # Very short mel-spectrogram (10 frames)
        mel = torch.randn(1, 128, 10)
        
        with torch.no_grad():
            output = model(mel)
        
        # Should still produce output
        assert output.shape[0] == 1
        assert output.shape[2] == 88
    
    @pytest.mark.slow
    def test_very_long_audio(self):
        """Test handling of very long audio."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        model.eval()
        
        # Very long mel-spectrogram (500 frames)
        mel = torch.randn(1, 128, 500)
        
        with torch.no_grad():
            output = model(mel)
        
        # Should produce output with corresponding time dimension
        assert output.shape[0] == 1
        assert output.shape[2] == 88


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not slow'])
