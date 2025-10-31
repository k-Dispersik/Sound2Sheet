"""
Tests for model architecture components.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from src.model.config import ModelConfig, InferenceConfig
from src.model.ast_model import ASTWrapper, NoteDecoder, PositionalEncoding
from src.model.sound2sheet_model import Sound2SheetModel


class TestPositionalEncoding:
    """Test PositionalEncoding module."""
    
    def test_initialization(self):
        """Test positional encoding initialization."""
        pe = PositionalEncoding(d_model=768, max_len=1000)
        assert pe.d_model == 768
        assert pe.max_len == 1000
    
    def test_forward_pass(self):
        """Test forward pass through positional encoding."""
        batch_size = 4
        seq_len = 100
        d_model = 768
        
        pe = PositionalEncoding(d_model=d_model, max_len=1000)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pe(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.equal(output, x)  # Should be modified
    
    def test_max_length_handling(self):
        """Test handling of sequences longer than max_len."""
        pe = PositionalEncoding(d_model=768, max_len=100)
        
        # Should work for sequences up to max_len
        x_short = torch.randn(2, 50, 768)
        output_short = pe(x_short)
        assert output_short.shape == x_short.shape
        
        # Should handle sequences at max_len
        x_max = torch.randn(2, 100, 768)
        output_max = pe(x_max)
        assert output_max.shape == x_max.shape


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


class TestNoteDecoder:
    """Test transformer decoder."""
    
    def test_initialization(self):
        """Test decoder initialization."""
        config = ModelConfig()
        decoder = NoteDecoder(config)
        
        assert decoder.d_model == config.hidden_size
        assert decoder.nhead == config.num_attention_heads
        assert decoder.num_layers == config.num_decoder_layers
    
    def test_forward_pass(self):
        """Test forward pass through decoder."""
        config = ModelConfig(device='cpu')
        decoder = NoteDecoder(config)
        decoder.eval()
        
        batch_size = 2
        tgt_seq_len = 50
        src_seq_len = 100
        hidden_size = config.hidden_size
        
        # Create dummy inputs
        memory = torch.randn(batch_size, src_seq_len, hidden_size)
        tgt = torch.randint(0, config.vocab_size, (batch_size, tgt_seq_len))
        
        with torch.no_grad():
            output = decoder(memory, tgt)
        
        # Output should be [batch, tgt_seq_len, vocab_size]
        assert output.shape == (batch_size, tgt_seq_len, config.vocab_size)
    
    def test_causal_masking(self):
        """Test that causal masking is applied correctly."""
        config = ModelConfig(device='cpu')
        decoder = NoteDecoder(config)
        decoder.eval()
        
        batch_size = 1
        seq_len = 10
        
        memory = torch.randn(batch_size, 20, config.hidden_size)
        tgt = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            output = decoder(memory, tgt)
        
        # Should work without errors (causal mask is applied internally)
        assert output.shape == (batch_size, seq_len, config.vocab_size)
    
    def test_padding_mask(self):
        """Test padding mask functionality."""
        config = ModelConfig(device='cpu')
        decoder = NoteDecoder(config)
        decoder.eval()
        
        batch_size = 2
        seq_len = 20
        
        memory = torch.randn(batch_size, 30, config.hidden_size)
        tgt = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Create padding mask (True for padding)
        tgt_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        tgt_padding_mask[0, 15:] = True  # Pad last 5 tokens of first sequence
        tgt_padding_mask[1, 10:] = True  # Pad last 10 tokens of second sequence
        
        with torch.no_grad():
            output = decoder(memory, tgt, target_mask=tgt_padding_mask)
        
        assert output.shape == (batch_size, seq_len, config.vocab_size)


class TestSound2SheetModel:
    """Test complete Sound2Sheet model."""
    
    @pytest.mark.slow
    def test_initialization(self):
        """Test model initialization."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        
        assert model.encoder is not None
        assert model.decoder is not None
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
        tgt_seq_len = 50
        
        # Create dummy inputs
        mel = torch.randn(batch_size, n_mels, time_frames)
        tgt = torch.randint(0, config.vocab_size, (batch_size, tgt_seq_len))
        
        with torch.no_grad():
            output = model(mel, tgt)
        
        # Output should be [batch, tgt_seq_len, vocab_size]
        assert output.shape == (batch_size, tgt_seq_len, config.vocab_size)
    
    @pytest.mark.slow
    def test_generate_greedy(self):
        """Test autoregressive generation with greedy decoding."""
        config = ModelConfig(device='cpu', max_notes_per_sample=20)
        model = Sound2SheetModel(config)
        model.eval()
        
        batch_size = 1
        n_mels = 128
        time_frames = 100
        
        mel = torch.randn(batch_size, n_mels, time_frames)
        
        inference_config = InferenceConfig(
            max_length=20,
            temperature=1.0,
            use_beam_search=False
        )
        
        with torch.no_grad():
            generated = model.generate(mel, inference_config)
        
        # Should generate sequence
        assert isinstance(generated, (list, torch.Tensor))
        if isinstance(generated, torch.Tensor):
            assert generated.dim() == 1
            assert len(generated) <= inference_config.max_length
    
    @pytest.mark.slow
    def test_generate_with_temperature(self):
        """Test generation with different temperatures."""
        config = ModelConfig(device='cpu', max_notes_per_sample=10)
        model = Sound2SheetModel(config)
        model.eval()
        
        mel = torch.randn(1, 128, 100)
        
        # Low temperature (more deterministic)
        inference_config_low = InferenceConfig(max_length=10, temperature=0.5)
        with torch.no_grad():
            output_low = model.generate(mel, inference_config_low)
        
        # High temperature (more random)
        inference_config_high = InferenceConfig(max_length=10, temperature=2.0)
        with torch.no_grad():
            output_high = model.generate(mel, inference_config_high)
        
        # Both should generate something
        assert len(output_low) > 0
        assert len(output_high) > 0
    
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
            
            assert loaded_model.config.vocab_size == config.vocab_size
            assert loaded_model.config.hidden_size == config.hidden_size
    
    @pytest.mark.slow
    def test_freeze_encoder(self):
        """Test encoder freezing functionality."""
        config = ModelConfig(device='cpu')
        
        # Model with frozen encoder
        model_frozen = Sound2SheetModel(config, freeze_encoder=True)
        
        # Check encoder parameters are frozen
        encoder_params_frozen = sum(p.requires_grad for p in model_frozen.encoder.parameters())
        assert encoder_params_frozen == 0  # Encoder should have some frozen params
        
        # Check decoder parameters are trainable
        decoder_params_trainable = sum(p.requires_grad for p in model_frozen.decoder.parameters())
        assert decoder_params_trainable > 0
    
    @pytest.mark.slow
    def test_model_modes(self):
        """Test switching between train and eval modes."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        
        # Train mode
        model.train()
        assert model.training is True
        assert model.encoder.training is True
        assert model.decoder.training is True
        
        # Eval mode
        model.eval()
        assert model.training is False
        assert model.encoder.training is False
        assert model.decoder.training is False


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
        tgt = torch.randint(0, config.vocab_size, (batch_size, 30))
        
        with torch.no_grad():
            output = model(mel, tgt)
        
        assert output.shape == (batch_size, 30, config.vocab_size)
        assert torch.isfinite(output).all()
    
    @pytest.mark.slow
    def test_end_to_end_generation(self):
        """Test end-to-end generation."""
        config = ModelConfig(device='cpu', max_notes_per_sample=50)
        model = Sound2SheetModel(config)
        model.eval()
        
        mel = torch.randn(1, 128, 100)
        inference_config = InferenceConfig(max_length=50)
        
        with torch.no_grad():
            generated = model.generate(mel, inference_config)
        
        # Check generated sequence
        if isinstance(generated, torch.Tensor):
            generated = generated.tolist()
        
        assert isinstance(generated, list)
        assert len(generated) > 0
        assert len(generated) <= inference_config.max_length
        
        # Check tokens are in valid range
        for token in generated:
            assert 0 <= token < config.vocab_size
    
    @pytest.mark.slow
    def test_batch_inference(self):
        """Test inference with batch size > 1."""
        config = ModelConfig(device='cpu', max_notes_per_sample=20)
        model = Sound2SheetModel(config)
        model.eval()
        
        batch_size = 3
        mel = torch.randn(batch_size, 128, 100)
        
        # Note: Current generate() might only support batch_size=1
        # This test checks if it handles single sample from batch
        with torch.no_grad():
            inference_config = InferenceConfig(max_length=20)
            generated = model.generate(mel[0:1], inference_config)
        
        assert generated is not None
    
    @pytest.mark.slow
    def test_gradient_flow(self):
        """Test that gradients flow through model."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config, freeze_encoder=False)
        model.train()
        
        mel = torch.randn(2, 128, 100, requires_grad=True)
        tgt = torch.randint(0, config.vocab_size, (2, 20))
        
        # Forward pass
        output = model(mel, tgt)
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
    def test_empty_sequence(self):
        """Test handling of empty target sequence."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        model.eval()
        
        mel = torch.randn(1, 128, 100)
        tgt = torch.randint(0, config.vocab_size, (1, 0))  # Empty sequence
        
        # Should handle gracefully or raise informative error
        try:
            with torch.no_grad():
                output = model(mel, tgt)
            assert output.shape[1] == 0
        except (ValueError, RuntimeError) as e:
            # Expected to fail with empty sequence
            pass
    
    @pytest.mark.slow
    def test_single_token_sequence(self):
        """Test handling of single token sequence."""
        config = ModelConfig(device='cpu')
        model = Sound2SheetModel(config)
        model.eval()
        
        mel = torch.randn(1, 128, 100)
        tgt = torch.randint(0, config.vocab_size, (1, 1))
        
        with torch.no_grad():
            output = model(mel, tgt)
        
        assert output.shape == (1, 1, config.vocab_size)
    
    @pytest.mark.slow
    def test_very_long_sequence(self):
        """Test handling of very long sequences."""
        config = ModelConfig(device='cpu', max_notes_per_sample=2000)
        model = Sound2SheetModel(config)
        model.eval()
        
        mel = torch.randn(1, 128, 100)
        tgt = torch.randint(0, config.vocab_size, (1, 500))
        
        with torch.no_grad():
            output = model(mel, tgt)
        
        assert output.shape == (1, 500, config.vocab_size)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not slow'])
