"""
Audio Spectrogram Transformer (AST) wrapper for Sound2Sheet.

Wraps pretrained AST model from Hugging Face for feature extraction
and Piano Roll classification head for frame-level note detection.
"""

import torch
import torch.nn as nn
from transformers import ASTModel, ASTConfig
from typing import Optional
import logging

from .config import ModelConfig


class ASTWrapper(nn.Module):
    """
    Wrapper for pretrained Audio Spectrogram Transformer.
    
    Uses AST to extract audio features from mel-spectrograms,
    then passes to decoder for note prediction.
    """
    
    def __init__(self, config: ModelConfig, freeze_encoder: bool = False):
        """
        Initialize AST wrapper.
        
        Args:
            config: Model configuration
            freeze_encoder: If True, freeze AST weights during training
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load pretrained AST
        self.logger.info(f"Loading AST model: {config.ast_model_name}")
        self.ast = ASTModel.from_pretrained(config.ast_model_name)
        
        # Optionally freeze encoder
        if freeze_encoder:
            self.logger.info("Freezing AST encoder weights")
            for param in self.ast.parameters():
                param.requires_grad = False
        
        self.hidden_size = self.ast.config.hidden_size
        self.d_model = self.hidden_size  # Alias for compatibility
        
        # Projection layer to match decoder dimensions if needed
        if self.hidden_size != config.hidden_size:
            self.projection = nn.Linear(self.hidden_size, config.hidden_size)
        else:
            self.projection = nn.Identity()
    
    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract features from mel-spectrogram.
        
        Args:
            mel_spectrogram: Input mel-spectrogram [batch, n_mels, time]
            attention_mask: Optional attention mask [batch, time]
            
        Returns:
            Encoded features [batch, seq_len, hidden_size]
        """
        # AST expects input shape [batch, num_mel_bins, time] with time=max_length (1024)
        # Pad or truncate to match expected length
        max_length = self.ast.config.max_length
        current_length = mel_spectrogram.shape[2]
        
        if current_length < max_length:
            # Pad with zeros on the right
            padding = max_length - current_length
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding))
            
            # Update attention mask if provided
            if attention_mask is not None:
                attention_mask = torch.nn.functional.pad(attention_mask, (0, padding))
        elif current_length > max_length:
            # Truncate
            mel_spectrogram = mel_spectrogram[:, :, :max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_length]
        
        # Forward through AST
        outputs = self.ast(mel_spectrogram, attention_mask=attention_mask)
        
        # Get sequence output (not pooled output)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Project if needed
        hidden_states = self.projection(hidden_states)
        
        return hidden_states
    
    def get_attention_mask(self, audio_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Create attention mask from audio lengths.
        
        Args:
            audio_lengths: Actual lengths of audio sequences [batch]
            max_length: Maximum sequence length
            
        Returns:
            Attention mask [batch, max_length]
        """
        batch_size = audio_lengths.size(0)
        mask = torch.arange(max_length, device=audio_lengths.device)[None, :] < audio_lengths[:, None]
        return mask.long()


class PianoRollClassifier(nn.Module):
    """
    Piano Roll classification head for frame-level note detection.
    
    Takes AST encoded features and predicts binary activations for each
    of 88 piano keys at each time frame.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Piano Roll classifier.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_piano_keys = config.num_piano_keys
        
        # Optional temporal convolution for context
        if config.use_temporal_conv:
            self.temporal_conv = nn.Conv1d(
                in_channels=config.hidden_size,
                out_channels=config.hidden_size,
                kernel_size=config.temporal_conv_kernel,
                padding=config.temporal_conv_kernel // 2,
                groups=1
            )
            self.temporal_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.temporal_conv = None
        
        # Multi-layer classifier
        layers = []
        in_dim = config.hidden_size
        
        for i in range(config.num_classifier_layers - 1):
            layers.extend([
                nn.Linear(in_dim, config.classifier_hidden_dim),
                nn.LayerNorm(config.classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            in_dim = config.classifier_hidden_dim
        
        # Final projection to piano keys (no activation - use BCEWithLogitsLoss)
        layers.append(nn.Linear(in_dim, config.num_piano_keys))
        
        self.classifier = nn.Sequential(*layers)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized PianoRollClassifier")
        self.logger.info(f"  Input dim: {config.hidden_size}")
        self.logger.info(f"  Output keys: {config.num_piano_keys}")
        self.logger.info(f"  Classifier layers: {config.num_classifier_layers}")
        self.logger.info(f"  Use temporal conv: {config.use_temporal_conv}")
    
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for piano roll classification.
        
        Args:
            encoder_output: Features from AST [batch, seq_len, hidden_size]
            
        Returns:
            Logits for piano roll [batch, seq_len, num_piano_keys]
            Each value represents probability of key being active at that frame
        """
        # Apply temporal convolution for context if enabled
        if self.temporal_conv is not None:
            # Conv1d expects [batch, channels, seq_len]
            x = encoder_output.transpose(1, 2)  # [batch, hidden_size, seq_len]
            x = self.temporal_conv(x)
            x = x.transpose(1, 2)  # [batch, seq_len, hidden_size]
            x = self.temporal_norm(x)
            x = x + encoder_output  # Residual connection
        else:
            x = encoder_output
        
        # Apply classifier to each time frame
        logits = self.classifier(x)  # [batch, seq_len, num_piano_keys]
        
        return logits
