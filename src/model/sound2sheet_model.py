"""
Sound2Sheet main model.

Combines AST encoder and Note decoder for end-to-end transcription.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, TYPE_CHECKING
import logging

from .config import ModelConfig
from .ast_model import ASTWrapper, NoteDecoder

if TYPE_CHECKING:
    from .config import InferenceConfig


class Sound2SheetModel(nn.Module):
    """
    Main Sound2Sheet model for piano transcription.
    
    Architecture:
        1. AST Encoder: Extracts features from mel-spectrogram
        2. Note Decoder: Predicts MIDI note sequence
    """
    
    def __init__(self, config: ModelConfig, freeze_encoder: bool = False):
        """
        Initialize Sound2Sheet model.
        
        Args:
            config: Model configuration
            freeze_encoder: If True, freeze AST encoder during training
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.encoder = ASTWrapper(config, freeze_encoder=freeze_encoder)
        self.decoder = NoteDecoder(config)
        
        self.logger.info(f"Initialized Sound2Sheet model")
        self.logger.info(f"  Encoder: AST ({config.ast_model_name})")
        self.logger.info(f"  Decoder: {config.num_decoder_layers} layers")
        self.logger.info(f"  Vocab size: {config.vocab_size}")
        self.logger.info(f"  Freeze encoder: {freeze_encoder}")
    
    def forward(
        self,
        mel: torch.Tensor,
        target_notes: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through model.
        
        Args:
            mel: Mel-spectrogram [batch, n_mels, time]
            target_notes: Target notes for teacher forcing [batch, max_notes]
            
        Returns:
            logits: Note predictions [batch, max_notes, vocab_size]
        """
        # Encode audio
        encoder_output = self.encoder(mel)  # [batch, enc_seq_len, hidden_size]
        
        # Decode notes (decoder expects: encoder_hidden_states, target_notes)
        logits = self.decoder(encoder_output, target_notes)  # [batch, max_notes, vocab_size]
        
        return logits
    
    def generate(
        self,
        mel: torch.Tensor,
        inference_config: 'InferenceConfig'
    ) -> List[int]:
        """
        Generate note sequence from audio (inference mode).
        
        Args:
            mel: Mel-spectrogram [batch, n_mels, time]
            inference_config: Inference configuration
            
        Returns:
            Generated note token IDs as list
        """
        self.eval()
        
        with torch.no_grad():
            # Encode audio
            encoder_output = self.encoder(mel)  # [batch, enc_seq_len, hidden_size]
            
            # Autoregressive generation (greedy decoding for now)
            batch_size = mel.size(0)
            current_tokens = torch.full(
                (batch_size, 1),
                self.config.sos_token_id,
                dtype=torch.long,
                device=mel.device
            )
            
            max_length = inference_config.max_length
            
            for _ in range(max_length):
                # Decode (decoder expects: encoder_hidden_states, target_notes)
                logits = self.decoder(encoder_output, current_tokens)  # [batch, seq_len, vocab_size]
                
                # Get next token (greedy decoding)
                next_token_logits = logits[:, -1, :] / inference_config.temperature  # [batch, vocab_size]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch, 1]
                
                # Append to sequence
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                
                # Stop if EOS produced
                if (next_token == self.config.eos_token_id).all():
                    break
            
            # Convert to list (remove SOS token)
            generated_notes = current_tokens[0, 1:].cpu().tolist()  # Take first batch item
        
        return generated_notes
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable and total parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
            'encoder_total': encoder_params,
            'encoder_trainable': encoder_trainable,
            'decoder_total': decoder_params,
            'decoder_trainable': decoder_trainable
        }
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: Optional[ModelConfig] = None) -> 'Sound2SheetModel':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Model configuration (if None, load from checkpoint)
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Use config from checkpoint if not provided
        if config is None:
            config = checkpoint.get('config')
            if config is None:
                raise ValueError("No config found in checkpoint and none provided")
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        **kwargs
    ):
        """
        Save model checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
            optimizer: Optional optimizer to save
            epoch: Current epoch number
            **kwargs: Additional items to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
