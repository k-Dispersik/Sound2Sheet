"""
Audio Spectrogram Transformer (AST) wrapper for Sound2Sheet.

Wraps pretrained AST model from Hugging Face for feature extraction.
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


class NoteDecoder(nn.Module):
    """
    Transformer decoder for predicting note sequences.
    
    Takes AST encoded audio features and generates MIDI note predictions.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize note decoder.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.d_model = self.hidden_size  # Alias for compatibility
        self.nhead = config.num_attention_heads  # Alias for compatibility
        self.num_layers = config.num_decoder_layers  # Alias for compatibility
        
        # Note embedding (for teacher forcing during training)
        self.note_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.hidden_size,
            max_len=config.max_sequence_length
        )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers
        )
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        target_notes: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode note sequence from audio features.
        
        Args:
            encoder_hidden_states: Features from AST [batch, enc_seq_len, hidden_size]
            target_notes: Target note sequence for teacher forcing [batch, dec_seq_len]
            target_mask: Mask for target notes [batch, dec_seq_len]
            encoder_attention_mask: Mask for encoder outputs [batch, enc_seq_len]
            
        Returns:
            Logits for note predictions [batch, dec_seq_len, vocab_size]
        """
        batch_size = encoder_hidden_states.size(0)
        
        # If target_notes provided, use teacher forcing
        if target_notes is not None:
            # Embed target notes
            tgt_embedded = self.note_embedding(target_notes)  # [batch, dec_seq_len, hidden_size]
            tgt_embedded = self.pos_encoder(tgt_embedded)
            tgt_embedded = self.dropout(tgt_embedded)
            
            # Create causal mask for decoder (prevent looking ahead)
            dec_seq_len = target_notes.size(1)
            causal_mask = self._generate_square_subsequent_mask(dec_seq_len).to(target_notes.device)
            
            # Create target key padding mask
            if target_mask is not None:
                tgt_key_padding_mask = ~target_mask
            else:
                tgt_key_padding_mask = None
            
            # Create memory key padding mask
            if encoder_attention_mask is not None:
                memory_key_padding_mask = ~encoder_attention_mask.bool()
            else:
                memory_key_padding_mask = None
            
            # Decode
            decoder_output = self.transformer_decoder(
                tgt=tgt_embedded,
                memory=encoder_hidden_states,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
        else:
            # Inference mode: use previous predictions
            # Start with SOS token
            current_tokens = torch.full(
                (batch_size, 1),
                self.config.sos_token_id,
                dtype=torch.long,
                device=encoder_hidden_states.device
            )
            
            decoder_output = self._decode_autoregressive(
                encoder_hidden_states,
                current_tokens,
                encoder_attention_mask
            )
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def _decode_autoregressive(
        self,
        encoder_hidden_states: torch.Tensor,
        start_tokens: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive decoding for inference.
        
        Args:
            encoder_hidden_states: Encoder outputs [batch, enc_seq_len, hidden_size]
            start_tokens: Initial tokens (usually SOS) [batch, 1]
            encoder_attention_mask: Mask for encoder [batch, enc_seq_len]
            max_steps: Maximum decoding steps (defaults to max_notes_per_sample)
            
        Returns:
            Decoder outputs [batch, num_steps, hidden_size]
        """
        if max_steps is None:
            max_steps = self.config.max_notes_per_sample
        
        batch_size = encoder_hidden_states.size(0)
        current_tokens = start_tokens
        outputs = []
        
        for step in range(max_steps):
            # Embed current tokens
            tgt_embedded = self.note_embedding(current_tokens)
            tgt_embedded = self.pos_encoder(tgt_embedded)
            tgt_embedded = self.dropout(tgt_embedded)
            
            # Create causal mask
            dec_seq_len = current_tokens.size(1)
            causal_mask = self._generate_square_subsequent_mask(dec_seq_len).to(current_tokens.device)
            
            # Create memory mask
            if encoder_attention_mask is not None:
                memory_key_padding_mask = ~encoder_attention_mask.bool()
            else:
                memory_key_padding_mask = None
            
            # Decode
            decoder_output = self.transformer_decoder(
                tgt=tgt_embedded,
                memory=encoder_hidden_states,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # Get last token prediction
            last_output = decoder_output[:, -1:, :]  # [batch, 1, hidden_size]
            outputs.append(last_output)
            
            # Project to vocabulary and get next token
            logits = self.output_projection(last_output)  # [batch, 1, vocab_size]
            next_token = torch.argmax(logits, dim=-1)  # [batch, 1]
            
            # Append to current tokens
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            # Stop if all sequences produced EOS
            if (next_token == self.config.eos_token_id).all():
                break
        
        # Concatenate all outputs
        decoder_output = torch.cat(outputs, dim=1)  # [batch, num_steps, hidden_size]
        
        return decoder_output
    
    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """
        Generate causal mask for decoder.
        
        Args:
            size: Sequence length
            
        Returns:
            Causal mask [size, size]
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float('-inf'))
        return mask


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    
    Adds sinusoidal position information to embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Input with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
