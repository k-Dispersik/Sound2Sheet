"""
Sound2Sheet main model.

Combines AST encoder and Piano Roll classifier for frame-level piano transcription.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING
import logging
import numpy as np

from .config import ModelConfig
from .ast_model import ASTWrapper, PianoRollClassifier

if TYPE_CHECKING:
    from .config import InferenceConfig


class Sound2SheetModel(nn.Module):
    """
    Main Sound2Sheet model for piano transcription using Piano Roll approach.
    
    Architecture:
        1. AST Encoder: Extracts features from mel-spectrogram
        2. Piano Roll Classifier: Predicts binary activation for each of 88 keys per frame
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
        self.classifier = PianoRollClassifier(config)
        
        self.logger.info(f"Initialized Sound2Sheet Piano Roll model")
        self.logger.info(f"  Encoder: AST ({config.ast_model_name})")
        self.logger.info(f"  Classifier: {config.num_classifier_layers} layers, {config.num_piano_keys} keys")
        self.logger.info(f"  Frame duration: {config.frame_duration_ms}ms")
        self.logger.info(f"  Freeze encoder: {freeze_encoder}")
    
    def forward(
        self,
        mel: torch.Tensor,
        piano_roll: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through model.
        
        Args:
            mel: Mel-spectrogram [batch, n_mels, time]
            piano_roll: Ground truth piano roll [batch, time_frames, num_piano_keys] (optional, for training)
            
        Returns:
            logits: Piano roll predictions [batch, time_frames, num_piano_keys]
        """
        # Encode audio
        encoder_output = self.encoder(mel)  # [batch, enc_seq_len, hidden_size]
        
        # Classify each frame
        logits = self.classifier(encoder_output)  # [batch, enc_seq_len, num_piano_keys]
        
        return logits
    
    def predict(
        self,
        mel: torch.Tensor,
        inference_config: 'InferenceConfig'
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Predict piano roll and extract note events from audio (inference mode).
        
        Args:
            mel: Mel-spectrogram [batch, n_mels, time]
            inference_config: Inference configuration
            
        Returns:
            - piano_roll: Binary piano roll [batch, time_frames, num_piano_keys]
            - events: List of note events per batch item, each event is a dict with:
                - pitch: MIDI note number (21-108)
                - onset_time_ms: Note start time in milliseconds
                - offset_time_ms: Note end time in milliseconds
                - velocity: MIDI velocity (if included)
        """
        self.eval()
        
        with torch.no_grad():
            # Get logits
            logits = self.forward(mel)  # [batch, time_frames, num_piano_keys]
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            
            # Apply median filter if enabled
            if inference_config.use_median_filter:
                probs = self._apply_median_filter(probs, inference_config.median_filter_size)
            
            # Threshold to get binary piano roll
            piano_roll = (probs >= inference_config.classification_threshold).float()
            
            # Extract note events if requested
            if inference_config.output_format == "events":
                events = self._piano_roll_to_events(
                    piano_roll,
                    inference_config
                )
            else:
                events = []
        
        return piano_roll, events
    
    def _apply_median_filter(
        self,
        probs: torch.Tensor,
        filter_size: int
    ) -> torch.Tensor:
        """
        Apply median filter along time dimension to smooth predictions.
        
        Args:
            probs: Probabilities [batch, time_frames, num_piano_keys]
            filter_size: Size of median filter
            
        Returns:
            Smoothed probabilities [batch, time_frames, num_piano_keys]
        """
        if filter_size <= 1:
            return probs
        
        # Use unfold to create sliding windows
        batch_size, time_frames, num_keys = probs.shape
        
        # Pad to handle edges
        padding = filter_size // 2
        probs_padded = torch.nn.functional.pad(probs, (0, 0, padding, padding), mode='replicate')
        
        # Apply median filter per key
        smoothed = []
        for b in range(batch_size):
            batch_smoothed = []
            for k in range(num_keys):
                key_probs = probs_padded[b, :, k]  # [time_frames + 2*padding]
                # Unfold creates windows
                windows = key_probs.unfold(0, filter_size, 1)  # [time_frames, filter_size]
                medians = windows.median(dim=1)[0]  # [time_frames]
                batch_smoothed.append(medians)
            smoothed.append(torch.stack(batch_smoothed, dim=1))  # [time_frames, num_keys]
        
        return torch.stack(smoothed, dim=0)  # [batch, time_frames, num_keys]
    
    def _piano_roll_to_events(
        self,
        piano_roll: torch.Tensor,
        inference_config: 'InferenceConfig'
    ) -> List[List[Dict]]:
        """
        Convert binary piano roll to note events.
        
        Args:
            piano_roll: Binary piano roll [batch, time_frames, num_piano_keys]
            inference_config: Inference configuration
            
        Returns:
            List of note events per batch item
        """
        batch_size = piano_roll.shape[0]
        frame_duration_ms = self.config.frame_duration_ms
        min_duration_frames = int(inference_config.min_note_duration_ms / frame_duration_ms)
        
        all_events = []
        
        for b in range(batch_size):
            batch_events = []
            roll = piano_roll[b].cpu().numpy()  # [time_frames, num_piano_keys]
            
            # Process each key
            for key_idx in range(self.config.num_piano_keys):
                key_activations = roll[:, key_idx]  # [time_frames]
                
                # Find note on/off transitions
                # Add padding to detect notes at boundaries
                padded = np.pad(key_activations, (1, 1), constant_values=0)
                diff = np.diff(padded)
                
                onsets = np.where(diff == 1)[0]  # Frames where note turns on
                offsets = np.where(diff == -1)[0]  # Frames where note turns off
                
                # Match onsets with offsets
                for onset_frame in onsets:
                    # Find corresponding offset
                    matching_offsets = offsets[offsets > onset_frame]
                    if len(matching_offsets) == 0:
                        # Note continues till end
                        offset_frame = len(key_activations)
                    else:
                        offset_frame = matching_offsets[0]
                    
                    # Check minimum duration
                    duration_frames = offset_frame - onset_frame
                    if duration_frames < min_duration_frames:
                        continue
                    
                    # Convert to MIDI note number
                    midi_note = self.config.min_midi_note + key_idx
                    
                    # Convert frames to milliseconds
                    onset_ms = onset_frame * frame_duration_ms
                    offset_ms = offset_frame * frame_duration_ms
                    
                    # Create event
                    event = {
                        'pitch': int(midi_note),
                        'onset_time_ms': float(onset_ms),
                        'offset_time_ms': float(offset_ms)
                    }
                    
                    if inference_config.events_include_velocity:
                        event['velocity'] = inference_config.default_velocity
                    
                    batch_events.append(event)
            
            # Sort by onset time
            batch_events.sort(key=lambda x: x['onset_time_ms'])
            all_events.append(batch_events)
        
        return all_events
    
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
