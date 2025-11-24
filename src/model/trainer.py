"""
Trainer for Sound2Sheet model.

Handles training loop, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from tqdm import tqdm
import json
from datetime import datetime

from .config import ModelConfig, TrainingConfig
from .sound2sheet_model import Sound2SheetModel


class Trainer:
    """
    Trainer for Sound2Sheet model.
    
    Implements training loop with:
    - Teacher forcing
    - Mixed precision training
    - Gradient clipping
    - Checkpointing
    - Learning rate scheduling
    - Early stopping
    """
    
    def __init__(
        self,
        model: Sound2SheetModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        resume_from: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Sound2Sheet model
            train_loader: Training data loader
            val_loader: Validation data loader
            model_config: Model configuration
            training_config: Training configuration
            resume_from: Optional checkpoint path to resume from
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_config = model_config
        self.config = training_config
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Device setup
        self.device = torch.device(model_config.device if torch.cuda.is_available() and model_config.device == "cuda" else "cpu")
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
        
        # Log model parameters
        param_counts = self.model.count_parameters()
        self.logger.info(f"Model parameters:")
        self.logger.info(f"  Total: {param_counts['total']:,}")
        self.logger.info(f"  Trainable: {param_counts['trainable']:,}")
        self.logger.info(f"  Frozen: {param_counts['frozen']:,}")
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if training_config.use_mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.config.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Logging to {log_file}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        num_training_steps = len(self.train_loader) * self.config.num_epochs
        
        if self.config.lr_scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=num_training_steps
            )
        
        elif self.config.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=1e-7
            )
        
        else:  # constant
            scheduler = None
        
        return scheduler
    
    def train(self):
        """
        Main training loop.
        
        Returns:
            Training history dictionary
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Epochs: {self.config.num_epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Learning rate: {self.config.learning_rate}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation epoch
            val_loss, val_metrics = self._validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
                f"LR: {current_lr:.2e}"
            )
            
            # Save checkpoint periodically (if save_every_n_epochs > 0)
            save_every = getattr(self.config, 'save_every_n_epochs', 0)
            if save_every > 0 and (epoch + 1) % save_every == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
                self.logger.info(f"Checkpoint saved at epoch {epoch + 1}")
            
            # Check for improvement and save best model
            if val_loss < self.best_val_loss - self.config.early_stopping_threshold:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint("best_model.pt")
                self.logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"({self.epochs_without_improvement} epochs without improvement)"
                )
                break
        
        # Save final checkpoint
        self._save_checkpoint("final_model.pt")
        
        # Save training history
        self._save_history()
        
        self.logger.info("Training completed!")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss
        }
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            mel = batch['mel'].to(self.device)
            piano_roll = batch['piano_roll'].to(self.device)  # [batch, time, num_piano_keys]
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                with autocast(self.device.type):
                    # Get logits from model
                    logits = self.model(mel, piano_roll)  # [batch, enc_time, num_piano_keys]
                    
                    # Resize piano_roll to match logits temporal dimension
                    # [batch, orig_time, keys] -> [batch, keys, orig_time] -> interpolate -> [batch, keys, enc_time] -> [batch, enc_time, keys]
                    if piano_roll.size(1) != logits.size(1):
                        piano_roll_resized = torch.nn.functional.interpolate(
                            piano_roll.transpose(1, 2),  # [batch, keys, time]
                            size=logits.size(1),
                            mode='nearest'
                        ).transpose(1, 2)  # [batch, enc_time, keys]
                    else:
                        piano_roll_resized = piano_roll
                    
                    # Compute loss - BCEWithLogitsLoss for multi-label classification
                    loss = self.criterion(logits, piano_roll_resized)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                logits = self.model(mel, piano_roll)
                
                # Resize piano_roll to match logits temporal dimension
                if piano_roll.size(1) != logits.size(1):
                    piano_roll_resized = torch.nn.functional.interpolate(
                        piano_roll.transpose(1, 2),
                        size=logits.size(1),
                        mode='nearest'
                    ).transpose(1, 2)
                else:
                    piano_roll_resized = piano_roll
                
                loss = self.criterion(logits, piano_roll_resized)
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Accumulate loss
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item() * self.config.gradient_accumulation_steps})
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Step {self.global_step} - "
                    f"Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f}, "
                    f"LR: {current_lr:.2e}"
                )
        
        # Handle empty dataloader
        if num_batches == 0:
            self.logger.warning("No batches in training loader")
            return 0.0
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average validation loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Metrics for binary classification
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                mel = batch['mel'].to(self.device)
                piano_roll = batch['piano_roll'].to(self.device)  # [batch, time, num_piano_keys]
                
                # Forward pass
                logits = self.model(mel, piano_roll)  # [batch, enc_time, num_piano_keys]
                
                # Resize piano_roll to match logits temporal dimension
                if piano_roll.size(1) != logits.size(1):
                    piano_roll_resized = torch.nn.functional.interpolate(
                        piano_roll.transpose(1, 2),
                        size=logits.size(1),
                        mode='nearest'
                    ).transpose(1, 2)
                else:
                    piano_roll_resized = piano_roll
                
                # Compute loss
                loss = self.criterion(logits, piano_roll_resized)
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions (apply sigmoid and threshold)
                probs = torch.sigmoid(logits)
                predictions = (probs >= self.model_config.classification_threshold).float()
                
                # Collect for metrics (use resized piano_roll for fair comparison)
                all_predictions.append(predictions.cpu())
                all_targets.append(piano_roll_resized.cpu())
        
        # Handle empty dataloader
        if num_batches == 0:
            self.logger.warning("No batches in validation loader")
            return 0.0, {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        avg_loss = total_loss / num_batches
        
        # Compute frame-level metrics
        all_predictions = torch.cat(all_predictions, dim=0)  # [total_samples, time, num_keys]
        all_targets = torch.cat(all_targets, dim=0)
        
        # Flatten for frame-level metrics
        predictions_flat = all_predictions.reshape(-1)
        targets_flat = all_targets.reshape(-1)
        
        # Calculate metrics
        true_positives = ((predictions_flat == 1) & (targets_flat == 1)).sum().item()
        false_positives = ((predictions_flat == 1) & (targets_flat == 0)).sum().item()
        false_negatives = ((predictions_flat == 0) & (targets_flat == 1)).sum().item()
        true_negatives = ((predictions_flat == 0) & (targets_flat == 0)).sum().item()
        
        # Frame-level accuracy
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        
        # Precision, Recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': avg_loss
        }
        
        return avg_loss, metrics
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / filename
        
        self.model.save_checkpoint(
            str(checkpoint_path),
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            global_step=self.global_step,
            best_val_loss=self.best_val_loss,
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            val_accuracies=self.val_accuracies
        )
        
        # Remove old checkpoints if exceeding limit
        if self.config.save_total_limit:
            checkpoints = sorted(
                self.config.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
                key=lambda x: x.stat().st_mtime
            )
            
            while len(checkpoints) > self.config.save_total_limit:
                oldest = checkpoints.pop(0)
                oldest.unlink()
                self.logger.info(f"Removed old checkpoint: {oldest}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        # PyTorch 2.6+ requires weights_only=False for custom objects
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        self.logger.info(f"Resumed from checkpoint: {checkpoint_path}")
        self.logger.info(f"Starting from epoch {self.current_epoch}")
    
    def _save_history(self):
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracy': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'total_steps': self.global_step
        }
        
        history_path = self.config.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")
