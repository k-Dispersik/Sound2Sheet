"""
Trainer for Sound2Sheet model.

Handles training loop, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp import GradScaler
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
        
        # Device setup
        self.device = torch.device(model_config.device if torch.cuda.is_available() and model_config.device == "cuda" else "cpu")
        self.model.to(self.device)
        
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
        self.scaler = GradScaler(self.device.type) if training_config.use_mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # History
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
    
    
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
        Main training loop. Returns training history dictionary.
        """
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Train
            train_loss, train_acc = self._train_epoch(current_lr)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_metrics = self._validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Learning rate
            self.learning_rates.append(current_lr)
            
            # Checkpointing
            self._handle_checkpointing(epoch, val_loss)
            
            # Early stopping
            if self._should_stop_early():
                break
        
        self._save_checkpoint("final_model.pt")
        self._save_history()
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss
        }

    def _handle_checkpointing(self, epoch, val_loss):
        """Handle checkpoint saving and best model tracking."""
        save_every = getattr(self.config, 'save_every_n_epochs', 0)
        if save_every > 0 and (epoch + 1) % save_every == 0:
            self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
        if val_loss < self.best_val_loss - self.config.early_stopping_threshold:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            self._save_checkpoint("best_model.pt")
        else:
            self.epochs_without_improvement += 1

    def _should_stop_early(self):
        """Check early stopping condition."""
        return self.epochs_without_improvement >= self.config.early_stopping_patience
    
    def _train_epoch(self, current_lr) -> Tuple[float, float]:
        """Train for one epoch. Returns (average training loss, average training accuracy)."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, batch in enumerate(progress_bar):
            loss, predictions, targets = self._process_train_batch(batch)
            total_loss += loss * self.config.gradient_accumulation_steps
            num_batches += 1
            all_predictions.append(predictions)
            all_targets.append(targets)
            
            # Compute running accuracy
            if len(all_predictions) > 0:
                temp_preds = torch.cat(all_predictions, dim=0)
                temp_targets = torch.cat(all_targets, dim=0)
                correct = (temp_preds.reshape(-1) == temp_targets.reshape(-1)).sum().item()
                total = temp_preds.numel()
                running_acc = correct / total if total > 0 else 0.0
            else:
                running_acc = 0.0
            
            avg_loss_so_far = total_loss / num_batches
            progress_bar.set_postfix({
                'tr_loss': f'{avg_loss_so_far:.3f}',
                'tr_acc': f'{running_acc:.2f}',
                'lr': f'{current_lr:.4f}'
            })
        
        if num_batches == 0:
            return 0.0, 0.0
        
        avg_loss = total_loss / num_batches
        
        # Compute final training accuracy
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        predictions_flat = all_predictions.reshape(-1)
        targets_flat = all_targets.reshape(-1)
        correct = (predictions_flat == targets_flat).sum().item()
        total = predictions_flat.numel()
        train_accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, train_accuracy

    def _process_train_batch(self, batch) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Process a single training batch and update model weights. Returns (loss, predictions, targets)."""
        mel = batch['mel'].to(self.device)
        piano_roll = batch['piano_roll'].to(self.device)
        
        if self.config.use_mixed_precision:
            with autocast(self.device.type):
                logits, piano_roll_resized = self._forward_and_resize(mel, piano_roll)
                loss = self.criterion(logits, piano_roll_resized)
                loss = loss / self.config.gradient_accumulation_steps
        else:
            logits, piano_roll_resized = self._forward_and_resize(mel, piano_roll)
            loss = self.criterion(logits, piano_roll_resized)
            loss = loss / self.config.gradient_accumulation_steps
        
        if self.config.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self._update_weights()
        
        self.global_step += 1
        
        # Get predictions for accuracy calculation (keep on GPU)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            predictions = (probs >= self.model_config.classification_threshold).float()
            targets = piano_roll_resized
        
        return loss.item(), predictions, targets

    def _forward_and_resize(self, mel, piano_roll):
        """Forward pass and resize piano_roll to match logits temporal dimension."""
        logits = self.model(mel, piano_roll)
        # Ensure piano_roll is float for interpolation
        piano_roll_float = piano_roll.float()
        if piano_roll.size(1) != logits.size(1):
            piano_roll_resized = torch.nn.functional.interpolate(
                piano_roll_float.transpose(1, 2),
                size=logits.size(1),
                mode='nearest'
            ).transpose(1, 2)
        else:
            piano_roll_resized = piano_roll_float
        return logits, piano_roll_resized

    def _update_weights(self):
        """Update model weights and learning rate."""
        if self.config.use_mixed_precision:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()
    
    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch. Returns (average validation loss, metrics dict)."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        with torch.no_grad():
            for batch in progress_bar:
                loss, predictions, targets = self._process_val_batch(batch)
                total_loss += loss
                num_batches += 1
                all_predictions.append(predictions)
                all_targets.append(targets)
                
                # Compute running accuracy
                if len(all_predictions) > 0:
                    temp_preds = torch.cat(all_predictions, dim=0)
                    temp_targets = torch.cat(all_targets, dim=0)
                    correct = (temp_preds.reshape(-1) == temp_targets.reshape(-1)).sum().item()
                    total = temp_preds.numel()
                    running_acc = correct / total if total > 0 else 0.0
                else:
                    running_acc = 0.0
                
                avg_loss_so_far = total_loss / num_batches
                progress_bar.set_postfix({
                    'val_loss': f'{avg_loss_so_far:.3f}',
                    'val_acc': f'{running_acc:.2f}'
                })
        
        if num_batches == 0:
            return 0.0, {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        avg_loss = total_loss / num_batches
        metrics = self._compute_metrics(all_predictions, all_targets, avg_loss)
        return avg_loss, metrics

    def _process_val_batch(self, batch):
        """Process a single validation batch and return loss, predictions, targets."""
        mel = batch['mel'].to(self.device)
        piano_roll = batch['piano_roll'].to(self.device)
        logits, piano_roll_resized = self._forward_and_resize(mel, piano_roll)
        loss = self.criterion(logits, piano_roll_resized).item()
        probs = torch.sigmoid(logits)
        predictions = (probs >= self.model_config.classification_threshold).float()
        targets = piano_roll_resized
        return loss, predictions, targets

    def _compute_metrics(self, all_predictions, all_targets, avg_loss):
        """Compute frame-level metrics from predictions and targets."""
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        predictions_flat = all_predictions.reshape(-1)
        targets_flat = all_targets.reshape(-1)
        true_positives = ((predictions_flat == 1) & (targets_flat == 1)).sum().item()
        false_positives = ((predictions_flat == 1) & (targets_flat == 0)).sum().item()
        false_negatives = ((predictions_flat == 0) & (targets_flat == 1)).sum().item()
        true_negatives = ((predictions_flat == 0) & (targets_flat == 0)).sum().item()
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': avg_loss
        }
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint and remove old ones if exceeding limit."""
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
        if self.config.save_total_limit:
            self._remove_old_checkpoints()

    def _remove_old_checkpoints(self):
        """Remove old checkpoints if exceeding save_total_limit."""
        checkpoints = sorted(
            self.config.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda x: x.stat().st_mtime
        )
        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            oldest.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
    
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
