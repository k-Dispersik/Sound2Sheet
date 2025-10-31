#!/usr/bin/env python3
"""
Training script for Sound2Sheet model.

Usage:
    python -m src.model.train --dataset data/datasets/my_dataset_*/ --epochs 50
"""

import argparse
import logging
from pathlib import Path
import torch

from .config import ModelConfig, TrainingConfig, DataConfig
from .sound2sheet_model import Sound2SheetModel
from .dataset import create_dataloaders
from .trainer import Trainer


def setup_logging(log_dir: Path):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / 'train.log')
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Sound2Sheet model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--use-augmentation',
        action='store_true',
        default=True,
        help='Use audio augmentation during training'
    )
    
    # Model arguments
    parser.add_argument(
        '--ast-model',
        type=str,
        default='MIT/ast-finetuned-audioset-10-10-0.4593',
        help='Pretrained AST model name from Hugging Face'
    )
    parser.add_argument(
        '--freeze-encoder',
        action='store_true',
        help='Freeze AST encoder weights'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=768,
        help='Hidden size for decoder'
    )
    parser.add_argument(
        '--num-decoder-layers',
        type=int,
        default=6,
        help='Number of decoder layers'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=1.0,
        help='Maximum gradient norm for clipping'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=1000,
        help='Number of warmup steps'
    )
    
    # Mixed precision
    parser.add_argument(
        '--no-mixed-precision',
        action='store_true',
        help='Disable mixed precision training'
    )
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps'
    )
    
    # Checkpointing
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=Path('checkpoints'),
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('logs'),
        help='Directory for logs'
    )
    parser.add_argument(
        '--save-steps',
        type=int,
        default=500,
        help='Save checkpoint every N steps'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        help='Path to checkpoint to resume from'
    )
    
    # Early stopping
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=5,
        help='Early stopping patience (epochs)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Sound2Sheet Training")
    logger.info("=" * 80)
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    logger.info(f"Using device: {args.device}")
    if args.device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Create configurations
    model_config = ModelConfig(
        ast_model_name=args.ast_model,
        hidden_size=args.hidden_size,
        num_decoder_layers=args.num_decoder_layers,
        device=args.device
    )
    
    data_config = DataConfig(
        dataset_dir=args.dataset,
        use_augmentation=args.use_augmentation
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        use_mixed_precision=not args.no_mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        save_steps=args.save_steps,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers
    )
    
    # Log configurations
    logger.info("\nModel Configuration:")
    logger.info(f"  AST Model: {model_config.ast_model_name}")
    logger.info(f"  Hidden Size: {model_config.hidden_size}")
    logger.info(f"  Decoder Layers: {model_config.num_decoder_layers}")
    logger.info(f"  Vocab Size: {model_config.vocab_size}")
    logger.info(f"  Max Notes: {model_config.max_notes_per_sample}")
    
    logger.info("\nTraining Configuration:")
    logger.info(f"  Epochs: {training_config.num_epochs}")
    logger.info(f"  Batch Size: {training_config.batch_size}")
    logger.info(f"  Learning Rate: {training_config.learning_rate}")
    logger.info(f"  Weight Decay: {training_config.weight_decay}")
    logger.info(f"  Mixed Precision: {training_config.use_mixed_precision}")
    logger.info(f"  Gradient Accumulation: {training_config.gradient_accumulation_steps}")
    
    logger.info("\nData Configuration:")
    logger.info(f"  Dataset: {data_config.dataset_dir}")
    logger.info(f"  Sample Rate: {data_config.sample_rate}")
    logger.info(f"  Augmentation: {data_config.use_augmentation}")
    
    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_config,
        model_config,
        training_config
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("\nInitializing model...")
    model = Sound2SheetModel(model_config, freeze_encoder=args.freeze_encoder)
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        training_config=training_config,
        resume_from=args.resume_from
    )
    
    # Start training
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80 + "\n")
    
    try:
        history = trainer.train()
        
        logger.info("\n" + "=" * 80)
        logger.info("Training Completed Successfully!")
        logger.info("=" * 80)
        logger.info(f"\nBest Validation Loss: {history['best_val_loss']:.4f}")
        logger.info(f"Checkpoints saved to: {training_config.checkpoint_dir}")
        logger.info(f"Logs saved to: {training_config.log_dir}")
        
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 80)
        logger.info("Training interrupted by user")
        logger.info("=" * 80)
        trainer._save_checkpoint("interrupted_model.pt")
        logger.info("Checkpoint saved before exit")
    
    except Exception as e:
        logger.error(f"\n" + "=" * 80)
        logger.error(f"Training failed with error: {e}")
        logger.error("=" * 80)
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
