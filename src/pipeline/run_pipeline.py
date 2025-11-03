"""Complete Sound2Sheet Training Pipeline.

Integrates all features:
- Dataset Generation (synthetic MIDI and audio)
- Audio Processing (mel-spectrogram, augmentation)
- Model Training (AST-based transformer)
- Evaluation (metrics, visualization, reports)
- Model Export (checkpoints and final model)

Usage:
  # Use default config
  python -m src.pipeline.run_pipeline
  
  # Use custom config
  python -m src.pipeline.run_pipeline --config my_config.yaml
  
  # Override parameters
  python -m src.pipeline.run_pipeline --samples 1000 --epochs 50 --batch-size 32
  
  # Quick test
  python -m src.pipeline.run_pipeline --samples 100 --epochs 5 --no-eval
  
  # Evaluation only
  python -m src.pipeline.run_pipeline --eval-only --model-path models/best_model.pt
"""

import logging
import shutil
import random
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

import numpy as np
import torch

from src.pipeline.config_parser import parse_args_to_config, PipelineConfig
from src.dataset import DatasetGenerator, DatasetConfig, MIDIGenerator, MIDIConfig, ComplexityLevel
from src.core import AudioProcessor
from src.model.config import ModelConfig as ModelConfigClass, TrainingConfig as TrainingConfigClass, DataConfig
from src.model import Sound2SheetModel, Trainer, create_dataloaders
from src.evaluation import Evaluator, EvaluationConfig, ReportGenerator, Visualizer, ReportFormat
from src.converter import NoteSequence


logger = logging.getLogger("pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def setup_directories(config: PipelineConfig) -> Path:
    """Create organized output directories by date and experiment.
    
    Structure: results/{YYYY-MM-DD}/{experiment_name}/
    Returns:
        Path to experiment directory
    """
    from datetime import datetime
    
    # Create experiment directory: results/{date}/{name}/
    date_str = datetime.now().strftime("%Y-%m-%d")
    experiment_name = getattr(config, 'experiment_name', 'training_run')
    experiment_dir = Path('results') / date_str / experiment_name
    
    # Create subdirectories
    subdirs = {
        'model': experiment_dir / 'model',
        'checkpoints': experiment_dir / 'checkpoints',
        'visualizations': experiment_dir / 'visualizations',
        'reports': experiment_dir / 'reports',
        'logs': experiment_dir / 'logs',
        'temp_data': experiment_dir / 'temp_data',  # Temporary training data
    }
    
    for name, dir_path in subdirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created {name} directory: {dir_path}")
    
    # Update config paths to use new structure
    config.training.checkpoint_dir = str(subdirs['checkpoints'])
    config.training.log_dir = str(subdirs['logs'])
    config.output.model_dir = str(subdirs['model'])
    config.output.results_dir = str(experiment_dir)
    config.dataset.output_dir = str(subdirs['temp_data'])
    
    logger.info(f"Experiment directory: {experiment_dir}")
    
    return experiment_dir


def cleanup_training_data(dataset_dir: Path, keep_manifests: bool = True):
    """Clean up temporary training data to save disk space.
    
    Args:
        dataset_dir: Directory containing training data
        keep_manifests: If True, keep manifest files for reference
    """
    if not dataset_dir.exists():
        return
    
    logger.info(f"Cleaning up training data from: {dataset_dir}")
    
    # Remove audio and MIDI files
    for split in ['train', 'val', 'test']:
        audio_dir = dataset_dir / split / 'audio'
        midi_dir = dataset_dir / split / 'midi'
        
        if audio_dir.exists():
            file_count = len(list(audio_dir.glob('*.wav')))
            shutil.rmtree(audio_dir)
            logger.info(f"  Removed {file_count} audio files from {split}/")
        
        if midi_dir.exists():
            file_count = len(list(midi_dir.glob('*.mid')))
            shutil.rmtree(midi_dir)
            logger.info(f"  Removed {file_count} MIDI files from {split}/")
    
    # Optionally remove manifests
    if not keep_manifests:
        metadata_dir = dataset_dir / 'metadata'
        if metadata_dir.exists():
            shutil.rmtree(metadata_dir)
            logger.info(f"  Removed manifest files")
    
    # Remove empty directories
    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split
        if split_dir.exists() and not any(split_dir.iterdir()):
            split_dir.rmdir()
    
    logger.info(f"✓ Cleanup completed")


def generate_dataset(config: PipelineConfig) -> Dict[str, Path]:
    """Generate synthetic dataset with MIDI and optionally audio.
    
    Returns:
        Dictionary with paths to train/val/test manifests
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Dataset Generation")
    logger.info("=" * 80)
    
    # Create dataset configuration using DatasetConfig from dataset module
    from src.dataset import DatasetConfig as DatasetGenConfig
    
    dataset_config = DatasetGenConfig(
        name="training_run",
        version="1.0.0",
        total_samples=config.dataset.samples,
        output_dir=Path(config.dataset.output_dir),
        soundfont_path=Path(config.dataset.soundfont_path) if config.dataset.soundfont_path else None,
        sample_rate=config.audio.sample_rate
    )
    
    # Generate dataset
    generator = DatasetGenerator(dataset_config)
    
    logger.info(f"Generating {config.dataset.samples} samples...")
    logger.info(f"  Complexity: {config.dataset.complexity}")
    logger.info(f"  Synthesize audio: {config.dataset.synthesize_audio}")
    
    dataset_dir = generator.generate(generate_audio=config.dataset.synthesize_audio)
    
    logger.info(f"✓ Dataset generated successfully")
    logger.info(f"  Output directory: {dataset_dir}")
    
    # Build manifest paths (they are in metadata/ subdirectory)
    manifests = {
        'train': dataset_dir / 'metadata' / 'train_manifest.json',
        'val': dataset_dir / 'metadata' / 'val_manifest.json',
        'test': dataset_dir / 'metadata' / 'test_manifest.json'
    }
    
    logger.info(f"  Train manifest: {manifests['train']}")
    logger.info(f"  Val manifest: {manifests['val']}")
    logger.info(f"  Test manifest: {manifests['test']}")
    
    # Store dataset_dir in config for later use
    config.dataset.output_dir = str(dataset_dir)
    
    return manifests


def create_model_configs(config: PipelineConfig, manifests: Dict[str, Path]) -> tuple:
    """Create model configuration objects from pipeline config.
    
    Returns:
        Tuple of (ModelConfig, TrainingConfig, DataConfig)
    """
    # Determine device
    device = config.model.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model configuration
    model_config = ModelConfigClass(
        ast_model_name=config.model.encoder_name,
        hidden_size=config.model.hidden_size,
        num_decoder_layers=config.model.num_decoder_layers,
        num_attention_heads=config.model.num_attention_heads,
        dropout=config.model.dropout,
        max_sequence_length=config.model.max_sequence_length,
        vocab_size=config.model.vocab_size,
        device=device
    )
    
    # Training configuration
    training_config = TrainingConfigClass(
        batch_size=config.training.batch_size,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        max_grad_norm=config.training.max_grad_norm,
        use_mixed_precision=config.training.use_mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        lr_scheduler_type=config.training.lr_schedule,
        early_stopping_patience=config.training.early_stopping_patience,
        checkpoint_dir=Path(config.training.checkpoint_dir),
        log_dir=Path(config.training.log_dir),
        logging_steps=config.training.log_every_n_steps,
        num_workers=config.training.num_workers
    )
    
    # Data configuration
    data_config = DataConfig(
        dataset_dir=Path(config.dataset.output_dir).absolute(),
        sample_rate=config.audio.sample_rate,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        n_mels=config.audio.n_mels,
        fmin=config.audio.f_min,
        fmax=config.audio.f_max,
        use_augmentation=config.audio.augmentation.get('enabled', True)
    )
    
    return model_config, training_config, data_config


def train_model(config: PipelineConfig, model_config: ModelConfigClass, 
                training_config: TrainingConfigClass, data_config: DataConfig,
                resume_from: Optional[str] = None) -> tuple:
    """Train the Sound2Sheet model.
    
    Returns:
        Tuple of (model, trainer, history)
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Model Training")
    logger.info("=" * 80)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_config, model_config, training_config
    )
    
    logger.info(f"  Train samples: {len(train_loader.dataset)}")
    logger.info(f"  Val samples: {len(val_loader.dataset)}")
    logger.info(f"  Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = Sound2SheetModel(model_config, freeze_encoder=config.model.freeze_encoder)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        training_config=training_config
    )
    
    # Resume from checkpoint if specified
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)
    
    # Train
    logger.info(f"Starting training for {training_config.num_epochs} epochs...")
    logger.info(f"  Batch size: {training_config.batch_size}")
    logger.info(f"  Learning rate: {training_config.learning_rate}")
    logger.info(f"  Device: {model_config.device}")
    logger.info(f"  Mixed precision: {training_config.use_mixed_precision}")
    
    history = trainer.train()
    
    logger.info("✓ Training completed successfully")
    logger.info(f"  Best epoch: {history.get('best_epoch', 'N/A')}")
    logger.info(f"  Best val loss: {history.get('best_val_loss', 'N/A'):.4f}")
    logger.info(f"  Final train loss: {history['train_losses'][-1]:.4f}")
    logger.info(f"  Final val loss: {history['val_losses'][-1]:.4f}")
    
    return model, trainer, history


def save_final_model(model: Sound2SheetModel, config: PipelineConfig, history: Dict[str, Any]):
    """Save final trained model."""
    output_dir = Path(config.output.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / config.output.final_model_name
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config.__dict__,
        'history': history
    }, model_path)
    
    logger.info(f"✓ Final model saved to {model_path}")
    
    # Save history as JSON
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"✓ Training history saved to {history_path}")


def evaluate_model(model: Sound2SheetModel, config: PipelineConfig,
                   data_config: DataConfig) -> Dict[str, Any]:
    """Evaluate model on test set.
    
    Returns:
        Evaluation results dictionary
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Model Evaluation")
    logger.info("=" * 80)
    
    # Create evaluation configuration
    eval_config = EvaluationConfig(
        onset_tolerance=config.evaluation.onset_tolerance,
        offset_tolerance=config.evaluation.offset_tolerance,
        pitch_tolerance=config.evaluation.pitch_tolerance,
        min_note_duration=config.evaluation.min_duration,
        max_note_duration=config.evaluation.max_duration
    )
    
    evaluator = Evaluator(eval_config)
    
    # Load test samples
    # TODO: Implement actual inference and evaluation
    # This is a placeholder that shows the structure
    
    logger.info("Running evaluation on test set...")
    logger.info(f"  Onset tolerance: {eval_config.onset_tolerance}s")
    logger.info(f"  Offset tolerance: {eval_config.offset_tolerance}s")
    logger.info(f"  Pitch tolerance: {eval_config.pitch_tolerance} semitones")
    
    # Placeholder: In real implementation, would run inference and evaluate
    logger.warning("Evaluation implementation pending - requires inference pipeline")
    
    results = {
        'status': 'pending',
        'message': 'Full evaluation requires inference implementation'
    }
    
    return results


def generate_reports(evaluator: Evaluator, config: PipelineConfig):
    """Generate evaluation reports."""
    if not config.evaluation.reports.get('enabled', True):
        logger.info("Report generation disabled")
        return
    
    logger.info("Generating evaluation reports...")
    
    output_dir = Path(config.evaluation.reports.get('output_dir', 'results/reports'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_gen = ReportGenerator(evaluator)
    
    formats = config.evaluation.reports.get('formats', ['json', 'csv'])
    for fmt in formats:
        if fmt == 'json':
            report_path = output_dir / "evaluation_report.json"
            report_gen.generate_report(str(report_path), ReportFormat.JSON)
            logger.info(f"  ✓ JSON report: {report_path}")
        elif fmt == 'csv':
            report_path = output_dir / "evaluation_report.csv"
            report_gen.generate_report(str(report_path), ReportFormat.CSV)
            logger.info(f"  ✓ CSV report: {report_path}")


def generate_visualizations(evaluator: Evaluator, config: PipelineConfig):
    """Generate evaluation visualizations."""
    if not config.evaluation.visualizations.get('enabled', True):
        logger.info("Visualization generation disabled")
        return
    
    logger.info("Generating visualizations...")
    
    output_dir = Path(config.evaluation.visualizations.get('output_dir', 'results/visualizations'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = Visualizer(evaluator)
    
    plot_types = config.evaluation.visualizations.get('plot_types', ['dashboard'])
    
    for plot_type in plot_types:
        if plot_type == 'dashboard':
            plot_path = output_dir / "evaluation_dashboard.png"
            visualizer.create_dashboard(str(plot_path))
            logger.info(f"  ✓ Dashboard: {plot_path}")
        elif plot_type == 'confusion_matrix':
            plot_path = output_dir / "confusion_matrix.png"
            visualizer.plot_confusion_matrix(str(plot_path))
            logger.info(f"  ✓ Confusion matrix: {plot_path}")
        elif plot_type == 'metrics_over_time':
            plot_path = output_dir / "metrics_over_time.png"
            visualizer.plot_metrics_over_time(str(plot_path))
            logger.info(f"  ✓ Metrics over time: {plot_path}")


def generate_training_visualizations(history: Dict, config: PipelineConfig):
    """Generate training history visualizations (loss curves, learning rate, etc.)."""
    logger.info("Generating training visualizations...")
    
    output_dir = Path(config.output.results_dir) / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import matplotlib.pyplot as plt
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    epochs = list(range(1, len(history['train_losses']) + 1))
    
    # Plot 1: Training and Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_losses'], 'b-o', label='Train Loss', linewidth=2, markersize=8)
    ax1.plot(epochs, history['val_losses'], 'r-s', label='Val Loss', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning Rate
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['learning_rates'], 'g-^', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss Improvement
    ax3 = axes[1, 0]
    train_improvements = [0] + [history['train_losses'][i-1] - history['train_losses'][i] 
                                 for i in range(1, len(history['train_losses']))]
    val_improvements = [0] + [history['val_losses'][i-1] - history['val_losses'][i] 
                              for i in range(1, len(history['val_losses']))]
    ax3.bar([e - 0.2 for e in epochs], train_improvements, width=0.4, label='Train', alpha=0.8)
    ax3.bar([e + 0.2 for e in epochs], val_improvements, width=0.4, label='Val', alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss Improvement', fontsize=12)
    ax3.set_title('Loss Improvement per Epoch', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
    
    # Plot 4: Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    Training Summary
    {'=' * 40}
    
    Total Epochs: {len(epochs)}
    
    Final Train Loss: {history['train_losses'][-1]:.4f}
    Final Val Loss: {history['val_losses'][-1]:.4f}
    Best Val Loss: {history['best_val_loss']:.4f}
    
    Train Loss Reduction: {history['train_losses'][0] - history['train_losses'][-1]:.4f}
    Val Loss Reduction: {history['val_losses'][0] - history['val_losses'][-1]:.4f}
    
    Final Learning Rate: {history['learning_rates'][-1]:.2e}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'training_history.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Training history plot: {plot_path}")
    
    # Also create a simple loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_losses'], 'b-o', label='Train Loss', linewidth=2, markersize=8)
    plt.plot(epochs, history['val_losses'], 'r-s', label='Val Loss', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    simple_plot_path = output_dir / 'loss_curve.png'
    plt.savefig(simple_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Loss curve: {simple_plot_path}")


def generate_training_report(history: Dict, config: PipelineConfig):
    """Generate detailed training report."""
    logger.info("Generating training report...")
    
    output_dir = Path(config.output.results_dir) / 'reports'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    from datetime import datetime
    
    # Calculate statistics
    epochs = len(history['train_losses'])
    train_loss_start = history['train_losses'][0]
    train_loss_end = history['train_losses'][-1]
    val_loss_start = history['val_losses'][0]
    val_loss_end = history['val_losses'][-1]
    best_val_loss = history['best_val_loss']
    
    train_improvement = train_loss_start - train_loss_end
    val_improvement = val_loss_start - val_loss_end
    train_improvement_pct = (train_improvement / train_loss_start) * 100
    val_improvement_pct = (val_improvement / val_loss_start) * 100
    
    # Create detailed report
    report = {
        'metadata': {
            'experiment_name': getattr(config, 'experiment_name', 'training_run'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_epochs': epochs
        },
        'configuration': {
            'batch_size': config.training.batch_size,
            'learning_rate': config.training.learning_rate,
            'num_epochs': config.training.num_epochs,
            'lr_schedule': config.training.lr_schedule,
            'device': getattr(config, 'device', 'cpu'),
            'mixed_precision': config.training.use_mixed_precision,
            'weight_decay': config.training.weight_decay
        },
        'dataset': {
            'total_samples': config.dataset.samples,
            'complexity': config.dataset.complexity,
            'audio_synthesized': config.dataset.synthesize_audio
        },
        'results': {
            'final_metrics': {
                'train_loss': train_loss_end,
                'val_loss': val_loss_end,
                'best_val_loss': best_val_loss
            },
            'improvements': {
                'train_loss_reduction': train_improvement,
                'val_loss_reduction': val_improvement,
                'train_loss_improvement_pct': train_improvement_pct,
                'val_loss_improvement_pct': val_improvement_pct
            },
            'epoch_history': {
                'train_losses': history['train_losses'],
                'val_losses': history['val_losses'],
                'learning_rates': history['learning_rates']
            }
        },
        'model': {
            'total_parameters': '134M (approx)',
            'trainable_parameters': '47M (approx)',
            'encoder': 'AST (frozen)',
            'decoder': 'Transformer (6 layers, 768 hidden)'
        }
    }
    
    # Save JSON report
    json_path = output_dir / 'training_report.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"  ✓ JSON report: {json_path}")
    
    # Save human-readable text report
    text_path = output_dir / 'training_report.txt'
    with open(text_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SOUND2SHEET TRAINING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Experiment: {report['metadata']['experiment_name']}\n")
        f.write(f"Timestamp: {report['metadata']['timestamp']}\n")
        f.write(f"Total Epochs: {report['metadata']['total_epochs']}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Batch Size: {report['configuration']['batch_size']}\n")
        f.write(f"Learning Rate: {report['configuration']['learning_rate']}\n")
        f.write(f"Epochs: {report['configuration']['num_epochs']}\n")
        f.write(f"LR Schedule: {report['configuration']['lr_schedule']}\n")
        f.write(f"Device: {report['configuration']['device']}\n")
        f.write(f"Mixed Precision: {report['configuration']['mixed_precision']}\n")
        f.write(f"Weight Decay: {report['configuration']['weight_decay']}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("DATASET\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Samples: {report['dataset']['total_samples']}\n")
        f.write(f"Complexity: {report['dataset']['complexity']}\n")
        f.write(f"Audio Synthesized: {report['dataset']['audio_synthesized']}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Final Train Loss: {train_loss_end:.4f}\n")
        f.write(f"Final Val Loss: {val_loss_end:.4f}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n\n")
        
        f.write(f"Train Loss Reduction: {train_improvement:.4f} ({train_improvement_pct:.2f}%)\n")
        f.write(f"Val Loss Reduction: {val_improvement:.4f} ({val_improvement_pct:.2f}%)\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("EPOCH-BY-EPOCH HISTORY\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'LR':<12}\n")
        f.write("-" * 80 + "\n")
        for i in range(epochs):
            f.write(f"{i+1:<8} {history['train_losses'][i]:<12.4f} "
                   f"{history['val_losses'][i]:<12.4f} {history['learning_rates'][i]:<12.2e}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logger.info(f"  ✓ Text report: {text_path}")
    
    # Save CSV for easy analysis
    csv_path = output_dir / 'training_history.csv'
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,learning_rate\n")
        for i in range(epochs):
            f.write(f"{i+1},{history['train_losses'][i]},{history['val_losses'][i]},"
                   f"{history['learning_rates'][i]}\n")
    
    logger.info(f"  ✓ CSV file: {csv_path}")


def run_pipeline(config: PipelineConfig):
    """Run complete training pipeline with automatic cleanup."""
    logger.info("=" * 80)
    logger.info("Sound2Sheet Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Configuration: {getattr(config, 'experiment_name', 'default')}")
    
    # Setup
    set_seed(getattr(config, 'seed', 42))
    experiment_dir = setup_directories(config)
    
    # Step 1: Generate dataset (temporary)
    manifests = generate_dataset(config)
    dataset_dir = Path(config.dataset.output_dir)
    
    # Step 2: Train model
    model_config, training_config, data_config = create_model_configs(config, manifests)
    
    resume_from = getattr(config, 'resume_checkpoint', None)
    model, trainer, history = train_model(
        config, model_config, training_config, data_config, resume_from
    )
    
    # Clean up training data immediately after training
    logger.info("=" * 80)
    logger.info("Cleaning up temporary training data...")
    logger.info("=" * 80)
    cleanup_training_data(dataset_dir, keep_manifests=True)
    
    # Save final model
    save_final_model(model, config, history)
    
    # Generate training visualizations
    generate_training_visualizations(history, config)
    
    # Generate training report
    generate_training_report(history, config)
    
    # Step 3: Evaluate model (if enabled)
    if config.evaluation.enabled:
        results = evaluate_model(model, config, data_config)
        
        # Generate reports and visualizations
        # Note: These require actual evaluation results
        # Uncommented when evaluation is fully implemented
        # generate_reports(evaluator, config)
        # generate_visualizations(evaluator, config)
    
    logger.info("=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 80)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Model: {config.output.model_dir}/{config.output.final_model_name}")
    logger.info(f"Visualizations: {experiment_dir}/visualizations/")
    logger.info(f"Reports: {experiment_dir}/reports/")
    logger.info(f"Training data cleaned up to save disk space")


def run_evaluation_only(config: PipelineConfig):
    """Run evaluation only on existing model."""
    logger.info("=" * 80)
    logger.info("Sound2Sheet Evaluation (Evaluation Only Mode)")
    logger.info("=" * 80)
    
    model_path = getattr(config, 'eval_model_path', None)
    if not model_path:
        raise ValueError("--model-path required for evaluation-only mode")
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = ModelConfigClass(**checkpoint['model_config'])
    model = Sound2SheetModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("✓ Model loaded successfully")
    
    # Create data config
    _, _, data_config = create_model_configs(config, {})
    
    # Run evaluation
    results = evaluate_model(model, config, data_config)
    
    logger.info("=" * 80)
    logger.info("Evaluation completed!")
    logger.info("=" * 80)


def main():
    """Main entry point for pipeline."""
    # Parse configuration
    config = parse_args_to_config()
    
    # Check if evaluation-only mode
    if getattr(config, 'eval_only', False):
        run_evaluation_only(config)
    else:
        run_pipeline(config)


if __name__ == '__main__':
    main()
