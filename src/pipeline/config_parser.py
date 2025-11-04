"""Configuration parser for Sound2Sheet pipeline.

Handles loading YAML config and merging with command-line arguments.
Command-line arguments have higher priority than config file values.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class DatasetConfig:
    """Dataset generation configuration."""
    samples: int = 1000
    complexity: str = "medium"
    min_notes: int = 10
    max_notes: int = 50
    min_duration: float = 5.0
    max_duration: float = 30.0
    key_signatures: Optional[list] = None
    time_signatures: Optional[list] = None
    output_dir: str = "data/datasets/training_run"
    synthesize_audio: bool = True
    soundfont_path: Optional[str] = None
    # Use existing dataset instead of generating new one
    use_existing_dataset: bool = False
    existing_dataset_path: Optional[str] = None


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float = 8000.0
    normalize: bool = True
    pre_emphasis: float = 0.97
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "pitch_shift_range": [-2, 2],
        "time_stretch_range": [0.9, 1.1],
        "noise_level": 0.005
    })


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    encoder_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    freeze_encoder: bool = True
    hidden_size: int = 768
    num_decoder_layers: int = 6
    num_attention_heads: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 512
    vocab_size: int = 388
    device: str = "cuda"


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    lr_schedule: str = "cosine"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    checkpoint_dir: str = "models/checkpoints"
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    log_dir: str = "logs"
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 1
    gpu_memory_fraction: Optional[float] = None
    resume_checkpoint: Optional[str] = None  # Path to checkpoint to resume from


@dataclass
class InferenceConfig:
    """Inference configuration."""
    strategy: str = "beam_search"
    max_length: int = 512
    num_beams: int = 5
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    enabled: bool = True
    onset_tolerance: float = 0.05
    offset_tolerance: float = 0.05
    pitch_tolerance: int = 0
    min_duration: float = 0.1
    max_duration: Optional[float] = None
    visualizations: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "output_dir": "results/visualizations",
        "plot_types": ["dashboard", "confusion_matrix", "metrics_over_time"]
    })
    reports: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "output_dir": "results/reports",
        "formats": ["json", "csv"]
    })


@dataclass
class OutputConfig:
    """Output paths configuration."""
    model_dir: str = "models/trained"
    results_dir: str = "results"
    final_model_name: str = "sound2sheet_model.pt"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict or {}


def merge_configs(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override config into base config."""
    merged = base_config.copy()
    
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        elif value is not None:  # Only override if value is not None
            merged[key] = value
    
    return merged


def dict_to_config(config_dict: Dict[str, Any]) -> PipelineConfig:
    """Convert dictionary to PipelineConfig dataclass."""
    return PipelineConfig(
        dataset=DatasetConfig(**config_dict.get('dataset', {})),
        audio=AudioConfig(**config_dict.get('audio', {})),
        model=ModelConfig(**config_dict.get('model', {})),
        training=TrainingConfig(**config_dict.get('training', {})),
        inference=InferenceConfig(**config_dict.get('inference', {})),
        evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
        output=OutputConfig(**config_dict.get('output', {}))
    )


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser with all pipeline options."""
    parser = argparse.ArgumentParser(
        description="Sound2Sheet Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python -m src.pipeline.run_pipeline
  
  # Use custom config
  python -m src.pipeline.run_pipeline --config my_config.yaml
  
  # Override specific parameters
  python -m src.pipeline.run_pipeline --epochs 100 --batch-size 32
  
  # Quick test run
  python -m src.pipeline.run_pipeline --samples 100 --epochs 5 --no-eval
        """
    )
    
    # General
    parser.add_argument('--config', type=str, 
                       default='src/pipeline/config.yaml',
                       help='Path to YAML configuration file')
    parser.add_argument('--name', type=str,
                       help='Experiment name (used for output directories)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Dataset generation
    dataset_group = parser.add_argument_group('Dataset Generation')
    dataset_group.add_argument('--samples', type=int,
                              help='Number of samples to generate')
    dataset_group.add_argument('--complexity', type=str, choices=['simple', 'medium', 'complex'],
                              help='MIDI generation complexity')
    dataset_group.add_argument('--min-notes', type=int,
                              help='Minimum notes per sample')
    dataset_group.add_argument('--max-notes', type=int,
                              help='Maximum notes per sample')
    dataset_group.add_argument('--min-duration', type=float,
                              help='Minimum sample duration (seconds)')
    dataset_group.add_argument('--max-duration', type=float,
                              help='Maximum sample duration (seconds)')
    dataset_group.add_argument('--soundfont', type=str,
                              help='Path to soundfont file for audio synthesis')
    dataset_group.add_argument('--use-existing-dataset', action='store_true',
                              help='Use existing dataset instead of generating new one')
    dataset_group.add_argument('--dataset-path', type=str,
                              help='Path to existing dataset directory (required if --use-existing-dataset)')
    
    # Audio processing
    audio_group = parser.add_argument_group('Audio Processing')
    audio_group.add_argument('--sample-rate', type=int,
                            help='Audio sample rate')
    audio_group.add_argument('--n-mels', type=int,
                            help='Number of mel bands')
    audio_group.add_argument('--no-augmentation', action='store_true',
                            help='Disable audio augmentation')
    
    # Model
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--encoder', type=str,
                            help='Encoder model name/path')
    model_group.add_argument('--freeze-encoder', action='store_true',
                            help='Freeze encoder weights')
    model_group.add_argument('--hidden-size', type=int,
                            help='Model hidden size')
    model_group.add_argument('--num-decoder-layers', type=int,
                            help='Number of decoder layers')
    model_group.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                            help='Device to use for training')
    
    # Training
    training_group = parser.add_argument_group('Training')
    training_group.add_argument('--batch-size', type=int,
                               help='Training batch size')
    training_group.add_argument('--epochs', type=int,
                               help='Number of training epochs')
    training_group.add_argument('--lr', '--learning-rate', type=float, dest='learning_rate',
                               help='Learning rate')
    training_group.add_argument('--warmup-steps', type=int,
                               help='Learning rate warmup steps')
    training_group.add_argument('--no-mixed-precision', action='store_true',
                               help='Disable mixed precision training')
    training_group.add_argument('--num-workers', type=int,
                               help='Number of dataloader workers')
    training_group.add_argument('--resume', '--resume-checkpoint', type=str, dest='resume_checkpoint',
                               help='Resume training from checkpoint (path to .pt file)')
    training_group.add_argument('--resume-from-epoch', type=int,
                               help='Specify which epoch checkpoint to resume from (e.g., 50 for checkpoint_epoch_50.pt)')
    
    # Evaluation
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument('--no-eval', action='store_true',
                           help='Skip evaluation after training')
    eval_group.add_argument('--no-viz', action='store_true',
                           help='Skip visualization generation')
    eval_group.add_argument('--eval-only', action='store_true',
                           help='Run evaluation only (skip training)')
    eval_group.add_argument('--model-path', type=str,
                           help='Model path for evaluation-only mode')
    
    # Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', type=str,
                             help='Base output directory')
    output_group.add_argument('--checkpoint-dir', type=str,
                             help='Checkpoint save directory')
    output_group.add_argument('--log-dir', type=str,
                             help='Log directory')
    
    return parser


def parse_args_to_config() -> PipelineConfig:
    """Parse command-line arguments and merge with YAML config.
    
    Priority: CLI args > YAML config > defaults
    """
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Load YAML config
    yaml_config = load_yaml_config(args.config)
    
    # Build overrides from CLI args
    overrides = {}
    
    # Dataset overrides
    if (args.samples is not None or args.complexity is not None or 
        args.use_existing_dataset or args.dataset_path is not None or
        args.min_notes is not None or args.max_notes is not None or
        args.min_duration is not None or args.max_duration is not None or
        args.soundfont is not None):
        overrides['dataset'] = {}
        if args.samples is not None:
            overrides['dataset']['samples'] = args.samples
        if args.complexity is not None:
            overrides['dataset']['complexity'] = args.complexity
        if args.min_notes is not None:
            overrides['dataset']['min_notes'] = args.min_notes
        if args.max_notes is not None:
            overrides['dataset']['max_notes'] = args.max_notes
        if args.min_duration is not None:
            overrides['dataset']['min_duration'] = args.min_duration
        if args.max_duration is not None:
            overrides['dataset']['max_duration'] = args.max_duration
        if args.soundfont is not None:
            overrides['dataset']['soundfont_path'] = args.soundfont
        if args.use_existing_dataset:
            overrides['dataset']['use_existing_dataset'] = True
        if args.dataset_path is not None:
            overrides['dataset']['existing_dataset_path'] = args.dataset_path
    
    # Audio overrides
    if args.sample_rate is not None or args.n_mels is not None or args.no_augmentation:
        overrides['audio'] = {}
        if args.sample_rate is not None:
            overrides['audio']['sample_rate'] = args.sample_rate
        if args.n_mels is not None:
            overrides['audio']['n_mels'] = args.n_mels
        if args.no_augmentation:
            overrides['audio']['augmentation'] = {'enabled': False}
    
    # Model overrides
    if any([args.encoder, args.freeze_encoder, args.hidden_size, args.num_decoder_layers, args.device]):
        overrides['model'] = {}
        if args.encoder is not None:
            overrides['model']['encoder_name'] = args.encoder
        if args.freeze_encoder:
            overrides['model']['freeze_encoder'] = True
        if args.hidden_size is not None:
            overrides['model']['hidden_size'] = args.hidden_size
        if args.num_decoder_layers is not None:
            overrides['model']['num_decoder_layers'] = args.num_decoder_layers
        if args.device is not None:
            overrides['model']['device'] = args.device
    
    # Training overrides
    training_overrides = {}
    if args.batch_size is not None:
        training_overrides['batch_size'] = args.batch_size
    if args.epochs is not None:
        training_overrides['num_epochs'] = args.epochs
    if args.learning_rate is not None:
        training_overrides['learning_rate'] = args.learning_rate
    if args.warmup_steps is not None:
        training_overrides['warmup_steps'] = args.warmup_steps
    if args.no_mixed_precision:
        training_overrides['use_mixed_precision'] = False
    if args.num_workers is not None:
        training_overrides['num_workers'] = args.num_workers
    if args.checkpoint_dir is not None:
        training_overrides['checkpoint_dir'] = args.checkpoint_dir
    if args.log_dir is not None:
        training_overrides['log_dir'] = args.log_dir
    if args.resume_checkpoint is not None:
        training_overrides['resume_checkpoint'] = args.resume_checkpoint
    
    if training_overrides:
        overrides['training'] = training_overrides
    
    # Evaluation overrides
    if args.no_eval or args.no_viz:
        overrides['evaluation'] = {}
        if args.no_eval:
            overrides['evaluation']['enabled'] = False
        if args.no_viz:
            overrides['evaluation']['visualizations'] = {'enabled': False}
    
    # Output overrides
    if args.output_dir is not None:
        overrides['output'] = {'model_dir': args.output_dir}
    
    # Merge configs
    merged_config = merge_configs(yaml_config, overrides)
    
    # Convert to dataclass
    config = dict_to_config(merged_config)
    
    # Store additional args
    config.experiment_name = args.name if args.name else 'test_run'
    config.seed = args.seed
    config.resume_checkpoint = args.resume_checkpoint if hasattr(args, 'resume_checkpoint') else args.resume if hasattr(args, 'resume') else None
    config.resume_from_epoch = args.resume_from_epoch if hasattr(args, 'resume_from_epoch') else None
    config.eval_only = args.eval_only
    config.eval_model_path = args.model_path
    
    return config


if __name__ == '__main__':
    # Test config parsing
    config = parse_args_to_config()
    print("Configuration loaded successfully:")
    print(f"Dataset samples: {config.dataset.samples}")
    print(f"Training epochs: {config.training.num_epochs}")
    print(f"Model device: {config.model.device}")
