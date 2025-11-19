"""
Model configuration for Sound2Sheet.

Defines configuration classes for model architecture, training, and inference.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the Sound2Sheet Piano Roll model."""
    
    # Model architecture
    ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    hidden_size: int = 768
    dropout: float = 0.1
    
    # Piano roll parameters
    num_piano_keys: int = 88  # Number of piano keys (A0 to C8)
    min_midi_note: int = 21  # A0
    max_midi_note: int = 108  # C8
    
    # Frame-level detection parameters
    frame_duration_ms: float = 10.0  # Duration of each frame in milliseconds (10ms = 100 FPS)
    classification_threshold: float = 0.5  # Threshold for binary classification
    
    # Classification head architecture
    num_classifier_layers: int = 2  # Number of fully connected layers in classifier head
    classifier_hidden_dim: int = 512  # Hidden dimension for classifier
    use_temporal_conv: bool = True  # Use 1D conv for temporal context before classification
    temporal_conv_kernel: int = 5  # Kernel size for temporal convolution
    
    # Training device
    device: str = "cuda"  # Will auto-detect in code
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.num_piano_keys != (self.max_midi_note - self.min_midi_note + 1):
            raise ValueError(f"num_piano_keys must equal max_midi_note - min_midi_note + 1")
        if self.frame_duration_ms <= 0:
            raise ValueError(f"frame_duration_ms must be positive, got {self.frame_duration_ms}")
        if not 0.0 <= self.classification_threshold <= 1.0:
            raise ValueError(f"classification_threshold must be in [0, 1], got {self.classification_threshold}")
        if self.num_classifier_layers <= 0:
            raise ValueError(f"num_classifier_layers must be positive, got {self.num_classifier_layers}")
    

@dataclass
class TrainingConfig:
    """Configuration for training the model."""
    
    # Training hyperparameters
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Mixed precision training
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"  # 'linear', 'cosine', 'constant'
    num_cycles: float = 0.5  # For cosine schedule
    
    # Checkpointing
    save_steps: int = 500  # Deprecated: use save_every_n_epochs instead
    save_every_n_epochs: int = 0  # 0 = don't save intermediate, only best + final
    save_total_limit: int = 3
    checkpoint_dir: Path = Path("checkpoints")
    
    # Logging
    logging_steps: int = 100
    eval_steps: int = 500
    log_dir: Path = Path("logs")
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # GPU memory fraction: float in (0.0, 1.0] to cap per-process GPU memory (optional)
    # If set, pipeline will call torch.cuda.set_per_process_memory_fraction(fraction)
    gpu_memory_fraction: Optional[float] = None
    
    # Validation - use smaller batch size to avoid OOM
    val_batch_size: int = 4
    eval_accumulation_steps: Optional[int] = None
    
    # Optimizer and scheduler
    optimizer: str = "adamw"  # 'adam', 'adamw', 'sgd'
    scheduler: str = "linear"  # 'linear', 'cosine', 'constant'
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")
        
        # Convert checkpoint_dir and log_dir to Path if needed
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)
        
        # Create directories if they don't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class InferenceConfig:
    """Configuration for Piano Roll inference."""
    
    # Inference parameters
    batch_size: int = 1
    classification_threshold: float = 0.5  # Threshold for note activation
    
    # Post-processing
    min_note_duration_ms: float = 30.0  # Minimum note duration in milliseconds
    onset_tolerance_ms: float = 50.0  # Tolerance for merging close onsets
    use_median_filter: bool = True  # Median filter to smooth predictions
    median_filter_size: int = 3  # Size of median filter (frames)
    
    # Output format
    output_format: str = "events"  # "events" or "piano_roll"
    events_include_velocity: bool = False  # Include velocity in events (if False, use default)
    default_velocity: int = 80  # Default MIDI velocity for notes
    
    # Device
    device: str = "cuda"  # Will auto-detect in code
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.classification_threshold <= 1.0:
            raise ValueError(f"classification_threshold must be in [0, 1], got {self.classification_threshold}")
        if self.min_note_duration_ms < 0:
            raise ValueError(f"min_note_duration_ms must be non-negative, got {self.min_note_duration_ms}")
        if self.onset_tolerance_ms < 0:
            raise ValueError(f"onset_tolerance_ms must be non-negative, got {self.onset_tolerance_ms}")
        if self.output_format not in ["events", "piano_roll"]:
            raise ValueError(f"output_format must be 'events' or 'piano_roll', got {self.output_format}")
        if not 0 <= self.default_velocity <= 127:
            raise ValueError(f"default_velocity must be in [0, 127], got {self.default_velocity}")


@dataclass
class DataConfig:
    """Configuration for dataset and data loading."""
    
    # Dataset paths
    dataset_dir: Path = Path("data/datasets")
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Audio processing
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float = 8000.0
    
    # Augmentation
    use_augmentation: bool = True
    noise_scale: float = 0.005
    noise_type: str = 'white'  # white, pink, brown, ambient, hum, or 'random'
    noise_types_pool: list = None  # List of noise types for random selection (if noise_type='random')
    pitch_shift_range: int = 2  # semitones
    
    # Caching
    use_cache: bool = False
    cache_dir: Path = Path("data/cache")
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert to Path if needed
        if isinstance(self.dataset_dir, str):
            self.dataset_dir = Path(self.dataset_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        
        # Validate splits
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"train/val/test splits must sum to 1.0, got {total_split}")
        
        # Validate audio parameters
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {self.n_mels}")
        
        # Create cache directory if needed
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

