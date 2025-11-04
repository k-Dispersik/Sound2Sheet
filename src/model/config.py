"""
Model configuration for Sound2Sheet.

Defines configuration classes for model architecture, training, and inference.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the Sound2Sheet model."""
    
    # Model architecture
    ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    hidden_size: int = 768
    num_decoder_layers: int = 6
    num_attention_heads: int = 8
    decoder_ffn_dim: int = 2048
    dropout: float = 0.1
    
    # Vocabulary (88 piano keys + special tokens)
    vocab_size: int = 92  # 88 keys + <pad>, <sos>, <eos>, <unk>
    pad_token_id: int = 0
    sos_token_id: int = 89
    eos_token_id: int = 90
    unk_token_id: int = 91
    
    # MIDI note range
    min_midi_note: int = 21  # A0
    max_midi_note: int = 108  # C8
    
    # Sequence parameters
    max_sequence_length: int = 512
    max_notes_per_sample: int = 256
    
    # Training device
    device: str = "cuda"  # Will auto-detect in code
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.num_decoder_layers <= 0:
            raise ValueError(f"num_decoder_layers must be positive, got {self.num_decoder_layers}")
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {self.num_attention_heads}")
    

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
    """Configuration for model inference."""
    
    # Inference parameters
    batch_size: int = 1
    max_length: int = 1000  # Maximum number of notes to generate
    
    # Decoding strategy
    use_beam_search: bool = False
    beam_size: int = 5
    temperature: float = 1.0
    top_k: int = 0  # 0 = disabled
    top_p: float = 0.0  # 0 = disabled
    
    # Post-processing
    remove_duplicates: bool = True
    min_note_duration: float = 0.05  # seconds
    quantize_timing: bool = True
    
    # Device
    device: str = "cuda"  # Will auto-detect in code
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.use_beam_search and self.beam_size <= 0:
            raise ValueError(f"beam_size must be positive when beam search is enabled, got {self.beam_size}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")


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

