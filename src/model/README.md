# Model Training Pipeline

## Overview

Complete training pipeline for Sound2Sheet model, combining Audio Spectrogram Transformer (AST) encoder with custom Note Decoder for end-to-end piano transcription.

## Class Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         MODEL ARCHITECTURE                      │
└────────────────────────────────────────────────────────────────┘

Input Audio
    │
    ▼
┌─────────────────────┐
│  AudioProcessor     │  (from Feature 1)
│  + generate_mel()   │
└──────────┬──────────┘
           │
           ▼
    Mel-Spectrogram
    [n_mels, time]
           │
           ▼
┌─────────────────────┐
│  ASTWrapper         │  Pretrained encoder
│  (HuggingFace)      │  from audioset
│  - 12 layers        │
│  - 768 hidden       │
└──────────┬──────────┘
           │
           ▼
    Encoder Features
    [batch, seq, 768]
           │
           ▼
┌─────────────────────┐
│  NoteDecoder        │  Transformer decoder
│  - N layers         │  with self-attention
│  - Causal masking   │  and cross-attention
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Output Layer       │
│  vocab_size logits  │
└──────────┬──────────┘
           │
           ▼
    Note Sequence
    [MIDI numbers]


┌────────────────────────────────────────────────────────────────┐
│                      TRAINING COMPONENTS                        │
└────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│  ModelConfig     │────────>│ Sound2SheetModel │
│  - vocab_size    │         │  + forward()     │
│  - hidden_size   │         │  + generate()    │
│  - num_layers    │         └────────┬─────────┘
└──────────────────┘                  │
                                      │
┌──────────────────┐                  │
│  DataConfig      │                  │
│  - sample_rate   │                  │
│  - n_mels        │                  │
│  - max_length    │                  │
└────────┬─────────┘                  │
         │                            │
         ▼                            │
┌──────────────────┐                  │
│  PianoDataset    │                  │
│  + __getitem__() │                  │
└────────┬─────────┘                  │
         │                            │
         ▼                            │
┌──────────────────┐                  │
│  DataLoader      │                  │
│  (PyTorch)       │                  │
└────────┬─────────┘                  │
         │                            │
         │                            │
         ▼                            ▼
┌──────────────────────────────────────────────┐
│  Trainer                                     │
│  + train_epoch()                             │
│  + validate()                                │
│  + save_checkpoint()                         │
│  + train()                                   │
└────────┬─────────────────────────────────────┘
         │
         │ uses
         ▼
┌──────────────────────────────────────────────┐
│  TrainingConfig                              │
│  - learning_rate                             │
│  - batch_size                                │
│  - epochs                                    │
│  - optimizer: AdamW                          │
│  - scheduler: Linear/Cosine/Constant         │
│  - mixed_precision: AMP                      │
│  - gradient_clipping                         │
│  - early_stopping                            │
└──────────────────────────────────────────────┘


┌────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                        │
└────────────────────────────────────────────────────────────────┘

Audio File
    │
    ▼
┌──────────────────┐
│  load_audio()    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  generate_mel()  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐         ┌──────────────────┐
│  InferenceConfig │────────>│  Inference       │
│  - strategy      │         │  + transcribe()  │
│  - temperature   │         └────────┬─────────┘
│  - beam_size     │                  │
└──────────────────┘                  │
                                      ▼
                               Note Predictions
                               [MIDI numbers]
```

## Class Dependencies

1. **ModelConfig** → **Sound2SheetModel**: Configuration defines model architecture
2. **DataConfig** → **PianoDataset**: Configuration for data loading
3. **PianoDataset** → **DataLoader**: PyTorch data loading
4. **DataLoader** + **Sound2SheetModel** → **Trainer**: Training orchestration
5. **TrainingConfig** → **Trainer**: Training hyperparameters
6. **InferenceConfig** → **Inference**: Generation strategies

## Core Components

### 1. Sound2SheetModel
Main model combining encoder and decoder:
- **ASTWrapper**: Pretrained Audio Spectrogram Transformer from HuggingFace
- **NoteDecoder**: Custom transformer decoder for note prediction
- **forward()**: Training mode with teacher forcing
- **generate()**: Inference mode with various decoding strategies

### 2. PianoDataset
PyTorch Dataset for loading audio-MIDI pairs:
- Loads from JSON manifest files
- Converts audio to mel-spectrograms
- Extracts MIDI notes as token sequences
- Handles padding and truncation
- Optional data augmentation

### 3. Trainer
Training orchestration:
- **Teacher forcing**: Use ground truth notes during training
- **Mixed precision**: AMP for faster training
- **Gradient clipping**: Prevent exploding gradients
- **Checkpointing**: Save best and periodic checkpoints
- **Early stopping**: Stop when validation loss plateaus
- **Learning rate scheduling**: Linear warmup, cosine decay

### 4. Inference
Transcription pipeline:
- **Greedy decoding**: Fast, deterministic
- **Beam search**: Better quality, slower
- **Sampling**: Temperature-based, top-k, nucleus (top-p)

## Usage Examples

### Basic Training

```python
from src.model import Sound2SheetModel, Trainer, PianoDataset
from src.model import ModelConfig, TrainingConfig, DataConfig
from torch.utils.data import DataLoader

# Configurations
model_config = ModelConfig(
    vocab_size=128,        # MIDI notes 0-127
    hidden_size=256,
    num_decoder_layers=4,
    freeze_encoder=True    # Fine-tune decoder only
)

training_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=8,
    num_epochs=50,
    optimizer="adamw",
    scheduler="cosine",
    use_amp=True,          # Mixed precision
    max_grad_norm=1.0      # Gradient clipping
)

data_config = DataConfig(
    sample_rate=16000,
    n_mels=128,
    max_audio_length=10.0,  # seconds
    max_notes=512
)

# Create datasets
train_dataset = PianoDataset(
    manifest_path="data/datasets/my_dataset_v1.0/train/manifest.json",
    data_config=data_config,
    model_config=model_config,
    is_training=True
)

val_dataset = PianoDataset(
    manifest_path="data/datasets/my_dataset_v1.0/val/manifest.json",
    data_config=data_config,
    model_config=model_config,
    is_training=False
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=training_config.batch_size,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=training_config.batch_size,
    shuffle=False,
    num_workers=4
)

# Initialize model
model = Sound2SheetModel(model_config, freeze_encoder=True)

# Initialize trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    model_config=model_config,
    training_config=training_config
)

# Train
trainer.train()
```

### Using CLI for Training

```bash
# Basic training
python -m src.model.train \
    --manifest-dir data/datasets/my_dataset_v1.0 \
    --output-dir checkpoints \
    --batch-size 8 \
    --epochs 50 \
    --learning-rate 1e-4

# With all options
python -m src.model.train \
    --manifest-dir data/datasets/my_dataset_v1.0 \
    --output-dir checkpoints \
    --batch-size 8 \
    --epochs 50 \
    --learning-rate 1e-4 \
    --hidden-size 256 \
    --num-decoder-layers 4 \
    --freeze-encoder \
    --use-amp \
    --max-grad-norm 1.0 \
    --patience 10 \
    --save-every 5
```

### Resume from Checkpoint

```python
from src.model import Trainer

# Resume training
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    model_config=model_config,
    training_config=training_config,
    resume_from="checkpoints/checkpoint_epoch_20.pt"
)

trainer.train()
```

```bash
# CLI
python -m src.model.train \
    --manifest-dir data/datasets/my_dataset_v1.0 \
    --output-dir checkpoints \
    --resume-from checkpoints/checkpoint_epoch_20.pt
```

### Inference (Transcription)

```python
from src.model import Inference, InferenceConfig
import torch

# Load model
checkpoint = torch.load("checkpoints/best_model.pt")
model = Sound2SheetModel(model_config)
model.load_state_dict(checkpoint['model_state_dict'])

# Inference configuration
inference_config = InferenceConfig(
    strategy="greedy",      # or "beam", "sample"
    max_length=512,
    temperature=1.0,
    beam_size=5,           # for beam search
    top_k=50,              # for sampling
    top_p=0.95             # nucleus sampling
)

# Transcribe audio
inference = Inference(model, model_config, inference_config)
predictions = inference.transcribe("piano_recording.wav")

print(f"Predicted notes: {predictions}")
```

### Using CLI for Inference

```bash
# Greedy decoding (fastest)
python -m src.model.inference \
    --audio-path piano_recording.wav \
    --checkpoint checkpoints/best_model.pt \
    --output-path transcription.json \
    --strategy greedy

# Beam search (better quality)
python -m src.model.inference \
    --audio-path piano_recording.wav \
    --checkpoint checkpoints/best_model.pt \
    --output-path transcription.json \
    --strategy beam \
    --beam-size 5

# Sampling with temperature
python -m src.model.inference \
    --audio-path piano_recording.wav \
    --checkpoint checkpoints/best_model.pt \
    --output-path transcription.json \
    --strategy sample \
    --temperature 0.8 \
    --top-k 50
```

### Custom Training Loop

```python
import torch
from tqdm import tqdm

model = Sound2SheetModel(model_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(50):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        mel = batch['mel'].to(device)           # [B, n_mels, T]
        notes = batch['notes'].to(device)       # [B, max_notes]
        
        # Forward pass
        logits = model(mel, notes)              # [B, max_notes, vocab]
        
        # Compute loss
        loss = criterion(
            logits.view(-1, model_config.vocab_size),
            notes.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

### Evaluation

```python
from src.model import Trainer
import torch

model.eval()
total_loss = 0
total_accuracy = 0

with torch.no_grad():
    for batch in val_loader:
        mel = batch['mel'].to(device)
        notes = batch['notes'].to(device)
        
        logits = model(mel, notes)
        
        # Loss
        loss = criterion(
            logits.view(-1, model_config.vocab_size),
            notes.view(-1)
        )
        total_loss += loss.item()
        
        # Accuracy
        predictions = logits.argmax(dim=-1)
        mask = notes != 0  # Ignore padding
        correct = (predictions == notes) & mask
        accuracy = correct.sum().item() / mask.sum().item()
        total_accuracy += accuracy

avg_loss = total_loss / len(val_loader)
avg_accuracy = total_accuracy / len(val_loader)
print(f"Val Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
```

## API Reference

### ModelConfig

```python
@dataclass
class ModelConfig:
    # Model architecture
    vocab_size: int = 128                           # MIDI notes (0-127)
    hidden_size: int = 256                          # Decoder hidden size
    num_decoder_layers: int = 4                     # Number of decoder layers
    num_attention_heads: int = 8                    # Attention heads
    intermediate_size: int = 1024                   # FFN intermediate size
    dropout: float = 0.1                            # Dropout rate
    
    # AST encoder
    ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    freeze_encoder: bool = True                     # Freeze AST encoder
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 128                         # Start token
    eos_token_id: int = 129                         # End token
    
    # Device
    device: str = "cuda"                            # cuda or cpu
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Optimization
    learning_rate: float = 1e-4                     # Learning rate
    weight_decay: float = 0.01                      # Weight decay
    optimizer: str = "adamw"                        # Optimizer type
    
    # Training
    batch_size: int = 8                             # Batch size
    num_epochs: int = 50                            # Number of epochs
    accumulation_steps: int = 1                     # Gradient accumulation
    
    # Regularization
    max_grad_norm: float = 1.0                      # Gradient clipping
    use_amp: bool = True                            # Mixed precision (AMP)
    
    # Scheduling
    scheduler: str = "cosine"                       # linear/cosine/constant
    warmup_steps: int = 1000                        # Warmup steps
    
    # Checkpointing
    save_dir: Path = Path("checkpoints")
    save_every: int = 5                             # Save every N epochs
    keep_last: int = 3                              # Keep last N checkpoints
    
    # Early stopping
    patience: int = 10                              # Early stopping patience
    min_delta: float = 1e-4                         # Minimum improvement
```

### InferenceConfig

```python
@dataclass
class InferenceConfig:
    strategy: str = "greedy"                        # greedy/beam/sample
    max_length: int = 512                           # Max sequence length
    
    # Beam search
    beam_size: int = 5                              # Beam width
    length_penalty: float = 1.0                     # Length penalty
    
    # Sampling
    temperature: float = 1.0                        # Temperature (0.1-2.0)
    top_k: int = 50                                 # Top-k sampling
    top_p: float = 0.95                             # Nucleus sampling
    
    # Generation
    num_return_sequences: int = 1                   # Number of outputs
```

### Sound2SheetModel

```python
Sound2SheetModel(config: ModelConfig, freeze_encoder: bool = False)
```

**Methods:**

- `forward(mel: Tensor, target_notes: Tensor) -> Tensor`
  - Training mode with teacher forcing
  - Returns: logits [batch, max_notes, vocab_size]

- `generate(mel: Tensor, inference_config: InferenceConfig) -> List[int]`
  - Inference mode with specified strategy
  - Returns: List of predicted MIDI note numbers

- `count_parameters() -> Dict[str, int]`
  - Count model parameters
  - Returns: {"total", "trainable", "frozen"}

### Trainer

```python
Trainer(
    model: Sound2SheetModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    resume_from: Optional[str] = None
)
```

**Methods:**

- `train() -> None`
  - Run complete training loop

- `train_epoch() -> float`
  - Train single epoch
  - Returns: Average training loss

- `validate() -> Tuple[float, float]`
  - Validate model
  - Returns: (val_loss, val_accuracy)

- `save_checkpoint(epoch: int, is_best: bool) -> None`
  - Save model checkpoint

### Inference

```python
Inference(
    model: Sound2SheetModel,
    model_config: ModelConfig,
    inference_config: InferenceConfig
)
```

**Methods:**

- `transcribe(audio_path: str) -> List[int]`
  - Transcribe audio file to MIDI notes
  - Returns: List of predicted note numbers

- `transcribe_batch(audio_paths: List[str]) -> List[List[int]]`
  - Batch transcription
  - Returns: List of predictions

## Checkpoint Format

```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'scheduler_state_dict': Optional[OrderedDict],
    'scaler_state_dict': Optional[OrderedDict],
    'train_loss': float,
    'val_loss': float,
    'val_accuracy': float,
    'best_val_loss': float,
    'model_config': dict,
    'training_config': dict,
    'timestamp': str
}
```

## Training History Format

```json
{
    "start_time": "2025-11-03T10:30:00",
    "end_time": "2025-11-03T14:25:00",
    "total_epochs": 50,
    "best_epoch": 35,
    "best_val_loss": 0.245,
    "history": [
        {
            "epoch": 1,
            "train_loss": 3.456,
            "val_loss": 2.987,
            "val_accuracy": 0.312,
            "learning_rate": 1e-4,
            "timestamp": "2025-11-03T10:35:00"
        }
    ]
}
```

## Testing

```bash
# Run all tests
pytest tests/model/ -v

# Only unit tests
pytest tests/model/ -v -m unit

# With coverage
pytest tests/model/ --cov=src.model --cov-report=html
```

**Coverage:** 95 tests, 100% code coverage

## Performance

- Training time (1000 samples, 50 epochs, single GPU): ~2 hours
- Inference time (single audio, greedy): ~100 ms
- Inference time (single audio, beam search): ~500 ms
- Model size: ~85M parameters (AST) + 5M parameters (decoder)

## Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 10 GB

**Recommended:**
- GPU: NVIDIA GPU with 8+ GB VRAM (RTX 3060 or better)
- CPU: 8+ cores
- RAM: 16+ GB
- Storage: 50+ GB (for datasets and checkpoints)

## Dependencies

```
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
librosa>=0.10.0
numpy>=1.24.0
tqdm>=4.65.0
```

## Model Architecture Details

### AST Encoder
- **Architecture**: Vision Transformer (ViT) adapted for audio
- **Pretrained on**: AudioSet (2M samples)
- **Input**: Mel-spectrogram patches
- **Output**: 768-dim contextual embeddings
- **Layers**: 12 transformer layers
- **Parameters**: ~85M

### Note Decoder
- **Architecture**: Transformer decoder with causal masking
- **Attention**: Self-attention + cross-attention to encoder
- **Layers**: Configurable (default: 4)
- **Hidden size**: Configurable (default: 256)
- **Parameters**: ~5M (default configuration)

### Training Strategy
- **Teacher forcing**: Use ground truth for next-token prediction
- **Loss**: Cross-entropy on note tokens
- **Optimization**: AdamW with cosine schedule
- **Regularization**: Dropout, gradient clipping, weight decay
