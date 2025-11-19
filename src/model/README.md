# Model Training Pipeline

## Overview

Complete training pipeline for Sound2Sheet model using **Piano Roll Classification** approach - combines Audio Spectrogram Transformer (AST) encoder with binary classifier for frame-level piano transcription.

## Architecture

```
Input Audio
    │
    ▼
┌─────────────────────┐
│  AudioProcessor     │  
│  + to_mel()         │
└──────────┬──────────┘
           │
           ▼
    Mel-Spectrogram
    [batch, n_mels, time]
           │
           ▼
┌─────────────────────┐
│  ASTWrapper         │  Pretrained encoder
│  (HuggingFace)      │  from AudioSet
│  - 12 layers        │
│  - 768 hidden       │
└──────────┬──────────┘
           │
           ▼
    Encoder Features
    [batch, time, 768]
           │
           ▼
┌─────────────────────┐
│ PianoRollClassifier │  Binary classifier
│ - Temporal Conv1D   │  for 88 piano keys
│ - Multi-layer FC    │
└──────────┬──────────┘
           │
           ▼
    Piano Roll Logits
    [batch, time, 88]
           │
           ▼ (Sigmoid + Threshold)
    Binary Piano Roll
    [batch, time, 88]
           │
           ▼ (Post-processing)
    Note Events
    [{pitch, onset_ms, offset_ms}]
```

## Key Concepts

### Piano Roll Representation
- **Binary matrix**: `[time_frames, 88]` where each value indicates if a piano key is active
- **Time resolution**: Configurable frame duration (default 10ms = 100 FPS)
- **Key range**: 88 piano keys (A0 to C8, MIDI 21-108)
- **Multi-label classification**: Each frame can have multiple active keys (chords)

### Training Approach
- **Loss**: BCEWithLogitsLoss for multi-label binary classification
- **Supervision**: Direct frame-level supervision from MIDI ground truth
- **No autoregressive decoding**: All frames predicted independently
- **Metrics**: Frame-level accuracy, precision, recall, F1 score

### Inference Process
1. Convert audio to mel-spectrogram
2. Pass through AST encoder → classifier
3. Apply sigmoid to get probabilities
4. Threshold to get binary piano roll
5. Apply median filter smoothing (optional)
6. Extract note events with onset/offset detection
7. Filter by minimum duration

## Model Components

### ModelConfig
```python
@dataclass
class ModelConfig:
    # AST Encoder
    ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    hidden_size: int = 768
    dropout: float = 0.1
    
    # Piano Roll
    num_piano_keys: int = 88
    min_midi_note: int = 21  # A0
    max_midi_note: int = 108  # C8
    frame_duration_ms: float = 10.0  # 10ms = 100 FPS
    classification_threshold: float = 0.5
    
    # Classifier Head
    num_classifier_layers: int = 2
    classifier_hidden_dim: int = 512
    use_temporal_conv: bool = True
    temporal_conv_kernel: int = 5
```

### Training Example

```python
from src.model import Sound2SheetModel, Trainer
from src.model import ModelConfig, TrainingConfig, DataConfig
from src.model.dataset import create_dataloaders

# Configurations
model_config = ModelConfig(
    num_piano_keys=88,
    frame_duration_ms=10.0,
    classification_threshold=0.5,
    use_temporal_conv=True,
    freeze_encoder=True  # Fine-tune classifier only
)

training_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=8,
    num_epochs=50,
    use_mixed_precision=True
)

data_config = DataConfig(
    dataset_dir=Path("data/datasets/my_experiment"),
    sample_rate=16000,
    n_mels=128,
    hop_length=160  # Matches frame_duration_ms
)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_config,
    model_config,
    training_config
)

# Initialize model
model = Sound2SheetModel(model_config, freeze_encoder=True)

# Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    model_config=model_config,
    training_config=training_config
)

history = trainer.train()
```

### Inference Example

```python
from src.model import Sound2SheetModel, InferenceConfig
from src.core import AudioProcessor
import torch

# Load model
checkpoint = torch.load("data/best_model.pt")
config = ModelConfig(
    num_piano_keys=88,
    frame_duration_ms=10.0,
    # ... use checkpoint params
)
model = Sound2SheetModel(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Process audio
processor = AudioProcessor()
audio = processor.load_audio("piano.wav")
mel = processor.to_mel_spectrogram(audio)
mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # Add batch dim

# Predict
inference_config = InferenceConfig(
    classification_threshold=0.5,
    min_note_duration_ms=30.0,
    use_median_filter=True,
    median_filter_size=3
)

piano_roll, events = model.predict(mel_tensor, inference_config)

# Events: List of {pitch, onset_time_ms, offset_time_ms}
for event in events[0]:  # First batch
    print(f"Note {event['pitch']} from {event['onset_time_ms']:.0f}ms to {event['offset_time_ms']:.0f}ms")
```

## Dataset Format

The `PianoDataset` generates piano roll ground truth from MIDI:

```python
# Sample structure
{
    'mel': torch.Tensor,  # [n_mels, time_frames]
    'piano_roll': torch.Tensor,  # [time_frames, 88]
    'audio_path': str
}

# Piano roll generation from MIDI notes
for note in midi_notes:
    key_idx = note['pitch'] - 21  # Map to 0-87
    onset_frame = int(note['onset'] / frame_duration_sec)
    offset_frame = int(note['offset'] / frame_duration_sec)
    piano_roll[onset_frame:offset_frame, key_idx] = 1.0
```

