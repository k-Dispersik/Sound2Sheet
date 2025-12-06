# Model: Piano Roll Classification

Frame-level piano transcription using AST encoder + binary classifier.

## Architecture

**Pipeline:**
```
Mel-Spectrogram [batch, 128, time]
    ↓
ASTWrapper (Pretrained Encoder)
    - MIT/ast-finetuned-audioset-10-10-0.4593
    - 12 transformer layers
    - 768 hidden dimensions
    ↓
Features [batch, time, 768]
    ↓
PianoRollClassifier
    - Optional temporal Conv1D (kernel=5)
    - 2-layer FC network
    ↓
Logits [batch, time, 88]
    ↓ (Sigmoid + Threshold)
Binary Piano Roll [batch, time, 88]
    ↓ (Post-processing)
Note Events [{pitch, onset_ms, offset_ms}]
```

**Key Details:**
- **Frame rate:** 100 FPS (10ms resolution)
- **Piano keys:** 88 keys (A0-C8, MIDI 21-108)
- **Loss:** BCEWithLogitsLoss (multi-label binary)
- **Optimizer:** AdamW with cosine scheduling
- **No autoregressive decoding** - all frames predicted independently

## Usage

### Training via run_pipeline.py

```bash
python run_pipeline.py  # Uses config.json
```

**config.json:**
```json
{
    "model_config": {
        "num_piano_keys": 88,
        "hidden_size": 512,
        "frame_duration_ms": 10.0,
        "classification_threshold": 0.5,
        "use_temporal_conv": true
    },
    "training": {
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 0.001,
        "use_mixed_precision": true
    }
}
```

### Manual Training

```python
from src.model import Sound2SheetModel, Trainer, ModelConfig, TrainingConfig

model_config = ModelConfig(
    num_piano_keys=88,
    frame_duration_ms=10.0,
    use_temporal_conv=True
)

training_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=8,
    num_epochs=50
)

model = Sound2SheetModel(model_config)
trainer = Trainer(model, train_loader, val_loader, model_config, training_config)
history = trainer.train()
```

### Inference

```python
from src.model import Sound2SheetModel, InferenceConfig
from src.core import AudioProcessor

# Load model
model = Sound2SheetModel.from_pretrained("models/best_model.pt")

# Process audio
processor = AudioProcessor()
audio = processor.load_audio("piano.wav")
mel = processor.to_mel_spectrogram(audio)

# Predict
config = InferenceConfig(
    classification_threshold=0.5,
    min_note_duration_ms=30.0,
    use_median_filter=True
)
piano_roll, events = model.predict(mel, config)

# events: [{pitch, onset_time_ms, offset_time_ms}]
```

## Components

### Sound2SheetModel
Main model combining encoder + classifier.

**Methods:**
- `forward(mel)` → logits [batch, time, 88]
- `predict(mel, config)` → (piano_roll, events)

### ASTWrapper
Pretrained Audio Spectrogram Transformer encoder.

### PianoRollClassifier
Binary classifier head with optional temporal convolution.

### PianoDataset
PyTorch dataset with piano roll ground truth from MIDI.

**Format:**
```python
{
    'mel': [n_mels, time],
    'piano_roll': [time, 88],
    'audio_path': str
}
```

### Trainer
Training loop with mixed precision, checkpointing, and validation.

## Configuration

**ModelConfig:**
```python
ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
num_piano_keys: int = 88
frame_duration_ms: float = 10.0
classification_threshold: float = 0.5
use_temporal_conv: bool = True
```

**TrainingConfig:**
```python
learning_rate: float = 1e-4
batch_size: int = 8
num_epochs: int = 50
use_mixed_precision: bool = True
optimizer: str = "adamw"
```

**InferenceConfig:**
```python
classification_threshold: float = 0.5
min_note_duration_ms: float = 30.0
use_median_filter: bool = True
median_filter_size: int = 3
```

## Testing

```bash
pytest tests/model/ -v --cov=src.model
```

**Coverage:** 95 tests, 89% coverage

