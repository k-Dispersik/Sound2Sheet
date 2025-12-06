# Sound2Sheet

AI-powered piano transcription system converting audio recordings into structured musical notation using deep learning.

**Version:** 0.7.2

## Overview

Sound2Sheet uses **Piano Roll Classification** with Audio Spectrogram Transformer (AST) for frame-level piano transcription. The system processes audio through mel-spectrograms and outputs binary piano roll predictions for 88 keys.

### Key Features
- **Piano Roll Classification:** Frame-level binary classification (100 FPS, 10ms resolution)
- **AST Encoder:** Pretrained Audio Spectrogram Transformer from MIT
- **Synthetic Dataset:** Automated MIDI generation and audio synthesis
- **JSON Configuration:** Simple pipeline setup and experiment tracking
- **Comprehensive Testing:** 531 tests with 93% coverage

## Quick Start

### Installation

```bash
git clone https://github.com/k-Dispersik/Sound2Sheet.git
cd Sound2Sheet
pip install -r requirements.txt
```

### Training Pipeline

```bash
# Configure experiment in config.json
python run_pipeline.py
```

**Configuration (`config.json`):**
```json
{
    "experiment_name": "data/my_experiment",
    "dataset": {
        "total_samples": 1000,
        "complexity_distribution": {
            "beginner": 0.5,
            "intermediate": 0.4,
            "advanced": 0.1
        }
    },
    "model_config": {
        "num_piano_keys": 88,
        "hidden_size": 512,
        "frame_duration_ms": 10.0,
        "classification_threshold": 0.5,
        "use_temporal_conv": true
    },
    "data_config": {
        "sample_rate": 16000,
        "n_mels": 128,
        "hop_length": 512
    },
    "training": {
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 0.001,
        "use_mixed_precision": true
    }
}
```

### Basic Usage

```python
from src.model import Sound2SheetModel, InferenceConfig
from src.core import AudioProcessor

# Load model
model = Sound2SheetModel.from_pretrained("models/best_model.pt")

# Process audio
processor = AudioProcessor(sample_rate=16000)
audio = processor.load_audio("piano.wav")
mel = processor.to_mel_spectrogram(audio)

# Transcribe
config = InferenceConfig(
    classification_threshold=0.5,
    use_median_filter=True,
    min_note_duration_ms=30.0
)
piano_roll, events = model.predict(mel, config)
```

## Architecture

### Model Pipeline

```
Audio → Mel-Spectrogram → AST Encoder → Piano Roll Classifier → Binary Predictions → Post-Processing → Note Events
```

**Components:**
1. **AST Encoder** (MIT/ast-finetuned-audioset-10-10-0.4593)
   - 12 transformer layers
   - 768 hidden dimensions
   - Pretrained on AudioSet
   
2. **Piano Roll Classifier**
   - Optional temporal Conv1D (5-kernel)
   - 2-layer FC network
   - 88 output keys (A0-C8, MIDI 21-108)
   
3. **Post-Processing**
   - Median filter smoothing
   - Onset/offset detection
   - Duration filtering

### Training Details

- **Loss:** BCEWithLogitsLoss (multi-label binary)
- **Optimizer:** AdamW with cosine scheduling
- **Mixed Precision:** torch.amp for memory efficiency
- **Frame Rate:** 100 FPS (10ms resolution)
- **Input:** Mel-spectrogram [128 bins, 16kHz]
- **Output:** Piano roll [time, 88 keys]

## Project Structure

```
Sound2Sheet/
├── src/
│   ├── core/              # Audio processing
│   │   ├── audio_processor.py
│   │   ├── audio_visualizer.py
│   │   └── noise_strategies.py
│   ├── dataset/           # Dataset generation
│   │   ├── midi_generator.py
│   │   ├── audio_synthesizer.py
│   │   └── dataset_generator.py
│   ├── model/             # Deep learning
│   │   ├── sound2sheet_model.py
│   │   ├── ast_model.py (ASTWrapper, PianoRollClassifier)
│   │   ├── trainer.py
│   │   ├── dataset.py (PianoDataset, create_dataloaders)
│   │   └── config.py (ModelConfig, TrainingConfig, InferenceConfig)
│   ├── converter/         # Notation export
│   └── evaluation/        # Metrics & visualization
├── tests/                 # 531 tests, 93% coverage
├── run_pipeline.py        # Main training script
├── config.json            # Pipeline configuration
└── CHANGELOG.md

```

## Testing

```bash
# All tests
pytest tests/ -v

# Specific modules
pytest tests/core/         # Audio (88 tests)
pytest tests/dataset/      # Dataset (40 tests)
pytest tests/model/        # Model (95 tests)
pytest tests/converter/    # Converter (329 tests)
pytest tests/evaluation/   # Evaluation (53 tests)

# With coverage
pytest tests/ --cov=src --cov-report=html
```

**Coverage:** 93% (531 tests across 31 files)

## Documentation

- **[Core (Audio)](src/core/README.md)** - Audio loading, mel-spectrograms, augmentation
- **[Dataset](src/dataset/README.md)** - MIDI generation, audio synthesis
- **[Model](src/model/README.md)** - Architecture, training, inference
- **[Converter](src/converter/README.md)** - Note sequences, MIDI/MusicXML
- **[Evaluation](src/evaluation/README.md)** - Metrics, visualization
- **[Changelog](CHANGELOG.md)** - Version history

## Key Changes in v0.8.0

- **JSON Configuration:** Replaced YAML with JSON for pipeline config
- **Piano Roll Classification:** Frame-level binary classification (v0.7.0)
- **Improved Training:** Fixed torch.amp deprecations, CPU/CUDA compatibility
- **Better DataLoaders:** Smart pin_memory handling

See [CHANGELOG.md](CHANGELOG.md) for complete history.

## License

MIT License - see [LICENSE](LICENSE) file.