# Sound2Sheet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-531%20passing-brightgreen.svg)](tests/)

AI-powered music transcription system that converts piano audio recordings into structured musical notation.

## Version
Current version: **0.5.0**

## 🎹 Overview

Sound2Sheet is a comprehensive machine learning pipeline for automatic music transcription. It processes piano audio and generates accurate musical notation with support for multiple output formats (JSON, MIDI, MusicXML).

### Key Features
- 🎵 **Audio Processing**: Multi-format support, noise augmentation, mel-spectrogram generation
- 🎼 **Dataset Generation**: Synthetic MIDI creation and audio synthesis
- 🧠 **Deep Learning**: AST-based transformer model for transcription
- 📝 **Notation Export**: JSON, MIDI, and MusicXML output formats
- 📊 **Evaluation System**: Comprehensive metrics and visualization tools

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Sound2Sheet.git
cd Sound2Sheet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install FluidSynth for audio synthesis:
```bash
# Ubuntu/Debian
sudo apt-get install fluidsynth fluid-soundfont-gm

# macOS
brew install fluidsynth

# Windows: Download from http://www.fluidsynth.org/
```

### Usage Examples

#### 1. Generate Training Dataset

```bash
# Generate 1000 samples with MIDI and audio
python -m src.dataset.cli generate --samples 1000 --name piano_v1

# MIDI only (faster, no FluidSynth required)
python -m src.dataset.cli generate --samples 500 --midi-only

# View dataset information
python -m src.dataset.cli info data/datasets/piano_v1_*/
```

#### 2. Train Model

```bash
# Train with default configuration
python -m src.model.train \
    --data-dir data/datasets/piano_v1_train \
    --output-dir models/piano_v1 \
    --epochs 50 \
    --batch-size 16

# Resume from checkpoint
python -m src.model.train \
    --data-dir data/datasets/piano_v1_train \
    --output-dir models/piano_v1 \
    --resume models/piano_v1/checkpoint_epoch_20.pt
```

#### 3. Evaluate Model Performance

```bash
# Run evaluation on test set
python -m src.evaluation.cli evaluate \
    --manifest data/evaluation/test_manifest.json \
    --output results/evaluation.json

# Generate CSV report
python -m src.evaluation.cli report \
    --results results/evaluation.json \
    --output results/report.csv \
    --format csv

# Create visualizations
python -m src.evaluation.cli visualize \
    --results results/evaluation.json \
    --output results/dashboard.png \
    --plot-type dashboard
```

#### 4. Transcribe Audio

```python
from src.model import Sound2SheetModel, InferenceConfig
from src.core import AudioProcessor
from src.converter import NoteSequence, MusicXMLConverter

# Load model
model = Sound2SheetModel.from_pretrained("models/piano_v1/best_model.pt")

# Process audio
processor = AudioProcessor(sample_rate=16000)
audio = processor.load_audio("piano_recording.wav")
mel_spec = processor.generate_mel_spectrogram(audio)

# Transcribe
config = InferenceConfig(max_length=512, strategy="beam_search", num_beams=5)
predictions = model.generate(mel_spec, config)

# Convert to notation
note_seq = NoteSequence.from_predictions(predictions)
converter = MusicXMLConverter(note_seq)
converter.export("output.musicxml")
```

## 📚 Documentation

### Module Documentation
Each module has comprehensive documentation with architecture details, usage examples, and API references:

- **[Core (Audio Processing)](src/core/README.md)** - Audio loading, preprocessing, augmentation, and visualization
- **[Dataset Generation](src/dataset/README.md)** - Synthetic MIDI creation and audio synthesis
- **[Model Training](src/model/README.md)** - AST-based architecture, training pipeline, and inference
- **[Converter (Notation)](src/converter/README.md)** - Note sequences, MIDI export, and MusicXML generation
- **[Evaluation System](src/evaluation/README.md)** - Metrics calculation, batch evaluation, and visualization

### Additional Documentation
- **[Product Requirements](spec/PRD.txt)** - Complete feature specifications
- **[TODO List](spec/docs/TODO_LIST.md)** - Development progress and roadmap
- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute
- **[Changelog](CHANGELOG.md)** - Version history and changes

## 🏗️ Architecture

```
Sound2Sheet/
├── src/
│   ├── core/              # Audio processing pipeline
│   │   ├── audio_processor.py
│   │   ├── audio_visualizer.py
│   │   └── README.md
│   ├── dataset/           # Dataset generation
│   │   ├── midi_generator.py
│   │   ├── audio_synthesizer.py
│   │   ├── dataset_generator.py
│   │   └── README.md
│   ├── model/             # Deep learning model
│   │   ├── sound2sheet_model.py
│   │   ├── ast_model.py
│   │   ├── trainer.py
│   │   └── README.md
│   ├── converter/         # Notation conversion
│   │   ├── note.py
│   │   ├── note_sequence.py
│   │   ├── musicxml_converter.py
│   │   └── README.md
│   ├── evaluation/        # Evaluation system
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   ├── visualizer.py
│   │   └── README.md
│   └── pipeline/          # Training pipeline
│       ├── run_pipeline.py
│       ├── config_parser.py
│       └── README.md
├── results/               # Training results (organized by date)
│   ├── {YYYY-MM-DD}/
│   │   └── {experiment_name}/
│   │       ├── model/          # Final trained model
│   │       ├── checkpoints/    # Training checkpoints
│   │       ├── visualizations/ # Loss curves, graphs
│   │       ├── reports/        # Training reports
│   │       ├── logs/           # Training logs
│   │       └── temp_data/      # Cleaned after training
│   └── README.md
├── tests/                 # Comprehensive test suite (531 tests)
├── spec/                  # Specifications and documentation
└── data/                  # Temporary datasets (auto-cleaned)
```

### Training Pipeline

The pipeline automatically organizes results by date and experiment name, and **cleans up training data** to save disk space:

```bash
# Run training
python -m src.pipeline.run_pipeline \
    --samples 1000 \
    --epochs 50 \
    --batch-size 32 \
    --name my_experiment

# Results saved to: results/2025-11-03/my_experiment/
# ✅ Model, checkpoints, visualizations, reports saved
# ❌ Audio/MIDI training files automatically deleted (saves GB of space!)
```

See **[Pipeline Documentation](src/pipeline/README.md)** and **[Results Structure](results/README.md)** for details.


## 🧪 Testing

This project uses `pytest` for comprehensive testing with **531 tests** and **90%+ coverage**.

### Run All Tests
```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Module Tests
```bash
# Audio processing (88 tests)
pytest tests/core/ -v

# Dataset generation (40 tests)
pytest tests/dataset/ -v

# Model training (95 tests)
pytest tests/model/ -v

# Converter (329 tests)
pytest tests/converter/ -v

# Evaluation system (53 tests)
pytest tests/evaluation/ -v
```

### Test Coverage by Module
| Module | Tests | Coverage |
|--------|-------|----------|
| Core (Audio) | 88 | 95% |
| Dataset | 40 | 92% |
| Model | 95 | 89% |
| Converter | 329 | 97% |
| Evaluation | 53 | 100% |
| **Total** | **531** | **93%** |

## 📊 Project Status

### ✅ Completed Features

#### Feature 1: Audio Processing
- Multi-format audio support (WAV, MP3, FLAC, OGG, M4A)
- Noise augmentation (white, pink, brown, ambient, hum)
- Mel-spectrogram generation with configurable parameters
- Audio visualization tools
- **88 tests passing**

#### Feature 2: Dataset Generation
- Realistic MIDI generation with 3 complexity levels
- Audio synthesis using FluidSynth
- Train/validation/test split management
- Metadata and statistics generation
- **40 tests passing**

#### Feature 3: Model Training Pipeline
- AST-based encoder with freezing support
- Custom transformer decoder
- Mixed precision training (AMP)
- Learning rate scheduling and early stopping
- Checkpoint management and history logging
- **95 tests passing**

#### Feature 4: Notation Converter
- Note and NoteSequence classes
- MIDI export functionality
- MusicXML export with full notation support
- Tied notes and expression markers
- Performance benchmarking utilities
- **329 tests passing**

#### Feature 5: Evaluation System
- Comprehensive metrics (accuracy, F1, precision, recall)
- Batch evaluation with progress tracking
- CSV and JSON report generation
- Visualization dashboard (6 plot types)
- Command-line interface
- **53 tests passing**

### 🚧 In Progress

#### Feature 6: Model Optimization
- Training on larger datasets (1000+ samples)
- End-to-end transcription quality evaluation
- Hyperparameter tuning based on evaluation metrics

### 📋 Planned Features

- **Feature 7**: REST API for transcription services
- **Feature 8**: Web interface for easy access
- Real piano recording support
- Multi-instrument transcription

## 🔧 Configuration

The system uses YAML configuration files for flexible setup:

```yaml
# config/audio.yaml
audio:
  sample_rate: 16000
  n_mels: 128
  hop_length: 512
  n_fft: 2048

# config/model.yaml
model:
  encoder_name: "MIT/ast-finetuned-audioset-10-10-0.4593"
  decoder_hidden_size: 768
  num_decoder_layers: 6
  max_sequence_length: 512

# config/training.yaml
training:
  batch_size: 16
  learning_rate: 0.0001
  num_epochs: 50
  early_stopping_patience: 10
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

### Development Workflow

This project follows **Git Flow**:
- `main`: Production releases
- `develop`: Integration branch
- `feature/*`: Feature development branches
- `bugfix/*`: Bug fix branches
- `release/*`: Release preparation branches

All code changes must include appropriate tests and documentation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ No warranty provided
- ⚠️ No liability accepted

## 📧 Contact

**Volodymyr** - Project Maintainer

- GitHub: [@yourusername](https://github.com/yourusername)
- Project Link: [https://github.com/yourusername/Sound2Sheet](https://github.com/yourusername/Sound2Sheet)

## 🙏 Acknowledgments

- **Hugging Face Transformers** - AST model implementation
- **librosa** - Audio processing utilities
- **PyTorch** - Deep learning framework
- **FluidSynth** - MIDI audio synthesis
- **music21** - Music theory utilities (inspiration)

---

**Made with ❤️ and Python**