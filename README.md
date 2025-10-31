# Sound2Sheet

AI-powered music transcription system that converts piano audio recordings into structured musical notation.

## Version
Current version: 0.1.0-dev

## Quick Start

### Installation

1. Clone and install dependencies:
```bash
git clone https://github.com/yourusername/Sound2Sheet.git
cd Sound2Sheet
pip install -r requirements.txt
```

2. (Optional) Install FluidSynth for audio synthesis:
```bash
# Ubuntu/Debian
sudo apt-get install fluidsynth fluid-soundfont-gm

# macOS
brew install fluidsynth

# Windows: Download from http://www.fluidsynth.org/
```

### Usage Examples

#### Generate Training Dataset

```bash
# Generate 1000 samples with MIDI and audio
python -m src.dataset.cli generate --samples 1000 --name piano_v1

# MIDI only (faster, no FluidSynth required)
python -m src.dataset.cli generate --samples 500 --midi-only

# View dataset information
python -m src.dataset.cli info data/datasets/piano_v1_*/
```

#### Audio Processing

```python
from src.core import AudioProcessor

# Initialize processor
processor = AudioProcessor(sample_rate=16000)

# Load and process audio
audio = processor.load_audio("piano.wav")

# Apply noise augmentation
augmented = processor.augment_audio(audio, noise_type="white", noise_level=0.05)

# Generate mel-spectrogram
mel_spec = processor.generate_mel_spectrogram(audio)
```

See [Dataset Generation Documentation](docs/dataset_generation.md) for detailed usage.

## Project Status
� In Development - Core Features Implementation

### Completed Features

#### ✅ Feature 1: Audio Processing
- **Noise Augmentation**: 5 noise types with Strategy pattern
  - White Noise, Pink Noise, Brown Noise, Ambient Noise, Hum
- **Audio Conversion**: Multiple format support (WAV, MP3, FLAC, OGG)
- **Mel-Spectrogram**: Configurable spectrogram generation
- **Visualization**: Waveform, spectrogram, mel-spectrogram plotting
- **Performance**: Caching and optimization utilities
- **Test Coverage**: 88 tests passing

#### ✅ Feature 2: Dataset Generation
- **MIDI Generation**: Realistic piano patterns with 3 complexity levels
  - 24 key signatures, multiple time signatures
  - Melody and chord progressions
- **Audio Synthesis**: FluidSynth-based MIDI-to-audio conversion
  - Programmatic synthesis (no subprocess)
  - Batch processing support
- **Dataset Management**: Hierarchical structure with train/val/test splits
  - Automatic metadata generation
  - Comprehensive statistics
- **CLI Tools**: Command-line interface for dataset operations
- **Test Coverage**: 40 tests passing (13 skipped without FluidSynth)

### In Progress

#### 🟡 Feature 3: Model Training Pipeline (Next)
- Training data loaders
- AST-based model architecture
- Training/validation loops
- Checkpointing and logging

### Planned

- **Feature 4**: Transcription Engine
- **Feature 5**: Output Formats (JSON/MIDI/MusicXML)
- **Feature 6**: REST API
- **Feature 7**: Web Interface

## Architecture
- **Audio Processing**: Noise augmentation, format conversion, mel-spectrograms
- **Dataset Generation**: MIDI creation, audio synthesis, metadata management
- **Model**: AST-based transcription with custom decoder (planned)
- **Output**: JSON/MIDI/MusicXML notation formats (planned)
- **API**: REST API for transcription services (planned)

## Testing

This project uses `pytest` for comprehensive testing.

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Feature Tests
```bash
# Audio processing tests (88 tests)
pytest tests/audio_processing/ -v

# Dataset generation tests (40 passing, 13 skipped)
pytest tests/dataset_generation/ -v
```

### Test Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

**Current Coverage:**
- Audio Processing: 88 tests passing
- Dataset Generation: 40 tests passing, 13 skipped (require FluidSynth)
- Total: 128 tests, 88% coverage

## Documentation

- **[Dataset Generation Guide](docs/dataset_generation.md)** - Complete guide to MIDI generation, audio synthesis, and dataset creation
- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to the project
- **[License](LICENSE)** - MIT License

## Development
This project follows Git Flow workflow:
- `main`: Production releases
- `develop`: Integration branch
- `feature/*`: Feature development branches

All code changes must include appropriate unit tests.

## License
MIT License