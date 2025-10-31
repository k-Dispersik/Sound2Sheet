# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

### Changed
- Nothing yet

### Fixed
- Nothing yet

## [0.4.0] - 2025-10-31
### Added
- **Feature 3: Model Training Pipeline (Complete)**
  - Model architecture implementation:
    - AST (Audio Spectrogram Transformer) encoder wrapper with freezing support
    - Custom Transformer decoder for note prediction
    - Positional encoding for sequence modeling
    - End-to-end Sound2Sheet model
  - Configuration system:
    - ModelConfig for architecture parameters
    - TrainingConfig for training hyperparameters
    - InferenceConfig for generation settings
    - DataConfig for dataset and audio processing
  - Dataset pipeline:
    - PianoDataset class for loading audio-MIDI pairs
    - Custom collate function for variable-length sequences
    - Audio augmentation in training mode
    - Efficient dataloader creation
  - Training infrastructure:
    - Full training loop with teacher forcing
    - Mixed precision training support (AMP)
    - Gradient accumulation for large batch simulation
    - Learning rate scheduling (linear, cosine, constant)
    - Early stopping with patience
    - Checkpoint management (best model + periodic saves)
    - Training history logging (JSON format)
  - Inference capabilities:
    - Autoregressive generation
    - Greedy decoding and beam search support
    - Temperature-based sampling
    - Top-k and nucleus (top-p) sampling
  - CLI scripts:
    - train.py for model training
    - inference.py for audio-to-MIDI transcription
  - Testing suite (95 tests, 100% passing):
    - Configuration tests (34 tests)
    - Dataset tests (18 tests)
    - Model architecture tests (26 tests)
    - Trainer tests (17 tests)
    - Integration and edge case coverage
  - Documentation:
    - Comprehensive model README with usage examples
    - Training and inference guides
    - API documentation
    - Model architecture description

### Changed
- Updated .gitignore to exclude logs/ directory
- Added pytest configuration with markers (slow, unit, integration)
- Updated requirements.txt with deep learning dependencies:
  - torch >= 2.0.0
  - transformers >= 4.30.0
  - accelerate >= 0.20.0
  - mido >= 1.3.0
  - pytest and plugins for testing

### Fixed
- AST input shape handling (3D [batch, n_mels, time] without channel dimension)
- Mel-spectrogram padding/truncation to AST max_length (1024 frames)
- Decoder parameter order (encoder_hidden_states, target_notes)
- PyTorch 2.6 compatibility (weights_only=False for torch.load)
- Empty dataloader handling in trainer
- Training history key naming consistency
- ZeroDivisionError in checkpoint saving with empty loaders

## [0.3.0] - 2025-10-31
### Added
- **Feature 2: Dataset Generation (Complete)**
  - AudioSynthesizer class for synthetic piano audio generation
  - MIDI-to-audio synthesis using soundfont (FluidSynth)
  - Multiple soundfont support (piano, electric piano, harpsichord)
  - Configurable synthesis parameters (sample rate, reverb, chorus)
  - MIDI generation from note sequences
  - DatasetGenerator class for automated dataset creation
  - Support for MAESTRO and other MIDI datasets
  - Train/validation/test split generation
  - JSON manifest generation with metadata
  - Batch processing with progress tracking
  - Testing suite (29 tests, 100% passing):
    - AudioSynthesizer tests (14 tests)
    - DatasetGenerator tests (15 tests)
    - Integration and validation tests
  - Comprehensive error handling and logging
  - Memory-efficient processing

### Changed
- Updated .gitignore patterns for data directories
- Added FluidSynth and soundfont dependencies to requirements.txt

### Fixed
- Data directory pattern handling in .gitignore
- MIDI file validation in dataset generation
- Audio synthesis error handling

## [0.2.0] - 2025-10-31
### Added
- **Feature 1: Audio Processing (Complete)**
  - AudioProcessor class with full audio pipeline implementation
  - Multi-format audio support (.wav, .mp3, .m4a)
  - Audio normalization with RMS-based algorithm
  - Pre-emphasis filtering for signal enhancement
  - Audio resampling to configurable sample rates
  - Mel-spectrogram generation with librosa
  - Audio augmentation suite:
    - Pitch shifting (±N semitones)
    - Time stretching (variable rates)
    - Gaussian noise injection
    - Volume adjustment
  - AudioVisualizer class for visualization:
    - Waveform plotting
    - Mel-spectrogram visualization
    - Augmentation comparison plots
  - Comprehensive error handling:
    - File format validation with header checks
    - Corrupted file detection
    - Memory usage estimation and OOM protection
    - Extensive logging throughout pipeline
  - Performance optimization:
    - Dynamic n_fft adjustment for short audio
    - Efficient soundfile backend for WAV files
    - Memory-conscious processing
  - Testing suite (48 tests, 100% passing):
    - Unit tests for all AudioProcessor methods
    - Integration tests for full pipeline
    - Performance benchmarking tests
    - Audio format and quality tests
    - Visualization tests
  - AudioConfig class for flexible configuration
  - YAML-based configuration support

### Changed
- Updated requirements.txt with audio processing dependencies:
  - librosa >= 0.10.0 for audio analysis
  - soundfile for WAV file handling
  - matplotlib and seaborn for visualization
- Optimized audio loading to use soundfile for WAV files (avoiding deprecation warnings)

### Fixed
- All audioread deprecation warnings properly handled
- librosa UserWarnings for n_fft parameter resolved
- Test validation methods now create proper audio file headers
- Memory estimation for large audio files

## [0.1.0-dev] - 2025-10-30
### Added
- Initial project setup
- Basic file structure
- Configuration management
- Requirements specification