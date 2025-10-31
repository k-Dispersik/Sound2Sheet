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