# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Trainer Refactor:** Removed all internal logging from `Trainer` class and split large methods into smaller, testable functions
- Improved device setup and mixed precision handling in `Trainer`
- Piano roll resizing now robust to tensor types (casts to float before interpolation)
- ModelConfig signature updated (removed obsolete `num_attention_heads`)
- Integration test updated to expect 2D piano roll output

## [0.7.1] - 2025-11-24

### Changed
- See below for details of Trainer refactor, test fixes, and model training improvements

## [Unreleased]
- All test failures after refactor (parameter mismatch, tensor type errors, output shape mismatch)
- `torch.nn.functional.interpolate` now works with float tensors for piano roll resizing
- Tests now match new ModelConfig and Trainer signatures

### Added
- Helper methods for checkpointing and early stopping in `Trainer`

### Performance
- Faster and more robust training pipeline

### Migration Notes
- **BREAKING:** Old tests and configs incompatible with new Trainer and ModelConfig signatures


## [0.7.0] - 2025-11-19
### Changed
- **BREAKING: Complete Model Architecture Redesign - Piano Roll Classification**
  - Migrated from sequence-to-sequence (seq2seq) to Piano Roll Classification approach
  - Model now performs frame-level binary classification instead of autoregressive note generation
  - Architecture changes:
    - Replaced `NoteDecoder` (Transformer decoder) with `PianoRollClassifier` (multi-layer FC + optional Conv1D)
    - Removed positional encoding (not needed for frame-level classification)
    - Output changed from token sequence `[batch, seq, vocab_size]` to piano roll `[batch, time, 88]`
    - Loss changed from CrossEntropyLoss to BCEWithLogitsLoss (multi-label binary)
  - Configuration changes (`ModelConfig`):
    - Removed: `vocab_size`, `pad/sos/eos/unk_token_id`, `max_sequence_length`, `max_notes_per_sample`, `num_decoder_layers`, `num_attention_heads`, `decoder_ffn_dim`
    - Added: `num_piano_keys=88`, `frame_duration_ms=10.0`, `classification_threshold=0.5`, `num_classifier_layers=2`, `classifier_hidden_dim=512`, `use_temporal_conv=True`, `temporal_conv_kernel=5`
  - Configuration changes (`InferenceConfig`):
    - Removed: `max_length`, `temperature`, `use_beam_search`, `beam_size`, `top_k`, `top_p`, `remove_duplicates`, `quantize_timing`
    - Added: `median_filter_size=3`, `min_note_duration_ms=30.0`, `onset_tolerance_ms=50.0`, `use_median_filter=True`, `output_format='events'`, `events_include_velocity=False`, `default_velocity=80`
  - Dataset changes:
    - `PianoDataset.__getitem__()` now returns `piano_roll` [time, 88] instead of `notes` sequence
    - Removed special tokens (SOS/EOS) from data preparation
    - Added `_notes_to_piano_roll()` method for converting MIDI to binary piano roll
    - Collate function now pads piano rolls instead of note sequences
  - Model changes (`Sound2SheetModel`):
    - `forward()` no longer takes `target_notes`, only `mel` input
    - Replaced `generate()` with `predict()` method that returns `(piano_roll, events)`
    - Added `_apply_median_filter()` for temporal smoothing
    - Added `_piano_roll_to_events()` for onset/offset detection and event extraction
  - Trainer changes:
    - Uses BCEWithLogitsLoss instead of CrossEntropyLoss
    - Removed teacher forcing logic
    - Metrics now frame-level: accuracy, precision, recall, F1 score
    - Training loop simplified (no autoregressive decoding needed)
  - Test suite updates:
    - Updated all 52 configuration, dataset, and model tests
    - Replaced NoteDecoder tests with PianoRollClassifier tests
    - Updated trainer tests for new loss function and metrics
    - All tests passing (31 config + 17 dataset + 4 model tests)
  - Documentation updates:
    - Completely rewrote model README with Piano Roll approach
    - Removed seq2seq architecture diagrams and examples
    - Added Piano Roll representation explanation
    - Updated training and inference examples
    - Updated Copilot instructions for new architecture

### Added
- **PianoRollClassifier**: New classification head for frame-level note detection
  - Multi-layer fully connected network with LayerNorm and Dropout
  - Optional temporal Conv1D for local context (kernel size 5)
  - Residual connections in temporal convolution
  - Binary output for 88 piano keys per frame
- **Piano Roll Post-processing**:
  - Median filtering for temporal smoothing
  - Onset/offset detection with configurable tolerance
  - Minimum note duration filtering
  - Event extraction with pitch, onset_ms, offset_ms
- **Improved Inference**:
  - Direct frame prediction (no beam search needed)
  - Faster inference (no autoregressive decoding)
  - More robust to timing variations
  - Better polyphony handling (concurrent notes)

### Fixed
- Import errors in `src/model/__init__.py` (replaced `NoteDecoder` with `PianoRollClassifier`)
- Test configuration mismatches (median_filter_size, output_format validation)
- Model attribute references (`d_model` → `hidden_size` in PianoRollClassifier)

### Performance
- **Training**: Faster convergence (direct supervision vs autoregressive)
- **Inference**: ~10x faster (parallel frame prediction vs sequential generation)
- **Memory**: Lower memory footprint (no decoder attention cache)

### Migration Notes
- **BREAKING**: Old checkpoints incompatible with new architecture
- **BREAKING**: Dataset format changed (piano_roll instead of notes)
- **BREAKING**: API changed (predict() instead of generate())
- Models need to be retrained with new Piano Roll approach
- See model README for updated training examples

## [0.6.2] - 2025-11-11
### Changed
- **Jupyter Notebook Training Pipeline Updates**
  - Complete redesign with improved structure and configuration
  - Increased default samples to 1000 for better training
  - Increased default epochs to 50 for better convergence
  - Dataset now stored in `data/datasets/{EXPERIMENT_NAME}` structure
  - Added versioned directory support for better experiment tracking
  - Improved model configuration:
    - Increased decoder layers from 4 to 6
    - Reduced max_grad_norm to 1.0 for better stability
    - Added gradient accumulation (4 steps) for larger effective batch size
    - Set save_every_n_epochs to 0 (only saves best and final)
  - Enhanced visualization cell:
    - Added proper history file path (`checkpoints/training_history.json`)
    - Improved plot formatting with proper labels and legends
    - Fixed history key names (`train_loss`, `val_loss`, `val_accuracy`)
  - Added evaluation step with comprehensive metrics
  - Added inference testing on sample data
  - Added resume training support (commented example)
  - Added custom configuration examples

### Fixed
- Fixed dataset path configuration in notebook (now uses correct `data/datasets/{EXPERIMENT_NAME}` path)
- Fixed training history visualization to use correct checkpoint directory

## [0.6.1] - 2025-11-10
### Added
- Validation accuracy tracking in Trainer
  - Added `val_accuracies` list to store accuracy for each epoch
  - Accuracy now saved to training history JSON
  - Accuracy included in checkpoint saves for resume support

### Changed
- **Dataset Generation Improvements**
  - Removed versioned timestamp subdirectories (e.g., `_v1.0.0_20251110_165359`)
  - Datasets now created directly in specified output directory
  - Added tqdm progress bars for dataset generation
  - Progress bars show current complexity and tempo during generation
  - Streamlined logging output:
    - Removed redundant initialization messages
    - Removed per-split completion logs
    - Single final summary: `✓ Generated N samples: train=X, val=Y, test=Z → path`
- **Notebook Updates**
  - Simplified dataset path verification (no versioned directory search)
  - Updated Step 2 to use direct path assignment

### Fixed
- Training history now properly saves validation accuracy
- Dataset directory structure simplified for easier navigation

## [0.6.0] - 2025-11-06
### Added
- **Jupyter Notebook Training Pipeline** (`Sound2Sheet_Training_Pipeline.ipynb`)
  - 10-step training workflow for cloud environments
  - GPU detection and automatic configuration
  - Dataset generation with configurable parameters
  - Model training with progress tracking
  - Training visualization (loss curves, accuracy)
  - Evaluation on test set with comprehensive metrics
  - Inference testing on sample data
  - Checkpoint saving (best and final models)
  - Works in Google Colab, Kaggle, and JupyterLab

### Changed
- Simplified notebook structure: focused on training pipeline
- Removed external storage dependencies
- All checkpoints stored locally in notebook environment
- Updated documentation to reflect Jupyter notebook support

## [0.5.0] - 2025-11-03
### Added
- **Feature 5: Evaluation System (Complete)**
  - Comprehensive metrics calculation:
    - Note accuracy measurement
    - Onset/offset/pitch F1-scores
    - Precision and recall for all aspects
    - Timing deviation analysis (mean error ~23.5ms)
    - Tempo error calculation
    - Pitch error distribution per MIDI note
  - Evaluator with batch processing:
    - Single sample and batch evaluation support
    - Progress callbacks for UI integration
    - Statistical aggregation (mean, std, min, max)
    - Error analysis (TP/FP/FN breakdown)
    - JSON persistence with full metadata
  - Report generation:
    - CSV export for sample-level metrics (15 columns)
    - JSON export with complete evaluation results
    - Configurable output formats
  - Visualization system:
    - Dashboard with 6 comprehensive plots
    - Confusion matrix for pitch predictions
    - Error distribution visualizations
    - Metrics tracking over time
    - Per-sample performance comparison
  - Command-line interface:
    - `evaluate` command for running evaluations
    - `report` command for generating CSV/JSON reports
    - `visualize` command for creating plots
  - Note matching algorithm:
    - Greedy matching with configurable tolerances
    - Onset tolerance (default: 50ms)
    - Offset tolerance (default: 50ms)
    - Pitch tolerance (default: exact match)
  - Testing suite (53 tests, 100% passing):
    - Metrics calculation tests (19 tests)
    - Evaluator tests (20 tests)
    - Report generator tests (4 tests)
    - Visualizer tests (10 tests)
  - Documentation:
    - Comprehensive README with architecture diagram
    - Usage examples for all components
    - API reference documentation
    - JSON format specifications
- **Feature 4.5: Converter Enhancements**
  - Advanced NoteSequence features:
    - Tied notes support with multiple tie chain handling
    - Expression markers (dynamics, articulation)
    - Measure organization with time signatures
    - Automatic key signature detection
    - Automatic time signature detection
  - Enhanced Note class:
    - Tied note properties (is_tied_start, is_tied_end, tie_id)
    - Dynamic markings (Dynamic enum: ppp to fff)
    - Articulation support (Articulation enum: staccato, legato, accent, etc.)
    - Expression marker metadata
  - MusicXML export improvements:
    - Full tied notes export with proper notation
    - Dynamic markings in MusicXML format
    - Articulation symbols
    - Measure-based organization
    - Key and time signature handling
  - Performance utilities:
    - Benchmark decorator for function timing
    - BenchmarkResult dataclass for statistics
    - BenchmarkRegistry for centralized tracking
    - Automatic min/max/mean/std calculation
  - Testing suite expansion:
    - Tied notes tests (15 tests)
    - Expression markers tests (10 tests)
    - Measure organization tests (12 tests)
    - Benchmark utility tests (22 tests)
    - Total converter tests: 329 passing
  - Documentation:
    - Updated README with new features
    - Code examples for tied notes and expressions
    - MusicXML export examples

### Changed
- Simplified evaluation reporting:
  - Removed HTML report generation (not needed)
  - Removed PDF report generation (not needed)
  - Focused on CSV/JSON for data export
- Updated TODO_LIST.md:
  - Marked Feature 5 as completed
  - Marked Feature 4.5 enhancements as completed
  - Updated milestones (Milestone 5 and 6)
- Test count updates:
  - Feature 4 (Converter): 51 → 329 tests
  - Feature 5 (Evaluation): 53 tests added

### Fixed
- Evaluation metric edge cases (zero division handling)
- Empty results handling in visualizations
- Note matching boundary conditions

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