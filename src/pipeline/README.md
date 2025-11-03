# Sound2Sheet Training Pipeline

Complete end-to-end training pipeline that integrates all Sound2Sheet features into a unified workflow.

## Overview

The pipeline automates the entire machine learning workflow:
1. **Dataset Generation** - Create synthetic MIDI and audio samples
2. **Model Training** - Train AST-based transformer model
3. **Evaluation** - Calculate metrics and analyze performance
4. **Reporting** - Generate CSV/JSON reports and visualizations

## Quick Start

### Basic Usage

```bash
# Run with default configuration
python -m src.pipeline.run_pipeline

# Use custom config file
python -m src.pipeline.run_pipeline --config my_config.yaml

# Quick test run
python -m src.pipeline.run_pipeline --samples 100 --epochs 5 --no-eval
```

### Common Scenarios

```bash
# Small training run for testing
python -m src.pipeline.run_pipeline \
    --samples 500 \
    --epochs 10 \
    --batch-size 8

# Full production training
python -m src.pipeline.run_pipeline \
    --samples 10000 \
    --epochs 100 \
    --batch-size 32 \
    --device cuda \
    --name production_run_v1

# Resume from checkpoint
python -m src.pipeline.run_pipeline \
    --resume models/checkpoints/checkpoint_epoch_50.pt \
    --epochs 100

# Evaluation only (skip training)
python -m src.pipeline.run_pipeline \
    --eval-only \
    --model-path models/trained/sound2sheet_model.pt
```

## Configuration

### Configuration File

The pipeline uses YAML configuration files. Default config is at `src/pipeline/config.yaml`.

```yaml
# Dataset Generation
dataset:
  samples: 1000
  complexity: "medium"  # simple, medium, complex
  min_notes: 10
  max_notes: 50
  min_duration: 5.0
  max_duration: 30.0
  synthesize_audio: true
  
# Model Architecture
model:
  encoder_name: "MIT/ast-finetuned-audioset-10-10-0.4593"
  freeze_encoder: true
  hidden_size: 768
  num_decoder_layers: 6
  device: "cuda"
  
# Training Configuration
training:
  batch_size: 16
  num_epochs: 50
  learning_rate: 0.0001
  use_mixed_precision: true
  early_stopping_patience: 10
  
# Evaluation
evaluation:
  enabled: true
  onset_tolerance: 0.05
  visualizations:
    enabled: true
    plot_types: ["dashboard", "confusion_matrix"]
```

### Command-Line Arguments

Command-line arguments override config file values.

#### General Options

```bash
--config PATH           # Path to YAML config file (default: src/pipeline/config.yaml)
--name NAME            # Experiment name for output directories
--seed INT             # Random seed for reproducibility (default: 42)
```

#### Dataset Generation

```bash
--samples INT          # Number of samples to generate
--complexity STR       # MIDI complexity: simple, medium, complex
--min-notes INT        # Minimum notes per sample
--max-notes INT        # Maximum notes per sample
--min-duration FLOAT   # Minimum sample duration (seconds)
--max-duration FLOAT   # Maximum sample duration (seconds)
--soundfont PATH      # Path to custom soundfont file
```

#### Audio Processing

```bash
--sample-rate INT      # Audio sample rate (default: 16000)
--n-mels INT          # Number of mel bands (default: 128)
--no-augmentation     # Disable audio augmentation
```

#### Model Architecture

```bash
--encoder STR          # Encoder model name/path
--freeze-encoder      # Freeze encoder weights during training
--hidden-size INT     # Model hidden size
--num-decoder-layers INT  # Number of decoder layers
--device STR          # Device: cuda, cpu, or auto
```

#### Training

```bash
--batch-size INT       # Training batch size
--epochs INT          # Number of training epochs
--lr FLOAT            # Learning rate
--warmup-steps INT    # Learning rate warmup steps
--no-mixed-precision  # Disable mixed precision training
--num-workers INT     # Number of dataloader workers
--resume PATH         # Resume training from checkpoint
```

#### Evaluation

```bash
--no-eval             # Skip evaluation after training
--no-viz              # Skip visualization generation
--eval-only           # Run evaluation only (skip training)
--model-path PATH     # Model path for evaluation-only mode
```

#### Output

```bash
--output-dir PATH      # Base output directory
--checkpoint-dir PATH  # Checkpoint save directory
--log-dir PATH        # Log directory
```

## Pipeline Workflow

### Step 1: Dataset Generation

Generates synthetic training data using:
- **MIDIGenerator**: Creates realistic piano MIDI sequences
- **AudioSynthesizer**: Converts MIDI to audio using FluidSynth

**Output:**
- `data/datasets/training_run/train/` - Training samples
- `data/datasets/training_run/val/` - Validation samples
- `data/datasets/training_run/test/` - Test samples
- `train_manifest.json`, `val_manifest.json`, `test_manifest.json`

### Step 2: Model Training

Trains the Sound2Sheet model:
- **AST Encoder**: Pretrained Audio Spectrogram Transformer
- **Transformer Decoder**: Custom decoder for note prediction
- **Training Loop**: Mixed precision, gradient accumulation, early stopping

**Output:**
- `models/checkpoints/` - Training checkpoints
- `models/trained/sound2sheet_model.pt` - Final trained model
- `models/trained/training_history.json` - Training metrics
- `logs/` - Training logs

### Step 3: Evaluation

Evaluates model performance:
- **Metrics**: Note accuracy, F1-scores, timing deviation
- **Reports**: CSV and JSON format evaluation reports
- **Visualizations**: Dashboards, confusion matrices, metric plots

**Output:**
- `results/reports/evaluation_report.json`
- `results/reports/evaluation_report.csv`
- `results/visualizations/evaluation_dashboard.png`
- `results/visualizations/confusion_matrix.png`

## Examples

### Example 1: Quick Test Run

```bash
python -m src.pipeline.run_pipeline \
    --samples 100 \
    --epochs 5 \
    --batch-size 4 \
    --device cpu \
    --no-eval
```

**Purpose**: Quick sanity check (5-10 minutes)  
**Use case**: Testing code changes, debugging

### Example 2: Small Training Run

```bash
python -m src.pipeline.run_pipeline \
    --config src/pipeline/config.yaml \
    --samples 1000 \
    --epochs 20 \
    --batch-size 16 \
    --name small_run_001
```

**Purpose**: Verify full pipeline (1-2 hours)  
**Use case**: Initial model validation

### Example 3: Production Training

```bash
python -m src.pipeline.run_pipeline \
    --samples 10000 \
    --complexity complex \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0001 \
    --device cuda \
    --name production_v1 \
    --seed 42
```

**Purpose**: Full-scale training (8-12 hours)  
**Use case**: Production model development

### Example 4: Resume Training

```bash
python -m src.pipeline.run_pipeline \
    --resume models/checkpoints/checkpoint_epoch_50.pt \
    --epochs 100 \
    --name production_v1_continued
```

**Purpose**: Continue interrupted training  
**Use case**: Extending training runs

### Example 5: Evaluation Only

```bash
python -m src.pipeline.run_pipeline \
    --eval-only \
    --model-path models/trained/sound2sheet_model.pt \
    --config src/pipeline/config.yaml
```

**Purpose**: Evaluate existing model  
**Use case**: Testing model on new data

### Example 6: Custom Configuration

Create `custom_config.yaml`:
```yaml
dataset:
  samples: 5000
  complexity: "complex"
  key_signatures: ["C", "G", "D", "A", "E"]
  
training:
  batch_size: 24
  num_epochs: 75
  learning_rate: 0.00005
```

Run with custom config:
```bash
python -m src.pipeline.run_pipeline \
    --config custom_config.yaml \
    --name custom_experiment
```

## Output Structure

```
project_root/
├── data/
│   └── datasets/
│       └── training_run/
│           ├── train/
│           ├── val/
│           ├── test/
│           ├── train_manifest.json
│           ├── val_manifest.json
│           └── test_manifest.json
├── models/
│   ├── checkpoints/
│   │   ├── checkpoint_epoch_5.pt
│   │   ├── checkpoint_epoch_10.pt
│   │   └── ...
│   └── trained/
│       ├── sound2sheet_model.pt
│       └── training_history.json
├── logs/
│   └── training.log
└── results/
    ├── reports/
    │   ├── evaluation_report.json
    │   └── evaluation_report.csv
    └── visualizations/
        ├── evaluation_dashboard.png
        ├── confusion_matrix.png
        └── metrics_over_time.png
```

## Configuration Priority

Configuration values are applied in the following order (highest to lowest priority):

1. **Command-line arguments** - `--epochs 100`
2. **YAML config file** - `config.yaml`
3. **Default values** - Built-in defaults

Example:
```bash
# config.yaml has epochs: 50
# This command will use epochs: 100 (CLI overrides config)
python -m src.pipeline.run_pipeline --config config.yaml --epochs 100
```

## Advanced Usage

### Custom Training Script

```python
from src.pipeline import run_pipeline, parse_args_to_config
from src.pipeline.config_parser import PipelineConfig, TrainingConfig

# Load and modify config programmatically
config = parse_args_to_config()
config.training.num_epochs = 200
config.dataset.samples = 15000

# Run pipeline
run_pipeline(config)
```

### Debugging

```bash
# Enable verbose logging
export PYTHONPATH=.
python -m src.pipeline.run_pipeline --samples 10 --epochs 1 2>&1 | tee debug.log

# Test individual components
python -m src.dataset.cli generate --samples 100 --name test
python -m src.model.train --data-dir data/datasets/test --epochs 1
```

## Performance Tips
### For Fast Iteration

- Reduce `--samples` to 100-500 for quick tests
- Set `--device cpu` if no GPU available
- Use `--no-eval` to skip evaluation

### For Production

- Use `--device cuda` for GPU acceleration
- Increase `--batch-size` to maximize GPU utilization
- Enable `--use-mixed-precision` for faster training
- Set `--num-workers 4` or higher for data loading

### For Memory Constraints

- Reduce `--batch-size` to 4 or 8
- Use `--gradient-accumulation-steps` to simulate larger batches
- Freeze encoder with `--freeze-encoder`
- Reduce `--hidden-size` if needed

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python -m src.pipeline.run_pipeline --batch-size 4

# Use gradient accumulation
python -m src.pipeline.run_pipeline --batch-size 4 --gradient-accumulation-steps 4
```

### FluidSynth Not Available

```bash
# Install FluidSynth first
# Ubuntu/Debian: sudo apt-get install fluidsynth
# macOS: brew install fluidsynth
```

### CUDA Out of Memory

```bash
# Use CPU
python -m src.pipeline.run_pipeline --device cpu

# Or reduce model size
python -m src.pipeline.run_pipeline --hidden-size 384 --num-decoder-layers 4
```

### Slow Data Loading

```bash
# Increase number of workers
python -m src.pipeline.run_pipeline --num-workers 8
```

## Integration with Other Tools

### Weights & Biases

```python
# Add to run_pipeline.py
import wandb

wandb.init(project="sound2sheet", config=config)
# Log metrics during training
```

### TensorBoard

```python
# Add to trainer
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=config.training.log_dir)
writer.add_scalar('Loss/train', loss, epoch)
```

## See Also

- [Dataset Generation](../dataset/README.md) - MIDI and audio synthesis
- [Model Training](../model/README.md) - Model architecture and training
- [Evaluation System](../evaluation/README.md) - Metrics and visualization
- [Core Audio Processing](../core/README.md) - Audio preprocessing

## Contributing

To extend the pipeline:

1. Add new configuration options to `config_parser.py`
2. Update YAML schema in `config.yaml`
3. Add command-line arguments to `create_arg_parser()`
4. Implement functionality in `run_pipeline.py`
5. Update this README with examples

## License

MIT License - See LICENSE file for details
