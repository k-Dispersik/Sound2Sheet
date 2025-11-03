# Synthetic Dataset Generation

## Overview

Module for generating synthetic musical datasets with MIDI files, realistic audio synthesis, and comprehensive metadata for training piano transcription models.

## Class Architecture

```
┌──────────────────────┐
│  ComplexityLevel     │
│  - BEGINNER          │
│  - INTERMEDIATE      │
│  - ADVANCED          │
└──────────────────────┘

┌──────────────────────┐
│  TimeSignature       │
│  - FOUR_FOUR (4/4)   │
│  - THREE_FOUR (3/4)  │
│  - SIX_EIGHT (6/8)   │
│  - TWO_FOUR (2/4)    │
└──────────────────────┘

┌──────────────────────┐
│  KeySignature        │
│  - C_MAJOR           │
│  - G_MAJOR           │
│  - A_MINOR           │
│  - ... (24 keys)     │
└──────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│  MIDIConfig          │────────>│  MIDIGenerator       │
│  - tempo             │         │  + generate_midi()   │
│  - time_signature    │         │  + batch_generate()  │
│  - key_signature     │         └──────────┬───────────┘
│  - complexity        │                    │
└──────────────────────┘                    │ creates
                                            │
┌──────────────────────┐                    ▼
│  DatasetConfig       │         ┌──────────────────────┐
│  - total_samples     │────────>│  MIDI File (.mid)    │
│  - train/val/test    │         └──────────────────────┘
│  - sample_rate       │                    │
└────────┬─────────────┘                    │
         │                                  │ synthesizes
         │                                  ▼
         │                      ┌──────────────────────┐
         │                      │  AudioSynthesizer    │
         │                      │  + synthesize()      │
         │                      │  + batch_synthesize()│
         │                      └──────────┬───────────┘
         │                                 │
         │                                 │ creates
         │                                 ▼
         │                      ┌──────────────────────┐
         │                      │  Audio File (.wav)   │
         │                      └──────────────────────┘
         │                                 │
         │                                 │
         ▼                                 ▼
┌─────────────────────────────────────────────────────┐
│  DatasetGenerator                                   │
│  + generate_dataset()                               │
│  + generate_split()                                 │
│  + create_manifest()                                │
│  + load_manifest()                                  │
└─────────────────────────────────────────────────────┘
         │
         │ creates
         ▼
┌──────────────────────┐
│  Dataset Structure   │
│  ├── train/          │
│  │   ├── audio/      │
│  │   ├── midi/       │
│  │   └── manifest.   │
│  ├── val/            │
│  └── test/           │
└──────────────────────┘
```

## Class Dependencies

1. **MIDIConfig** → **MIDIGenerator**: Configuration for MIDI generation
2. **MIDIGenerator** → **MIDI Files**: Generates .mid files
3. **MIDI Files** → **AudioSynthesizer**: Synthesizes audio from MIDI
4. **AudioSynthesizer** → **Audio Files**: Creates .wav files
5. **DatasetConfig** → **DatasetGenerator**: Configuration for dataset
6. **DatasetGenerator** → **All Components**: Orchestrates generation pipeline

## Core Components

### 1. MIDIGenerator
Generates realistic MIDI sequences:
- **Chord progressions**: Common patterns (I-IV-V, ii-V-I, etc.)
- **Melody generation**: Scale-based patterns
- **Rhythmic patterns**: Quarter notes, eighth notes, triplets
- **Complexity levels**: Beginner (simple) → Advanced (complex)
- **24 key signatures**: All major and minor keys
- **4 time signatures**: 4/4, 3/4, 6/8, 2/4

### 2. AudioSynthesizer
Converts MIDI to realistic piano audio:
- **FluidSynth integration**: High-quality synthesis
- **Soundfont support**: FluidR3_GM (default) or custom
- **Configurable parameters**: Sample rate, gain
- **Batch processing**: Efficient multi-file synthesis

### 3. DatasetGenerator
Creates complete datasets:
- **Hierarchical structure**: train/val/test splits
- **Metadata generation**: JSON manifests
- **Validation**: Integrity checks
- **Statistics**: Diversity analysis

### 4. CLI Interface
Command-line tool for dataset operations:
- `generate`: Create new dataset
- `info`: Display dataset statistics
- `validate`: Check dataset integrity

## Usage Examples

### Basic Dataset Generation

```python
from src.dataset import DatasetGenerator, DatasetConfig

# Create configuration
config = DatasetConfig(
    name="my_piano_dataset",
    total_samples=1000,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Initialize generator
generator = DatasetGenerator(config)

# Generate complete dataset
generator.generate_dataset()
```

### Custom MIDI Configuration

```python
from src.dataset import MIDIGenerator, MIDIConfig
from src.dataset import ComplexityLevel, TimeSignature, KeySignature

# Create custom MIDI configuration
midi_config = MIDIConfig(
    tempo=120,
    time_signature=TimeSignature.FOUR_FOUR,
    key_signature=KeySignature.C_MAJOR,
    num_measures=8,
    complexity=ComplexityLevel.INTERMEDIATE,
    include_chords=True,
    include_melody=True
)

# Generate single MIDI file
generator = MIDIGenerator(midi_config)
generator.generate_midi("output.mid")
```

### Batch MIDI Generation

```python
from src.dataset import MIDIGenerator

generator = MIDIGenerator()

# Generate 100 MIDI files with random parameters
midi_files = generator.batch_generate(
    num_files=100,
    output_dir="midi_output",
    randomize_params=True
)

print(f"Generated {len(midi_files)} MIDI files")
```

### Audio Synthesis

```python
from src.dataset import AudioSynthesizer

# Initialize synthesizer
synthesizer = AudioSynthesizer(
    soundfont_path="/usr/share/sounds/sf2/FluidR3_GM.sf2",
    sample_rate=16000
)

# Synthesize single file
synthesizer.synthesize(
    midi_path="input.mid",
    output_path="output.wav"
)

# Get audio metadata
metadata = synthesizer.get_audio_metadata("output.wav")
print(f"Duration: {metadata['duration']}s")
print(f"Sample rate: {metadata['sample_rate']} Hz")
```

### Batch Audio Synthesis

```python
from pathlib import Path

synthesizer = AudioSynthesizer()

# Synthesize all MIDI files in directory
midi_dir = Path("midi_files")
audio_dir = Path("audio_files")
audio_dir.mkdir(exist_ok=True)

for midi_file in midi_dir.glob("*.mid"):
    output_path = audio_dir / f"{midi_file.stem}.wav"
    synthesizer.synthesize(str(midi_file), str(output_path))
    print(f"Synthesized: {midi_file.name}")
```

### Dataset with Custom Complexity Distribution

```python
from src.dataset import DatasetGenerator, DatasetConfig

config = DatasetConfig(
    name="advanced_dataset",
    total_samples=500,
    complexity_distribution={
        "beginner": 0.2,      # 20% beginner
        "intermediate": 0.3,  # 30% intermediate
        "advanced": 0.5       # 50% advanced
    }
)

generator = DatasetGenerator(config)
generator.generate_dataset()
```

### Loading and Using Dataset Manifest

```python
from src.dataset import DatasetGenerator

generator = DatasetGenerator()

# Load existing dataset manifest
manifest = generator.load_manifest("data/datasets/my_dataset_v1.0/train/manifest.json")

print(f"Dataset: {manifest['dataset_name']}")
print(f"Split: {manifest['split']}")
print(f"Total samples: {len(manifest['samples'])}")

# Access sample information
for sample in manifest['samples'][:5]:
    print(f"Audio: {sample['audio_path']}")
    print(f"MIDI: {sample['midi_path']}")
    print(f"Tempo: {sample['tempo']} BPM")
    print(f"Key: {sample['key_signature']}")
    print(f"Complexity: {sample['complexity']}")
    print("---")
```

### Using CLI

```bash
# Generate dataset
python -m src.dataset.cli generate \
    --name my_dataset \
    --total-samples 1000 \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --output-dir data/datasets

# Display dataset info
python -m src.dataset.cli info data/datasets/my_dataset_v1.0

# Validate dataset
python -m src.dataset.cli validate data/datasets/my_dataset_v1.0
```

### Advanced: Custom Generation Pipeline

```python
from src.dataset import MIDIGenerator, AudioSynthesizer, DatasetGenerator
from src.dataset import MIDIConfig, ComplexityLevel
import random

# Custom pipeline with specific parameters
midi_generator = MIDIGenerator()
audio_synthesizer = AudioSynthesizer(sample_rate=22050)

output_dir = Path("custom_dataset")
output_dir.mkdir(exist_ok=True)

for i in range(100):
    # Random parameters for diversity
    midi_config = MIDIConfig(
        tempo=random.randint(60, 180),
        complexity=random.choice(list(ComplexityLevel)),
        num_measures=random.randint(4, 16)
    )
    
    # Generate MIDI
    midi_path = output_dir / f"sample_{i:04d}.mid"
    midi_generator.config = midi_config
    midi_generator.generate_midi(str(midi_path))
    
    # Synthesize audio
    audio_path = output_dir / f"sample_{i:04d}.wav"
    audio_synthesizer.synthesize(str(midi_path), str(audio_path))
    
    print(f"Generated sample {i+1}/100")
```

## API Reference

### MIDIConfig

```python
@dataclass
class MIDIConfig:
    tempo: int = 120                              # BPM
    time_signature: TimeSignature = FOUR_FOUR     # Time signature
    key_signature: KeySignature = C_MAJOR         # Key signature
    num_measures: int = 8                         # Number of measures
    complexity: ComplexityLevel = INTERMEDIATE    # Complexity level
    include_chords: bool = True                   # Include chord accompaniment
    include_melody: bool = True                   # Include melody
    velocity_variation: float = 0.2               # Velocity randomization (0-1)
    chord_octave_low: int = 3                     # Chord lowest octave
    chord_octave_high: int = 4                    # Chord highest octave
    melody_octave_low: int = 4                    # Melody lowest octave
    melody_octave_high: int = 6                   # Melody highest octave
```

### MIDIGenerator

```python
MIDIGenerator(config: Optional[MIDIConfig] = None)
```

**Methods:**

- `generate_midi(output_path: str) -> None`
  - Generate single MIDI file

- `batch_generate(num_files: int, output_dir: str, randomize_params: bool = True) -> List[Path]`
  - Generate multiple MIDI files
  - Returns: List of generated file paths

### AudioSynthesizer

```python
AudioSynthesizer(
    soundfont_path: Optional[str] = None,
    sample_rate: int = 44100,
    gain: float = 0.5
)
```

**Methods:**

- `synthesize(midi_path: str, output_path: str) -> None`
  - Synthesize single audio file from MIDI

- `get_audio_metadata(audio_path: str) -> Dict[str, Any]`
  - Get audio file metadata (duration, sample rate, channels)

### DatasetConfig

```python
@dataclass
class DatasetConfig:
    name: str = "piano_dataset"                   # Dataset name
    version: str = "1.0.0"                        # Version string
    total_samples: int = 1000                     # Total number of samples
    train_ratio: float = 0.7                      # Training split ratio
    val_ratio: float = 0.15                       # Validation split ratio
    test_ratio: float = 0.15                      # Test split ratio
    sample_rate: int = 16000                      # Audio sample rate
    audio_format: str = "wav"                     # Audio format
    complexity_distribution: Dict[str, float]     # Complexity distribution
    output_dir: Path = "data/datasets"            # Output directory
    soundfont_path: Optional[Path] = None         # Soundfont path
```

### DatasetGenerator

```python
DatasetGenerator(config: Optional[DatasetConfig] = None)
```

**Methods:**

- `generate_dataset() -> Path`
  - Generate complete dataset with all splits
  - Returns: Path to dataset directory

- `load_manifest(manifest_path: str) -> Dict`
  - Load dataset manifest JSON
  - Returns: Manifest dictionary

## Dataset Structure

```
data/datasets/my_dataset_v1.0/
├── train/
│   ├── audio/
│   │   ├── sample_0000.wav
│   │   ├── sample_0001.wav
│   │   └── ...
│   ├── midi/
│   │   ├── sample_0000.mid
│   │   ├── sample_0001.mid
│   │   └── ...
│   └── manifest.json
├── val/
│   ├── audio/
│   ├── midi/
│   └── manifest.json
├── test/
│   ├── audio/
│   ├── midi/
│   └── manifest.json
└── metadata.json
```

### Manifest Format

```json
{
    "dataset_name": "my_dataset",
    "version": "1.0.0",
    "split": "train",
    "total_samples": 700,
    "samples": [
        {
            "id": "sample_0000",
            "audio_path": "train/audio/sample_0000.wav",
            "midi_path": "train/midi/sample_0000.mid",
            "tempo": 120,
            "time_signature": [4, 4],
            "key_signature": "C major",
            "complexity": "intermediate",
            "num_measures": 8,
            "duration": 16.0
        }
    ]
}
```

## Testing

```bash
# Run all tests
pytest tests/dataset/ -v

# Only unit tests
pytest tests/dataset/ -v -m unit

# With coverage
pytest tests/dataset/ --cov=src.dataset --cov-report=html
```

**Coverage:** 53 tests, 100% code coverage

## Performance

- MIDI generation: ~5 ms per file
- Audio synthesis (16kHz, 8 measures): ~200 ms per file
- Complete dataset (1000 samples): ~5-10 minutes

## Musical Features

### Complexity Levels

**Beginner:**
- Simple chord progressions (I-IV-V)
- Whole notes and half notes
- Single-note melodies
- Limited range (1 octave)

**Intermediate:**
- Extended progressions (ii-V-I, vi-IV-I-V)
- Quarter notes and eighth notes
- Simple harmonies
- Medium range (2 octaves)

**Advanced:**
- Complex progressions (circle of fifths, jazz)
- Sixteenth notes, triplets, syncopation
- Chord inversions
- Wide range (3+ octaves)

### Supported Keys

- **Major keys:** C, G, D, A, E, B, F♯, C♯, F, B♭, E♭, A♭, D♭, G♭
- **Minor keys:** A, E, B, F♯, C♯, G♯, D, G, C, F, B♭, E♭

### Supported Time Signatures

- **4/4**: Most common
- **3/4**: Waltz time
- **6/8**: Compound meter
- **2/4**: March time

## Dependencies

```
midiutil>=1.2.1
mido>=1.2.10
midi2audio>=0.1.1
numpy>=1.24.0
soundfile>=0.12.0
```

**System Requirements:**
- FluidSynth must be installed
- Ubuntu/Debian: `sudo apt-get install fluidsynth`
- macOS: `brew install fluid-synth`
- Windows: Download from fluidsynth.org
