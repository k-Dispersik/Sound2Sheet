# Dataset: Synthetic Piano Data Generation

Automated MIDI generation and audio synthesis for training piano transcription models.

## Components

### MIDIGenerator
Generates realistic piano MIDI sequences.

**Features:**
- Chord progressions (I-IV-V, ii-V-I, etc.)
- Melody generation (scale-based)
- 3 complexity levels (beginner/intermediate/advanced)
- 24 key signatures, 4 time signatures

### AudioSynthesizer
Converts MIDI to realistic piano audio via FluidSynth.

**Features:**
- High-quality synthesis (FluidR3_GM soundfont)
- Configurable sample rate
- Batch processing support

### DatasetGenerator
Creates complete datasets with train/val/test splits.

**Features:**
- Hierarchical structure
- JSON manifests with metadata
- Complexity distribution control

## Usage

### Generate Dataset via run_pipeline.py

```python
# config.json
{
    "dataset": {
        "total_samples": 1000,
        "complexity_distribution": {
            "beginner": 0.5,
            "intermediate": 0.4,
            "advanced": 0.1
        }
    }
}
```

```bash
python run_pipeline.py  # Generates synthetic data automatically
```

### Manual MIDI Generation

```python
from src.dataset import MIDIGenerator, MIDIConfig, ComplexityLevel

config = MIDIConfig(
    tempo=120,
    complexity=ComplexityLevel.INTERMEDIATE,
    num_measures=8
)

generator = MIDIGenerator(config)
generator.generate_midi("output.mid")
```

### Audio Synthesis

```python
from src.dataset import AudioSynthesizer

synthesizer = AudioSynthesizer(sample_rate=16000)
synthesizer.synthesize("input.mid", "output.wav")
```

## Complexity Levels

**Beginner:**
- Simple I-IV-V progressions
- Whole/half notes
- Single-note melodies
- 1 octave range

**Intermediate:**
- Extended progressions (ii-V-I)
- Quarter/eighth notes
- Simple harmonies
- 2 octave range

**Advanced:**
- Complex progressions (circle of fifths)
- Sixteenth notes, triplets
- Chord inversions
- 3+ octave range

## Testing

```bash
pytest tests/dataset/ -v --cov=src.dataset
```

**Coverage:** 40 tests, 92% coverage

## Requirements

**System:**
- FluidSynth installed
- Ubuntu: `sudo apt-get install fluidsynth`
- macOS: `brew install fluidsynth`

**Python:**
```
midiutil>=1.2.1
mido>=1.2.10
midi2audio>=0.1.1
```
