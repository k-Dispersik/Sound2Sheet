# Converter (Note Builder)

## Overview

Module for converting model predictions into structured musical notation with timing analysis, quantization, and multi-format export (JSON, MIDI, MusicXML).

## Class Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        DATA STRUCTURES                          │
└────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Dynamic         │  Expression dynamics
│  - PPP...FFF     │  (8 velocity levels)
└──────────────────┘

┌──────────────────┐
│  Articulation    │  Note articulation
│  - STACCATO      │  (6 types)
│  - LEGATO        │
│  - ACCENT        │
└──────────────────┘

┌──────────────────────────────────────────────────────┐
│  Note                                                │
│  - pitch: int (MIDI number)                          │
│  - start_time: float (seconds)                       │
│  - duration: float (seconds)                         │
│  - velocity: int (0-127)                             │
│  - is_tied_start: bool                               │
│  - is_tied_end: bool                                 │
│  - tie_group_id: Optional[int]                       │
│  - dynamic: Optional[Dynamic]                        │
│  - articulation: Optional[Articulation]              │
│  + to_dict()                                         │
│  + infer_dynamic_from_velocity()                     │
└──────────────────────────────────────────────────────┘
         │
         │ contains
         ▼
┌──────────────────────────────────────────────────────┐
│  Measure                                             │
│  - number: int                                       │
│  - start_time: float                                 │
│  - duration: float                                   │
│  - time_signature: Tuple[int, int]                   │
│  - notes: List[Note]                                 │
│  + to_dict()                                         │
└──────────────────────────────────────────────────────┘
         │
         │ contains
         ▼
┌──────────────────────────────────────────────────────┐
│  NoteSequence                                        │
│  - notes: List[Note]                                 │
│  - measures: List[Measure]                           │
│  - tempo: int (BPM)                                  │
│  - time_signature: Tuple[int, int]                   │
│  - key_signature: str                                │
│  + organize_into_measures()                          │
│  + apply_tied_notes()                                │
│  + infer_expression_marks()                          │
│  + to_dict()                                         │
└──────────────────────────────────────────────────────┘


┌────────────────────────────────────────────────────────────────┐
│                      PROCESSING PIPELINE                        │
└────────────────────────────────────────────────────────────────┘

Model Predictions
[MIDI numbers]
    │
    ▼
┌──────────────────┐
│  NoteBuilder     │
│  + build_from_   │
│    predictions() │
└────────┬─────────┘
         │
         │ uses
         ▼
┌──────────────────┐         ┌──────────────────┐
│ QuantizationConf │────────>│  Quantizer       │
│ - resolution     │         │  + quantize()    │
│ - auto_tempo     │         │  + detect_tempo()│
└──────────────────┘         └────────┬─────────┘
                                      │
                                      │ creates
                                      ▼
                             ┌────────────────┐
                             │  Quantized     │
                             │  NoteSequence  │
                             └────────┬───────┘
                                      │
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌────────────────┐         ┌──────────────────┐       ┌──────────────────┐
│ JSONConverter  │         │  MIDIConverter   │       │ MusicXMLConverter│
│ + export()     │         │  + export()      │       │ + export()       │
└────────┬───────┘         └────────┬─────────┘       └────────┬─────────┘
         │                          │                           │
         ▼                          ▼                           ▼
┌────────────────┐         ┌──────────────────┐       ┌──────────────────┐
│  JSON File     │         │  MIDI File       │       │  MusicXML File   │
│  (metadata +   │         │  (playback)      │       │  (score)         │
│   notes)       │         └──────────────────┘       └──────────────────┘
└────────────────┘


┌────────────────────────────────────────────────────────────────┐
│                     DETECTION ALGORITHMS                        │
└────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  Tempo Detection                                             │
│  1. Calculate inter-onset intervals (IOIs)                   │
│  2. Find common divisors (beat candidates)                   │
│  3. Select most frequent tempo (60-180 BPM range)            │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  Key Signature Detection (Krumhansl-Schmuckler)              │
│  1. Build pitch class distribution                           │
│  2. Correlate with major/minor profiles                      │
│  3. Select key with highest correlation                      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  Time Signature Detection                                    │
│  1. Quantize notes to beat grid                              │
│  2. Find repeating patterns                                  │
│  3. Detect measure boundaries                                │
│  4. Infer time signature (4/4, 3/4, 6/8, 2/4)               │
└──────────────────────────────────────────────────────────────┘
```

## Class Dependencies

1. **Note**: Basic note representation with expression
2. **Measure** → **Note**: Contains multiple notes
3. **NoteSequence** → **Note** + **Measure**: Contains notes and measures
4. **QuantizationConfig** → **Quantizer**: Configuration for quantization
5. **Quantizer** → **NoteSequence**: Quantizes timing
6. **NoteBuilder** → **Quantizer** + **NoteSequence**: Orchestrates conversion
7. **Converters** → **NoteSequence**: Export to different formats

## Core Components

### 1. Note
Represents a single musical note with:
- **Basic attributes**: pitch, start_time, duration, velocity
- **Tied notes**: is_tied_start, is_tied_end, tie_group_id
- **Expression**: dynamic (ppp-fff), articulation (staccato, legato, etc.)
- **Methods**: to_dict(), infer_dynamic_from_velocity()

### 2. Measure
Represents a musical bar/measure:
- **number**: 1-indexed measure number
- **start_time**: Measure start in seconds
- **duration**: Measure duration in seconds
- **time_signature**: (numerator, denominator)
- **notes**: List of notes in this measure

### 3. NoteSequence
Collection of notes with metadata:
- **notes**: List of all notes
- **measures**: List of measures (optional)
- **tempo**: BPM
- **time_signature**: (numerator, denominator)
- **key_signature**: Key string (e.g., "C major")
- **Advanced methods**:
  - organize_into_measures(): Create measure structure
  - apply_tied_notes(): Split notes across measure boundaries
  - infer_expression_marks(): Automatic dynamics/articulation

### 4. Quantizer
Timing quantization and tempo detection:
- **quantize()**: Snap note timings to grid
- **detect_tempo()**: Automatic tempo detection from IOIs
- **detect_key_signature()**: Krumhansl-Schmuckler algorithm
- **detect_time_signature()**: Pattern-based detection

### 5. NoteBuilder
Orchestrates conversion pipeline:
- Parse predictions to notes
- Detect tempo (optional)
- Quantize timing
- Validate musical correctness
- Build final sequence

### 6. Converters
Export to multiple formats:
- **JSONConverter**: Full metadata + notes
- **MIDIConverter**: Playback-ready MIDI
- **MusicXMLConverter**: Score notation with ties and expression

## Usage Examples

### Basic: Predictions to JSON

```python
from src.converter import NoteBuilder

# Initialize builder
builder = NoteBuilder()

# Convert predictions (MIDI numbers) to structured sequence
predictions = [60, 64, 67, 72, 67, 64, 60]  # C-E-G-C-G-E-C
sequence = builder.build_from_predictions(
    predictions,
    tempo=120,
    time_signature=(4, 4),
    key_signature="C major"
)

# Export to JSON
json_data = sequence.to_dict()
print(json_data)
```

### With Automatic Tempo Detection

```python
from src.converter import NoteBuilder, QuantizationConfig

# Enable auto tempo detection
config = QuantizationConfig(auto_tempo_detection=True)
builder = NoteBuilder(quantization_config=config)

# Builder will detect tempo from note timings
sequence = builder.build_from_predictions(predictions)
print(f"Detected tempo: {sequence.tempo} BPM")
```

### Quantization with Different Resolutions

```python
from src.converter import QuantizationConfig

# 16th note resolution (more detailed)
config = QuantizationConfig(
    resolution=16,
    snap_threshold=0.1
)
builder = NoteBuilder(quantization_config=config)

# Or 8th note resolution (simpler)
config = QuantizationConfig(resolution=8)
builder = NoteBuilder(quantization_config=config)
```

### Export to MIDI

```python
from src.converter import MIDIConverter

# Build sequence
sequence = builder.build_from_predictions(predictions, tempo=120)

# Export to MIDI
converter = MIDIConverter()
converter.export(sequence, "output.mid")
print("MIDI file saved!")
```

### Export to MusicXML (Score)

```python
from src.converter import MusicXMLConverter

# Build sequence with expression
sequence = builder.build_from_predictions(predictions, tempo=120)

# Infer expression marks (dynamics, articulation)
sequence.infer_expression_marks()

# Export to MusicXML
converter = MusicXMLConverter()
converter.export(sequence, "output.musicxml")
print("MusicXML score saved!")
```

### Advanced: Measures and Tied Notes

```python
from src.converter import NoteBuilder

builder = NoteBuilder()

# Build sequence
sequence = builder.build_from_predictions(
    predictions,
    tempo=120,
    time_signature=(4, 4)
)

# Organize into measures
sequence.organize_into_measures()
print(f"Total measures: {len(sequence.measures)}")

# Apply tied notes (split notes across measure boundaries)
sequence.apply_tied_notes()

# Export with ties
from src.converter import MusicXMLConverter
converter = MusicXMLConverter()
converter.export(sequence, "output_with_ties.musicxml")
```

### Expression Marks (Dynamics & Articulation)

```python
from src.converter import Note, Dynamic, Articulation

# Manual expression
note = Note(
    pitch=60,
    start_time=0.0,
    duration=0.5,
    velocity=100,
    dynamic=Dynamic.F,           # Forte
    articulation=Articulation.STACCATO
)

# Or infer from velocity
note = Note(pitch=60, start_time=0.0, duration=0.5, velocity=100)
note.infer_dynamic_from_velocity()
print(f"Inferred dynamic: {note.dynamic}")  # F (forte)

# Automatic inference for all notes
sequence.infer_expression_marks()
```

### Complete Pipeline with All Features

```python
from src.converter import (
    NoteBuilder,
    QuantizationConfig,
    MIDIConverter,
    MusicXMLConverter,
    JSONConverter
)

# Configuration
config = QuantizationConfig(
    resolution=16,
    auto_tempo_detection=True,
    snap_threshold=0.1
)

# Build sequence
builder = NoteBuilder(quantization_config=config)
sequence = builder.build_from_predictions(predictions)

# Add advanced features
sequence.organize_into_measures()      # Create measures
sequence.apply_tied_notes()            # Handle ties
sequence.infer_expression_marks()      # Add dynamics/articulation

# Export to all formats
json_converter = JSONConverter()
midi_converter = MIDIConverter()
musicxml_converter = MusicXMLConverter()

json_converter.export(sequence, "output.json")
midi_converter.export(sequence, "output.mid")
musicxml_converter.export(sequence, "output.musicxml")

print("Exported to JSON, MIDI, and MusicXML!")
```

### Using CLI

```bash
# Predictions to MIDI
python -m src.converter.cli predict-to-midi \
    predictions.txt \
    output.mid \
    --tempo 120 \
    --time-signature 4 4

# Predictions to JSON
python -m src.converter.cli predict-to-json \
    predictions.txt \
    output.json \
    --tempo 120 \
    --key-signature "C major"

# With quantization
python -m src.converter.cli predict-to-midi \
    predictions.txt \
    output.mid \
    --tempo 120 \
    --resolution 16 \
    --auto-tempo

# Full pipeline
python -m src.converter.cli predict-to-midi \
    predictions.txt \
    output.mid \
    --tempo 120 \
    --measures \
    --tied-notes \
    --expression
```

### Key Signature Detection

```python
from src.converter import Quantizer

quantizer = Quantizer()

# Detect key from notes
notes = [
    Note(pitch=60, start_time=0.0, duration=0.5, velocity=80),  # C
    Note(pitch=64, start_time=0.5, duration=0.5, velocity=80),  # E
    Note(pitch=67, start_time=1.0, duration=0.5, velocity=80),  # G
]

key = quantizer.detect_key_signature(notes)
print(f"Detected key: {key}")  # "C major"
```

### Time Signature Detection

```python
from src.converter import Quantizer

quantizer = Quantizer()

# Detect time signature from note patterns
time_sig = quantizer.detect_time_signature(notes, tempo=120)
print(f"Detected time signature: {time_sig[0]}/{time_sig[1]}")  # 4/4
```

### Validation

```python
from src.converter import NoteSequence

sequence = NoteSequence(notes=notes, tempo=120)

# Check for overlapping notes (chord detection)
has_chords = any(
    n1.start_time == n2.start_time
    for i, n1 in enumerate(notes)
    for n2 in notes[i+1:]
)

# Check for rests (gaps between notes)
for i in range(len(notes) - 1):
    gap = notes[i+1].start_time - (notes[i].start_time + notes[i].duration)
    if gap > 0.1:  # More than 100ms gap
        print(f"Rest detected at {notes[i].start_time}s, duration {gap}s")
```

## API Reference

### Note

```python
@dataclass
class Note:
    pitch: int                              # MIDI number (0-127)
    start_time: float                       # Seconds
    duration: float                         # Seconds
    velocity: int                           # MIDI velocity (0-127)
    is_tied_start: bool = False             # Tie to next note
    is_tied_end: bool = False               # Tie from previous note
    tie_group_id: Optional[int] = None      # Tie group identifier
    dynamic: Optional[Dynamic] = None       # ppp/pp/p/mp/mf/f/ff/fff
    articulation: Optional[Articulation] = None  # staccato/legato/etc
```

**Methods:**
- `to_dict() -> Dict`: Serialize to dictionary
- `infer_dynamic_from_velocity() -> Dynamic`: Infer dynamic from velocity

### Measure

```python
@dataclass
class Measure:
    number: int                             # 1-indexed measure number
    start_time: float                       # Seconds
    duration: float                         # Seconds
    time_signature: Tuple[int, int]         # (numerator, denominator)
    notes: List[Note] = field(default_factory=list)
```

**Methods:**
- `to_dict() -> Dict`: Serialize to dictionary

### NoteSequence

```python
@dataclass
class NoteSequence:
    notes: List[Note] = field(default_factory=list)
    measures: List[Measure] = field(default_factory=list)
    tempo: int = 120                        # BPM
    time_signature: Tuple[int, int] = (4, 4)
    key_signature: str = "C major"
```

**Methods:**
- `organize_into_measures() -> None`: Create measure structure
- `apply_tied_notes() -> None`: Split notes across measure boundaries
- `infer_expression_marks() -> None`: Infer dynamics and articulation
- `to_dict() -> Dict`: Serialize to dictionary

### QuantizationConfig

```python
@dataclass
class QuantizationConfig:
    resolution: int = 16                    # Notes per measure
    snap_threshold: float = 0.1             # Snap tolerance (seconds)
    auto_tempo_detection: bool = False      # Auto-detect tempo
    min_tempo: int = 60                     # Min BPM
    max_tempo: int = 180                    # Max BPM
```

### Quantizer

```python
Quantizer(config: QuantizationConfig)
```

**Methods:**
- `quantize(notes: List[Note], tempo: int) -> List[Note]`
  - Quantize note timings to grid

- `detect_tempo(notes: List[Note]) -> int`
  - Detect tempo from inter-onset intervals
  - Returns: BPM (60-180 range)

- `detect_key_signature(notes: List[Note]) -> str`
  - Detect key using Krumhansl-Schmuckler algorithm
  - Returns: Key string (e.g., "C major")

- `detect_time_signature(notes: List[Note], tempo: int) -> Tuple[int, int]`
  - Detect time signature from patterns
  - Returns: (numerator, denominator)

### NoteBuilder

```python
NoteBuilder(quantization_config: Optional[QuantizationConfig] = None)
```

**Methods:**
- `build_from_predictions(predictions, tempo, time_signature, key_signature, ...) -> NoteSequence`
  - Build structured sequence from model predictions

### Converters

```python
JSONConverter().export(sequence: NoteSequence, output_path: str)
MIDIConverter().export(sequence: NoteSequence, output_path: str)
MusicXMLConverter().export(sequence: NoteSequence, output_path: str)
```

## JSON Output Format

```json
{
    "tempo": 120,
    "time_signature": [4, 4],
    "key_signature": "C major",
    "measures": [
        {
            "number": 1,
            "start_time": 0.0,
            "duration": 2.0,
            "time_signature": [4, 4],
            "notes": [...]
        }
    ],
    "notes": [
        {
            "pitch": 60,
            "start_time": 0.0,
            "duration": 0.5,
            "velocity": 80,
            "is_tied_start": false,
            "is_tied_end": false,
            "tie_group_id": null,
            "dynamic": "mf",
            "articulation": null
        }
    ]
}
```

## Testing

```bash
# Run all tests
pytest tests/converter/ -v

# Only unit tests
pytest tests/converter/ -v -m unit

# With coverage
pytest tests/converter/ --cov=src.converter --cov-report=html
```

**Coverage:** 329 tests, 100% code coverage

## Performance

- Parse predictions (100 notes): ~5 ms
- Quantize notes: ~10 ms
- Tempo detection: ~15 ms
- Key detection: ~20 ms
- Export to JSON: ~2 ms
- Export to MIDI: ~10 ms
- Export to MusicXML: ~50 ms

## Musical Features

### Dynamics (8 levels)
- **PPP** (pianississimo): velocity 0-15
- **PP** (pianissimo): velocity 16-31
- **P** (piano): velocity 32-47
- **MP** (mezzo-piano): velocity 48-63
- **MF** (mezzo-forte): velocity 64-79
- **F** (forte): velocity 80-95
- **FF** (fortissimo): velocity 96-111
- **FFF** (fortississimo): velocity 112-127

### Articulation (6 types)
- **STACCATO**: Short, detached (duration < 0.3 beats)
- **STACCATISSIMO**: Very short, very detached
- **TENUTO**: Full value (duration > 0.9 beats)
- **ACCENT**: Emphasized (velocity > 100)
- **MARCATO**: Strongly accented
- **LEGATO**: Smooth, connected

### Tied Notes
- Automatically split notes across measure boundaries
- Maintain note continuity with tie markers
- Proper MusicXML rendering with start/stop/continue ties

## Dependencies

```
numpy>=1.24.0
mido>=1.2.10
music21>=9.1.0  # Optional, for MusicXML export
```

**Note:** MusicXML export requires `music21`. Install with:
```bash
pip install music21
```
