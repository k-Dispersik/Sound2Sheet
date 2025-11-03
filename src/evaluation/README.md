# Evaluation System

## Overview

Comprehensive evaluation system for assessing music transcription model performance. Provides metrics calculation, batch evaluation, statistical analysis, report generation, and visualization tools.

## Architecture

```
┌─────────────────────┐
│  EvaluationConfig   │
│  - onset_tolerance  │
│  - offset_tolerance │
│  - pitch_tolerance  │
└──────────┬──────────┘
           │
           │ configures
           ▼
┌─────────────────────┐         ┌──────────────────────┐
│  MetricCalculator   │◄────────│  EvaluationMetrics   │
│  + calculate_metrics│         │  (dataclass)         │
│  + match_notes      │         │  - note_accuracy     │
│  + calculate_f1     │         │  - onset_f1          │
└──────────┬──────────┘         │  - pitch_f1          │
           │                    │  - timing_error      │
           │                    └──────────────────────┘
           │ uses
           ▼
┌─────────────────────────────┐         ┌──────────────────────┐
│  Evaluator                  │◄────────│  SampleEvaluation    │
│  + evaluate_sample()        │         │  - sample_id         │
│  + evaluate_batch()         │         │  - metrics           │
│  + get_aggregated_metrics() │         │  - predicted_notes   │
│  + get_error_analysis()     │         │  - ground_truth      │
│  + save_results()           │         └──────────────────────┘
└─────────────┬───────────────┘
              │
              │ used by
              ├──────────────────┬────────────────────┐
              ▼                  ▼                    ▼
    ┌──────────────────┐  ┌──────────────┐  ┌─────────────────┐
    │ ReportGenerator  │  │  Visualizer  │  │  CLI Interface  │
    │  + generate()    │  │  + dashboard │  │  - evaluate     │
    │  - CSV export    │  │  + confusion │  │  - report       │
    │  - JSON export   │  │  + plots     │  │  - visualize    │
    └──────────────────┘  └──────────────┘  └─────────────────┘
```

## Class Dependencies

1. **EvaluationConfig** → **MetricCalculator**: Configuration used by calculator
2. **EvaluationConfig** → **Evaluator**: Configuration used by evaluator
3. **MetricCalculator** → **EvaluationMetrics**: Calculator produces metrics
4. **Evaluator** → **MetricCalculator**: Evaluator uses calculator internally
5. **Evaluator** → **SampleEvaluation**: Evaluator creates sample results
6. **ReportGenerator** ← **Evaluator**: Generator reads evaluator results
7. **Visualizer** ← **Evaluator**: Visualizer reads evaluator results

## Core Components

### 1. Note (dataclass)
Simple note representation for evaluation:
- `pitch`: MIDI note number (0-127)
- `onset`: Start time in seconds
- `offset`: End time in seconds
- `velocity`: MIDI velocity (0-127)

### 2. EvaluationMetrics (dataclass)
Container for all evaluation metrics:
- **Note-level**: `note_accuracy` (percentage of correct notes)
- **Onset metrics**: `onset_precision`, `onset_recall`, `onset_f1`
- **Offset metrics**: `offset_precision`, `offset_recall`, `offset_f1`
- **Pitch metrics**: `pitch_precision`, `pitch_recall`, `pitch_f1`
- **Timing**: `timing_error_mean`, `timing_error_std`
- **Tempo**: `tempo_error` (relative error)
- **Error counts**: `true_positives`, `false_positives`, `false_negatives`
- **Distributions**: `pitch_errors` (per-pitch error counts)

### 3. MetricCalculator
Calculates comprehensive evaluation metrics:
- Note matching with configurable tolerances
- F1-score, precision, recall calculation
- Timing deviation analysis
- Pitch error distribution
- Tempo accuracy evaluation

### 4. Evaluator
Orchestrates evaluation process:
- Single sample evaluation
- Batch evaluation with progress tracking
- Aggregated statistics (mean, std, min, max)
- Error analysis and distributions
- Results persistence (JSON)

### 5. ReportGenerator
Generates evaluation reports:
- **CSV format**: Sample-level metrics table
- **JSON format**: Complete results with metadata

### 6. Visualizer
Creates visual analysis:
- Comprehensive dashboard (6 plots in one figure)
- Confusion matrix for pitch predictions
- Metrics over time plots
- Error distribution charts

### 7. CLI Interface
Command-line tool for evaluation:
- `evaluate`: Run evaluation on samples
- `report`: Generate reports from results
- `visualize`: Create visualization plots

## Usage Examples

### Basic Evaluation

```python
from src.evaluation import Evaluator, MetricCalculator, Note

# Create evaluator
evaluator = Evaluator()

# Define notes
predicted = [
    Note(pitch=60, onset=0.0, offset=1.0, velocity=80),
    Note(pitch=62, onset=1.0, offset=2.0, velocity=75),
]

ground_truth = [
    Note(pitch=60, onset=0.0, offset=1.0, velocity=80),
    Note(pitch=62, onset=1.02, offset=2.0, velocity=70),  # Slightly off onset
]

# Evaluate single sample
result = evaluator.evaluate_sample(
    sample_id="test_001",
    predicted_notes=predicted,
    ground_truth_notes=ground_truth
)

print(f"Accuracy: {result.metrics.note_accuracy:.3f}")
print(f"Onset F1: {result.metrics.onset_f1:.3f}")
print(f"Pitch F1: {result.metrics.pitch_f1:.3f}")
```

### Batch Evaluation

```python
from src.evaluation import Evaluator, EvaluationConfig

# Create evaluator with custom tolerances
config = EvaluationConfig(
    onset_tolerance=0.05,  # 50ms tolerance for onset
    offset_tolerance=0.05,
    pitch_tolerance=0      # Exact pitch matching
)
evaluator = Evaluator(config)

# Prepare samples
samples = [
    {
        'sample_id': f'sample_{i:03d}',
        'predicted_notes': load_predicted_notes(f'pred_{i}.json'),
        'ground_truth_notes': load_ground_truth_notes(f'gt_{i}.json'),
        'predicted_tempo': 120.0,
        'ground_truth_tempo': 120.0,
    }
    for i in range(100)
]

# Evaluate batch with progress tracking
def progress(current, total):
    print(f"Progress: {current}/{total}")

results = evaluator.evaluate_batch(samples, progress_callback=progress)

# Get aggregated metrics
agg = evaluator.get_aggregated_metrics()
print(f"Mean Accuracy: {agg.mean_metrics.note_accuracy:.3f} ± {agg.std_metrics.note_accuracy:.3f}")
print(f"Mean Onset F1: {agg.mean_metrics.onset_f1:.3f} ± {agg.std_metrics.onset_f1:.3f}")

# Get error analysis
error_analysis = evaluator.get_error_analysis()
print(f"Total TP: {error_analysis['total_true_positives']}")
print(f"Total FP: {error_analysis['total_false_positives']}")
print(f"Total FN: {error_analysis['total_false_negatives']}")

# Save results
evaluator.save_results('evaluation_results.json')
```

### Custom Metric Calculator

```python
from src.evaluation.metrics import MetricCalculator, Note

# Create calculator with custom tolerances
calculator = MetricCalculator(
    onset_tolerance=0.1,   # 100ms tolerance
    offset_tolerance=0.1,
    pitch_tolerance=1      # ±1 semitone tolerance
)

predicted = [Note(pitch=60, onset=0.05, offset=1.0)]
ground_truth = [Note(pitch=60, onset=0.0, offset=1.0)]

metrics = calculator.calculate_metrics(
    predicted_notes=predicted,
    ground_truth_notes=ground_truth,
    predicted_tempo=120.0,
    ground_truth_tempo=118.0
)

print(f"Note Accuracy: {metrics.note_accuracy:.3f}")
print(f"Timing Error: {metrics.timing_error_mean*1000:.1f}ms")
print(f"Tempo Error: {metrics.tempo_error:.3f}")
```

### Generate Reports

```python
from src.evaluation import Evaluator, ReportGenerator, ReportFormat

# Load or create evaluator with results
evaluator = Evaluator()
# ... perform evaluation ...

# Create report generator
reporter = ReportGenerator(evaluator)

# Generate CSV report
reporter.generate_report(
    output_path='evaluation_report.csv',
    format=ReportFormat.CSV
)

# Generate JSON report
reporter.generate_report(
    output_path='evaluation_report.json',
    format=ReportFormat.JSON
)
```

### Create Visualizations

```python
from src.evaluation import Evaluator, Visualizer

# Load evaluator with results
evaluator = Evaluator()
# ... perform evaluation ...

visualizer = Visualizer(evaluator)

# Create comprehensive dashboard (6 plots)
visualizer.create_dashboard('evaluation_dashboard.png')

# Create confusion matrix
visualizer.plot_confusion_matrix(
    output_path='confusion_matrix.png',
    pitch_range=(48, 84)  # C3 to C6
)

# Plot metrics over time
visualizer.plot_metrics_over_time(
    output_path='metrics_over_time.png',
    metric_names=['note_accuracy', 'onset_f1', 'pitch_f1']
)
```

### Using CLI

```bash
# Evaluate samples from manifest
python -m src.evaluation.cli evaluate \
    --manifest evaluation_manifest.json \
    --output results.json \
    --onset-tolerance 0.05 \
    --pitch-tolerance 0

# Generate CSV report
python -m src.evaluation.cli report \
    --results results.json \
    --output report.csv \
    --format csv

# Create visualizations
python -m src.evaluation.cli visualize \
    --results results.json \
    --dashboard dashboard.png \
    --confusion-matrix confusion.png \
    --metrics-over-time metrics.png
```

### Evaluation Manifest Format

```json
{
  "samples": [
    {
      "sample_id": "test_001",
      "predicted_path": "predictions/test_001.json",
      "ground_truth_path": "ground_truth/test_001.json",
      "predicted_tempo": 120.0,
      "ground_truth_tempo": 120.0,
      "metadata": {
        "duration": 3.0,
        "audio_path": "audio/test_001.wav"
      }
    }
  ]
}
```

### Notes JSON Format

```json
{
  "notes": [
    {
      "pitch": 60,
      "onset": 0.0,
      "offset": 1.0,
      "velocity": 80
    },
    {
      "pitch": 62,
      "onset": 1.0,
      "offset": 2.0,
      "velocity": 75
    }
  ]
}
```

### Evaluation Summary

```python
# Print human-readable summary
print(evaluator.get_summary())
```

Output:
```
============================================================
EVALUATION SUMMARY
============================================================
Number of samples: 100
Total predicted notes: 1523
Total ground truth notes: 1500

AGGREGATED METRICS (Mean ± Std):
------------------------------------------------------------
Note Accuracy:     0.876 ± 0.043
Onset F1:          0.912 ± 0.038
Offset F1:         0.854 ± 0.052
Pitch F1:          0.923 ± 0.031
Timing Error (ms): 23.5 ± 12.3

ERROR ANALYSIS:
------------------------------------------------------------
True Positives:    1368
False Positives:   155
False Negatives:   132
============================================================
```

## API Reference

### MetricCalculator

```python
MetricCalculator(
    onset_tolerance: float = 0.05,
    offset_tolerance: float = 0.05,
    pitch_tolerance: int = 0
)
```

**Methods:**
- `calculate_metrics(predicted_notes, ground_truth_notes, predicted_tempo=None, ground_truth_tempo=None) -> EvaluationMetrics`

### Evaluator

```python
Evaluator(config: Optional[EvaluationConfig] = None)
```

**Methods:**
- `evaluate_sample(sample_id, predicted_notes, ground_truth_notes, predicted_tempo=None, ground_truth_tempo=None, metadata=None) -> SampleEvaluation`
- `evaluate_batch(samples, progress_callback=None) -> List[SampleEvaluation]`
- `get_aggregated_metrics() -> AggregatedMetrics`
- `get_error_analysis() -> Dict[str, Any]`
- `save_results(output_path: str) -> None`
- `load_results(input_path: str) -> None`
- `clear_results() -> None`
- `get_summary() -> str`

### ReportGenerator

```python
ReportGenerator(evaluator: Evaluator)
```

**Methods:**
- `generate_report(output_path: str, format: ReportFormat = ReportFormat.JSON) -> None`

**Formats:**
- `ReportFormat.JSON`: Complete results with all metadata
- `ReportFormat.CSV`: Sample-level metrics table

### Visualizer

```python
Visualizer(evaluator: Evaluator)
```

**Methods:**
- `create_dashboard(output_path: str, figsize=(16, 12)) -> None`
- `plot_confusion_matrix(output_path: str, pitch_range=None, figsize=(12, 10)) -> None`
- `plot_metrics_over_time(output_path: str, metric_names=None, figsize=(12, 6)) -> None`

## Metrics Explained

### Note Accuracy
Percentage of notes that match exactly in pitch, onset, and offset (within tolerances).

### Onset F1-Score
Harmonic mean of precision and recall for note onsets. Measures how well the model detects when notes start.

### Offset F1-Score
Harmonic mean of precision and recall for note offsets. Measures how well the model detects when notes end.

### Pitch F1-Score
Harmonic mean of precision and recall for pitch detection. Measures how well the model identifies correct pitches.

### Timing Error
Mean absolute difference between predicted and ground truth onset times (in seconds).

### Tempo Error
Relative error in tempo detection: `|predicted - ground_truth| / ground_truth`

## Testing

```bash
# Run all evaluation tests
pytest tests/evaluation/ -v

# Run with coverage
pytest tests/evaluation/ --cov=src.evaluation --cov-report=html

# Run specific test file
pytest tests/evaluation/test_metrics.py -v
```

**Test Coverage:** 53 tests, 100% code coverage

## Performance

- Single sample evaluation: ~1-5ms
- Batch evaluation (100 samples): ~0.5-2s
- Dashboard generation: ~2-3s
- Confusion matrix: ~1-2s

## Configuration Options

### EvaluationConfig

```python
EvaluationConfig(
    onset_tolerance: float = 0.05,      # 50ms tolerance
    offset_tolerance: float = 0.05,     # 50ms tolerance
    pitch_tolerance: int = 0,           # Exact match
    min_note_duration: float = 0.01,    # 10ms minimum
    max_note_duration: float = 10.0,    # 10s maximum
    tempo_tolerance: float = 5.0        # ±5 BPM
)
```

## Error Handling

```python
# Handle evaluation errors gracefully
try:
    result = evaluator.evaluate_sample(
        sample_id="test_001",
        predicted_notes=predicted,
        ground_truth_notes=ground_truth
    )
except Exception as e:
    print(f"Evaluation failed: {e}")
```

## Best Practices

1. **Choose appropriate tolerances**: 50ms (0.05s) is typical for onset/offset
2. **Use batch evaluation**: More efficient than evaluating one by one
3. **Save results regularly**: Use `save_results()` to persist evaluation data
4. **Monitor timing errors**: High timing errors indicate synchronization issues
5. **Check pitch error distribution**: Identifies which pitches are problematic
6. **Use visualizations**: Dashboards reveal patterns not obvious in numbers

## Integration with Model

```python
from src.model import Sound2SheetModel, InferenceConfig
from src.evaluation import Evaluator, Note

# Load model
model = Sound2SheetModel.load_from_checkpoint('model.ckpt')

# Run inference
predictions = model.generate(mel_spec, inference_config)

# Convert to Note objects
predicted_notes = [
    Note(pitch=p, onset=o, offset=f, velocity=v)
    for p, o, f, v in predictions
]

# Evaluate
evaluator = Evaluator()
result = evaluator.evaluate_sample(
    sample_id="audio_001",
    predicted_notes=predicted_notes,
    ground_truth_notes=ground_truth_notes
)
```

## Dependencies

```
numpy>=1.24.0
matplotlib>=3.7.0
```

## Future Enhancements (Optional)

- ❌ HTML report generation (not implemented - not needed)
- ❌ PDF report generation (not implemented - not needed)
- ✅ Real-time evaluation dashboard
- ✅ Per-instrument metrics (future if needed)
- ✅ Multi-pitch detection metrics (future if needed)

## Related Modules

- **src.converter**: Provides note structures and conversion tools
- **src.model**: Model inference for predictions
- **src.core**: Audio processing for input preparation
