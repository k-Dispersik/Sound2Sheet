# Evaluation: Metrics & Visualization

Comprehensive evaluation system for piano transcription model assessment.

## Components

### MetricCalculator
Computes transcription metrics with configurable tolerances.

**Metrics:**
- Note-level: accuracy, precision, recall, F1
- Onset metrics: precision, recall, F1
- Offset metrics: precision, recall, F1
- Pitch metrics: precision, recall, F1
- Timing errors: mean, std deviation

### Evaluator
Orchestrates batch evaluation and aggregation.

**Features:**
- Single sample evaluation
- Batch processing with progress tracking
- Aggregated statistics (mean/std/min/max)
- Error analysis and distributions
- Results persistence (JSON)

### Visualizer
Visual analysis tools.

**Plots:**
- Comprehensive dashboard (6 plots)
- Confusion matrix for pitch predictions
- Metrics over time
- Error distributions

## Usage

### Basic Evaluation

```python
from src.evaluation import Evaluator, Note

evaluator = Evaluator()

predicted = [
    Note(pitch=60, onset=0.0, offset=1.0, velocity=80),
    Note(pitch=62, onset=1.0, offset=2.0, velocity=75),
]

ground_truth = [
    Note(pitch=60, onset=0.0, offset=1.0, velocity=80),
    Note(pitch=62, onset=1.02, offset=2.0, velocity=70),
]

result = evaluator.evaluate_sample("test_001", predicted, ground_truth)

print(f"Accuracy: {result.metrics.note_accuracy:.3f}")
print(f"Onset F1: {result.metrics.onset_f1:.3f}")
```

### Batch Evaluation

```python
from src.evaluation import Evaluator, EvaluationConfig

config = EvaluationConfig(
    onset_tolerance=0.05,  # 50ms
    offset_tolerance=0.05
)
evaluator = Evaluator(config)

samples = [
    {
        'sample_id': f'sample_{i}',
        'predicted_notes': pred_notes[i],
        'ground_truth_notes': gt_notes[i]
    }
    for i in range(100)
]

results = evaluator.evaluate_batch(samples)
agg = evaluator.get_aggregated_metrics()

print(f"Mean F1: {agg.mean_metrics.onset_f1:.3f}")
```

### Visualization

```python
from src.evaluation import Visualizer

viz = Visualizer()
viz.plot_dashboard(evaluator.results, save_path="dashboard.png")
viz.plot_confusion_matrix(evaluator.results, save_path="confusion.png")
```
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
## Configuration

**EvaluationConfig:**
```python
onset_tolerance: float = 0.05    # 50ms onset tolerance
offset_tolerance: float = 0.05   # 50ms offset tolerance
pitch_tolerance: int = 0         # Exact pitch matching
```

## Metrics

**Calculated:**
- `note_accuracy` - Percentage of correctly matched notes
- `onset_f1` - F1 score for onset detection
- `offset_f1` - F1 score for offset detection
- `pitch_f1` - F1 score for pitch detection
- `timing_error_mean` - Average timing deviation (ms)
- `timing_error_std` - Timing deviation std (ms)
- `tempo_error` - Relative tempo error

**Error Counts:**
- `true_positives` - Correctly predicted notes
- `false_positives` - Incorrectly predicted notes
- `false_negatives` - Missed notes

## Testing

```bash
pytest tests/evaluation/ -v --cov=src.evaluation
```

**Coverage:** 53 tests, 100% coverage

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
