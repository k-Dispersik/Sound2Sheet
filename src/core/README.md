# Core: Audio Processing

Audio loading, preprocessing, and mel-spectrogram generation for piano transcription.

## Components

### AudioProcessor
Main audio processing class.

**Key Methods:**
- `load_audio(path)` - Load audio file (.wav, .mp3, .m4a)
- `to_mel_spectrogram(audio)` - Convert to mel-spectrogram [128 bins, time]
- `augment_audio(audio, type, **params)` - Apply augmentation
- `normalize_audio(audio)` - Normalize to [-1, 1]

### NoiseStrategy
Factory pattern for noise augmentation.

**Noise Types:**
- `WhiteNoise` - Uniform across frequencies
- `PinkNoise` - 1/f natural noise
- `BrownNoise` - 1/f² deep noise
- `AmbientNoise` - Room ambient simulation
- `ElectricalHum` - 50/60 Hz hum

### AudioVisualizer
Visualization tools for audio analysis.

**Plots:**
- Waveform visualization
- Mel-spectrogram heatmaps
- Augmentation comparison

## Usage

### Basic Processing

```python
from src.core import AudioProcessor

processor = AudioProcessor(sample_rate=16000, n_mels=128)

# Load and process
audio = processor.load_audio("piano.wav")
mel = processor.to_mel_spectrogram(audio)  # Shape: [128, time]
```

### Augmentation

```python
# Pitch shift ±2 semitones
shifted = processor.augment_audio(audio, "pitch_shift", n_steps=2)

# Time stretch (1.2x speed)
stretched = processor.augment_audio(audio, "time_stretch", rate=1.2)

# Add pink noise (20dB SNR)
noisy = processor.augment_audio(audio, "noise", snr_db=20, noise_type="pink")

# Volume adjustment
louder = processor.augment_audio(audio, "volume", volume_factor=1.5)
```

### Visualization

```python
from src.core import AudioVisualizer

viz = AudioVisualizer()
viz.plot_waveform(audio, sample_rate=16000)
viz.plot_mel_spectrogram(mel)
viz.plot_augmentation_comparison({
    "Original": audio,
    "Noisy": noisy,
    "Shifted": shifted
}, sample_rate=16000)
```

## Configuration

Default settings optimized for piano transcription:

```python
sample_rate = 16000    # 16 kHz audio
n_mels = 128          # 128 mel bins (required for AST)
hop_length = 512      # ~32ms per frame
n_fft = 2048          # FFT window size
```

## Testing

```bash
pytest tests/core/ -v --cov=src.core
```

**Coverage:** 88 tests, 95% coverage
