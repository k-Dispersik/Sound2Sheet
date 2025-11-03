# Audio Processing

## Overview

Module for loading, preprocessing, and converting audio files into mel-spectrograms suitable for machine learning model input.

## Class Architecture

```
┌─────────────────┐
│  AudioConfig    │
│  - sample_rate  │
│  - n_fft        │
│  - hop_length   │
│  - n_mels       │
└────────┬────────┘
         │
         │ uses
         ▼
┌─────────────────────────┐         ┌──────────────────────────┐
│  AudioProcessor         │◄────────│  NoiseStrategyFactory    │
│  + load_audio()         │         │  + create()              │
│  + generate_mel_spec()  │         └──────────┬───────────────┘
│  + augment_audio()      │                    │
│  + normalize_audio()    │                    │ creates
└─────────────────────────┘                    ▼
                                    ┌──────────────────────────┐
         ▲                          │  NoiseStrategy           │
         │                          │  - WhiteNoiseStrategy    │
         │ uses                     │  - PinkNoiseStrategy     │
         │                          │  - BrownNoiseStrategy    │
┌─────────────────────────┐         │  - AmbientNoiseStrategy  │
│  AudioVisualizer        │         │  - ElectricalHumStrategy │
│  + plot_waveform()      │         └──────────────────────────┘
│  + plot_spectrogram()   │
│  + plot_augmentation()  │
└─────────────────────────┘
```

## Class Dependencies

1. **AudioConfig** → **AudioProcessor**: Configuration is used by processor
2. **AudioProcessor** → **NoiseStrategyFactory**: Processor creates noise strategies via factory
3. **NoiseStrategyFactory** → **NoiseStrategy**: Factory creates concrete noise strategies
4. **AudioVisualizer** → **AudioProcessor**: Visualizer uses processor for analysis

## Core Components

### 1. AudioConfig
Configuration for audio processing parameters:
- `sample_rate`: Sampling frequency (16000 Hz)
- `n_fft`: FFT window size (1024)
- `hop_length`: Hop length between windows (320)
- `n_mels`: Number of mel filter banks (128)

### 2. AudioProcessor
Main class for audio processing:
- Load audio files (.wav, .mp3, .m4a)
- Generate mel-spectrograms
- Audio augmentation (pitch shift, time stretch, noise, volume)
- Audio normalization

### 3. NoiseStrategyFactory + NoiseStrategy
Factory for creating different noise types:
- **WhiteNoise**: Uniform noise across all frequencies
- **PinkNoise**: 1/f noise (more natural)
- **BrownNoise**: 1/f² noise (deeper)
- **AmbientNoise**: Room ambient noise simulation
- **ElectricalHum**: Electrical hum simulation (50/60 Hz)

### 4. AudioVisualizer
Visualization for audio and augmentations:
- Waveform plots
- Mel-spectrogram plots
- Augmentation comparison plots

## Usage Examples

### Basic Audio Processing

```python
from src.core import AudioProcessor, AudioConfig

# Initialize with default parameters
processor = AudioProcessor()

# Load audio
audio, sr = processor.load_audio("piano.wav")

# Generate mel-spectrogram
mel_spec = processor.generate_mel_spectrogram(audio)
print(f"Mel-spectrogram shape: {mel_spec.shape}")  # (128, time_steps)
```

### Custom Configuration

```python
# Create custom configuration
config = AudioConfig()
config.sample_rate = 22050  # Higher quality
config.n_mels = 256         # More mel bins

# Processor with custom configuration
processor = AudioProcessor(config)
```

### Audio Augmentation

```python
# Pitch shift by ±2 semitones
pitch_shifted = processor.augment_audio(
    audio,
    augmentation_type="pitch_shift",
    n_steps=2
)

# Time stretching (speed up/slow down)
time_stretched = processor.augment_audio(
    audio,
    augmentation_type="time_stretch",
    rate=1.2  # 20% faster
)

# Add noise (SNR=20 dB)
noisy_audio = processor.augment_audio(
    audio,
    augmentation_type="noise",
    snr_db=20,
    noise_type="pink"
)

# Volume adjustment
louder_audio = processor.augment_audio(
    audio,
    augmentation_type="volume",
    volume_factor=1.5
)
```

### Processing Pipeline with Visualization

```python
from src.core import AudioProcessor, AudioVisualizer

processor = AudioProcessor()
visualizer = AudioVisualizer()

# Load audio
audio, sr = processor.load_audio("piano.mp3")

# Visualize original
visualizer.plot_waveform(audio, sr, title="Original Audio")
visualizer.plot_mel_spectrogram(
    processor.generate_mel_spectrogram(audio),
    title="Mel-Spectrogram"
)

# Compare augmentations
augmentations = {
    "Original": audio,
    "Pitch +2": processor.augment_audio(audio, "pitch_shift", n_steps=2),
    "Time x1.2": processor.augment_audio(audio, "time_stretch", rate=1.2),
    "Noise 20dB": processor.augment_audio(audio, "noise", snr_db=20)
}

visualizer.plot_augmentation_comparison(augmentations, sr)
```

### Batch Processing

```python
from pathlib import Path
import numpy as np

processor = AudioProcessor()
audio_dir = Path("data/audio")

for audio_file in audio_dir.glob("*.wav"):
    try:
        # Load audio
        audio, sr = processor.load_audio(str(audio_file))
        
        # Normalize
        audio = processor.normalize_audio(audio)
        
        # Generate mel-spectrogram
        mel = processor.generate_mel_spectrogram(audio)
        
        # Save result
        output_path = audio_file.stem + "_mel.npy"
        np.save(output_path, mel)
        
        print(f"Processed: {audio_file.name}")
    except Exception as e:
        print(f"Error processing {audio_file.name}: {e}")
```

### Using Different Noise Types

```python
from src.core.noise_strategies import NoiseStrategyFactory

processor = AudioProcessor()
audio, sr = processor.load_audio("piano.wav")

# White noise (sharp, uniform)
white_noise_audio = processor.augment_audio(
    audio, "noise", snr_db=15, noise_type="white"
)

# Pink noise (more natural, softer)
pink_noise_audio = processor.augment_audio(
    audio, "noise", snr_db=15, noise_type="pink"
)

# Brown noise (deep, low frequencies)
brown_noise_audio = processor.augment_audio(
    audio, "noise", snr_db=15, noise_type="brown"
)

# Ambient noise (room simulation)
ambient_noise_audio = processor.augment_audio(
    audio, "noise", snr_db=15, noise_type="ambient"
)

# Electrical hum (50 Hz hum)
hum_noise_audio = processor.augment_audio(
    audio, "noise", snr_db=15, noise_type="hum"
)
```

## API Reference

### AudioConfig

```python
AudioConfig(config_path: Optional[str] = None)
```

**Parameters:**
- `config_path`: Path to YAML configuration file (optional)

**Attributes:**
- `sample_rate`: int = 16000 - Sampling rate
- `n_fft`: int = 1024 - FFT window size
- `hop_length`: int = 320 - Hop length between windows
- `n_mels`: int = 128 - Number of mel bins
- `f_min`: float = 0.0 - Minimum frequency
- `f_max`: float = 8000.0 - Maximum frequency
- `normalize`: bool = True - Whether to normalize audio
- `pre_emphasis`: float = 0.97 - Pre-emphasis coefficient

### AudioProcessor

```python
AudioProcessor(config: Optional[AudioConfig] = None)
```

**Methods:**

- `load_audio(file_path: str) -> Tuple[np.ndarray, int]`
  - Load audio from file
  - Returns: (audio_data, sample_rate)

- `generate_mel_spectrogram(audio: np.ndarray) -> np.ndarray`
  - Generate mel-spectrogram
  - Returns: mel_spec shape (n_mels, time_steps)

- `augment_audio(audio, augmentation_type, **kwargs) -> np.ndarray`
  - Augment audio
  - Types: "pitch_shift", "time_stretch", "noise", "volume"

- `normalize_audio(audio: np.ndarray) -> np.ndarray`
  - Normalize audio to [-1, 1]

### AudioVisualizer

```python
AudioVisualizer()
```

**Methods:**

- `plot_waveform(audio, sr, title="")`
  - Plot waveform

- `plot_mel_spectrogram(mel_spec, title="")`
  - Plot mel-spectrogram

- `plot_augmentation_comparison(augmentations: Dict, sr)`
  - Compare different augmentations

## Testing

```bash
# Run all tests
pytest tests/core/ -v

# Only unit tests
pytest tests/core/ -v -m unit

# With code coverage
pytest tests/core/ --cov=src.core --cov-report=html
```

**Coverage:** 48 tests, 100% code coverage

## Performance Benchmarks

- Load audio (3 min): ~50 ms
- Generate mel-spectrogram: ~20 ms
- Pitch shift augmentation: ~100 ms
- Time stretch augmentation: ~80 ms
- Add noise: ~10 ms

## Supported Formats

- ✅ WAV (via soundfile)
- ✅ MP3 (via audioread/ffmpeg)
- ✅ M4A (via audioread/ffmpeg)
- ✅ FLAC (via soundfile)
- ✅ OGG (via soundfile)

## Dependencies

```
librosa>=0.10.0
numpy>=1.24.0
soundfile>=0.12.0
audioread>=3.0.0
scipy>=1.10.0
matplotlib>=3.7.0
pyyaml>=6.0
```
