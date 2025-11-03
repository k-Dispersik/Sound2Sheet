"""Batched generate->train->cleanup pipeline for prototyping.

This script creates small synthetic datasets (WAV + MIDI), trains a tiny model on
each batch and removes the files to keep disk use low.

Usage (from repo root):
  python -m src.pipeline.run_pipeline --batches 5 --batch-size 100 --epochs 1

Notes:
 - Uses simple sine-wave synthesis for audio (no external synth required).
 - Manifests use absolute paths so the dataset loader picks files correctly.
 - Defaults are conservative to run on CPU and limited RAM.
"""

import argparse
import logging
import tempfile
import shutil
from pathlib import Path
import json
import math
import numpy as np
import soundfile as sf
import mido

import torch

from src.model.config import ModelConfig, TrainingConfig, DataConfig
from src.model.dataset import create_dataloaders
from src.model.sound2sheet_model import Sound2SheetModel
from src.model.trainer import Trainer


logger = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


NOTE_FREQ = {i: 440.0 * 2 ** ((i - 69) / 12) for i in range(21, 109)}  # MIDI -> freq


def synth_sine_notes(pitches, durations, sr=16000):
    """Synthesize a sequence of sine notes concatenated into one waveform."""
    parts = []
    for p, d in zip(pitches, durations):
        f = NOTE_FREQ.get(p, 440.0)
        t = np.linspace(0, d, int(sr * d), endpoint=False)
        w = 0.2 * np.sin(2 * math.pi * f * t)
        parts.append(w)
    return np.concatenate(parts).astype(np.float32)


def write_midi(pitches, durations, path: Path):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # simple fixed tempo and ticks_per_beat
    mid.ticks_per_beat = 480
    tempo = mido.bpm2tempo(120)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    ticks_per_second = mid.ticks_per_beat * (120.0 / 60.0) / 60.0  # approximate

    current_time_ticks = 0
    for pitch, dur in zip(pitches, durations):
        # note_on
        track.append(mido.Message('note_on', note=int(pitch), velocity=64, time=0))
        # note_off after dur seconds
        delta_ticks = int(dur * mid.ticks_per_beat * 2)  # rough conversion
        track.append(mido.Message('note_off', note=int(pitch), velocity=0, time=delta_ticks))

    mid.save(path)


def generate_sample(sample_idx: int, out_dir: Path, sample_rate=16000, duration=1.0):
    """Generate one synthetic WAV + MIDI sample and return relative file paths.

    The audio is a short sequence of 1-3 notes.
    Returns relative paths (just filenames) for manifest.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_filename = f"sample_{sample_idx:04d}.wav"
    midi_filename = f"sample_{sample_idx:04d}.mid"
    
    wav_path = out_dir / wav_filename
    midi_path = out_dir / midi_filename

    # random small melody: 1-3 notes within piano range
    num_notes = np.random.randint(1, 4)
    pitches = np.random.randint(48, 72, size=num_notes).tolist()  # C3..C5
    # split duration evenly
    per = max(0.1, duration / num_notes)
    durations = [per] * num_notes

    audio = synth_sine_notes(pitches, durations, sr=sample_rate)
    sf.write(str(wav_path), audio, samplerate=sample_rate)

    write_midi(pitches, durations, midi_path)

    # Return relative paths (just filenames) - dataset_dir will be prepended by DataConfig
    return wav_filename, midi_filename


def create_manifests(base_dir: Path, batch_subdir: str, entries):
    # entries: list of {'audio_path': str (relative), 'midi_path': str (relative), 'duration': float}
    # Manifests are created in base_dir (tmp_batches/), entries have paths like "batch_001/sample_0000.wav"
    train_manifest = base_dir / 'train_manifest.json'
    val_manifest = base_dir / 'val_manifest.json'
    test_manifest = base_dir / 'test_manifest.json'

    # simple split: 90/5/5
    n = len(entries)
    n_train = int(n * 0.9)
    n_val = int(n * 0.05)

    with open(train_manifest, 'w') as f:
        json.dump(entries[:n_train], f)
    with open(val_manifest, 'w') as f:
        json.dump(entries[n_train:n_train + n_val], f)
    with open(test_manifest, 'w') as f:
        json.dump(entries[n_train + n_val:], f)

    return train_manifest, val_manifest, test_manifest


def run_batch(batch_idx: int, args):
    logger.info(f"Starting batch {batch_idx + 1}/{args.batches}")
    batch_name = f"batch_{batch_idx + 1:03d}"
    base_dir = Path(args.tmp_dir)
    batch_dir = base_dir / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for i in range(args.batch_size):
        wav, midi = generate_sample(i, batch_dir, duration=args.sample_duration)
        # Store paths relative to base_dir (with batch_xxx/ prefix)
        entries.append({'audio_path': f"{batch_name}/{wav}", 'midi_path': f"{batch_name}/{midi}", 'duration': args.sample_duration})

    create_manifests(base_dir, batch_name, entries)

    # create dataloaders and train - dataset_dir should be parent of batch_xxx
    data_config = DataConfig(dataset_dir=str(Path(args.tmp_dir).absolute()))
    model_config = ModelConfig(device='cpu', hidden_size=args.hidden_size, num_decoder_layers=1, num_attention_heads=2)
    training_config = TrainingConfig(batch_size=args.train_batch_size, num_epochs=args.epochs, use_mixed_precision=False, num_workers=0)

    train_loader, val_loader, test_loader = create_dataloaders(data_config, model_config, training_config)

    model = Sound2SheetModel(model_config, freeze_encoder=True)
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, model_config=model_config, training_config=training_config)

    history = trainer.train()

    logger.info(f"Batch {batch_idx + 1} finished. Train losses: {history.get('train_losses')}")

    # cleanup
    try:
        shutil.rmtree(batch_dir)
        logger.info(f"Removed temporary batch dir {batch_dir}")
    except Exception as e:
        logger.warning(f"Failed to remove {batch_dir}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batched generate->train->cleanup pipeline")
    parser.add_argument('--batches', type=int, default=1, help='Number of batches to process')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of samples to generate per batch')
    parser.add_argument('--sample-duration', type=float, default=1.0, help='Duration (s) per sample')
    parser.add_argument('--tmp-dir', type=str, default='tmp_batches', help='Temporary directory for batches')
    parser.add_argument('--epochs', type=int, default=1, help='Epochs to train per batch')
    parser.add_argument('--train-batch-size', type=int, default=4, help='Training dataloader batch size')
    parser.add_argument('--hidden-size', type=int, default=128, help='Model hidden size for prototyping')
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Pipeline starting")
    for b in range(args.batches):
        run_batch(b, args)
    logger.info("Pipeline completed")


if __name__ == '__main__':
    main()
