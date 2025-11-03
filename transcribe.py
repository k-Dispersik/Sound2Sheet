#!/usr/bin/env python3
"""
Sound2Sheet Transcription Script

Transcribes audio file to musical notation using trained model.

Usage:
    # Print to terminal
    python transcribe.py --model path/to/model.pt --audio piano.wav
    
    # Save to file
    python transcribe.py --model path/to/model.pt --audio piano.wav --output result.musicxml
    
    # With custom config
    python transcribe.py --model path/to/model.pt --audio piano.wav --output result.mid --beam-size 5
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import Sound2SheetModel
from src.core import AudioProcessor, AudioConfig
from src.converter import NoteSequence, MusicXMLConverter, MIDIConverter


def load_model(model_path: str, device: str = 'cpu') -> Sound2SheetModel:
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model config
    if 'model_config' in checkpoint:
        from src.model.config import ModelConfig
        model_config = ModelConfig(**checkpoint['model_config'])
    else:
        raise ValueError("Model config not found in checkpoint")
    
    # Create and load model
    model_config.device = device
    model = Sound2SheetModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Device: {device}")
    print(f"  Vocab size: {model_config.vocab_size}")
    
    return model


def process_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load and process audio file to mel-spectrogram."""
    print(f"Processing audio: {audio_path}")
    
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Create audio processor
    audio_config = AudioConfig()
    audio_config.sample_rate = sample_rate
    audio_config.n_mels = 128
    audio_config.n_fft = 1024
    audio_config.hop_length = 320
    
    processor = AudioProcessor(config=audio_config)
    
    # Load and process
    audio = processor.load_audio(audio_path)
    mel_spec = processor.to_mel_spectrogram(audio)
    
    # Convert to torch tensor if numpy array
    if isinstance(mel_spec, np.ndarray):
        mel_spec = torch.from_numpy(mel_spec).float()
    
    # Get duration
    duration = len(audio) / sample_rate
    
    print(f"✓ Audio processed")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Mel-spectrogram shape: {mel_spec.shape}")
    
    return mel_spec


def transcribe(
    model: Sound2SheetModel,
    mel_spec: torch.Tensor,
    beam_size: int = 3,
    max_length: int = 512
) -> list:
    """Transcribe mel-spectrogram to note sequence."""
    print(f"Transcribing...")
    print(f"  Max length: {max_length}")
    
    # Add batch dimension if needed
    if mel_spec.dim() == 2:
        mel_spec = mel_spec.unsqueeze(0)
    
    # Move to model device
    device = next(model.parameters()).device
    mel_spec = mel_spec.to(device)
    
    # Create inference config
    from src.model.config import InferenceConfig
    inference_config = InferenceConfig(
        max_length=max_length,
        temperature=1.0
    )
    
    # Generate predictions using model's generate method
    with torch.no_grad():
        predictions = model.generate(mel_spec, inference_config)
    
    print(f"✓ Transcription complete")
    print(f"  Generated {len(predictions)} tokens")
    
    return predictions


def predictions_to_notes(predictions: list, vocab) -> NoteSequence:
    """Convert model predictions to note sequence."""
    from src.converter import Note
    
    notes = []
    current_time = 0.0
    
    # Parse predictions (simplified - actual implementation depends on vocabulary)
    for token in predictions:
        # Skip special tokens
        if token < 4:  # <pad>, <sos>, <eos>, <unk>
            continue
        
        # TODO: Implement actual token-to-note conversion based on vocabulary
        # This is a placeholder that needs to be implemented based on your model's output format
        
        # Example: if tokens encode pitch, duration, etc.
        # note = Note(
        #     pitch=pitch_from_token(token),
        #     start_time=current_time,
        #     duration=duration_from_token(token),
        #     velocity=80
        # )
        # notes.append(note)
        # current_time += note.duration
    
    # For now, create a placeholder note sequence
    note_seq = NoteSequence(notes=notes)
    
    return note_seq


def print_results(note_seq: NoteSequence):
    """Print transcription results to terminal."""
    print("\n" + "=" * 80)
    print("TRANSCRIPTION RESULTS")
    print("=" * 80)
    
    if len(note_seq.notes) == 0:
        print("No notes detected")
        print("\nNote: Full inference implementation is pending.")
        print("The model generates tokens that need to be converted to notes.")
        return
    
    print(f"Total notes: {len(note_seq.notes)}")
    print(f"Duration: {note_seq.total_duration:.2f}s")
    
    print("\n" + "-" * 80)
    print(f"{'#':<5} {'Pitch':<8} {'Start':<10} {'Duration':<10} {'Velocity':<10}")
    print("-" * 80)
    
    for i, note in enumerate(note_seq.notes[:50], 1):  # Show first 50 notes
        pitch_name = f"{note.pitch_class}{note.octave}"
        print(f"{i:<5} {pitch_name:<8} {note.start_time:<10.3f} {note.duration:<10.3f} {note.velocity:<10}")
    
    if len(note_seq.notes) > 50:
        print(f"... and {len(note_seq.notes) - 50} more notes")
    
    print("=" * 80)


def save_results(note_seq: NoteSequence, output_path: str):
    """Save transcription results to file."""
    output_path = Path(output_path)
    ext = output_path.suffix.lower()
    
    print(f"\nSaving results to: {output_path}")
    
    if ext == '.mid' or ext == '.midi':
        # Save as MIDI
        converter = MIDIConverter(note_seq)
        converter.export(str(output_path))
        print(f"✓ Saved MIDI file")
        
    elif ext == '.xml' or ext == '.musicxml' or ext == '.mxl':
        # Save as MusicXML
        converter = MusicXMLConverter(note_seq)
        converter.export(str(output_path))
        print(f"✓ Saved MusicXML file")
        
    else:
        raise ValueError(f"Unsupported output format: {ext}. Use .mid or .musicxml")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio to musical notation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print to terminal
  python transcribe.py --model results/2025-11-03/my_experiment/model/sound2sheet_model.pt --audio piano.wav
  
  # Save as MIDI
  python transcribe.py --model results/2025-11-03/my_experiment/model/sound2sheet_model.pt --audio piano.wav --output result.mid
  
  # Save as MusicXML with beam search
  python transcribe.py --model model.pt --audio piano.wav --output result.musicxml --beam-size 5
        """
    )
    
    # Required arguments
    parser.add_argument('--model', '-m', required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--audio', '-a', required=True, help='Path to audio file to transcribe')
    
    # Optional arguments
    parser.add_argument('--output', '-o', default=None, help='Output file path (.mid or .musicxml). If not specified, prints to terminal')
    parser.add_argument('--device', '-d', default='cpu', choices=['cpu', 'cuda'], help='Device to use for inference (default: cpu)')
    parser.add_argument('--beam-size', '-b', type=int, default=3, help='Beam size for beam search (default: 3, use 1 for greedy)')
    parser.add_argument('--max-length', '-l', type=int, default=512, help='Maximum sequence length (default: 512)')
    parser.add_argument('--sample-rate', '-s', type=int, default=16000, help='Audio sample rate (default: 16000)')
    
    args = parser.parse_args()
    
    try:
        # Print header
        print("=" * 80)
        print("SOUND2SHEET TRANSCRIPTION")
        print("=" * 80)
        print()
        
        # 1. Load model
        model = load_model(args.model, device=args.device)
        print()
        
        # 2. Process audio
        mel_spec = process_audio(args.audio, sample_rate=args.sample_rate)
        print()
        
        # 3. Transcribe
        predictions = transcribe(
            model,
            mel_spec,
            beam_size=args.beam_size,
            max_length=args.max_length
        )
        print()
        
        # 4. Convert predictions to notes
        print("Converting predictions to notes...")
        # Note: This requires proper implementation based on model's vocabulary
        note_seq = NoteSequence(notes=[])  # Placeholder
        print(f"⚠️  Note conversion pending - full inference pipeline needs implementation")
        print(f"   Generated {len(predictions)} tokens from model")
        print()
        
        # 5. Output results
        if args.output:
            # Save to file
            if len(note_seq.notes) > 0:
                save_results(note_seq, args.output)
            else:
                print(f"⚠️  Cannot save empty note sequence")
                print(f"   Predictions: {predictions[:20]}...")  # Show first 20 tokens
        else:
            # Print to terminal
            print_results(note_seq)
            print(f"\nRaw predictions (first 50 tokens): {predictions[:50]}")
        
        print("\n" + "=" * 80)
        print("Done!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
