#!/usr/bin/env python3
"""
Inference script for Sound2Sheet model - transcribe audio to MIDI.

Usage:
    python -m src.model.inference --checkpoint checkpoints/best_model.pt --input audio.wav --output output.mid
"""

import argparse
import logging
from pathlib import Path
from typing import List
import torch
import mido
from mido import MidiFile, MidiTrack, Message

from .config import ModelConfig, InferenceConfig
from .sound2sheet_model import Sound2SheetModel
from src.core.audio_processor import AudioProcessor


# Token to note mapping (reverse of dataset encoding)
NOTE_START = 21  # MIDI note A0
NOTE_END = 108   # MIDI note C8

# Special tokens
PAD_TOKEN = 0
SOS_TOKEN = 89
EOS_TOKEN = 90
UNK_TOKEN = 91


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio to MIDI using Sound2Sheet model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input audio file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output MIDI file'
    )
    
    # Inference arguments
    parser.add_argument(
        '--max-length',
        type=int,
        default=1000,
        help='Maximum number of notes to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (higher = more random)'
    )
    parser.add_argument(
        '--use-beam-search',
        action='store_true',
        help='Use beam search instead of greedy decoding'
    )
    parser.add_argument(
        '--beam-size',
        type=int,
        default=5,
        help='Beam size for beam search'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=0,
        help='Top-k sampling (0 = disabled)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.0,
        help='Nucleus sampling threshold (0 = disabled)'
    )
    
    # MIDI generation arguments
    parser.add_argument(
        '--default-velocity',
        type=int,
        default=80,
        help='Default MIDI velocity (0-127)'
    )
    parser.add_argument(
        '--default-duration',
        type=float,
        default=0.5,
        help='Default note duration in seconds'
    )
    parser.add_argument(
        '--tempo',
        type=int,
        default=120,
        help='MIDI tempo in BPM'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )
    
    return parser.parse_args()


def tokens_to_midi_notes(tokens: List[int], 
                         default_velocity: int = 80,
                         default_duration: float = 0.5) -> List[dict]:
    """
    Convert predicted tokens to MIDI note dictionaries.
    
    Args:
        tokens: List of predicted token IDs
        default_velocity: Default MIDI velocity
        default_duration: Default note duration in seconds
        
    Returns:
        List of note dictionaries with keys: pitch, onset, offset, velocity
    """
    notes = []
    current_time = 0.0
    
    for token in tokens:
        # Skip special tokens
        if token in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
            if token == EOS_TOKEN:
                break
            continue
        
        # Convert token to MIDI pitch (tokens 1-88 -> MIDI notes 21-108)
        pitch = token - 1 + NOTE_START
        
        if NOTE_START <= pitch <= NOTE_END:
            notes.append({
                'pitch': pitch,
                'onset': current_time,
                'offset': current_time + default_duration,
                'velocity': default_velocity
            })
            current_time += default_duration
    
    return notes


def notes_to_midi_file(notes: List[dict], 
                       output_path: Path,
                       tempo: int = 120):
    """
    Convert note dictionaries to MIDI file.
    
    Args:
        notes: List of note dictionaries
        output_path: Output MIDI file path
        tempo: Tempo in BPM
    """
    # Create MIDI file
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
    
    # Sort notes by onset time
    notes = sorted(notes, key=lambda n: n['onset'])
    
    # Convert to MIDI messages
    events = []
    
    for note in notes:
        # Note on event
        events.append({
            'time': note['onset'],
            'type': 'note_on',
            'note': note['pitch'],
            'velocity': note['velocity']
        })
        
        # Note off event
        events.append({
            'time': note['offset'],
            'type': 'note_off',
            'note': note['pitch'],
            'velocity': 0
        })
    
    # Sort events by time
    events = sorted(events, key=lambda e: e['time'])
    
    # Convert absolute times to delta times
    current_time = 0.0
    for event in events:
        delta_time = event['time'] - current_time
        delta_ticks = int(mido.second2tick(delta_time, mid.ticks_per_beat, mido.bpm2tempo(tempo)))
        
        if event['type'] == 'note_on':
            track.append(Message('note_on', 
                               note=event['note'],
                               velocity=event['velocity'],
                               time=delta_ticks))
        else:
            track.append(Message('note_off',
                               note=event['note'],
                               velocity=0,
                               time=delta_ticks))
        
        current_time = event['time']
    
    # Save MIDI file
    mid.save(output_path)


def main():
    """Main inference function."""
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Sound2Sheet Inference")
    logger.info("=" * 80)
    
    # Check input file
    if not args.input.exists():
        raise FileNotFoundError(f"Input audio file not found: {args.input}")
    
    # Check checkpoint
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    logger.info(f"Using device: {args.device}")
    
    # Load model
    logger.info(f"\nLoading model from: {args.checkpoint}")
    model = Sound2SheetModel.from_pretrained(str(args.checkpoint))
    model = model.to(args.device)
    model.eval()
    
    logger.info("Model loaded successfully")
    logger.info(f"Total parameters: {model.count_parameters()['total']:,}")
    
    # Create inference config
    inference_config = InferenceConfig(
        max_length=args.max_length,
        temperature=args.temperature,
        use_beam_search=args.use_beam_search,
        beam_size=args.beam_size,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # Load and process audio
    logger.info(f"\nLoading audio from: {args.input}")
    audio_processor = AudioProcessor()
    audio, sr = audio_processor.load_audio(str(args.input))
    
    logger.info(f"Audio duration: {len(audio) / sr:.2f}s")
    logger.info(f"Sample rate: {sr} Hz")
    
    # Generate mel-spectrogram
    logger.info("\nGenerating mel-spectrogram...")
    mel = audio_processor.audio_to_mel(audio)
    
    # Convert to tensor
    mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(args.device)
    
    logger.info(f"Mel-spectrogram shape: {mel_tensor.shape}")
    
    # Generate transcription
    logger.info("\nTranscribing audio...")
    with torch.no_grad():
        predicted_tokens = model.generate(
            mel_tensor,
            inference_config
        )
    
    # Convert to list
    if isinstance(predicted_tokens, torch.Tensor):
        predicted_tokens = predicted_tokens.cpu().tolist()
    
    logger.info(f"Generated {len(predicted_tokens)} tokens")
    
    # Convert tokens to MIDI notes
    logger.info("\nConverting to MIDI notes...")
    notes = tokens_to_midi_notes(
        predicted_tokens,
        default_velocity=args.default_velocity,
        default_duration=args.default_duration
    )
    
    logger.info(f"Detected {len(notes)} notes")
    
    if len(notes) == 0:
        logger.warning("No notes detected! The model may need more training.")
        return
    
    # Create MIDI file
    logger.info(f"\nSaving MIDI to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    notes_to_midi_file(notes, args.output, tempo=args.tempo)
    
    logger.info("\n" + "=" * 80)
    logger.info("Transcription completed successfully!")
    logger.info("=" * 80)
    logger.info(f"\nInput:  {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Notes:  {len(notes)}")
    logger.info(f"Duration: ~{notes[-1]['offset']:.2f}s")


if __name__ == '__main__':
    main()
