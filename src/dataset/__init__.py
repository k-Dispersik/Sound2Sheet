"""
Dataset generation module for Sound2Sheet.

This module provides functionality for generating synthetic datasets
including MIDI generation, audio synthesis, and dataset management.
"""

from .midi_generator import MIDIGenerator

__all__ = [
    'MIDIGenerator',
]
