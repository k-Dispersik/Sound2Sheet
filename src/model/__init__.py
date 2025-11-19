"""
Model package for Sound2Sheet.

Contains the model architecture, training, and inference code.
"""

from .config import ModelConfig, TrainingConfig, InferenceConfig, DataConfig
from .sound2sheet_model import Sound2SheetModel
from .ast_model import ASTWrapper, PianoRollClassifier
from .dataset import PianoDataset, create_dataloaders
from .trainer import Trainer

# Scripts are importable but not in __all__ (they have main() functions)
from . import train
from . import inference

__all__ = [
    'ModelConfig',
    'TrainingConfig', 
    'InferenceConfig',
    'DataConfig',
    'Sound2SheetModel',
    'ASTWrapper',
    'PianoRollClassifier',
    'PianoDataset',
    'create_dataloaders',
    'Trainer',
]