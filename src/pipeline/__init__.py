"""
Pipeline package for Sound2Sheet.

Provides complete end-to-end training pipeline:
- Dataset generation (MIDI + audio synthesis)
- Model training with AST-based architecture
- Evaluation with comprehensive metrics
- Report and visualization generation
"""

from .run_pipeline import main, run_pipeline, run_evaluation_only
from .config_parser import (
    PipelineConfig,
    DatasetConfig,
    AudioConfig,
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    EvaluationConfig,
    OutputConfig,
    parse_args_to_config,
    load_yaml_config
)

__all__ = [
    'main',
    'run_pipeline',
    'run_evaluation_only',
    'PipelineConfig',
    'DatasetConfig',
    'AudioConfig',
    'ModelConfig',
    'TrainingConfig',
    'InferenceConfig',
    'EvaluationConfig',
    'OutputConfig',
    'parse_args_to_config',
    'load_yaml_config'
]
