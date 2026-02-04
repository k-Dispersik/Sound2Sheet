"""
Dataset generation module for creating synthetic training datasets.

This module provides functionality for generating complete datasets
with MIDI files, synthesized audio, and metadata for training.
"""

from typing import Optional, List, Dict, Tuple
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .midi_generator import MIDIGenerator, MIDIConfig, ComplexityLevel
from .audio_synthesizer import AudioSynthesizer


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    name: str = "piano_dataset"
    version: str = "1.0.0"
    total_samples: int = 1000
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Audio parameters
    sample_rate: int = 16000
    audio_format: str = "wav"
    
    # MIDI parameters
    complexity_distribution: Dict[str, float] = None
    
    # Paths
    output_dir: Path = Path("data/datasets")
    soundfont_path: Optional[Path] = None
    num_workers: int = 0
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.complexity_distribution is None:
            # Default: balanced distribution
            self.complexity_distribution = {
                "beginner": 0.33,
                "intermediate": 0.34,
                "advanced": 0.33
            }
        
        # Validate split ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not 0.99 <= total_ratio <= 1.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # Ensure output_dir is Path
        self.output_dir = Path(self.output_dir)


@dataclass
class DatasetSample:
    """Metadata for a single dataset sample."""
    id: str
    midi_path: str
    audio_path: str
    complexity: str
    tempo: int
    time_signature: str
    key_signature: str
    num_measures: int
    duration: float
    split: str  # 'train', 'val', or 'test'


class DatasetGenerator:
    """
    Generator for creating complete synthetic piano datasets.
    
    Orchestrates MIDI generation, audio synthesis, and metadata management
    to create structured datasets for model training.
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        """
        Initialize DatasetGenerator.
        
        Args:
            config: Dataset configuration. If None, uses defaults.
        """
        self.config = config or DatasetConfig()
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize generators
        self.midi_generator = MIDIGenerator()
        
        try:
            self.audio_synthesizer = AudioSynthesizer(
                soundfont_path=self.config.soundfont_path,
                sample_rate=self.config.sample_rate
            )
            self.synthesis_available = True
        except (ImportError, FileNotFoundError) as e:
            self.logger.warning(f"Audio synthesis not available: {e}")
            self.synthesis_available = False
        
        self.samples: List[DatasetSample] = []
    
    def generate(self, generate_audio: bool = True) -> Path:
        """
        Generate complete dataset.
        
        Args:
            generate_audio: Whether to synthesize audio (requires FluidSynth)
            
        Returns:
            Path to generated dataset directory
        """
        # Create dataset directory structure
        dataset_dir = self._create_directory_structure()
        
        # Calculate split sizes
        train_size, val_size, test_size = self._calculate_splits()
        
        # Generate samples for each split
        self._generate_split("train", train_size, dataset_dir, generate_audio)
        self._generate_split("val", val_size, dataset_dir, generate_audio)
        self._generate_split("test", test_size, dataset_dir, generate_audio)
        
        # Generate metadata
        self._generate_metadata(dataset_dir)
        
        return dataset_dir
    
    def _create_directory_structure(self) -> Path:
        """Create hierarchical directory structure for dataset."""
        # Create base directory directly (no timestamp subdirectory)
        dataset_dir = self.config.output_dir
        
        # Create subdirectories
        for split in ['train', 'val', 'test']:
            (dataset_dir / split / 'midi').mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / 'audio').mkdir(parents=True, exist_ok=True)
        
        (dataset_dir / 'metadata').mkdir(parents=True, exist_ok=True)
        
        self.logger.debug(f"Created directory structure: {dataset_dir}")
        return dataset_dir
    
    def _calculate_splits(self) -> Tuple[int, int, int]:
        """Calculate number of samples for each split."""
        total = self.config.total_samples
        
        train_size = int(total * self.config.train_ratio)
        val_size = int(total * self.config.val_ratio)
        test_size = total - train_size - val_size  # Remainder goes to test
        
        return train_size, val_size, test_size
    
    def _generate_split(
        self,
        split: str,
        size: int,
        dataset_dir: Path,
        generate_audio: bool
    ) -> None:
        """Generate samples for a specific split."""
        midi_dir = dataset_dir / split / 'midi'
        audio_dir = dataset_dir / split / 'audio'
        
        total_audio_size = 0
        total_midi_size = 0
        success_count = 0
        num_workers = max(0, int(getattr(self.config, "num_workers", 0)))

        if num_workers > 0:
            futures = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for i in range(size):
                    futures.append(
                        executor.submit(
                            self._generate_single_sample,
                            split,
                            i,
                            dataset_dir,
                            midi_dir,
                            audio_dir,
                            generate_audio
                        )
                    )
                pbar = tqdm(as_completed(futures), total=size, desc=f"Generating {split}", unit="sample")
                for future in pbar:
                    result = future.result()
                    if not result:
                        continue
                    sample, midi_size, audio_size = result
                    self.samples.append(sample)
                    total_midi_size += midi_size
                    total_audio_size += audio_size
                    success_count += 1
                    pbar.set_postfix({
                        'complexity': sample.complexity[:3],
                        'tempo': sample.tempo
                    })
                pbar.close()
        else:
            pbar = tqdm(range(size), desc=f"Generating {split}", unit="sample")
            for i in pbar:
                result = self._generate_single_sample(
                    split,
                    i,
                    dataset_dir,
                    midi_dir,
                    audio_dir,
                    generate_audio
                )
                if not result:
                    continue
                sample, midi_size, audio_size = result
                self.samples.append(sample)
                total_midi_size += midi_size
                total_audio_size += audio_size
                success_count += 1
                pbar.set_postfix({
                    'complexity': sample.complexity[:3],
                    'tempo': sample.tempo
                })
            pbar.close()

        self.logger.info(
            f"{split} split: generated {success_count}/{size} samples "
            f"(midi {total_midi_size / (1024 * 1024):.2f} MB, "
            f"audio {total_audio_size / (1024 * 1024):.2f} MB)"
        )

    def _generate_single_sample(
        self,
        split: str,
        index: int,
        dataset_dir: Path,
        midi_dir: Path,
        audio_dir: Path,
        generate_audio: bool
    ) -> Optional[Tuple[DatasetSample, int, int]]:
        """Generate a single sample and return metadata plus file sizes."""
        complexity = self._sample_complexity()
        midi_config = self._create_midi_config(complexity)

        midi_generator = MIDIGenerator(midi_config)

        sample_id = f"{split}_{index:05d}"
        midi_path = midi_dir / f"{sample_id}.mid"
        audio_path = audio_dir / f"{sample_id}.{self.config.audio_format}"

        try:
            midi_generator.generate(midi_path)
        except Exception as exc:
            self.logger.error(f"Failed to generate MIDI {sample_id}: {exc}")
            return None

        midi_size = midi_path.stat().st_size if midi_path.exists() else 0
        duration = 0.0
        audio_size = 0

        if generate_audio and self.synthesis_available:
            try:
                synthesizer = AudioSynthesizer(
                    soundfont_path=self.config.soundfont_path,
                    sample_rate=self.config.sample_rate
                )
                audio = synthesizer.synthesize(midi_path, audio_path)
                duration = len(audio) / self.config.sample_rate
                if audio_path.exists():
                    audio_size = audio_path.stat().st_size
            except Exception as exc:
                self.logger.error(f"Failed to synthesize {sample_id}: {exc}")
                return None

        sample = DatasetSample(
            id=sample_id,
            midi_path=str(midi_path.relative_to(dataset_dir)),
            audio_path=str(audio_path.relative_to(dataset_dir)) if generate_audio else "",
            complexity=complexity.value,
            tempo=midi_config.tempo,
            time_signature=f"{midi_config.time_signature.value[0]}/{midi_config.time_signature.value[1]}",
            key_signature=midi_config.key_signature.name,
            num_measures=midi_config.num_measures,
            duration=duration,
            split=split
        )

        return sample, midi_size, audio_size
    
    def _sample_complexity(self) -> ComplexityLevel:
        """Sample complexity level based on distribution."""
        levels = list(ComplexityLevel)
        weights = [
            self.config.complexity_distribution.get(level.value, 0.33)
            for level in levels
        ]
        
        return random.choices(levels, weights=weights)[0]
    
    def _create_midi_config(self, complexity: ComplexityLevel) -> MIDIConfig:
        """Create randomized MIDI configuration."""
        from .midi_generator import TimeSignature, KeySignature
        
        config = MIDIConfig()
        config.complexity = complexity
        config.tempo = random.randint(60, 180)
        config.time_signature = random.choice(list(TimeSignature))
        config.key_signature = random.choice(list(KeySignature))
        config.num_measures = random.randint(4, 16)
        
        return config
    
    def _generate_metadata(self, dataset_dir: Path) -> None:
        """Generate metadata files for the dataset."""
        metadata_dir = dataset_dir / 'metadata'
        
        # 1. Dataset info
        dataset_info = {
            'name': self.config.name,
            'version': self.config.version,
            'generated_at': datetime.now().isoformat(),
            'total_samples': len(self.samples),
            'splits': {
                'train': sum(1 for s in self.samples if s.split == 'train'),
                'val': sum(1 for s in self.samples if s.split == 'val'),
                'test': sum(1 for s in self.samples if s.split == 'test'),
            },
            'config': {
                'sample_rate': self.config.sample_rate,
                'audio_format': self.config.audio_format,
                'complexity_distribution': self.config.complexity_distribution,
            },
            'statistics': self._calculate_statistics()
        }
        
        with open(metadata_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # 2. Samples manifest (all samples)
        samples_data = [asdict(sample) for sample in self.samples]
        with open(metadata_dir / 'samples.json', 'w') as f:
            json.dump(samples_data, f, indent=2)
        
        # 3. Split-specific manifests
        for split in ['train', 'val', 'test']:
            split_samples = [s for s in self.samples if s.split == split]
            split_data = [asdict(sample) for sample in split_samples]
            
            with open(metadata_dir / f'{split}_manifest.json', 'w') as f:
                json.dump(split_data, f, indent=2)
    
    def _calculate_statistics(self) -> Dict:
        """Calculate dataset statistics."""
        if not self.samples:
            return {}
        
        stats = {
            'complexity': {},
            'tempo': {'min': float('inf'), 'max': 0, 'avg': 0},
            'measures': {'min': float('inf'), 'max': 0, 'avg': 0},
            'duration': {'total': 0, 'avg': 0},
            'time_signatures': {},
            'key_signatures': {}
        }
        
        # Count by complexity
        for level in ['beginner', 'intermediate', 'advanced']:
            count = sum(1 for s in self.samples if s.complexity == level)
            stats['complexity'][level] = count
        
        # Tempo statistics
        tempos = [s.tempo for s in self.samples]
        stats['tempo']['min'] = min(tempos)
        stats['tempo']['max'] = max(tempos)
        stats['tempo']['avg'] = sum(tempos) / len(tempos)
        
        # Measures statistics
        measures = [s.num_measures for s in self.samples]
        stats['measures']['min'] = min(measures)
        stats['measures']['max'] = max(measures)
        stats['measures']['avg'] = sum(measures) / len(measures)
        
        # Duration statistics
        durations = [s.duration for s in self.samples if s.duration > 0]
        if durations:
            stats['duration']['total'] = sum(durations)
            stats['duration']['avg'] = sum(durations) / len(durations)
        
        # Time signature distribution
        for sample in self.samples:
            ts = sample.time_signature
            stats['time_signatures'][ts] = stats['time_signatures'].get(ts, 0) + 1
        
        # Key signature distribution
        for sample in self.samples:
            key = sample.key_signature
            stats['key_signatures'][key] = stats['key_signatures'].get(key, 0) + 1
        
        return stats
    
    def load_manifest(self, manifest_path: Path) -> List[DatasetSample]:
        """
        Load samples from manifest file.
        
        Args:
            manifest_path: Path to manifest JSON file
            
        Returns:
            List of DatasetSample objects
        """
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        
        samples = [DatasetSample(**item) for item in data]
        return samples
