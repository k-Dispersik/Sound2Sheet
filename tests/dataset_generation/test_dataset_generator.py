"""
Tests for DatasetGenerator class.
"""

import pytest
from pathlib import Path
import json
import shutil

from src.dataset import (
    DatasetGenerator, DatasetConfig, DatasetSample,
    ComplexityLevel
)


class TestDatasetConfig:
    """Tests for DatasetConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DatasetConfig()
        
        assert config.name == "piano_dataset"
        assert config.version == "1.0.0"
        assert config.total_samples == 1000
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.15
        assert config.test_ratio == 0.15
        assert config.sample_rate == 16000
        assert config.audio_format == "wav"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = DatasetConfig(
            name="custom_dataset",
            version="2.0.0",
            total_samples=500,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        assert config.name == "custom_dataset"
        assert config.version == "2.0.0"
        assert config.total_samples == 500
        assert config.train_ratio == 0.8
    
    def test_default_complexity_distribution(self):
        """Test default complexity distribution."""
        config = DatasetConfig()
        
        assert "beginner" in config.complexity_distribution
        assert "intermediate" in config.complexity_distribution
        assert "advanced" in config.complexity_distribution
        
        # Should be roughly balanced
        assert 0.3 <= config.complexity_distribution["beginner"] <= 0.35
    
    def test_custom_complexity_distribution(self):
        """Test custom complexity distribution."""
        dist = {
            "beginner": 0.5,
            "intermediate": 0.3,
            "advanced": 0.2
        }
        config = DatasetConfig(complexity_distribution=dist)
        
        assert config.complexity_distribution == dist
    
    def test_invalid_split_ratios(self):
        """Test validation of split ratios."""
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            DatasetConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


class TestDatasetSample:
    """Tests for DatasetSample dataclass."""
    
    def test_create_sample(self):
        """Test creating a dataset sample."""
        sample = DatasetSample(
            id="train_00001",
            midi_path="train/midi/train_00001.mid",
            audio_path="train/audio/train_00001.wav",
            complexity="intermediate",
            tempo=120,
            time_signature="4/4",
            key_signature="C_MAJOR",
            num_measures=8,
            duration=16.5,
            split="train"
        )
        
        assert sample.id == "train_00001"
        assert sample.complexity == "intermediate"
        assert sample.tempo == 120
        assert sample.split == "train"


class TestDatasetGenerator:
    """Tests for DatasetGenerator."""
    
    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "datasets"
        output_dir.mkdir()
        yield output_dir
        # Cleanup
        if output_dir.exists():
            shutil.rmtree(output_dir)
    
    @pytest.fixture
    def small_config(self, temp_output_dir):
        """Create config with small dataset for testing."""
        return DatasetConfig(
            name="test_dataset",
            version="0.1.0",
            total_samples=10,  # Small for fast tests
            output_dir=temp_output_dir
        )
    
    def test_initialization(self, small_config):
        """Test DatasetGenerator initialization."""
        generator = DatasetGenerator(small_config)
        
        assert generator.config.name == "test_dataset"
        assert generator.midi_generator is not None
        assert isinstance(generator.samples, list)
        assert len(generator.samples) == 0
    
    def test_initialization_default_config(self):
        """Test initialization with default config."""
        generator = DatasetGenerator()
        
        assert generator.config.name == "piano_dataset"
        assert generator.config.total_samples == 1000
    
    def test_calculate_splits(self, small_config):
        """Test split calculation."""
        generator = DatasetGenerator(small_config)
        
        train, val, test = generator._calculate_splits()
        
        assert train == 7  # 70% of 10
        assert val == 1    # 15% of 10
        assert test == 2   # Remainder
        assert train + val + test == 10
    
    def test_sample_complexity(self, small_config):
        """Test complexity sampling."""
        generator = DatasetGenerator(small_config)
        
        # Sample multiple times to check distribution
        samples = [generator._sample_complexity() for _ in range(100)]
        
        # All should be valid ComplexityLevel
        assert all(isinstance(s, ComplexityLevel) for s in samples)
        
        # Should have variety (not all same)
        unique_levels = set(samples)
        assert len(unique_levels) >= 2
    
    def test_create_midi_config(self, small_config):
        """Test MIDI configuration creation."""
        generator = DatasetGenerator(small_config)
        
        config = generator._create_midi_config(ComplexityLevel.INTERMEDIATE)
        
        assert config.complexity == ComplexityLevel.INTERMEDIATE
        assert 60 <= config.tempo <= 180
        assert 4 <= config.num_measures <= 16
    
    def test_generate_midi_only(self, small_config):
        """Test generating dataset with MIDI only (no audio synthesis)."""
        generator = DatasetGenerator(small_config)
        
        dataset_dir = generator.generate(generate_audio=False)
        
        # Check directory structure
        assert dataset_dir.exists()
        assert (dataset_dir / 'train' / 'midi').exists()
        assert (dataset_dir / 'val' / 'midi').exists()
        assert (dataset_dir / 'test' / 'midi').exists()
        assert (dataset_dir / 'metadata').exists()
        
        # Check samples were generated
        assert len(generator.samples) == 10
        
        # Check MIDI files exist
        train_samples = [s for s in generator.samples if s.split == 'train']
        for sample in train_samples:
            midi_path = dataset_dir / sample.midi_path
            assert midi_path.exists()
    
    def test_generate_with_audio(self, small_config):
        """Test generating dataset with audio synthesis."""
        generator = DatasetGenerator(small_config)
        
        # Only run if synthesis is available
        if not generator.synthesis_available:
            pytest.skip("Audio synthesis not available")
        
        dataset_dir = generator.generate(generate_audio=True)
        
        # Check audio directories
        assert (dataset_dir / 'train' / 'audio').exists()
        
        # Check audio files exist
        train_samples = [s for s in generator.samples if s.split == 'train']
        for sample in train_samples:
            if sample.audio_path:  # May be empty if synthesis failed
                audio_path = dataset_dir / sample.audio_path
                assert audio_path.exists()
    
    def test_metadata_generation(self, small_config):
        """Test metadata file generation."""
        generator = DatasetGenerator(small_config)
        dataset_dir = generator.generate(generate_audio=False)
        
        metadata_dir = dataset_dir / 'metadata'
        
        # Check metadata files exist
        assert (metadata_dir / 'dataset_info.json').exists()
        assert (metadata_dir / 'samples.json').exists()
        assert (metadata_dir / 'train_manifest.json').exists()
        assert (metadata_dir / 'val_manifest.json').exists()
        assert (metadata_dir / 'test_manifest.json').exists()
    
    def test_dataset_info_content(self, small_config):
        """Test dataset info JSON content."""
        generator = DatasetGenerator(small_config)
        dataset_dir = generator.generate(generate_audio=False)
        
        with open(dataset_dir / 'metadata' / 'dataset_info.json') as f:
            info = json.load(f)
        
        assert info['name'] == "test_dataset"
        assert info['version'] == "0.1.0"
        assert info['total_samples'] == 10
        assert 'splits' in info
        assert info['splits']['train'] == 7
        assert info['splits']['val'] == 1
        assert info['splits']['test'] == 2
        assert 'statistics' in info
    
    def test_samples_manifest_content(self, small_config):
        """Test samples manifest JSON content."""
        generator = DatasetGenerator(small_config)
        dataset_dir = generator.generate(generate_audio=False)
        
        with open(dataset_dir / 'metadata' / 'samples.json') as f:
            samples = json.load(f)
        
        assert len(samples) == 10
        
        # Check first sample structure
        sample = samples[0]
        assert 'id' in sample
        assert 'midi_path' in sample
        assert 'complexity' in sample
        assert 'tempo' in sample
        assert 'split' in sample
    
    def test_split_manifests(self, small_config):
        """Test split-specific manifest files."""
        generator = DatasetGenerator(small_config)
        dataset_dir = generator.generate(generate_audio=False)
        
        metadata_dir = dataset_dir / 'metadata'
        
        # Load each split manifest
        with open(metadata_dir / 'train_manifest.json') as f:
            train_samples = json.load(f)
        
        with open(metadata_dir / 'val_manifest.json') as f:
            val_samples = json.load(f)
        
        with open(metadata_dir / 'test_manifest.json') as f:
            test_samples = json.load(f)
        
        # Check counts match expected splits
        assert len(train_samples) == 7
        assert len(val_samples) == 1
        assert len(test_samples) == 2
        
        # Check all samples have correct split
        assert all(s['split'] == 'train' for s in train_samples)
        assert all(s['split'] == 'val' for s in val_samples)
        assert all(s['split'] == 'test' for s in test_samples)
    
    def test_statistics_calculation(self, small_config):
        """Test dataset statistics calculation."""
        generator = DatasetGenerator(small_config)
        dataset_dir = generator.generate(generate_audio=False)
        
        with open(dataset_dir / 'metadata' / 'dataset_info.json') as f:
            info = json.load(f)
        
        stats = info['statistics']
        
        # Check statistics structure
        assert 'complexity' in stats
        assert 'tempo' in stats
        assert 'measures' in stats
        assert 'time_signatures' in stats
        assert 'key_signatures' in stats
        
        # Check complexity counts
        complexity_total = sum(stats['complexity'].values())
        assert complexity_total == 10
        
        # Check tempo range
        assert stats['tempo']['min'] >= 60
        assert stats['tempo']['max'] <= 180
        assert stats['tempo']['min'] <= stats['tempo']['avg'] <= stats['tempo']['max']
    
    def test_load_manifest(self, small_config):
        """Test loading manifest file."""
        generator = DatasetGenerator(small_config)
        dataset_dir = generator.generate(generate_audio=False)
        
        manifest_path = dataset_dir / 'metadata' / 'train_manifest.json'
        samples = generator.load_manifest(manifest_path)
        
        assert len(samples) == 7
        assert all(isinstance(s, DatasetSample) for s in samples)
        assert all(s.split == 'train' for s in samples)
    
    def test_directory_structure_hierarchy(self, small_config):
        """Test complete directory hierarchy."""
        generator = DatasetGenerator(small_config)
        dataset_dir = generator.generate(generate_audio=False)
        
        # Expected structure
        expected_dirs = [
            'train/midi',
            'train/audio',
            'val/midi',
            'val/audio',
            'test/midi',
            'test/audio',
            'metadata'
        ]
        
        for dir_path in expected_dirs:
            assert (dataset_dir / dir_path).exists()
            assert (dataset_dir / dir_path).is_dir()
    
    def test_unique_sample_ids(self, small_config):
        """Test that all sample IDs are unique."""
        generator = DatasetGenerator(small_config)
        generator.generate(generate_audio=False)
        
        ids = [s.id for s in generator.samples]
        assert len(ids) == len(set(ids)), "Sample IDs are not unique"
    
    def test_relative_paths_in_metadata(self, small_config):
        """Test that metadata contains relative paths."""
        generator = DatasetGenerator(small_config)
        dataset_dir = generator.generate(generate_audio=False)
        
        with open(dataset_dir / 'metadata' / 'samples.json') as f:
            samples = json.load(f)
        
        for sample in samples:
            # Paths should be relative (not absolute)
            midi_path = sample['midi_path']
            assert not Path(midi_path).is_absolute()
            assert midi_path.startswith(sample['split'])
