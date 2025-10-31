#!/usr/bin/env python3
"""
Sound2Sheet - AI-powered music transcription system

Main entry point for the Sound2Sheet application.

Author: Development Team
Version: 0.1.0-dev
"""

import sys
from pathlib import Path
import yaml

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def load_version() -> str:
    """Load version from VERSION file."""
    version_file = Path(__file__).parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "unknown"


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Configuration file {config_path} not found. Using defaults.")
        return {}
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def show_status():
    """Show project development status."""
    print("Sound2Sheet Development Status")
    print("=" * 40)
    print(f"Version: {load_version()}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--version":
        print(f"Sound2Sheet {load_version()}")
        return 0
    
    show_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())