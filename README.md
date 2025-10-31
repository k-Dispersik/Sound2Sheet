# Sound2Sheet

AI-powered music transcription system that converts piano audio recordings into structured musical notation.

## Version
Current version: 0.1.0-dev

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py --help
```

## Project Status
🔴 In Development - Planning Phase

## Architecture
- **Audio Processing**: Convert audio to mel-spectrograms
- **Model**: AST-based transcription with custom decoder
- **Output**: JSON/MIDI/MusicXML notation formats
- **API**: REST API for transcription services

## Development
This project follows Git Flow workflow:
- `main`: Production releases
- `develop`: Integration branch
- `feature/*`: Feature development branches

## License
MIT License