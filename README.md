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
python main.py --version
```

3. Run tests:
```bash
python -m unittest discover tests -v
```

## Project Status
🔴 In Development - Planning Phase

## Architecture
- **Audio Processing**: Convert audio to mel-spectrograms
- **Model**: AST-based transcription with custom decoder
- **Output**: JSON/MIDI/MusicXML notation formats
- **API**: REST API for transcription services

## Testing
This project uses Python's built-in `unittest` framework for testing.

Run all tests:
```bash
python -m unittest discover tests -v
```

Run specific test file:
```bash
python -m unittest tests.test_audio_processor -v
```

## Development
This project follows Git Flow workflow:
- `main`: Production releases
- `develop`: Integration branch
- `feature/*`: Feature development branches

All code changes must include appropriate unit tests.

## License
MIT License