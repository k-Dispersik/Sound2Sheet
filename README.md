# Sound2Sheet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ No warranty provided
- ⚠️ No liability accepted

Copyright (c) 2025 Volodymyr