# Contributing to Speaker Recognition System

Thank you for your interest in contributing to the Speaker Recognition System! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/speaker-recognition-system.git
   cd speaker-recognition-system
   ```

### 2. Set Up Development Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install development dependencies:
   ```bash
   pip install flake8 black pytest
   ```

### 3. Make Your Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the code style guidelines
3. Test your changes thoroughly

### 4. Submit Your Contribution

1. Commit your changes:
   ```bash
   git add .
   git commit -m "Add: descriptive commit message"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request on GitHub

## 📋 Contribution Guidelines

### Code Style

- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Use type hints where appropriate

### Example Code Style:

```python
def extract_features(audio_file: str, sample_rate: int = 22050) -> np.ndarray:
    """
    Extract audio features from an audio file.
    
    Args:
        audio_file: Path to the audio file
        sample_rate: Sample rate for processing
        
    Returns:
        Extracted feature vector
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio file is corrupted
    """
    pass
```

### Testing

- Write tests for new features
- Ensure all existing tests pass
- Test with different audio formats and qualities
- Test edge cases and error conditions

### Documentation

- Update README.md if you add new features
- Add docstrings to all new functions
- Update configuration documentation
- Include examples for new functionality

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear description** of the problem
2. **Steps to reproduce** the issue
3. **Expected behavior** vs actual behavior
4. **System information** (OS, Python version, etc.)
5. **Audio file details** (format, duration, quality)
6. **Error messages** and stack traces

### Bug Report Template:

```
**Bug Description:**
A clear description of the bug.

**Steps to Reproduce:**
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior:**
What should happen.

**Actual Behavior:**
What actually happens.

**System Information:**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- Browser: [e.g., Chrome 96]

**Audio Details:**
- Format: [e.g., WAV, MP3]
- Duration: [e.g., 5 seconds]
- Quality: [e.g., 16kHz, 44.1kHz]

**Error Messages:**
Paste any error messages here.
```

## 💡 Feature Requests

When suggesting new features:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case** and why it's needed
3. **Provide examples** of how it would work
4. **Consider implementation** challenges
5. **Discuss alternatives** you've considered

## 🎯 Areas for Contribution

### High Priority
- [ ] Performance optimizations
- [ ] Better error handling
- [ ] More audio format support
- [ ] Improved real-time processing
- [ ] Enhanced security features

### Medium Priority
- [ ] Additional ML models
- [ ] Better visualization
- [ ] Mobile app integration
- [ ] Cloud deployment guides
- [ ] Multi-language support

### Low Priority
- [ ] UI/UX improvements
- [ ] Additional datasets
- [ ] Advanced configuration options
- [ ] Integration with other tools
- [ ] Performance benchmarks

## 🔄 Development Workflow

### 1. Planning
- Discuss major changes in Issues first
- Break down large features into smaller tasks
- Consider backward compatibility

### 2. Development
- Write clean, readable code
- Follow existing patterns and conventions
- Add appropriate logging and error handling

### 3. Testing
- Test manually with the Streamlit app
- Test with different audio files and formats
- Verify performance doesn't degrade

### 4. Documentation
- Update docstrings and comments
- Update README if necessary
- Add examples for new features

## 📞 Getting Help

- **Issues**: For bug reports and feature requests
- **Discussions**: For general questions and ideas
- **Email**: For private inquiries or security issues

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for their contributions
- Special mentions for significant features

## 📜 Code of Conduct

Please be respectful and constructive in all interactions. We strive to maintain a welcoming environment for all contributors.

### Guidelines:
- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow
- Follow the Golden Rule

Thank you for contributing to the Speaker Recognition System! 🎉
