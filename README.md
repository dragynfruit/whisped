# Whisped

A Rust CLI tool for Whisper ASR with automatic model download.

## About

Whisped is a command-line interface for OpenAI's Whisper automatic speech recognition (ASR) system. It supports transcribing audio from various file formats, with automatic model downloading and caching for convenience.

## Features

- Automatic model downloading and caching
- Support for WAV, MP3, and OGG audio formats
- Multiple model sizes (tiny, base, small, medium, large)
- Optional quantization for faster inference
- Language detection or specification
- Translation to English
- Optional timestamp inclusion

## Installation

### Prerequisites

- Rust and Cargo (https://rustup.rs/)

### Build from source

```bash
git clone https://github.com/dragynfruit/whisped.git
cd whisped
cargo build --release
```

The binary will be available at `target/release/whisped`.

## Usage

```bash
# Basic usage
whisped --input audio.mp3 --output transcript.txt

# Specify a model
whisped --model medium --input audio.mp3 --output transcript.txt

# Use quantization for faster inference
whisped --model medium --quant q5_0 --input audio.mp3 --output transcript.txt

# Specify language and include timestamps
whisped --input audio.mp3 --output transcript.txt --language en --timestamps

# Translate to English
whisped --input audio_in_any_language.mp3 --output transcript.txt --translate
```

## Command Line Options

- `--model`: Model to use (tiny, base, small, medium, large). Default: base
- `--quant`: Quantization level (e.g., q4_0, q5_0, q8_0). Default: none
- `--input`: Path to the input audio file (required)
- `--output`: Path to the output text file (required)
- `--language`: Language code (e.g., en, fr, de). Default: auto-detect
- `--translate`: Enable translation to English
- `--timestamps`: Include timestamps in the output

## Acknowledgements

- This tool uses [whisper-rs](https://github.com/tazz4843/whisper-rs) for Rust bindings to OpenAI's Whisper model
- Models are automatically downloaded from [HuggingFace](https://huggingface.co/ggerganov/whisper.cpp)
