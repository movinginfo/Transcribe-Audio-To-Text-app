# Transcribe Audio To Text

> **v0.1.2** — Real-time speech-to-text on Windows using [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with **Intel NPU / GPU / CPU** acceleration via [OpenVINO](https://github.com/openvinotoolkit/openvino).

Supports **99 languages** including Ukrainian, English, German, French, and more.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What's New in v0.1.2

- **MULTI:NPU,GPU default** — encoder runs on NPU and GPU simultaneously; fastest result wins
- **Flash Attention** — 20–40% faster decoder, enabled automatically
- **VAD (Voice Activity Detection)** — skip silence; major speedup for mic recordings with pauses
- **Language lock-in** — detected language is locked before transcription, eliminating mid-stream drift between Ukrainian / English / German
- **Temperature fallback** — automatic retry on uncertain segments for better accuracy on noisy audio

---

## Requirements

| Component | Version |
|-----------|---------|
| Windows | 10 / 11 |
| Visual Studio (MSVC) | 2019 or 2022 |
| CMake | 3.14+ |
| Python | 3.9+ |
| Intel OpenVINO | 2026.0 (installed via pip) |
| Intel NPU driver | 15.x+ (optional, for NPU acceleration) |

Install Python dependencies:
```cmd
pip install openvino==2026.0.0 openai-whisper torch
```

---

## Quick Start

### 1. Build

Open **Developer Command Prompt for Visual Studio** and run:

```cmd
cd "C:\Work\Transcribe Audio To Text app"
cmake -B build
cmake --build build --config Release
```

### 2. Download a model + VAD

```cmd
REM Recommended starting model
py -3 setup_models.py small

REM Also download the VAD model (enables --vad silence skipping)
py -3 setup_models.py --vad
```

### 3. Rebuild to copy models into build\Release

```cmd
cmake --build build --config Release
```

### 4. Transcribe

```cmd
cd build\Release

REM Ukrainian microphone — fast + accurate
simple_whisper.exe small mic --language uk

REM With VAD (skip silence — even faster)
simple_whisper.exe small mic --language uk --vad

REM From audio file, auto-detect language
simple_whisper.exe small audio.wav
```

---

## Command Reference

```
simple_whisper.exe <model> <audio|mic> [options]

Arguments:
  model          Model short name: tiny, base, small, medium, large-v3-turbo
                 Full names also work: ggml-base.bin
  audio|mic      Path to .wav / .mp3 / .flac file, or "mic" to record

Options:
  --device       MULTI:NPU,GPU | NPU | GPU | CPU | AUTO   (default: MULTI:NPU,GPU)
  --language     auto | uk | en | de | fr | ...           (default: auto)
  --beam N       Beam search width: higher = more accurate (default: 5)
                 Use --beam 1 for greedy decoding (fastest)
  --threads N    CPU decode threads                       (default: all cores, max 16)
  --vad          Skip silence using Voice Activity Detection
  --vad-model    Path to silero VAD model                 (default: auto-search)
  --loop N       Repeat N times – keeps NPU busy for Task Manager view
  --ov-model     Path to custom encoder .xml file
```

### Device selection

| Flag | Description |
|------|-------------|
| `--device MULTI:NPU,GPU` | NPU + GPU run in parallel, fastest result used **(default)** |
| `--device NPU` | Intel AI Boost (lowest power) |
| `--device GPU` | Intel integrated or discrete GPU |
| `--device CPU` | CPU only (no OpenVINO needed) |
| `--device AUTO` | OpenVINO picks automatically |

### Language codes (common)

| Code | Language | Code | Language |
|------|----------|------|----------|
| `uk` | Ukrainian | `en` | English |
| `de` | German | `fr` | French |
| `pl` | Polish | `es` | Spanish |
| `ru` | Russian | `zh` | Chinese |
| `auto` | Auto-detect | — | 99 total |

---

## Model Management

See [models.md](models.md) for a detailed guide on all models, quality vs. speed trade-offs, and NPU compatibility.

```cmd
REM List current status of all models + VAD:
py -3 setup_models.py --list

REM Download specific models:
py -3 setup_models.py small medium large-v3-turbo

REM Download VAD model (for --vad flag):
py -3 setup_models.py --vad

REM Download all models (~20 GB):
py -3 setup_models.py
```

After downloading, always rebuild:
```cmd
cmake --build build --config Release
```

---

## Voice Activity Detection (VAD)

VAD uses the [Silero VAD](https://github.com/snakers4/silero-vad) model to detect speech segments and skip silence before it reaches the encoder. This provides the biggest speed improvement for microphone recordings with natural pauses.

```cmd
REM Download the VAD model (0.9 MB, one-time)
py -3 setup_models.py --vad
cmake --build build --config Release

REM Use VAD in transcription
simple_whisper.exe small mic --language uk --vad
```

**When to use VAD:**
- Microphone recordings with pauses between sentences
- Long audio files with sections of silence
- Interviews, meetings, lectures

**When VAD is not needed:**
- Dense speech with no pauses
- Short audio clips (under 10 seconds)

---

## OpenVINO Encoder Conversion

OpenVINO acceleration requires a pre-converted encoder file (`ggml-<model>-encoder-openvino.xml`).
`setup_models.py` handles this automatically. To convert manually:

```cmd
py -3 convert_encoder_to_openvino.py --model small
cmake --build build --config Release
```

---

## Project Structure

```
├── main.cpp                          Application source
├── CMakeLists.txt                    Build configuration
├── setup_models.py                   Download + convert all models + VAD
├── convert_encoder_to_openvino.py    Convert single encoder to OpenVINO
├── cmake/
│   └── copy_encoders.cmake           Copies models/encoders/VAD into build\Release
├── ggml-*.bin                        Downloaded model weights
├── ggml-*-encoder-openvino.xml/.bin  Converted OpenVINO encoders
├── ggml-silero-v5.1.2.bin            Silero VAD model (optional)
└── build/Release/
    └── simple_whisper.exe            Compiled executable
```

---

## Examples

```cmd
cd "C:\Work\Transcribe Audio To Text app\build\Release"

REM Ukrainian — microphone — MULTI:NPU+GPU (default)
simple_whisper.exe small mic --language uk

REM Ukrainian — with VAD (skip silence)
simple_whisper.exe small mic --language uk --vad

REM English — audio file
simple_whisper.exe small.en interview.wav

REM German — best quality
simple_whisper.exe medium audio.flac --language de

REM Best possible quality
simple_whisper.exe large-v3-turbo lecture.wav --language en

REM Benchmark: run 20 loops to measure NPU throughput
simple_whisper.exe base test_audio.wav --device NPU --loop 20
```

---

## License

MIT — see [LICENSE](LICENSE)
