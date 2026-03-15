# Transcribe Audio To Text

Real-time speech-to-text on Windows using [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with **Intel NPU / GPU / CPU** acceleration via [OpenVINO](https://github.com/openvinotoolkit/openvino).

Supports **99 languages** including Ukrainian, English, German, French, and more.

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

### 2. Download a model

```cmd
py -3 setup_models.py base
```

Or for better quality:

```cmd
py -3 setup_models.py small
```

### 3. Rebuild to copy models into build\Release

```cmd
cmake --build build --config Release
```

### 4. Transcribe

```cmd
cd build\Release

REM From microphone (Ukrainian):
simple_whisper.exe base mic --device NPU --language uk

REM From audio file:
simple_whisper.exe base audio.wav --device NPU --language uk

REM Auto-detect language:
simple_whisper.exe base mic --device NPU
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
  --device       AUTO | NPU | GPU | CPU | MULTI:NPU,GPU   (default: AUTO)
  --language     auto | uk | en | de | fr | ...           (default: auto)
  --beam N       Beam search width: higher = more accurate (default: 5)
                 Use --beam 1 for greedy decoding (faster)
  --threads N    CPU decode threads                       (default: 8)
  --loop N       Repeat N times – keeps NPU busy for Task Manager view
  --ov-model     Path to custom encoder .xml file
```

### Device selection

| Flag | Description |
|------|-------------|
| `--device AUTO` | OpenVINO picks the fastest available device (default) |
| `--device NPU` | Intel AI Boost (fastest encoder, lowest power) |
| `--device GPU` | Intel integrated or discrete GPU |
| `--device CPU` | CPU only (no OpenVINO needed) |
| `--device MULTI:NPU,GPU` | Run on NPU and GPU simultaneously |

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
REM List current status of all models:
py -3 setup_models.py --list

REM Download specific models:
py -3 setup_models.py small medium large-v3-turbo

REM Download all models (~20 GB):
py -3 setup_models.py
```

After downloading, always rebuild:
```cmd
cmake --build build --config Release
```

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
├── setup_models.py                   Download + convert all models
├── convert_encoder_to_openvino.py    Convert single encoder to OpenVINO
├── cmake/
│   └── copy_encoders.cmake           Copies models into build\Release at build time
├── ggml-*.bin                        Downloaded model weights
├── ggml-*-encoder-openvino.xml/.bin  Converted OpenVINO encoders
└── build/Release/
    └── simple_whisper.exe            Compiled executable
```

---

## Examples

```cmd
cd "C:\Work\Transcribe Audio To Text app\build\Release"

REM Ukrainian — microphone — NPU
simple_whisper.exe base mic --device NPU --language uk

REM English — audio file — auto device
simple_whisper.exe small.en interview.wav

REM German — best quality, CPU fallback
simple_whisper.exe medium audio.flac --device CPU --language de

REM English — highest accuracy
simple_whisper.exe large-v3-turbo lecture.wav --device GPU --language en

REM Benchmark: run 20 loops to measure NPU throughput
simple_whisper.exe base test_audio.wav --device NPU --loop 20
```
