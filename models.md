# Whisper Models — Installation & Usage Guide

This guide covers the four recommended models: **base**, **small**, **medium**, and **large-v3-turbo**.

---

## Model Comparison

| Model | Size | Languages | Quality | NPU | GPU | CPU | Best for |
|-------|------|-----------|---------|-----|-----|-----|----------|
| `tiny` | 75 MB | 99 | ★★☆☆☆ | ✅ Fast | ✅ | ✅ | Quick tests |
| `base` | 141 MB | 99 | ★★★☆☆ | ✅ Fast | ✅ | ✅ | Everyday use, low-resource |
| `small` | 466 MB | 99 | ★★★★☆ | ✅ Good | ✅ | ✅ | **Recommended for Ukrainian** |
| `medium` | 1.5 GB | 99 | ★★★★½ | ⚠️ Maybe | ✅ | ✅ | High accuracy |
| `large-v3-turbo` | 1.5 GB | 99 | ★★★★★ | ⚠️ Maybe | ✅ | ✅ | **Best quality** |
| `large-v3` | 2.9 GB | 99 | ★★★★★ | ❌ Too large | ✅ | ✅ | Maximum accuracy |

> **NPU memory on Intel AI Boost (Ultra series):** ~1 GB usable.
> Models up to `small` (466 MB encoder) run comfortably.
> `medium` and `large-v3-turbo` may or may not fit depending on driver version.

---

## Installation

### Prerequisites

```cmd
pip install openvino==2026.0.0 openai-whisper torch
```

### Step 1 — Download + convert encoder

```cmd
cd "C:\Work\Transcribe Audio To Text app"

REM One model:
py -3 setup_models.py small

REM Multiple models at once:
py -3 setup_models.py base small medium large-v3-turbo
```

`setup_models.py` does three things automatically:
1. Downloads the GGML weight file from HuggingFace
2. Converts the encoder to OpenVINO IR (FP16)
3. Clears the OpenVINO compile cache so the device recompiles

### Step 2 — Rebuild

```cmd
cmake --build build --config Release
```

This copies the downloaded models and converted encoders into `build\Release\`.

### Step 3 — Run

```cmd
cd build\Release
simple_whisper.exe <model> mic --device NPU --language uk
```

---

## base — Lightweight Everyday Model

**Size:** 141 MB &nbsp;|&nbsp; **Encoder:** 41 MB &nbsp;|&nbsp; **NPU:** ✅ Excellent

The `base` multilingual model is already included in the project.
It runs fast on NPU and works well for clear speech in any of 99 languages.

### Install

Already downloaded. Nothing extra needed.

### Verify

```cmd
py -3 setup_models.py --list
```

Look for `base` showing `encoder: ready`.

### Use

```cmd
cd build\Release

REM Ukrainian microphone
simple_whisper.exe base mic --device NPU --language uk

REM English audio file
simple_whisper.exe base recording.wav --device NPU --language en

REM Auto-detect language (shows detected language + confidence before transcription)
simple_whisper.exe base mic --device NPU

REM English-only variant (slightly better English accuracy)
simple_whisper.exe base.en mic --device NPU
```

### Performance (Intel Core Ultra 7 255U)

| Device | Encode | Real-time factor |
|--------|--------|-----------------|
| NPU | ~170 ms | ~50–70× |
| GPU | ~250 ms | ~35× |
| CPU | ~400 ms | ~20× |

---

## small — Recommended for Ukrainian

**Size:** 466 MB &nbsp;|&nbsp; **Encoder:** 176 MB &nbsp;|&nbsp; **NPU:** ✅ Good

The `small` model is **5× more parameters** than `base` and significantly more accurate
for non-English languages, especially Ukrainian, German, and Polish.

### Install

```cmd
cd "C:\Work\Transcribe Audio To Text app"
py -3 setup_models.py small
cmake --build build --config Release
```

Download: ~466 MB GGML + ~466 MB PyTorch (for encoder conversion) = ~1 GB total traffic.

### Use

```cmd
cd build\Release

REM Ukrainian — recommended command
simple_whisper.exe small mic --device NPU --language uk

REM German
simple_whisper.exe small mic --device NPU --language de

REM Auto-detect
simple_whisper.exe small mic --device NPU

REM English-only variant (best English with small size)
simple_whisper.exe small.en mic --device NPU
```

### Performance (Intel Core Ultra 7 255U)

| Device | Encode | Real-time factor |
|--------|--------|-----------------|
| NPU | ~400 ms | ~25–35× |
| GPU | ~600 ms | ~15× |
| CPU | ~1200 ms | ~8× |

---

## medium — High Accuracy

**Size:** 1.5 GB &nbsp;|&nbsp; **Encoder:** ~600 MB &nbsp;|&nbsp; **NPU:** ⚠️ May not fit

The `medium` model has **6× more parameters** than `small`. Recommended for:
- Professional transcription where accuracy matters
- Languages with complex phonetics
- Noisy recordings

### Install

```cmd
cd "C:\Work\Transcribe Audio To Text app"
py -3 setup_models.py medium
cmake --build build --config Release
```

Download: ~1.5 GB GGML + ~1.5 GB PyTorch = ~3 GB total traffic.

### NPU note

The `medium` encoder (~600 MB) may exceed the NPU's available memory.
Try NPU first; if it fails or gives a warning, fall back to GPU or CPU:

```cmd
REM Try NPU first
simple_whisper.exe medium mic --device NPU --language uk

REM If NPU fails, use GPU
simple_whisper.exe medium mic --device GPU --language uk

REM CPU always works
simple_whisper.exe medium mic --device CPU --language uk
```

### Use

```cmd
cd build\Release

REM Ukrainian
simple_whisper.exe medium mic --device GPU --language uk

REM German interview recording
simple_whisper.exe medium interview.flac --device GPU --language de

REM English-only variant (best medium-size English)
simple_whisper.exe medium.en lecture.wav --device GPU
```

### Performance (Intel Core Ultra 7 255U, GPU)

| Device | Encode | Real-time factor |
|--------|--------|-----------------|
| GPU | ~1.5 s | ~8–12× |
| CPU | ~4 s | ~3–5× |

---

## large-v3-turbo — Best Quality

**Size:** 1.5 GB &nbsp;|&nbsp; **Encoder:** ~600 MB &nbsp;|&nbsp; **NPU:** ⚠️ May not fit

`large-v3-turbo` delivers **large-v3 quality** (the best Whisper model) at the speed
of `medium`. It achieves this by using a trimmed decoder (4 layers instead of 32)
while keeping the full large encoder.

Recommended for:
- Highest accuracy transcription
- Difficult audio (accents, noise, fast speech)
- Professional use where quality > speed

### Install

```cmd
cd "C:\Work\Transcribe Audio To Text app"
py -3 setup_models.py large-v3-turbo
cmake --build build --config Release
```

Download: ~1.5 GB GGML + ~1.5 GB PyTorch = ~3 GB total traffic.

### Quantised variant (smaller, slightly lower quality)

```cmd
py -3 setup_models.py large-v3-turbo-q5_0
cmake --build build --config Release
```

`large-v3-turbo-q5_0` is only **547 MB** and uses the same encoder.
Quality is nearly identical to full precision.

### NPU note

Same as `medium` — try NPU; fall back to GPU if the encoder is too large.

### Use

```cmd
cd build\Release

REM Ukrainian — highest quality
simple_whisper.exe large-v3-turbo mic --device GPU --language uk

REM Quantised variant (smaller, nearly same quality)
simple_whisper.exe large-v3-turbo-q5_0 mic --device GPU --language uk

REM English transcription — best possible
simple_whisper.exe large-v3-turbo lecture.wav --device GPU --language en

REM Auto-detect language
simple_whisper.exe large-v3-turbo mic --device GPU
```

### Performance (Intel Core Ultra 7 255U, GPU)

| Device | Encode | Real-time factor |
|--------|--------|-----------------|
| GPU | ~1.5 s | ~8–10× |
| CPU | ~4 s | ~3× |

---

## Choosing the Right Model

```
Fast, low memory needed?
  └── tiny / base         (NPU: excellent)

Good Ukrainian / non-English?
  └── small               (NPU: good)      ← START HERE

High accuracy, professional?
  └── medium              (GPU recommended)

Best possible quality?
  └── large-v3-turbo      (GPU recommended)
  └── large-v3-turbo-q5_0 (GPU, smaller download)

Maximum accuracy (no speed concern)?
  └── large-v3            (CPU or GPU only)
```

---

## Beam Search vs Greedy

| Mode | Flag | Speed | Accuracy |
|------|------|-------|----------|
| Beam search (default) | `--beam 5` | Slower | Higher |
| Greedy | `--beam 1` | Fastest | Lower |

For the best transcription quality, keep the default `--beam 5`.
Use `--beam 1` only when speed is critical and accuracy is secondary.

---

## Full Install Example (small model, Ukrainian)

```cmd
cd "C:\Work\Transcribe Audio To Text app"

REM 1. Download + convert encoder
py -3 setup_models.py small

REM 2. Rebuild
cmake --build build --config Release

REM 3. Transcribe
cd build\Release
simple_whisper.exe small mic --device NPU --language uk
```

Speak into the microphone, press **ENTER** to stop, and the transcription appears.

---

## Troubleshooting

### "no tensors loaded from model file"
A corrupt or stub model file is in `build\Release`. Delete it and rebuild:
```cmd
del "build\Release\ggml-<model>.bin"
cmake --build build --config Release
```

### "OpenVINO encoder init failed"
The encoder XML is missing. Run:
```cmd
py -3 setup_models.py <model>
cmake --build build --config Release
```

### NPU not detected / not used
Check driver: `simple_whisper.exe base test_audio.wav --device NPU`
Look for `NPU driver version` in output. If missing, install the latest Intel NPU driver.

### Poor transcription quality
1. Use a larger model: `base` → `small` → `medium` → `large-v3-turbo`
2. Specify language explicitly: `--language uk` instead of auto-detect
3. Ensure microphone is not muted and is the default recording device

### Auto-detect wrong language
Add `--language <code>` explicitly, e.g. `--language uk` for Ukrainian.
