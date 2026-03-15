"""
setup_models.py  –  Download + convert ALL whisper.cpp models

Downloads every GGML model file and converts its encoder to OpenVINO IR.
Quantised variants (q5_0 / q8_0) reuse the encoder of their full-precision
parent so there is no duplicate conversion.

Usage
-----
    py -3 setup_models.py              # all models
    py -3 setup_models.py tiny base    # specific models only

After running, rebuild the project:
    cmake --build build --config Release

Disk requirements (approximate)
---------------------------------
  tiny           75 MiB  + encoder ~10 MiB
  tiny.en        75 MiB  + encoder ~10 MiB
  base          142 MiB  + encoder ~41 MiB  (already converted)
  base.en       142 MiB  + encoder ~41 MiB  (already converted)
  small         466 MiB  + encoder ~176 MiB
  small.en      466 MiB  + encoder ~176 MiB
  small.en-tdrz 465 MiB  (encoder = copy of small.en)
  medium        1.5 GiB  + encoder ~600 MiB
  medium.en     1.5 GiB  + encoder ~600 MiB
  large-v1      2.9 GiB  + encoder ~1.1 GiB
  large-v2      2.9 GiB  + encoder ~1.1 GiB
  large-v2-q5_0 1.1 GiB  (encoder = copy of large-v2)
  large-v3      2.9 GiB  + encoder ~1.1 GiB
  large-v3-q5_0 1.1 GiB  (encoder = copy of large-v3)
  large-v3-turbo 1.5 GiB + encoder ~600 MiB
  large-v3-turbo-q5_0 547 MiB (encoder = copy of large-v3-turbo)

NPU memory limits
-----------------
  tiny / base            work great on NPU
  small                  works on NPU
  medium                 may work on NPU (try; fallback to GPU/CPU)
  large-*                likely too large for NPU; use --device GPU or CPU
"""

import os, sys, shutil, urllib.request, argparse
from pathlib import Path

HERE = Path(__file__).parent.resolve()

# ── Model catalogue ───────────────────────────────────────────────────────────
# (name, ggml_file, download_url, encoder_source)
#   encoder_source = None  → convert fresh from openai-whisper
#   encoder_source = str   → copy encoder from that model (quantised reuse)

HF_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
HF_TDRZ = "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp/resolve/main"

MODELS = [
    # name                ggml file                          url                                       encoder_source
    ("tiny",              "ggml-tiny.bin",                   f"{HF_BASE}/ggml-tiny.bin",               None),
    ("tiny.en",           "ggml-tiny.en.bin",                f"{HF_BASE}/ggml-tiny.en.bin",            None),
    ("base",              "ggml-base.bin",                   f"{HF_BASE}/ggml-base.bin",               None),
    ("base.en",           "ggml-base.en.bin",                f"{HF_BASE}/ggml-base.en.bin",            None),
    ("small",             "ggml-small.bin",                  f"{HF_BASE}/ggml-small.bin",              None),
    ("small.en",          "ggml-small.en.bin",               f"{HF_BASE}/ggml-small.en.bin",           None),
    ("small.en-tdrz",     "ggml-small.en-tdrz.bin",         f"{HF_TDRZ}/ggml-small.en-tdrz.bin",     "small.en"),
    ("medium",            "ggml-medium.bin",                 f"{HF_BASE}/ggml-medium.bin",             None),
    ("medium.en",         "ggml-medium.en.bin",              f"{HF_BASE}/ggml-medium.en.bin",          None),
    ("large-v1",          "ggml-large-v1.bin",               f"{HF_BASE}/ggml-large-v1.bin",           None),
    ("large-v2",          "ggml-large-v2.bin",               f"{HF_BASE}/ggml-large-v2.bin",           None),
    ("large-v2-q5_0",     "ggml-large-v2-q5_0.bin",         f"{HF_BASE}/ggml-large-v2-q5_0.bin",     "large-v2"),
    ("large-v3",          "ggml-large-v3.bin",               f"{HF_BASE}/ggml-large-v3.bin",           None),
    ("large-v3-q5_0",     "ggml-large-v3-q5_0.bin",         f"{HF_BASE}/ggml-large-v3-q5_0.bin",     "large-v3"),
    ("large-v3-turbo",    "ggml-large-v3-turbo.bin",         f"{HF_BASE}/ggml-large-v3-turbo.bin",    None),
    ("large-v3-turbo-q5_0","ggml-large-v3-turbo-q5_0.bin", f"{HF_BASE}/ggml-large-v3-turbo-q5_0.bin","large-v3-turbo"),
]

MODEL_NAMES = [m[0] for m in MODELS]
MODEL_MAP   = {m[0]: m for m in MODELS}


# ── Helpers ───────────────────────────────────────────────────────────────────

def encoder_xml(model_name: str) -> Path:
    return HERE / f"ggml-{model_name}-encoder-openvino.xml"

def encoder_bin(model_name: str) -> Path:
    return HERE / f"ggml-{model_name}-encoder-openvino.bin"


def download(url: str, dest: Path):
    """Download url → dest, printing progress every 10%."""
    print(f"  Downloading {dest.name} ...", flush=True)
    tmp = dest.with_suffix(".tmp")
    last_pct = -1
    try:
        def _progress(block, block_size, total):
            nonlocal last_pct
            if total <= 0:
                return
            pct = min(100, block * block_size * 100 // total)
            pct = (pct // 10) * 10   # snap to nearest 10%
            if pct != last_pct:
                last_pct = pct
                mb_done = min(block * block_size, total) / 1e6
                print(f"    {pct:3d}%  ({mb_done:.0f} / {total/1e6:.0f} MB)", flush=True)
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        tmp.rename(dest)
        print(f"  Saved {dest.name}  ({dest.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {e}") from e


def convert_encoder(model_name: str):
    """Convert encoder for model_name using convert_encoder_to_openvino.py."""
    script = HERE / "convert_encoder_to_openvino.py"
    import subprocess
    cmd = [sys.executable, str(script), "--model", model_name, "--output-dir", str(HERE)]
    print(f"  Converting encoder for {model_name} ...")
    result = subprocess.run(cmd, cwd=str(HERE))
    if result.returncode != 0:
        raise RuntimeError(f"Encoder conversion failed for {model_name}")


def copy_encoder(src_name: str, dst_name: str):
    """Copy encoder files from src_name to dst_name (for quantised reuse)."""
    for ext in (".xml", ".bin"):
        src = HERE / f"ggml-{src_name}-encoder-openvino{ext}"
        dst = HERE / f"ggml-{dst_name}-encoder-openvino{ext}"
        if not src.exists():
            raise RuntimeError(f"Source encoder not found: {src.name}  "
                               f"(convert '{src_name}' first)")
        print(f"  Copying encoder {src.name} -> {dst.name}")
        shutil.copy2(src, dst)


# ── Main ──────────────────────────────────────────────────────────────────────

def process(name: str, force_download=False, force_convert=False):
    _, ggml_file, url, enc_src = MODEL_MAP[name]
    ggml_path = HERE / ggml_file

    # ── 1. Download GGML weights ──────────────────────────────────────────
    if ggml_path.exists() and not force_download:
        print(f"  [{name}] {ggml_file} already present ({ggml_path.stat().st_size/1e6:.0f} MB) – skip download")
    else:
        download(url, ggml_path)

    # ── 2. Encoder ────────────────────────────────────────────────────────
    xml = encoder_xml(name)
    if xml.exists() and not force_convert:
        print(f"  [{name}] encoder already exists – skip conversion")
        return

    if enc_src is not None:
        # Quantised / tdrz variant: reuse parent's encoder
        parent_xml = encoder_xml(enc_src)
        if not parent_xml.exists():
            print(f"  [{name}] parent encoder '{enc_src}' not yet converted – converting it first ...")
            _, _, _, parent_enc_src = MODEL_MAP[enc_src]
            if parent_enc_src is None:
                convert_encoder(enc_src)
            else:
                copy_encoder(parent_enc_src, enc_src)
        copy_encoder(enc_src, name)
    else:
        convert_encoder(name)


def main():
    parser = argparse.ArgumentParser(description="Download + convert all whisper models")
    parser.add_argument("models", nargs="*",
                        help="Model names to process (default: all). "
                             f"Choices: {', '.join(MODEL_NAMES)}")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download even if file already exists")
    parser.add_argument("--force-convert",  action="store_true",
                        help="Re-convert encoder even if .xml already exists")
    parser.add_argument("--list", action="store_true",
                        help="List all models and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        print(f"  {'Name':<24} {'GGML file':<34} {'Encoder'}")
        print(f"  {'-'*24} {'-'*34} {'-'*20}")
        for name, ggml, _, enc_src in MODELS:
            enc_note = f"copy of {enc_src}" if enc_src else "convert (fresh)"
            ggml_path = HERE / ggml
            status = f"{ggml_path.stat().st_size/1e6:.0f} MB" if ggml_path.exists() else "not downloaded"
            xml = encoder_xml(name)
            enc_status = "ready" if xml.exists() else "missing"
            print(f"  {name:<24} {status:<34} encoder: {enc_status}  ({enc_note})")
        return

    targets = args.models if args.models else MODEL_NAMES

    # Validate names
    invalid = [n for n in targets if n not in MODEL_MAP]
    if invalid:
        print(f"ERROR: unknown model(s): {invalid}")
        print(f"Valid: {MODEL_NAMES}")
        sys.exit(1)

    print(f"\nProcessing {len(targets)} model(s): {', '.join(targets)}")
    print(f"Output directory: {HERE}\n")

    failed = []
    for name in targets:
        print(f"\n{'='*60}")
        print(f"  MODEL: {name}")
        print(f"{'='*60}")
        try:
            process(name, args.force_download, args.force_convert)
            print(f"  [OK] {name} done")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append(name)

    # ── Clear ov-cache so NPU recompiles ────────────────────────────────
    ov_cache = HERE / "build" / "Release" / "ov-cache"
    if ov_cache.exists():
        shutil.rmtree(ov_cache, ignore_errors=True)
        print("\nCleared ov-cache (devices will recompile on first use).")

    print(f"\n{'='*60}")
    if failed:
        print(f"DONE with errors. Failed: {failed}")
    else:
        print(f"ALL DONE. {len(targets)} model(s) processed.")
    print()
    print("Next step — rebuild to copy new models into build/Release:")
    print("  cmake --build build --config Release")


if __name__ == "__main__":
    main()
