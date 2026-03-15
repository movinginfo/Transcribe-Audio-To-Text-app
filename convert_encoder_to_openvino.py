"""
Convert a whisper.cpp ggml model's encoder to OpenVINO IR format (.xml + .bin)
so it can be offloaded to CPU / GPU / NPU via the WHISPER_OPENVINO build.

Usage:
    python convert_encoder_to_openvino.py --model tiny.en

Requirements:
    pip install openvino==2026.0.0 openai-whisper torch

Output files (same folder as this script):
    ggml-<model>-encoder-openvino.xml
    ggml-<model>-encoder-openvino.bin
"""

import argparse
import os
import sys
import shutil
import torch

VALID_MODELS = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "large-v3-turbo",
]

def convert(model_name: str, output_dir: str):
    try:
        import whisper as openai_whisper
    except ImportError:
        print("ERROR: 'openai-whisper' not installed.")
        print("       Run:  pip install openai-whisper torch")
        sys.exit(1)

    # OpenVINO 2026 API  (openvino.runtime namespace was removed in 2026.0)
    from openvino import serialize, save_model
    from openvino.tools.ovc import convert_model

    print(f"Loading openai-whisper model: {model_name} ...")
    model = openai_whisper.load_model(model_name).cpu()
    encoder = model.encoder
    encoder.eval()

    n_mels = model.dims.n_mels
    mel_input = torch.zeros((1, n_mels, 3000))

    # ── Step 1: export to ONNX ────────────────────────────────────────────
    tmp_dir  = os.path.join(output_dir, "_ov_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    onnx_path = os.path.join(tmp_dir, "whisper_encoder.onnx")

    print("Exporting encoder to ONNX ...")
    torch.onnx.export(
        encoder,
        mel_input,
        onnx_path,
        input_names=["mel"],
        output_names=["output_features"],
        opset_version=14,
        dynamo=False,   # force legacy exporter (avoids onnxscript Unicode issues)
    )

    # ── Step 2: ONNX → OpenVINO IR (FP16) ────────────────────────────────
    xml_out = os.path.join(output_dir, f"ggml-{model_name}-encoder-openvino.xml")
    print("Converting ONNX -> OpenVINO IR (FP16 weights) ...")
    print("  compress_to_fp16=True  -- NPU/GPU native precision, ~2x smaller model")

    # compress_to_fp16=True stores all weights in FP16:
    #   * NPU runs natively in FP16 -> no runtime cast, faster inference
    #   * GPU  benefits from reduced memory bandwidth
    #   * Model file is ~2x smaller than FP32 baseline
    # Note: compress_to_fp16 is a save_model parameter in OpenVINO 2026,
    #       not a convert_model parameter.
    ov_model = convert_model(onnx_path)
    save_model(ov_model, xml_out, compress_to_fp16=True)

    # ── Cleanup ───────────────────────────────────────────────────────────
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nDone!  OpenVINO encoder (FP16) saved to:")
    print(f"  {xml_out}")
    print(f"  {xml_out.replace('.xml', '.bin')}")
    print()
    print("IMPORTANT: delete ov-cache so devices recompile with the new FP16 model:")
    print("  rmdir /s /q build\\Release\\ov-cache")
    print()
    print("Now run the transcriber with:")
    print(f"  simple_whisper.exe {model_name} audio.wav --device NPU")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert whisper encoder to OpenVINO IR")
    parser.add_argument("--model", required=True, choices=VALID_MODELS,
                        help="Whisper model name, e.g. tiny.en")
    parser.add_argument("--output-dir", default=".",
                        help="Where to write the .xml/.bin files (default: current dir)")
    args = parser.parse_args()

    convert(args.model, os.path.abspath(args.output_dir))
