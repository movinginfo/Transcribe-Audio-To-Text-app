# copy_encoders.cmake
# Called at BUILD time (not configure time) to copy:
#   1. ggml-*-encoder-openvino.xml/.bin  – OpenVINO encoder models
#   2. ggml-*.bin (>10 MB)               – real GGML model weights (skip test stubs)
#   3. ggml-silero-*.bin                 – Silero VAD model (if downloaded)
# from the project root into the executable output directory.
#
# Variables injected by the caller:
#   SRC_DIR  – project source root (where the .xml/.bin files are generated)
#   DST_DIR  – target output dir  (e.g. build/Release)

# ── OpenVINO encoder files ─────────────────────────────────────────────────
file(GLOB ENCODER_XML "${SRC_DIR}/ggml-*-encoder-openvino.xml")
file(GLOB ENCODER_BIN "${SRC_DIR}/ggml-*-encoder-openvino.bin")

foreach(f IN LISTS ENCODER_XML ENCODER_BIN)
    get_filename_component(fname "${f}" NAME)
    set(dst "${DST_DIR}/${fname}")
    if(NOT "${f}" IS_NEWER_THAN "${dst}")
        # already up to date – skip
    else()
        message(STATUS "Copying encoder: ${fname} -> ${DST_DIR}")
        file(COPY "${f}" DESTINATION "${DST_DIR}")
    endif()
endforeach()

# ── Silero VAD model (optional – only copied if downloaded) ───────────────
foreach(vad_name "ggml-silero-v5.1.2.bin" "ggml-silero-v6.2.0.bin")
    set(_vad_src "${SRC_DIR}/${vad_name}")
    if(EXISTS "${_vad_src}")
        set(_vad_dst "${DST_DIR}/${vad_name}")
        if(NOT "${_vad_src}" IS_NEWER_THAN "${_vad_dst}")
            # already up to date
        else()
            message(STATUS "Copying VAD model: ${vad_name} -> ${DST_DIR}")
            file(COPY "${_vad_src}" DESTINATION "${DST_DIR}")
        endif()
    endif()
endforeach()

# ── GGML model weights (skip test stubs smaller than 10 MB) ───────────────
file(GLOB GGML_MODELS "${SRC_DIR}/ggml-*.bin")
foreach(f IN LISTS GGML_MODELS)
    # skip encoder openvino files (already handled above)
    string(FIND "${f}" "-encoder-openvino" _ov_idx)
    if(NOT _ov_idx EQUAL -1)
        continue()
    endif()
    # skip test stubs (< 10 MB = 10485760 bytes)
    file(SIZE "${f}" _fsize)
    if(_fsize LESS 10485760)
        continue()
    endif()
    get_filename_component(fname "${f}" NAME)
    set(dst "${DST_DIR}/${fname}")
    if(NOT "${f}" IS_NEWER_THAN "${dst}")
        # already up to date – skip
    else()
        message(STATUS "Copying model: ${fname} -> ${DST_DIR}")
        file(COPY "${f}" DESTINATION "${DST_DIR}")
    endif()
endforeach()
