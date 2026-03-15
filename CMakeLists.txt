cmake_minimum_required(VERSION 3.14)
project(SimpleWhisperApp C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────────────────────────────────────
# OpenVINO 2026.0.0  (installed via:  pip install openvino==2026.0.0)
# The cmake configs live inside the Python site-packages tree.
# ─────────────────────────────────────────────────────────────────────────────
set(OpenVINO_DIR
    "$ENV{LOCALAPPDATA}/Programs/Python/Python314/Lib/site-packages/openvino/cmake"
    CACHE PATH "OpenVINO CMake config directory")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
message(STATUS "OpenVINO ${OpenVINO_VERSION} found at ${OpenVINO_DIR}")

# Where all the OpenVINO / NPU .dll files live (needed at runtime)
set(OV_LIBS_DIR
    "$ENV{LOCALAPPDATA}/Programs/Python/Python314/Lib/site-packages/openvino/libs")

# ─────────────────────────────────────────────────────────────────────────────
# whisper.cpp  (master branch from local zip, already extracted to deps/)
# ─────────────────────────────────────────────────────────────────────────────
include(FetchContent)

# Disable extras we don't need
set(WHISPER_BUILD_TESTS     OFF CACHE BOOL "Disable whisper tests"     FORCE)
set(WHISPER_BUILD_EXAMPLES  OFF CACHE BOOL "Disable whisper examples"  FORCE)
set(BUILD_SHARED_LIBS       OFF CACHE BOOL "Build static libs"         FORCE)

# KEY: enable the OpenVINO encoder offload (CPU / GPU / NPU)
set(WHISPER_OPENVINO        ON  CACHE BOOL "Enable OpenVINO encoder"   FORCE)

FetchContent_Declare(
    whisper
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/whisper.cpp-master"
)
FetchContent_MakeAvailable(whisper)

# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────
add_executable(simple_whisper main.cpp)

target_link_libraries(simple_whisper
    PRIVATE
        whisper
        openvino::runtime   # direct link so main.cpp can use ov::Core
)

# Let main.cpp know OpenVINO is compiled in
target_compile_definitions(simple_whisper PRIVATE WHISPER_USE_OPENVINO)

# ─────────────────────────────────────────────────────────────────────────────
# Copy every OpenVINO .dll (including NPU plugin) next to the .exe
# ─────────────────────────────────────────────────────────────────────────────
file(GLOB OV_DLLS "${OV_LIBS_DIR}/*.dll")
foreach(dll IN LISTS OV_DLLS)
    add_custom_command(TARGET simple_whisper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}"
                "$<TARGET_FILE_DIR:simple_whisper>"
        COMMENT "Copying OpenVINO DLL: ${dll}"
    )
endforeach()

# ─────────────────────────────────────────────────────────────────────────────
# Auto-copy any generated OpenVINO encoder models (ggml-*-encoder-openvino.*)
# from the project root to the output directory at every build.
# Run the helper script at BUILD time so newly-converted models are picked up
# without needing to re-run cmake.
# ─────────────────────────────────────────────────────────────────────────────
add_custom_command(TARGET simple_whisper POST_BUILD
    COMMAND ${CMAKE_COMMAND}
            -DSRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
            -DDST_DIR="$<TARGET_FILE_DIR:simple_whisper>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_encoders.cmake"
    COMMENT "Syncing OpenVINO encoder models to output dir"
)
