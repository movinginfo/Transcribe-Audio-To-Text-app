#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#ifdef _WIN32
#define NOMINMAX           // prevent windows.h from defining min/max macros
#define WIN32_LEAN_AND_MEAN
#include <windows.h>       // SetConsoleOutputCP / CP_UTF8
#undef min
#undef max
#endif

// ─────────────────────────────────────────────────────────────────────────────
// miniaudio – audio decoder (WAV, MP3, FLAC) + microphone capture (WASAPI)
// Already ships inside whisper.cpp deps; no extra download needed.
// Full device I/O enabled so mic capture works via ma_device_type_capture.
// ─────────────────────────────────────────────────────────────────────────────
#define MINIAUDIO_IMPLEMENTATION
#include "deps/whisper.cpp-master/examples/miniaudio.h"

#include "whisper.h"

// OpenVINO headers for NPU device query (only when compiled with OpenVINO)
#ifdef WHISPER_USE_OPENVINO
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Print a compact NPU / OpenVINO device summary
// ─────────────────────────────────────────────────────────────────────────────
#ifdef WHISPER_USE_OPENVINO
static void print_openvino_device_info(const std::string& target_device)
{
    ov::Core core;

    std::vector<std::string> available = core.get_available_devices();

    std::cout << "\n+======================================================+\n";
    std::cout <<   "|         OpenVINO 2026 -- Device Report               |\n";
    std::cout <<   "+======================================================+\n";

    std::cout << "Available devices:";
    for (const auto& d : available) std::cout << "  " << d;
    std::cout << "\n";

    // Per-device short info
    for (const auto& d : available) {
        try {
            std::string full_name = core.get_property(d, ov::device::full_name);
            std::cout << "  [" << d << "]  " << full_name << "\n";
        } catch (...) {
            std::cout << "  [" << d << "]  (no name property)\n";
        }
    }

    // NPU-specific block
    bool npu_available = (std::find(available.begin(), available.end(), "NPU") != available.end());

    if (npu_available) {
        std::cout << "\n  [*] NPU detected -- encoder can run on Neural Processing Unit\n";
        try {
            uint32_t driver = core.get_property("NPU", ov::intel_npu::driver_version);
            std::cout << "    NPU driver version : "
                      << (driver >> 16) << "."
                      << ((driver >> 8) & 0xFF) << "."
                      << (driver & 0xFF) << "\n";
        } catch (...) {}
        try {
            auto caps = core.get_property("NPU", ov::device::capabilities);
            std::cout << "    NPU capabilities   :";
            for (const auto& c : caps) std::cout << " " << c;
            std::cout << "\n";
        } catch (...) {}
    }

    // AUTO / MULTI / HETERO hints
    if (target_device == "AUTO") {
        std::cout << "\n  [*] AUTO -- OpenVINO selects the fastest available device\n";
    } else if (target_device.size() > 6 && target_device.substr(0, 6) == "MULTI:") {
        std::cout << "\n  [*] MULTI -- load-balanced: " << target_device.substr(6) << "\n";
    } else if (target_device.size() > 7 && target_device.substr(0, 7) == "HETERO:") {
        std::cout << "\n  [*] HETERO -- primary/fallback: " << target_device.substr(7) << "\n";
    } else if (!npu_available && target_device == "NPU") {
        std::cout << "\n  [!] NPU not found -- check Intel driver or NPU plugin install.\n";
        std::cout << "    Falling back to CPU for the OpenVINO encoder.\n";
    }

    std::cout << "\n  Requested device: " << target_device << "\n";
    std::cout << "+======================================================+\n\n";
}
#endif // WHISPER_USE_OPENVINO

// ─────────────────────────────────────────────────────────────────────────────
// Universal audio loader  WAV / MP3 / FLAC  ->  16 kHz mono float32
//
// miniaudio handles format detection, channel mixing, and resampling
// automatically so no hand-rolled resampler is needed.
// ─────────────────────────────────────────────────────────────────────────────
static bool load_audio(const std::string& path, std::vector<float>& pcmf32)
{
    // Request: 32-bit float, mono, 16 kHz  (whisper's native format)
    ma_decoder_config cfg = ma_decoder_config_init(
        ma_format_f32, /*channels=*/1, /*sampleRate=*/WHISPER_SAMPLE_RATE);

    ma_decoder dec;
    if (ma_decoder_init_file(path.c_str(), &cfg, &dec) != MA_SUCCESS) {
        std::cerr << "Error: cannot open " << path
                  << "\n  Supported formats: .wav  .mp3  .flac\n";
        return false;
    }

    // Read in chunks (robust for all formats, incl. VBR MP3 with unknown length)
    const ma_uint64 CHUNK = 8192;
    std::vector<float> tmp(CHUNK);
    ma_uint64 got = 0;
    do {
        ma_result res = ma_decoder_read_pcm_frames(&dec, tmp.data(), CHUNK, &got);
        pcmf32.insert(pcmf32.end(), tmp.begin(), tmp.begin() + (size_t)got);
        if (res != MA_SUCCESS) break;
    } while (got == CHUNK);

    ma_decoder_uninit(&dec);

    if (pcmf32.empty()) {
        std::cerr << "Error: no audio data read from " << path << "\n";
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Microphone capture
//
// MicState is shared between the miniaudio capture callback (runs on a
// background thread) and the main thread that joins/moves the samples.
// ─────────────────────────────────────────────────────────────────────────────
struct MicState {
    std::vector<float>  samples;
    std::mutex          mtx;
    std::atomic<bool>   active{true};
};

// miniaudio calls this on its internal capture thread for every audio block.
static void mic_callback(ma_device* pDevice, void* /*pOutput*/,
                         const void* pInput, ma_uint32 frameCount)
{
    MicState* state = reinterpret_cast<MicState*>(pDevice->pUserData);
    if (!state || !state->active.load(std::memory_order_relaxed)) return;
    const float* src = reinterpret_cast<const float*>(pInput);
    std::lock_guard<std::mutex> lock(state->mtx);
    state->samples.insert(state->samples.end(), src, src + frameCount);
}

// Records from the default microphone at 16 kHz mono float32 until the user
// presses ENTER.  Returns the captured PCM data in pcmf32.
static bool record_from_mic(std::vector<float>& pcmf32)
{
    MicState state;

    ma_device_config cfg  = ma_device_config_init(ma_device_type_capture);
    cfg.capture.format    = ma_format_f32;
    cfg.capture.channels  = 1;
    cfg.sampleRate        = WHISPER_SAMPLE_RATE;
    cfg.dataCallback      = mic_callback;
    cfg.pUserData         = &state;

    ma_device device;
    if (ma_device_init(nullptr, &cfg, &device) != MA_SUCCESS) {
        std::cerr << "Error: cannot open microphone.\n"
                  << "  Make sure a microphone is connected and permitted.\n";
        return false;
    }

    if (ma_device_start(&device) != MA_SUCCESS) {
        std::cerr << "Error: cannot start microphone capture.\n";
        ma_device_uninit(&device);
        return false;
    }

    std::cout << "\n[REC] Recording ... press ENTER to stop.\n";

    // Elapsed-time display – runs on a separate thread so it doesn't block
    // the cin.get() call below.
    std::atomic<bool> timer_done{false};
    std::thread timer_thread([&]() {
        while (!timer_done.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            size_t n;
            {
                std::lock_guard<std::mutex> lock(state.mtx);
                n = state.samples.size();
            }
            float sec = static_cast<float>(n) / WHISPER_SAMPLE_RATE;
            printf("\r  [REC]  %.1f s ...   ", sec);
            fflush(stdout);
        }
    });

    // Block until the user presses Enter
    std::cin.get();

    // Signal everything to stop
    state.active.store(false, std::memory_order_relaxed);
    timer_done.store(true,    std::memory_order_relaxed);
    timer_thread.join();

    ma_device_stop(&device);
    ma_device_uninit(&device);

    // Move the captured samples into the output vector
    {
        std::lock_guard<std::mutex> lock(state.mtx);
        pcmf32 = std::move(state.samples);
    }

    float sec = static_cast<float>(pcmf32.size()) / WHISPER_SAMPLE_RATE;
    printf("\r  Recording stopped.  %.1f s captured.           \n\n", sec);

    if (pcmf32.empty()) {
        std::cerr << "Error: no audio captured from microphone.\n";
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Model search helpers
// ─────────────────────────────────────────────────────────────────────────────
static bool file_exists(const std::string& path)
{
#ifdef _WIN32
    FILE* f = nullptr;
    fopen_s(&f, path.c_str(), "rb");   // fopen_s avoids MSVC C4996 warning
#else
    FILE* f = fopen(path.c_str(), "rb");
#endif
    if (f) { fclose(f); return true; }
    return false;
}

// Returns the directory that contains the running .exe  (Windows only).
// e.g. "C:\Work\project\build\Release\"
static std::string exe_dir()
{
#ifdef _WIN32
    char buf[MAX_PATH + 1] = {};
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string p(buf);
    size_t sep = p.find_last_of("/\\");
    return (sep != std::string::npos) ? p.substr(0, sep + 1) : "";
#else
    return "";
#endif
}

// Go up N directory levels from a path like "C:\a\b\c\" → "C:\a\" (N=2).
static std::string dir_up(const std::string& dir, int levels)
{
    std::string p = dir;
    for (int i = 0; i < levels; ++i) {
        if (!p.empty() && (p.back() == '/' || p.back() == '\\'))
            p.pop_back();                          // strip trailing slash
        size_t sep = p.find_last_of("/\\");
        if (sep == std::string::npos) return "";
        p = p.substr(0, sep + 1);                 // keep the slash
    }
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// Model short-name resolver + multi-directory search
//
//   Short names accepted (no path, no .bin):
//     "tiny"    ->  "ggml-tiny.bin"
//     "tiny.en" ->  "ggml-tiny.en.bin"
//     "base"    ->  "ggml-base.bin"
//
//   Search order (first match wins):
//     1. As-is relative to CWD           (build\Release\ggml-tiny.en.bin)
//     2. Next to the .exe                (same as CWD when run from Release\)
//     3. Project root = exe/../..        (C:\Work\project\ggml-tiny.en.bin)
//        Models can simply be placed here; no manual copying needed.
//
//   Explicit paths (containing / or \) are returned unchanged.
// ─────────────────────────────────────────────────────────────────────────────
static std::string resolve_model(const std::string& arg)
{
    // Explicit path – return as-is
    if (arg.find('/') != std::string::npos || arg.find('\\') != std::string::npos)
        return arg;

    // Build the canonical filename from a short name
    std::string fname;
    if (arg.size() >= 4 && arg.compare(arg.size() - 4, 4, ".bin") == 0)
        fname = arg;
    else if (arg.size() >= 5 && arg.compare(0, 5, "ggml-") == 0)
        fname = arg + ".bin";
    else
        fname = "ggml-" + arg + ".bin";

    // 1. CWD
    if (file_exists(fname)) return fname;

    std::string exd = exe_dir();
    if (!exd.empty()) {
        // 2. Exe directory  (usually identical to CWD, but not always)
        std::string p2 = exd + fname;
        if (file_exists(p2)) return p2;

        // 3. Project root = exe/../../   (build\Release\ → project root)
        std::string root = dir_up(exd, 2);
        if (!root.empty()) {
            std::string p3 = root + fname;
            if (file_exists(p3)) return p3;
        }
    }

    return fname;   // not found – whisper will emit a clear error
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
#ifdef _WIN32
    // Switch the Windows console to UTF-8 so any remaining Unicode prints
    // correctly, and avoids Cyrillic garbage on cp1251 / cp866 systems.
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    // Default thread count: all logical cores, capped at 8
    const int hw_threads     = static_cast<int>(std::thread::hardware_concurrency());
    const int default_threads = std::max(1, std::min(hw_threads, 8));

    // ── Argument parsing ─────────────────────────────────────────────────
    if (argc < 3) {
        std::cerr
            << "Usage: " << argv[0]
            << " <model> <audio.wav|audio.mp3|audio.flac|mic>\n"
               "         [--device AUTO|NPU|GPU|CPU|MULTI:NPU,GPU]  (default: AUTO)\n"
               "         [--language auto|en|uk|de|fr|...]           (default: auto)\n"
               "         [--beam N]     beam search width (default: 5, use 1 for greedy)\n"
               "         [--threads N]                               (default: "
            << default_threads << ")\n"
               "         [--loop N]     repeat inference N times (default: 1)\n"
               "                        use --loop 20 to see NPU in Task Manager\n"
               "         [--ov-model encoder.xml]\n"
               "\n"
               "  Model short names (no 'ggml-' prefix or '.bin' needed):\n"
               "    tiny / tiny.en / base / base.en / small / medium\n"
               "    e.g. 'tiny' -> 'ggml-tiny.bin'\n"
               "\n"
               "  Microphone:\n"
               "    Use 'mic' as the audio argument to record from the default\n"
               "    microphone.  Press ENTER to stop recording.\n"
               "\n"
               "  NOTE: '.en' models are English-ONLY.\n"
               "  For Ukrainian (uk), German (de) or auto-detect, use a\n"
               "  multilingual model: tiny / base / small / medium  (no .en)\n";
        return 1;
    }

    std::string model_arg    = argv[1];
    std::string model_path   = resolve_model(argv[1]);
    std::string audio_path   = argv[2];
    std::string ov_device    = "AUTO";   // OpenVINO picks the fastest device
    std::string ov_model_xml = "";
    std::string language     = "auto";   // whisper auto-detects the language
    int         n_threads    = default_threads;
    int         n_loops      = 1;        // repeat inference N times (benchmark / Task Manager)
    int         beam_size    = 5;        // beam search width (1 = greedy)

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--device"   || arg == "-d") && i + 1 < argc)
            ov_device = argv[++i];
        else if ((arg == "--ov-model" || arg == "-m") && i + 1 < argc)
            ov_model_xml = argv[++i];
        else if ((arg == "--language" || arg == "-l") && i + 1 < argc)
            language = argv[++i];
        else if ((arg == "--threads" || arg == "-t") && i + 1 < argc)
            n_threads = std::max(1, std::atoi(argv[++i]));
        else if ((arg == "--loop" || arg == "-n") && i + 1 < argc)
            n_loops = std::max(1, std::atoi(argv[++i]));
        else if ((arg == "--beam" || arg == "-b") && i + 1 < argc)
            beam_size = std::max(1, std::atoi(argv[++i]));
    }

    // OpenVINO device names are CASE-SENSITIVE uppercase (CPU, GPU, NPU, AUTO).
    // Auto-uppercase so --device cpu / gpu / npu all work correctly.
    std::transform(ov_device.begin(), ov_device.end(), ov_device.begin(), ::toupper);

    // ── Warn if English-only model used with non-English language ─────────
    if (language != "en" && language != "auto") {
        if (model_path.find(".en.bin") != std::string::npos) {
            std::string base = model_path;
            size_t sep = base.find_last_of("/\\");
            if (sep != std::string::npos) base = base.substr(sep + 1);
            if (base.size() >= 5 && base.compare(0, 5, "ggml-") == 0) base = base.substr(5);
            size_t e = base.find(".en");
            if (e != std::string::npos) base = base.substr(0, e);
            std::cerr << "[WARNING] '" << model_path << "' is English-only.\n"
                      << "  For language '" << language
                      << "' use multilingual model: ggml-" << base << ".bin\n\n";
        }
    }

    // ── OpenVINO device info ─────────────────────────────────────────────
#ifdef WHISPER_USE_OPENVINO
    print_openvino_device_info(ov_device);
#endif

    // ── Load / capture audio ─────────────────────────────────────────────
    std::vector<float> pcmf32;
    const bool is_mic = (audio_path == "mic");

    if (is_mic) {
        if (!record_from_mic(pcmf32))
            return 1;
    } else {
        if (!load_audio(audio_path, pcmf32))
            return 1;
    }

    float audio_sec = static_cast<float>(pcmf32.size()) / WHISPER_SAMPLE_RATE;
    if (is_mic)
        printf("Audio:  microphone  (%.1f s recorded, %zu samples)\n",
               audio_sec, pcmf32.size());
    else
        printf("Audio:  %s  (%.1f s, %zu samples)\n",
               audio_path.c_str(), audio_sec, pcmf32.size());

    // ── Init whisper context ─────────────────────────────────────────────
    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context* ctx =
        whisper_init_from_file_with_params(model_path.c_str(), cparams);

    if (!ctx) {
        std::cerr << "Error: failed to load model '" << model_path << "'\n"
                  << "  Tip: use short name, e.g.  tiny  tiny.en  base  small\n";
        return 1;
    }

    // ── Attach OpenVINO encoder (AUTO / NPU / GPU / CPU) ─────────────────
#ifdef WHISPER_USE_OPENVINO
    {
        const char* xml_path  = ov_model_xml.empty() ? nullptr : ov_model_xml.c_str();
        const char* cache_dir = "ov-cache";   // speed up 2nd+ runs

        std::cout << "Initialising OpenVINO encoder on device: "
                  << ov_device << " ...\n";

        int ret = whisper_ctx_init_openvino_encoder(
                      ctx, xml_path, ov_device.c_str(), cache_dir);
        if (ret != 0) {
            std::cerr << "  [WARNING] OpenVINO encoder init failed (ret="
                      << ret << ") - using ggml CPU fallback.\n";
        } else {
            std::cout << "  OpenVINO encoder ready.\n";
        }
    }
#endif

    // ── Explicit language detection (auto mode only) ─────────────────────
    // When language=="auto", run whisper_lang_auto_detect to show detected
    // language + confidence BEFORE transcription, and warn if confidence is low.
    std::string detected_lang_str = language;  // used in summary
    if (language == "auto") {
        int n_lang = whisper_lang_max_id() + 1;
        std::vector<float> lang_probs(n_lang, 0.0f);
        int detected_id = whisper_lang_auto_detect(ctx, 0, n_threads, lang_probs.data());

        if (detected_id >= 0) {
            const char* det = whisper_lang_str(detected_id);
            float conf = lang_probs[detected_id];
            detected_lang_str = det ? det : "?";

            // Find top-5 candidates for display
            std::vector<std::pair<float,int>> ranked;
            ranked.reserve(n_lang);
            for (int i = 0; i < n_lang; ++i)
                ranked.push_back({lang_probs[i], i});
            std::sort(ranked.begin(), ranked.end(),
                      [](const auto& a, const auto& b){ return a.first > b.first; });

            printf("Detected language: %s (%.0f%%)  --  top candidates: ",
                   detected_lang_str.c_str(), conf * 100.0f);
            for (int k = 0; k < 3 && k < (int)ranked.size(); ++k) {
                const char* s = whisper_lang_str(ranked[k].second);
                if (s) printf("%s %.0f%%  ", s, ranked[k].first * 100.0f);
            }
            printf("\n");

            if (conf < 0.50f) {
                fprintf(stderr,
                    "[WARNING] Low detection confidence (%.0f%%).\n"
                    "  If the result is wrong, specify the language explicitly, e.g.:\n"
                    "    simple_whisper.exe %s mic --device %s --language uk\n\n",
                    conf * 100.0f,
                    model_arg.c_str(), ov_device.c_str());
            }
        }
    }

    // ── Inference parameters ─────────────────────────────────────────────
    const bool use_beam = (beam_size > 1);
    struct whisper_full_params wparams =
        whisper_full_default_params(use_beam ? WHISPER_SAMPLING_BEAM_SEARCH
                                             : WHISPER_SAMPLING_GREEDY);

    wparams.print_progress   = false;
    wparams.print_special    = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = true;
    wparams.language         = language.c_str();  // "auto" = whisper detects language
    wparams.n_threads        = n_threads;

    // Beam search
    if (use_beam)
        wparams.beam_search.beam_size = beam_size;

    // Suppress non-speech tokens like [музика], [аплодисменти], etc.
    wparams.suppress_nst     = true;
    // Suppress blank-only segments
    wparams.suppress_blank   = true;

    if (n_loops > 1)
        printf("Language: %-6s  |  Beam: %d  |  Threads: %d  |  Device: %s  |  Loops: %d\n\n",
               language.c_str(), beam_size, n_threads, ov_device.c_str(), n_loops);
    else
        printf("Language: %-6s  |  Beam: %d  |  Threads: %d  |  Device: %s\n\n",
               language.c_str(), beam_size, n_threads, ov_device.c_str());

    std::cout << "Running transcription...\n\n";

    // ── Inference loop ───────────────────────────────────────────────────
    // Run n_loops times so the device (NPU/GPU) stays busy long enough to
    // appear in Windows Task Manager (which samples at ~1 second intervals).
    // Timing is accumulated across all loops; transcript is printed after
    // the final loop only.
    double total_wall_sec = 0.0;
    double total_encode_ms = 0.0;
    double total_decode_ms = 0.0;
    int    total_tokens    = 0;
    int    n_segments      = 0;

    for (int loop = 0; loop < n_loops; ++loop) {
        if (n_loops > 1)
            printf("  [loop %d/%d]\n", loop + 1, n_loops);

        auto t_start = std::chrono::steady_clock::now();

        if (whisper_full(ctx, wparams, pcmf32.data(), (int)pcmf32.size()) != 0) {
            std::cerr << "Error: transcription failed\n";
            whisper_free(ctx);
            return 1;
        }

        auto t_end = std::chrono::steady_clock::now();
        double loop_sec = std::chrono::duration<double>(t_end - t_start).count();
        total_wall_sec += loop_sec;

        // Accumulate internal timing counters
        struct whisper_timings* tim_loop = whisper_get_timings(ctx);
        if (tim_loop) {
            total_encode_ms += tim_loop->encode_ms;
            total_decode_ms += tim_loop->decode_ms;
        }

        // Count tokens from this loop
        n_segments = whisper_full_n_segments(ctx);
        int loop_tokens = 0;
        for (int i = 0; i < n_segments; ++i)
            loop_tokens += whisper_full_n_tokens(ctx, i);
        total_tokens += loop_tokens;
    }

    // ── Detected language ────────────────────────────────────────────────
    int         lang_id  = whisper_full_lang_id(ctx);
    const char* lang_str = (lang_id >= 0) ? whisper_lang_str(lang_id) : detected_lang_str.c_str();

    // ── Print transcript (from the last loop) ────────────────────────────
    printf("\n");
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        int64_t t0 = whisper_full_get_segment_t0(ctx, i);
        int64_t t1 = whisper_full_get_segment_t1(ctx, i);

        printf("[%02d:%02d.%03d --> %02d:%02d.%03d]  %s\n",
               (int)(t0 / 6000), (int)((t0 / 100) % 60), (int)(t0 % 100) * 10,
               (int)(t1 / 6000), (int)((t1 / 100) % 60), (int)(t1 % 100) * 10,
               text);
    }

    // ── Performance summary ───────────────────────────────────────────────
    // Average per-run values for multi-loop mode (first run includes warm-up
    // overhead, subsequent runs are steady-state; average is shown for all).
    double avg_wall_sec   = total_wall_sec   / n_loops;
    double avg_encode_ms  = total_encode_ms  / n_loops;
    double avg_decode_ms  = total_decode_ms  / n_loops;
    int    tokens_per_run = total_tokens     / n_loops;

    double tok_per_sec = (avg_wall_sec > 0.0) ? tokens_per_run / avg_wall_sec : 0.0;
    double rtf         = (avg_wall_sec > 0.0) ? audio_sec      / avg_wall_sec : 0.0;

    printf("\n+======================================================+\n");
    printf("|              Performance Summary                     |\n");
    printf("+======================================================+\n");

#ifdef WHISPER_USE_OPENVINO
    printf("|  OpenVINO device  : %-32s|\n", ov_device.c_str());
#else
    printf("|  Backend          : %-32s|\n", "ggml CPU");
#endif

    printf("|  Language         : %-32s|\n", lang_str);
    printf("|  Threads (decode) : %-32d|\n", n_threads);
    if (n_loops > 1)
        printf("|  Loops            : %-32d|\n", n_loops);
    printf("|  Segments         : %-32d|\n", n_segments);
    printf("|  Tokens / run     : %-32d|\n", tokens_per_run);
    printf("|  Wall time (avg)  : %-28.2f s  |\n", avg_wall_sec);
    printf("|  Encode (avg)     : %-28.1f ms |\n", avg_encode_ms);
    printf("|  Decode (avg)     : %-28.1f ms |\n", avg_decode_ms);
    printf("|  Speed (avg)      : %-25.1f tok/s  |\n", tok_per_sec);
    printf("|  RT factor (avg)  : %-25.2f x      |\n", rtf);

    printf("+======================================================+\n");

    whisper_free(ctx);
    return 0;
}
