/* 
 * This file introduces a bit of common boilerplate so the user can jump straight into making
 * a project file.
 */

using namespace std;

#include "Timer.h"
#include "MicroblockPlan.h"
#include "Smoketest.h"
#include "../IO/Writer.h"
#include "../Scenes/Scene.h"
#include "State/GlobalState.h"

void render_video(); // Forward declaration, provided by the user in their project file

void parse_args(int argc, char* argv[], int& w, int& h, int& framerate, int& samplerate, bool& include_audio, bool& audio_sfx, string& microblock_plan_path, bool& record_microblock_plan) {
    cout << "Parsing command line arguments... " << endl;

    if (argc != 9) {
        throw runtime_error("Expected 8 arguments: width height framerate samplerate smoketest/render include_audio audio_sfx microblock_plan_path");
    }

    if (sscanf(argv[1], "%d", &w) != 1 || w < 1 || w > 10000) {
        throw runtime_error("Invalid width argument: " + string(argv[1]) );
    }
    set_global_state("VIDEO_WIDTH", w);
    cout << "Width: " << w << ", " << flush;

    if (sscanf(argv[2], "%d", &h) != 1 || h < 1 || h > 10000) {
        throw runtime_error("Invalid height argument: " + string(argv[2]) );
    }
    set_global_state("VIDEO_HEIGHT", h);
    cout << "Height: " << h << ", " << flush;

    if (sscanf(argv[3], "%d", &framerate) != 1 || framerate < 1 || framerate > 240) {
        throw runtime_error("Invalid framerate argument: " + string(argv[3]) );
    }
    cout << "Framerate: " << framerate << ", " << flush;

    if (sscanf(argv[4], "%d", &samplerate) != 1 || samplerate < 8000 || samplerate > 192000) {
        throw runtime_error("Invalid samplerate argument: " + string(argv[4]) );
    }
    cout << "Samplerate: " << samplerate << ", " << flush;

    string smoketest_arg = argv[5];
    if (smoketest_arg == "smoketest") {
        set_smoketest(true);
        record_microblock_plan = true;
    } else if (smoketest_arg == "render") {
        set_smoketest(false);
        record_microblock_plan = false;
    } else {
        throw runtime_error("Invalid smoketest flag argument: " + smoketest_arg);
    }
    cout << "Smoketest: " << (is_smoketest() ? "true" : "false") << ", " << flush;

    if(samplerate % framerate != 0) {
        throw runtime_error("Video framerate must be divisible by audio sample rate.");
    }

    int include_audio_i;
    if (sscanf(argv[6], "%d", &include_audio_i) != 1 || (include_audio_i != 0 && include_audio_i != 1)) {
        throw runtime_error("Invalid include audio argument: " + string(argv[6]) );
    }
    include_audio = (include_audio_i != 0);
    cout << "Include Audio: " << (include_audio ? "true" : "false") << ", " << flush;

    int audio_sfx_i;
    if (sscanf(argv[7], "%d", &audio_sfx_i) != 1 || (audio_sfx_i != 0 && audio_sfx_i != 1)) {
        throw runtime_error("Invalid audio sfx argument: " + string(argv[7]) );
    }
    audio_sfx = (audio_sfx_i != 0);
    cout << "Audio SFX: " << (audio_sfx ? "true" : "false") << ", " << flush;

    microblock_plan_path = argv[8];
    if (microblock_plan_path.empty()) {
        throw runtime_error("Microblock plan path cannot be empty");
    }
    cout << "Microblock Counts: " << microblock_plan_path << endl << endl;
}

inline void signal_handler(int signal) {
    throw runtime_error("Interrupt detected. Exiting gracefully.");
}

void setup_output_subfolders() {
    cout << "Setting up output subfolders... " << endl;
    string cmd = "mkdir -p io_out/frames io_out/data io_out/plots";
    system(cmd.c_str());
}

int main(int argc, char* argv[]) {
    int VIDEO_WIDTH, VIDEO_HEIGHT, FRAMERATE, SAMPLERATE;
    bool INCLUDE_AUDIO, AUDIO_SFX;
    bool record_microblock_plan;
    string microblock_plan_path;
    parse_args(argc, argv, VIDEO_WIDTH, VIDEO_HEIGHT, FRAMERATE, SAMPLERATE, INCLUDE_AUDIO, AUDIO_SFX, microblock_plan_path, record_microblock_plan);
    Timer timer;

    // Main Render Loop
    signal(SIGINT, signal_handler);
    try {
        setup_output_subfolders();
        init_writer(VIDEO_WIDTH, VIDEO_HEIGHT, FRAMERATE, SAMPLERATE, 0xff000044, INCLUDE_AUDIO, AUDIO_SFX);
        initialize_microblock_plan(microblock_plan_path, record_microblock_plan);
        if (is_smoketest()) {
            cout << "Smoketesting video... " << endl;
        } else {
            cout << "Rendering video... " << endl;
        }
        render_video();
        finalize_macroblock_sequence();
    } catch(std::exception& e) {
        // Change to red text
        cout << "\033[1;31m";

        cout << endl << "====================" << endl;
        cout << "EXCEPTION CAUGHT IN RUNTIME: " << endl;
        cout << e.what() << endl;
        cout << "Last written subtitle: " << get_writer().subtitle->get_last_written_subtitle() << endl;
        cout << "====================" << endl;

        // Change back to normal text
        cout << "\033[0m" << endl;
        return 1;
    }

    get_writer().destroy();

    cout << "\033[1;32m" << endl << "====================" << endl;
    cout << "Completed successfully!" << endl;
    cout << "====================" << "\033[0m" << endl << endl;
    return 0;
}
