/* 
 * This file introduces a bit of common boilerplate so the user can jump straight into making
 * a project file.
 */

using namespace std;
#include <string>
#include <stdexcept>
#include <iostream>
#include <signal.h>

static int VIDEO_WIDTH;
static int VIDEO_HEIGHT;
static int FRAMERATE;
static int SAMPLERATE;

static bool SAVE_FRAME_PNGS = true;   // Whether to save every keyframe as a PNG for debugging
static bool PRINT_TO_TERMINAL = true;

static bool FOR_REAL = true; // Flag exposed to the project definition to disable sections of video
static bool SMOKETEST= false;// Overall smoketest flag
static bool AVOID_CUDA=false;// Indicates not to use CUDA to add video background

bool rendering_on() { return FOR_REAL && !SMOKETEST; }

enum TransitionType {
    MICRO,
    MACRO
};

const string project_name = PROJECT_NAME_MACRO;

int VIDEO_BACKGROUND_COLOR = 0xff000000;

#include "../io/AudioWriter.cpp"
#include "../io/VideoWriter.cpp"
#include "../io/SubtitleWriter.cpp"
#include "../io/ShtookaWriter.cpp"
#include "Timer.cpp"

// Initialize globals
static AVFormatContext* FORMAT_CONTEXT = nullptr;
static AudioWriter* AUDIO_WRITER = nullptr;
static VideoWriter* VIDEO_WRITER = nullptr;
static SubtitleWriter* SUBTITLE_WRITER = nullptr;
static ShtookaWriter* SHTOOKA_WRITER = nullptr;

// The go.sh script temporarily moves the current project to this location:
#include "../Projects/.active_project.cpp"
// The project file is expected to do two things:
// (1) include all relevant scenes for the video
// (2) define a function "render_video" which uses those scenes to define the video timeline.

void parse_args(int argc, char* argv[], int& w, int& h, int& framerate, int& samplerate, bool& smoketest) {
    if (argc != 6) {
        throw runtime_error("Expected 5 arguments: width height framerate samplerate smoketest_flag");
    }

    if (sscanf(argv[1], "%d", &w) != 1 || w < 1 || w > 10000) {
        throw runtime_error("Invalid width argument: " + string(argv[1]) );
    }

    if (sscanf(argv[2], "%d", &h) != 1 || h < 1 || h > 10000) {
        throw runtime_error("Invalid height argument: " + string(argv[2]) );
    }

    if (sscanf(argv[3], "%d", &framerate) != 1 || framerate < 1 || framerate > 240) {
        throw runtime_error("Invalid framerate argument: " + string(argv[3]) );
    }

    if (sscanf(argv[4], "%d", &samplerate) != 1 || samplerate < 8000 || samplerate > 192000) {
        throw runtime_error("Invalid samplerate argument: " + string(argv[4]) );
    }

    string smoketest_arg = argv[5];
    if (smoketest_arg == "smoketest") {
        smoketest = true;
    } else if (smoketest_arg == "render") {
        smoketest = false;
    } else {
        throw runtime_error("Invalid smoketest flag argument: " + smoketest_arg);
    }

    if(SAMPLERATE % FRAMERATE != 0){
        throw runtime_error("Video framerate must be divisible by audio sample rate.");
    }
}

void initialize_rendering_environment() {
    cout << "Initializing rendering environment... " << flush;
    int ret = avformat_alloc_output_context2(&FORMAT_CONTEXT, NULL, NULL, PATH_MANAGER.video_output.c_str());
    if (ret < 0) throw runtime_error("Failed to allocate output format context");
    if (FORMAT_CONTEXT == nullptr) throw runtime_error("Failed to allocate output format context");
    AUDIO_WRITER = new AudioWriter(FORMAT_CONTEXT);
    VIDEO_WRITER = new VideoWriter(FORMAT_CONTEXT);
    SUBTITLE_WRITER = new SubtitleWriter();
    SHTOOKA_WRITER = new ShtookaWriter();
    cout << "Done!" << endl;
}
void finalize_rendering_environment() {
    delete AUDIO_WRITER;
    delete VIDEO_WRITER; // This also finalizes FORMAT_CONTEXT
    delete SUBTITLE_WRITER;
    delete SHTOOKA_WRITER;
}

inline void signal_handler(int signal) {
    throw runtime_error("Interrupt detected. Exiting gracefully.");
}

int main(int argc, char* argv[]) {
    parse_args(argc, argv, VIDEO_WIDTH, VIDEO_HEIGHT, FRAMERATE, SAMPLERATE, SMOKETEST);
    cout << "Rendering video with resolution " << VIDEO_WIDTH << "x" << VIDEO_HEIGHT 
         << " at " << FRAMERATE << " FPS and " << SAMPLERATE << " Hz audio." << endl;

    Timer timer;

    // Main Render Loop
    signal(SIGINT, signal_handler);
    try {
        initialize_rendering_environment();
        render_video();
    } catch(std::exception& e) {
        // Change to red text
        cout << "\033[1;31m";

        cout << endl << "====================" << endl;
        cout << "EXCEPTION CAUGHT IN RUNTIME: " << endl;
        cout << e.what() << endl;
        cout << "Last written subtitle: " << SUBTITLE_WRITER->get_last_written_subtitle() << endl;
        cout << "====================" << endl;

        // Change back to normal text
        cout << "\033[0m" << endl;
        finalize_rendering_environment();
        return 1;
    }

    cout << "\033[1;32m" << endl << "====================" << endl;
    cout << "Completed successfully!" << endl;
    cout << "====================" << "\033[0m" << endl << endl;
    finalize_rendering_environment();
    return 0;
}
