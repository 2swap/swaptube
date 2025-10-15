/* 
 * This file introduces a bit of common boilerplate so the user can jump straight into making
 * a project file.
 */

using namespace std;
#include <string>
#include <stdexcept>
#include <iostream>
#include <signal.h>

static bool SAVE_FRAME_PNGS = true;   // Whether to save every 30th frame to disk as PNG
static bool PRINT_TO_TERMINAL = true;

static bool FOR_REAL = true; // Flag exposed to the project definition to disable sections of video
static bool SMOKETEST= false;// Overall smoketest flag

bool rendering_on() { return FOR_REAL && !SMOKETEST; }

enum TransitionType {
    MICRO,
    MACRO
};

const string project_name = PROJECT_NAME_MACRO;

//const int VIDEO_BACKGROUND_COLOR = 0xff000020; // Lambda Blue
//const int VIDEO_BACKGROUND_COLOR = 0xff202020; // Charcoal Grey
const int VIDEO_BACKGROUND_COLOR = 0xff101010; // Charcoal Grey...er
//const int VIDEO_BACKGROUND_COLOR = 0xff000000; // Black

#include "../io/AudioWriter.cpp"
#include "../io/VideoWriter.cpp"
#include "../io/SubtitleWriter.cpp"
#include "../io/ShtookaWriter.cpp"
#include "Timer.cpp"

// Initializer struct for FORMAT_CONTEXT and dependent objects
struct FCInitializer {
    AVFormatContext* format_context;
    AudioWriter* audio_writer;
    VideoWriter* video_writer;
    Timer timer;

    FCInitializer() {
        format_context = nullptr;
        int ret = avformat_alloc_output_context2(&format_context, NULL, NULL, PATH_MANAGER.video_output.c_str());
        if (ret < 0) throw runtime_error("Failed to allocate output format context");
        if (format_context == nullptr) throw runtime_error("Failed to allocate output format context");
        audio_writer = new AudioWriter(format_context);
        cout << "Initialized AudioWriter" << endl;
        video_writer = new VideoWriter(format_context);
        cout << "Initialized VideoWriter" << endl;
    }

    ~FCInitializer() {
        // These destructors have to happen in this specific order
        cout << "Cleaning up resources" << endl;
        delete audio_writer;
        cout << "Cleaning up audio" << endl;
        delete video_writer; // This also frees the FC
        cout << "Done cleanup" << endl;
    }
};

// Initialize globals
static FCInitializer FCINIT;
static AVFormatContext* FORMAT_CONTEXT = FCINIT.format_context;
static AudioWriter& AUDIO_WRITER = *FCINIT.audio_writer;
static VideoWriter& VIDEO_WRITER = *FCINIT.video_writer;
static SubtitleWriter SUBTITLE_WRITER;
static ShtookaWriter SHTOOKA_WRITER;

inline void signal_handler(int signal) {
    throw runtime_error("Interrupt detected. Exiting gracefully.");
}

#include "../io/SubtitleWriter.cpp"

// The go.sh script temporarily moves the current project to this location:
#include "../Projects/.active_project.cpp"
// The project file is expected to do two things:
// (1) include all relevant scenes for the video
// (2) define a function "render_video" which uses those scenes to define the video timeline.

int main(int argc, char* argv[]) {
    if (argc > 1) {
        string arg1 = argv[1];
        if (arg1 == "smoketest") {
            cout << "Running in smoketest mode" << endl;
            SMOKETEST = true;
        } else if (arg1 == "no_smoketest") {
            cout << "Running in no_smoketest mode" << endl;
            SMOKETEST = false;
        } else {
            throw runtime_error("Invalid argument. Use 'smoketest' or 'no_smoketest'.");
        }
    } else {
        cout << "No argument provided, defaulting to for_real mode" << endl;
    }

    if (SAMPLERATE % FRAMERATE != 0){
        throw runtime_error("Video framerate must be divisible by audio sample rate.");
    }

    // Main Render Loop
    signal(SIGINT, signal_handler);
    try {
        render_video();
    } catch(std::exception& e) {
        // Change to red text
        cout << "\033[1;31m";

        cout << endl << "====================" << endl;
        cout << "EXCEPTION CAUGHT IN RUNTIME: " << endl;
        cout << e.what() << endl;
        cout << "Last written subtitle: " << SUBTITLE_WRITER.get_last_written_subtitle() << endl;
        cout << "====================" << endl;

        // Change back to normal text
        cout << "\033[0m" << endl;
        return 1;
    }

    cout << "\033[1;32m" << endl << "====================" << endl;
    cout << "Completed successfully!" << endl;
    cout << "====================" << "\033[0m" << endl << endl;

    return 0;
}
