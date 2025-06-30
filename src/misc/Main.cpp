/* 
 * This file introduces a bit of common boilerplate so the user can jump straight into making
 * a project file.
 */

using namespace std;

static bool SAVE_FRAME_PNGS = true;   // Whether to save every 30th frame to disk as PNG
static bool PRINT_TO_TERMINAL = true;

#include <string>
#include <stdexcept>
#include <iostream>
#include <signal.h>
const string project_name = PROJECT_NAME_MACRO;

const int VIDEO_BACKGROUND_COLOR = 0xff000020; // Lambda Blue
//const int VIDEO_BACKGROUND_COLOR = 0xff202020; // Charcoal Grey

bool FOR_REAL = !SMOKETEST;

#include "../io/PathManager.cpp"
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
        video_writer = new VideoWriter(format_context);
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

#include "../io/SubtitleWriter.cpp"

// The go.sh script temporarily moves the current project to this location:
#include "../Projects/.active_project.cpp"
// The project file is expected to do two things:
// (1) include all relevant scenes for the video
// (2) define a function "render_video" which uses those scenes to define the video timeline.

int main() {
    if (SAMPLERATE % FRAMERATE != 0){
        throw runtime_error("Video framerate must be divisible by audio sample rate.");
    }

    // Main Render Loop
    signal(SIGINT, signal_handler);
    try {
        render_video();
    } catch(std::exception& e) {
        cout << "====================" << endl;
        cout << "EXCEPTION CAUGHT IN RUNTIME: " << endl;
        cout << e.what() << endl;
    }
    cout << "Done" << endl;

    return 0;
}
