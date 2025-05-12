// This file introduces a bit of common boilerplate so the user doesn't need to every time.

using namespace std;

static bool FOR_REAL = true;          // Whether to actually write any AV output
static bool PRINT_TO_TERMINAL = true; // Whether to print every 5th frame for user
static bool SAVE_FRAME_PNGS = true;   // Whether to save every 30th frame to disk as PNG

#include <string>
const string project_name = PROJECT_NAME_MACRO;
const int VIDEO_FRAMERATE = 30;

const int VIDEO_BACKGROUND_COLOR = 0xff0a0a1e; // Lambda Blue
//const int VIDEO_BACKGROUND_COLOR = 0xff202020; // Charcoal Grey

#include "../io/writer.cpp"
#include "Timer.cpp"

// The go.sh script temporarily moves the current project to this location:
#include "../Projects/.active_project.cpp"
// The project file is expected to do two things:
// (1) include all relevant scenes for the video
// (2) define a function "render_video" which uses those scenes to define the video timeline.

int main() {
    if (44100%VIDEO_FRAMERATE != 0){
        throw runtime_error("Video framerate must be divisible by audio sample rate.");
    }

    // Setup Output Format Context
    AVFormatContext* fc = nullptr;
    int ret = avformat_alloc_output_context2(&fc, NULL, NULL, PATH_MANAGER.video_output.c_str());
    if(ret<0) throw runtime_error("Failed to allocate AVFormatContext: " + to_string(ret));
    if(!fc) throw runtime_error("AVFormatContext was null upon creation!");

    // Main Render Loop
    Timer timer;
    signal(SIGINT, signal_handler);
    try {
        render_video();
    } catch(std::exception& e) {
        cout << "EXCEPTION CAUGHT IN RUNTIME: " << endl;
        cout << e.what() << endl;
        cout << "Last Shtooka Entry: " << WRITER.get_last_shtooka() << endl;
    }

    // We babysit these cleanup functions instead of delegating to destructors because
    // the shared fc resource complicates things and ordering of cleanup is crucial.
    AUDIO_WRITER.cleanup();
    VIDEO_WRITER.cleanup();

    return 0;
}
