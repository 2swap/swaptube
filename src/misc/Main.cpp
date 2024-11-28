// This file introduces a bit of common boilerplate so the user doesn't need to every time.

using namespace std;

static bool FOR_REAL = true;          // Whether to actually write any AV output
static bool PRINT_TO_TERMINAL = true; // Whether to print every 5th frame for user
static bool SAVE_FRAME_PNGS = true;   // Whether to save every 30th frame to disk as PNG

#include <string>
const string project_name = PROJECT_NAME_MACRO;
const int VIDEO_FRAMERATE = 30;
#include "../io/writer.cpp"
#include "Timer.cpp"

// The go.sh script temporarily moves the current project to this location:
#include "../Projects/.active_project.cpp"
// The project file is expected to do two things:
// (1) include all relevant scenes for the video
// (2) define a function "render_video" which uses those scenes to define the video timeline.

int main() {
    Timer timer;
    signal(SIGINT, signal_handler);
    try {
        render_video();
    }
    catch(std::exception& e) {
        movecache.WriteCache();
        fhourstonesCache.WriteCache();
        cout << "EXCEPTION CAUGHT IN RUNTIME: " << endl;
        cout << e.what() << endl;
    }
    return 0;
}
