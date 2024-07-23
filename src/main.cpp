using namespace std;

int WIDTH_BASE = 640;
int HEIGHT_BASE = 360;
int MULT = 1;
int VIDEO_WIDTH = WIDTH_BASE*MULT;
int VIDEO_HEIGHT = HEIGHT_BASE*MULT;
int VIDEO_FRAMERATE = 30;
bool FOR_REAL = true; // Whether we should actually write any AV output
bool PRINT_TO_TERMINAL = true;
int video_num_frames = 0;

#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <fstream>

string project_name = "to_be_populated";
string media_folder = "to_be_populated";
string output_folder = "to_be_populated";

#include "misc/inlines.h"
#include "io/writer.cpp"
#include "misc/Timer.cpp"

#include "projects/.active_project.cpp"

void run_unit_tests(){
    run_inlines_unit_tests();
    run_color_unit_tests();
    test_dagger();
    run_c4_unit_tests();
}

int main(int argc, char* argv[]) {
    cout << "0" << endl;
    run_unit_tests();
    cout << "1" << endl;

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <config_file_without_.json>" << endl;
        exit(1);
    }
    cout << "2" << endl;
    project_name = string(argv[1]);
    media_folder = "../media/" + project_name + "/";
    output_folder = "../out/" + project_name + "/" + get_timestamp() + "/";

    DebugPlot time_per_frame_plot("render_time_per_frame");
    DebugPlot memutil_plot("memutil");
    DebugPlot dag_time_plot("Time-based metrics", vector<string>{"t", "transition_fraction", "subscene_transition_fraction"});

    Timer timer;
    {
        WRITER = new MovieWriter(project_name, media_folder, output_folder);
        {
            render_video(); // from the project file
        }
        delete WRITER;
    }
    timer.stop_timer();

    return 0;
}
