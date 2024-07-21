using namespace std;

#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <fstream>

int WIDTH_BASE = 640;
int HEIGHT_BASE = 360;
int MULT = 1;
int VIDEO_WIDTH = WIDTH_BASE*MULT;
int VIDEO_HEIGHT = HEIGHT_BASE*MULT;
int VIDEO_FRAMERATE = 30;
bool FOR_REAL = true; // Whether we should actually write any AV output
bool PRINT_TO_TERMINAL = true;
int video_num_frames = 0;
string project_name = "to_be_populated";

#include "misc/inlines.h"
#include "misc/color.cpp"
#include "misc/dagger.cpp"
#include "audio_video/AudioSegment.cpp"
#include "audio_video/writer.cpp"
#include "scenes/Connect4/c4.h"
#include "misc/pixels.h"
#include "misc/convolver.cpp"
#include "misc/convolution_tests.cpp"
#include "misc/Timer.cpp"

void setup_writer(const string& project_name){
    // Create a new MovieWriter object and assign it to the pointer
    WRITER = new MovieWriter(project_name);
    WRITER->init("../media/testaudio.mp3");
}

#include "projects/.active_project.cpp"

void run_unit_tests(){
    run_inlines_unit_tests();
    run_color_unit_tests();
    test_dagger();
    run_convolution_unit_tests();
    run_c4_unit_tests();
}

int main(int argc, char* argv[]) {
    run_unit_tests();

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <config_file_without_.json>" << endl;
        exit(1);
    }
    project_name = string(argv[1]);

    Timer timer;
    {
        setup_writer(project_name);
        {
            render_video(); // from the project file
        }
        delete WRITER;
    }
    timer.stop_timer(dag["t"]);

    return 0;
}
