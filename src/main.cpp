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
double video_time_s = 0;
int video_num_frames = 0;

#include "audio_video/AudioSegment.cpp"
#include "audio_video/writer.cpp"
#include "misc/inlines.h"
#include "scenes/Connect4/c4.h"
#include "misc/pixels.h"
#include "misc/convolver.cpp"
#include "misc/convolution_tests.cpp"
#include "scenes/scene.cpp"
#include "misc/Timer.cpp"

void setup_writer(const string& project_name){
    cout << "Setting up static writer" << endl;
    // Create a new MovieWriter object and assign it to the pointer
    WRITER = new MovieWriter(project_name);
    cout << "Initializing static writer" << endl;
    WRITER->init("../media/testaudio.mp3");
    cout << "Static writer ready" << endl;
}

#include "examples/C4_Manual_Tree.cpp"


void run_unit_tests(){
    run_inlines_unit_tests();
    run_convolution_unit_tests();
    run_c4_unit_tests();
}

int main(int argc, char* argv[]) {
    run_unit_tests();

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <config_file_without_.json>" << endl;
        exit(1);
    }
    const string project_name = string(argv[1]);

    Timer timer;
    {
        setup_writer(project_name);
        {
            render_video();
        }
        cout << "Deleting the writer" << endl;
        delete WRITER;
        cout << "Deleted the writer" << endl;
    }
    timer.stop_timer(video_time_s);
    cout << "Done" << endl;

    return 0;
}