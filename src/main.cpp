using namespace std;

#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <chrono> // chrono library for timing.

int WIDTH_BASE = 640;
int HEIGHT_BASE = 360;
int MULT = 2;

int VIDEO_WIDTH = WIDTH_BASE*MULT;
int VIDEO_HEIGHT = HEIGHT_BASE*MULT;
int VIDEO_FRAMERATE = 30;
double video_time_s = 0;
int video_num_frames = 0;

#include "audio_video/AudioSegment.cpp"
#include "audio_video/writer.cpp"

MovieWriter* WRITER;

#include "misc/inlines.h"
#include "scenes/Connect4/c4.h"
#include "misc/pixels.h"
#include "misc/convolver.cpp"
#include "misc/convolution_tests.cpp"
#include "scenes/scene.cpp"

void run_unit_tests(){
    run_inlines_unit_tests();
    run_convolution_unit_tests();
    run_c4_unit_tests();
}

void setup_writer(const string& project_name){
    // Create a new MovieWriter object and assign it to the pointer
    WRITER = new MovieWriter("../out/" + project_name + ".mp4",
                             "../out/" + project_name + ".srt",
                           "../media/" + project_name + "/record_list.tsv",
                           "../media/" + project_name + "/");
    WRITER->init("../media/testaudio.mp3");
}

void render_video() {
    ComplexPlotScene coefficients(VIDEO_WIDTH/2, VIDEO_HEIGHT);
    ComplexPlotScene roots(VIDEO_WIDTH/2, VIDEO_HEIGHT);
    coefficients.add_point(0,0);
    coefficients.add_point(0,0);
    coefficients.add_point(0,0);
    coefficients.set_mode(COEFFICIENTS);
    roots.add_point(0,0);
    roots.add_point(0,0);
    roots.add_point(0,0);
    roots.set_mode(ROOTS);

    CompositeScene composite;
    composite.add_scene(&coefficients, 0, 0, 1);
    composite.add_scene(&roots, .5, 0, 1);

    VariableScene v;
    v.set_subscene(&composite);
    v.insert_variable("r0", "t sin");
    v.insert_variable("i0", "t cos");
    v.insert_variable("r1", "t 2 * sin");
    v.insert_variable("i1", "t 3 * cos");
    v.insert_variable("r2", "t 4 * sin");
    v.insert_variable("i2", "t 5 * cos");
    v.inject_audio_and_render(AudioSegment(6.28));
}

int main(int argc, char* argv[]) {
    run_unit_tests();

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <config_file_without_.json>" << endl;
        exit(1);
    }

    const string project_name = string(argv[1]);

    setup_writer(project_name);

    // Start the timer.
    auto start = std::chrono::high_resolution_clock::now();
    render_video();
    // Stop the timer.
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // Print out time stats
    double render_time_minutes = duration.count() / 60000000.0;
    double video_length_minutes = video_time_s/60.;
    cout << "Render time:  " << render_time_minutes << " minutes." << endl;
    cout << "Video length: " << video_length_minutes << " minutes." << endl;

    delete WRITER;

    return 0;
}