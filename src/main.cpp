using namespace std;

#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <chrono> // chrono library for timing.

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
    ThreeDimensionScene tds;

    C4Scene c40("43334444343773");

    tds.add_surface(Surface(glm::vec3(0,0,-1),glm::vec3(-8,2,8),glm::vec3(-2,-9,0),&c40));

    VariableScene v(&tds);
    v.set_variables(std::unordered_map<std::string, std::string>{
        {"x", "t sin 30 *"},
        {"y", "t 3 * cos"},
        {"z", "t cos 30 *"},
        {"q1", "t 4 / cos"},
        {"q2", "0"},
        {"q3", "t -4 / sin"},
        {"q4", "0"}
    });
    v.inject_audio_and_render(AudioSegment(3));
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