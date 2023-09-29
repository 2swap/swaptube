using namespace std;

#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <chrono> // chrono library for timing.

int WIDTH_BASE = 640;
int HEIGHT_BASE = 360;
int MULT = 3;

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
                           "../media/" + project_name + "/");
    WRITER->init("../media/testaudio.mp3");
}

void render_claimeven() {
    C4Scene c4(VIDEO_WIDTH, VIDEO_HEIGHT);
    c4.representation = "43334444343773";
    AudioSegment a("intro_1.mp3", "To the untrained eye, this connect 4 position might look unassuming.");
    c4.render(a);
    TwoswapScene swap(VIDEO_WIDTH, VIDEO_HEIGHT);
    swap.render(a);
}

void render_latex() {
    AudioSegment a("intro_1.mp3", "blabla.");
    LatexScene ls1(VIDEO_WIDTH, VIDEO_HEIGHT, "奴");
    LatexScene ls2(VIDEO_WIDTH, VIDEO_HEIGHT, "奵");
    LatexScene ls3(VIDEO_WIDTH, VIDEO_HEIGHT, "奶");
    LatexScene ls4(VIDEO_WIDTH, VIDEO_HEIGHT, "姺");
    LatexScene ls5(VIDEO_WIDTH, VIDEO_HEIGHT, "姻");
    LatexScene ls6(VIDEO_WIDTH, VIDEO_HEIGHT, "姼");
    LatexScene ls7(VIDEO_WIDTH, VIDEO_HEIGHT, "姽");
    LatexScene ls8(VIDEO_WIDTH, VIDEO_HEIGHT, "姾");
    LatexTransitionScene lst1(ls1, ls2);
    LatexTransitionScene lst2(ls2, ls3);
    LatexTransitionScene lst3(ls3, ls4);
    LatexTransitionScene lst4(ls4, ls5);
    LatexTransitionScene lst5(ls5, ls6);
    LatexTransitionScene lst6(ls6, ls7);
    LatexTransitionScene lst7(ls7, ls8);
    ls1.render(a);
    lst1.render(a);
    ls2.render(a);
    lst2.render(a);
    ls3.render(a);
    lst3.render(a);
    ls4.render(a);
    lst4.render(a);
    ls5.render(a);
    lst5.render(a);
    ls6.render(a);
    lst6.render(a);
    ls7.render(a);
    lst7.render(a);
    ls8.render(a);
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
    render_latex();
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