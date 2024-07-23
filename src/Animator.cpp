using namespace std;

#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include "misc/inlines.h"
#include "io/writer.cpp"
#include "misc/Timer.cpp"

class Animator {
private:
    const int VIDEO_WIDTH;
    const int VIDEO_HEIGHT;
    const int VIDEO_FRAMERATE;
    const string project_name;
    const string output_folder;
    const string media_folder;
    MovieWriter writer;

public:
    Animator(int w, int h, int fr, string proj)
        : VIDEO_WIDTH(w), VIDEO_HEIGHT(h), VIDEO_FRAMERATE(fr),
          project_name(proj),
          output_folder("../media/" + proj + "/"),
          media_folder("../out/" + proj + "/" + get_timestamp() + "/"),
          writer(w, h, fr, project_name, output_folder, media_folder) { }

};

