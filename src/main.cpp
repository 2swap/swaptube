#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include "misc/inlines.h"
#include "scenes/Connect4/c4.h"
#include "misc/json.hpp" // nlohmann json library
#include "misc/pixels.h"
#include "misc/convolver.cpp"
#include "misc/convolution_tests.cpp"
#include "audio_video/writer.cpp"
#include "scenes/scene.cpp"

using json = nlohmann::json;
using namespace std;

const double transition_duration_seconds = 2.;

json parse_json(string filename) {
    // Parse the json file
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        exit(1);
    }

    json video_json;
    try {
        file >> video_json;
    } catch (const exception& e) {
        cerr << "Failed to parse file: " << filename << endl;
        exit(1);
    }

    return video_json;
}

int main(int argc, char* argv[]) {
    run_inlines_unit_tests();
    run_convolution_unit_tests();
    run_c4_unit_tests();

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <config_file_without_.json>" << endl;
        exit(1);
    }

    json video = parse_json("../in/" + string(argv[1]) + ".json");

    // Extract overall information from the config
    json config = video["config"];
    int width = config["width"];
    int height = config["height"];
    int framerate = config.value("framerate", 30);
    string name = "../out/" + string(argv[1]) + ".mp4";

    cout << "Project name: " << name << endl;

    // Create the MovieWriter object
    MovieWriter writer(name, width, height, framerate, "../media/" + string(argv[1]) + "/");

    cout << "Initializing Writer" << endl;

    writer.init("../media/testaudio.mp3");

    cout << "Beginning Render" << endl;

    double time_s = 0;
    int i = 0;
    // Process each scene in the config
    for (auto& scene_json : video["scenes"]) {
        writer.set_audiotime(time_s);
        Scene* scene = create_scene_determine_type(config, scene_json, writer);
        if (scene != nullptr) {
            int frames_left = -1;
            while (frames_left != 0) {
                time_s += 1./framerate;
                Pixels p = scene->query(frames_left);
                if((++i)%5 == 0) p.print_to_terminal();
                writer.addFrame(p);
                if (frames_left == -1){
                    cout << "frames_left was not set" << endl;
                    exit(1);
                }
            }

            // clean up the dynamically allocated scene object
            delete scene;
        }
    }

    return 0;
}