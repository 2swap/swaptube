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
#include "misc/writer.cpp" // moviemaker-cpp video writer
#include "scenes/scene.cpp"

using json = nlohmann::json;
using namespace std;

const double transition_duration_seconds = 2.;

json parse_json(string filename) {
    // Parse the json file
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return 1;
    }

    json video_json;
    try {
        file >> video_json;
    } catch (const exception& e) {
        cerr << "Failed to parse file: " << e.what() << endl;
        return 1;
    }

    return video_json;
}

int main(int argc, char* argv[]) {
    run_inlines_unit_tests();
    run_c4_unit_tests();

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <config_file>" << endl;
        return 1;
    }

    string json_filename = argv[1];

    json video = parse_json(json_filename);

    // Extract overall information from the config
    json config = video["config"];
    int width = config["width"];
    int height = config["height"];
    int framerate = config["framerate"];
    string name = "../out/" + config["name"].get<string>();

    // Create the MovieWriter object
    MovieWriter writer(name, width, height, framerate);

    // Process each scene in the config
    for (auto& scene_json : video["scenes"]) {
        Scene* scene = create_scene_determine_type(config, scene_json);
        if (scene != nullptr) {
            int frames_left = -1;
            while (frames_left != 0) {
                Pixels result = scene->query(frames_left);
                if (frames_left == -1){
                    cout << "frames_left was not set" << endl;
                    exit(1);
                }
                writer.addFrame(&result.pixels[0]);
            }

            // clean up the dynamically allocated scene object
            delete scene;
        }

    }

    //One last black frame at the end, else adding more audio afterwards leaves an afterimage.
    Pixels pix(width, height);
    pix.fill(0);
    writer.addFrame(&pix.pixels[0]);

    return 0;
}