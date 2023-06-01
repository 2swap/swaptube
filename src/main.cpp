#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include "misc/inlines.h"
#include "scenes/Connect4/c4.h"
#include "misc/json.hpp" // nlohmann json library
#include "misc/pixels.h"
#include "misc/writer.cpp" // moviemaker-cpp video writer


// Scenes
#include "scenes/scene.cpp"
#include "scenes/mandelbrot_scene.cpp"
#include "scenes/latex_scene.cpp"
#include "scenes/header_scene.cpp"
#include "scenes/c4_scene.cpp"
#include "scenes/twoswap_scene.cpp"

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
    for (auto& scene : video["scenes"]) {
        string scene_type = scene["type"];
        cout << "Rendering a " << scene_type << " scene" << endl;
        Scene* sceneObject = nullptr;

        if (scene_type == "latex") {
            sceneObject = new LatexScene(config, scene);
        }
        else if (scene_type == "c4") {
            sceneObject = new C4Scene(config, scene);
        }
        else if (scene_type == "mandelbrot") {
            sceneObject = new MandelbrotScene(config, scene);
        }
        else if (scene_type == "header") {
            sceneObject = new HeaderScene(config, scene);
        }
        else if (scene_type == "2swap") {
            sceneObject = new TwoswapScene(config, scene);
        }
        else {
            cerr << "Unknown scene type: " << scene_type << endl;
        }
        if (sceneObject != nullptr) {
            int frames_left = -1;
            while (frames_left != 0) {
                Pixels result = sceneObject->query(frames_left);
                if (frames_left == -1){
                    cout << "frames_left was not set" << endl;
                    exit(1);
                }
                writer.addFrame(&result.pixels[0]);
            }

            // Don't forget to clean up the dynamically allocated scene object
            delete sceneObject;
        }

    }

    //One last black frame at the end, else adding more audio afterwards leaves an afterimage.
    Pixels pix(width, height);
    pix.fill(0);
    writer.addFrame(&pix.pixels[0]);

    return 0;
}