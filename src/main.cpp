#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <chrono> // chrono library for timing.
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
    } catch (const json::parse_error& e) {
        cerr << "Parse error at position " << e.byte << ": " << e.what() << endl;
        exit(1);
    } catch (const exception& e) {
        cerr << "Failed to parse file: " << filename << endl;
        exit(1);
    }

    return video_json;
}

int main(int argc, char* argv[]) {
    // Start the timer.
    auto start = std::chrono::high_resolution_clock::now();

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
    int framerate = config["framerate"];
    string name = "../out/" + string(argv[1]);

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
        if (scene_json.value("omit", false))
            continue;
        Scene* scene = create_scene_determine_type(config, scene_json, &writer);
        if (scene != nullptr) {
            bool done_scene = false;
            while (!done_scene) {
                writer.set_audiotime(time_s);
                Pixels p = scene->query(done_scene);
                time_s += 1./framerate;
                if((i++)%15 == 0) p.print_to_terminal();
                writer.set_audiotime(0.0);
                writer.addFrame(p);
            }
            cout << "d" << endl;

            // clean up the dynamically allocated scene object
            delete scene;
            cout << "d" << endl;
        }
    }

    // Stop the timer.
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // Print out time stats
    double render_time_minutes = duration.count() / 60000000.0;
    double video_length_minutes = time_s/60.;
    cout << "Program execution time: " << render_time_minutes << " minutes." << endl;
    cout << "Video length          : " << video_length_minutes << " minutes." << endl;

    return 0;
}