using namespace std;

#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <chrono> // chrono library for timing.

int VIDEO_WIDTH = 0;
int VIDEO_HEIGHT = 0;
int VIDEO_FRAMERATE = 0;
double video_time_s = 0;
int video_num_frames = 0;

#include "audio_video/writer.cpp"

MovieWriter* WRITER;

#include "misc/inlines.h"
#include "scenes/Connect4/c4.h"
#include "misc/json.hpp" // nlohmann json library
#include "misc/pixels.h"
#include "misc/convolver.cpp"
#include "misc/convolution_tests.cpp"
#include "scenes/scene.cpp"

using json = nlohmann::json;

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

void render_a_scene(Scene* scene){
    bool done_scene = false;
    while (!done_scene) {
        WRITER->set_audiotime(video_time_s);
        const Pixels& p = scene->query(done_scene);
        video_time_s += 1./VIDEO_FRAMERATE;
        if((video_num_frames++)%15 == 0) p.print_to_terminal();
        WRITER->set_audiotime(0.0);
        WRITER->addFrame(p);
    }
    
    // clean up the dynamically allocated scene object
    delete scene;
}

void render(const json& scenes){
    cout << "Beginning Render" << endl;
    // Start the timer.
    auto start = std::chrono::high_resolution_clock::now();

    // Process each scene in the config
    for (auto& scene_json : scenes) {
        if (scene_json.value("omit", false))
            continue;
        Scene* scene = create_scene_determine_type(VIDEO_WIDTH, VIDEO_HEIGHT, scene_json);
        if (scene != nullptr) {
            render_a_scene(scene);
        }
    }

    // Stop the timer.
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // Print out time stats
    double render_time_minutes = duration.count() / 60000000.0;
    double video_length_minutes = video_time_s/60.;
    cout << "Render time:  " << render_time_minutes << " minutes." << endl;
    cout << "Video length: " << video_length_minutes << " minutes." << endl;
}

void extract_config_info(const json& config) {
    VIDEO_WIDTH = config["width"];
    VIDEO_HEIGHT = config["height"];
    VIDEO_FRAMERATE = config["framerate"];
}

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

int main(int argc, char* argv[]) {
    run_unit_tests();

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <config_file_without_.json>" << endl;
        exit(1);
    }

    const string project_name = string(argv[1]);

    json video = parse_json("../in/" + project_name + ".json");

    extract_config_info(video["config"]);

    setup_writer(project_name);

    render(video["scenes"]);

    delete WRITER;

    return 0;
}