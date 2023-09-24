#include "mandelbrot_scene.cpp"
#include "latex_scene.cpp"
#include "header_scene.cpp"
#include "c4_scene.cpp"
#include "twoswap_scene.cpp"
#include "composite_scene.cpp"
#include "variable_scene.cpp"

// Implementation of the Scene class constructor
Scene::Scene(const int width, const int height, const json& c) : w(width), h(height), contents(c), pix(width, height){
    if (contents.contains("duration_seconds"))
        scene_duration_frames = contents["duration_seconds"].get<int>() * VIDEO_FRAMERATE;
}

void Scene::add_audio(const json& contents) {
    double duration_seconds = 0;

    if (contents.find("audio") != contents.end()) {
        duration_seconds = WRITER->add_audio_get_length(contents["audio"].get<string>());
        WRITER->add_subtitle(duration_seconds, contents["script"]);
    } else {
        duration_seconds = contents["duration_seconds"].get<int>();
        WRITER->add_silence(duration_seconds);
    }
    scene_duration_frames = duration_seconds * VIDEO_FRAMERATE;
}

static Scene* create_scene_determine_type(const int width, const int height, const json& scene_json) {
    string scene_type = scene_json["type"];
    cout << endl << "Creating a " << scene_type << " scene" << endl;
    if (scene_type == "latex") {
        return new LatexScene(width, height, scene_json);
    }
    else if (scene_type == "c4") {
        return new SequentialScene<C4Scene>(width, height, scene_json);
    }
    else if (scene_type == "mandelbrot") {
        return new MandelbrotScene(width, height, scene_json);
    }
    else if (scene_type == "header") {
        return new HeaderScene(width, height, scene_json);
    }
    else if (scene_type == "2swap") {
        return new TwoswapScene(width, height, scene_json);
    }
    else if (scene_type == "composite") {
        return new CompositeScene(width, height, scene_json);
    }
    else if (scene_type == "variable") {
        return new VariableScene(width, height, scene_json);
    }
    else {
        cerr << "Unknown scene type: " << scene_type << endl;
        exit(1);
    }
}

