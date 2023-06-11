#include "mandelbrot_scene.cpp"
#include "latex_scene.cpp"
#include "header_scene.cpp"
#include "c4_scene.cpp"
#include "twoswap_scene.cpp"
#include "composite_scene.cpp"

// Implementation of the Scene class constructor
Scene::Scene(const json& config, const json& c) : contents(c), framerate(config["framerate"].get<int>()), pix(config["width"].get<int>(), config["height"].get<int>()){
    if (contents.contains("duration_seconds"))
        scene_duration_frames = contents["duration_seconds"].get<int>() * framerate;
}

static Scene* create_scene_determine_type(const json& config, const json& scene_json) {
    string scene_type = scene_json["type"];
    cout << endl << "Creating a " << scene_type << " scene" << endl;
    if (scene_type == "latex") {
        return new LatexScene(config, scene_json);
    }
    else if (scene_type == "c4") {
        return new C4Scene(config, scene_json);
    }
    else if (scene_type == "mandelbrot") {
        return new MandelbrotScene(config, scene_json);
    }
    else if (scene_type == "header") {
        return new HeaderScene(config, scene_json);
    }
    else if (scene_type == "2swap") {
        return new TwoswapScene(config, scene_json);
    }
    else if (scene_type == "composite") {
        return new CompositeScene(config, scene_json);
    }
    else {
        cerr << "Unknown scene type: " << scene_type << endl;
        exit(1);
    }
}

