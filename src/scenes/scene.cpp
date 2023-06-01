#pragma once

using json = nlohmann::json;

class Scene {
public:
    Scene(const json& config, const json& contents);
    virtual ~Scene() = default;
    virtual Pixels query(int& frames_left) = 0;
  
protected:
    Pixels pix;
    json contents;
    int framerate = 0;
    int time = 0;
    int scene_duration_frames = 0;
};

// Implementation of the Scene class constructor
Scene::Scene(const json& config, const json& c) : contents(c), framerate(config["framerate"].get<int>()), pix(config["width"].get<int>(), config["height"].get<int>()){
    if (contents.contains("duration_seconds"))
        scene_duration_frames = contents["duration_seconds"].get<int>() * framerate;
}

// Implementation of the query function for the Scene class
Pixels Scene::query(int& frames_left) {
    // the caller errors on this
    frames_left = -1;
    // Do nothing in the base class
    return pix;
}