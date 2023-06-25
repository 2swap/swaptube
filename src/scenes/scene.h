#pragma once

using json = nlohmann::json;

class Scene {
public:
    Scene(const json& config, const json& contents, MovieWriter* writer);
    virtual Scene* createScene(const json& config, const json& contents, MovieWriter* writer) = 0;
    virtual Pixels query(int& frames_left) = 0;
    virtual void update_variables(const map<string, double>& variables) {};
    void frontload_audio(const json& contents, MovieWriter* writer);
  
protected:
    Pixels pix;
    json contents;
    int framerate = 0;
    int time = 0;
    int scene_duration_frames = 0;
};

static Scene* create_scene_determine_type(const json& config, const json& scene_json, MovieWriter* writer);