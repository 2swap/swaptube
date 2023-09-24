#pragma once

using json = nlohmann::json;

class Scene {
public:
    Scene(const int width, const int height, const json& contents);
    virtual Scene* createScene(const int width, const int height, const json& contents) = 0;
    virtual const Pixels& query(bool& done_scene) = 0;
    virtual void update_variables(const map<string, double>& variables) {};
    void add_audio(const json& contents);
  
protected:
    Pixels pix;
    json contents;
    int w = 0;
    int h = 0;
    int time = 0;
    int scene_duration_frames = 0;
};

static Scene* create_scene_determine_type(const int width, const int height, const json& scene_json);