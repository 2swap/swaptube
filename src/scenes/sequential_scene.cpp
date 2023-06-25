#pragma once

#include "scene.h"
using json = nlohmann::json;

class SequentialScene : public Scene {
public:
    SequentialScene(const json& config, const json& contents, MovieWriter* writer);
    Pixels query(int& frames_left) override;
    void frontload_audio(const json& contents, MovieWriter* writer);
    // Pure virtual methods for rendering
    virtual void render_non_transition(Pixels& p, int which) = 0;
    virtual void render_transition(Pixels& p, int which, double weight) = 0;

private:
    vector<double> durations;
};

SequentialScene::SequentialScene(const json& config, const json& contents, MovieWriter* writer) : Scene(config, contents, writer) {
}

void SequentialScene::frontload_audio(const json& contents, MovieWriter* writer) {
    if (writer == nullptr) return;
    vector<json> sequence_json = contents["sequence"];
    if(contents.find("audio") != contents.end()){
        cout << "This scene has a single audio for all of its subscenes." << endl;
        double duration = writer->add_audio_get_length(contents["audio"].get<string>());
        double ct = sequence_json.size();
        for (int i = 0; i < sequence_json.size(); i++) {
            json& element_json = sequence_json[i];
            if (element_json.find("duration_seconds") != element_json.end()) {
                duration -= element_json["duration_seconds"].get<double>();
                ct--;
            }
        }
        for (int i = 0; i < sequence_json.size(); i++){
            json& element_json = sequence_json[i];
            if (element_json.find("duration_seconds") != element_json.end()) {
                durations.push_back(element_json["duration_seconds"].get<double>());
            }
            else {
                durations.push_back(duration/ct);
            }
        }
    }
    else{
        cout << "This scene has a unique audio for each of its subscenes." << endl;
        for (int i = 0; i < sequence_json.size(); i++) {
            json& element_json = sequence_json[i];
            if (element_json.find("audio") != element_json.end()) {
                string audio_path = element_json["audio"].get<string>();
                durations.push_back(writer->add_audio_get_length(audio_path));
            } else {
                double duration = element_json["duration_seconds"].get<double>();
                writer->add_silence(duration);
                durations.push_back(duration);
            }
        }
    }
}

Pixels SequentialScene::query(int& frames_left) {
    vector<json> sequence_json = contents["sequence"];
    int time_left = 0;
    int time_spent = time;
    bool rendered = false;
    int content_index = -1;
    for (int i = 0; i < sequence_json.size(); i++) {
        json element_json = sequence_json[i];
        bool is_transition = element_json.find("transition") != element_json.end();

        double frames = durations[i] * framerate;
        if (rendered) time_left += frames;
        else if (time_spent < frames) {
            if(is_transition)render_transition(pix, content_index, time_spent/frames);
            else render_non_transition(pix, content_index);
            rendered = true;
            time_left += frames - time_spent;
        } else {
            time_spent -= frames;
        }
        if(is_transition) content_index += 1;
    }
    frames_left = time_left;
    time++;
    return pix;
}
