#pragma once

#include "scene.h"
using json = nlohmann::json;

class SequentialScene : public Scene {
public:
    SequentialScene(const json& config, const json& contents, MovieWriter* writer);
    Pixels query(bool& done_scene) override;
    void query_subscene(int content_index, double transition_fraction);
    void add_subscene_audio(int i, const json& element_json);
    // Pure virtual methods for rendering
    virtual void render_non_transition(Pixels& p, int content_index) = 0;
    virtual void render_transition(Pixels& p, int transition_index, double weight) = 0;

private:
    MovieWriter* audio_writer = nullptr;
    vector<double> subsequence_durations_in_frames;
    int index_in_sequence = -1;
    int whole_sequence_duration_in_frames = 0;
    void append_duration(double duration_seconds);
};

int content_index_to_json_index(int b){
    return b*2+1;
}
int json_index_to_transition_index(int b){
    return b/2;
}
int json_index_to_content_index(int b){
    return b/2;
}

SequentialScene::SequentialScene(const json& config, const json& contents, MovieWriter* writer) : Scene(config, contents, writer) {
    audio_writer = writer;
}

void SequentialScene::append_duration(double duration_seconds){
    int duration_frames = duration_seconds * framerate;
    subsequence_durations_in_frames.push_back(duration_frames);
    whole_sequence_duration_in_frames += duration_frames;
}

void SequentialScene::add_subscene_audio(int i, const json& element_json) {
    if (audio_writer == nullptr) return;
    if (element_json.find("audio") != element_json.end()) {
        string audio_path = element_json["audio"].get<string>();
        append_duration(audio_writer->add_audio_get_length(audio_path));
    } else {
        double duration = element_json["duration_seconds"].get<double>();
        audio_writer->add_silence(duration);
        append_duration(duration);
    }
}

/* deprecated
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
                append_duration(element_json["duration_seconds"].get<double>());
            }
            else {
                append_duration(duration/ct);
            }
        }
    } else {
        cout << "This scene has a unique audio for each of its subscenes." << endl;
        for (int i = 0; i < sequence_json.size(); i++) {
            json& element_json = sequence_json[i];
            if (element_json.find("audio") != element_json.end()) {
                string audio_path = element_json["audio"].get<string>();
                append_duration(writer->add_audio_get_length(audio_path));
            } else {
                double duration = element_json["duration_seconds"].get<double>();
                writer->add_silence(duration);
                append_duration(duration);
            }
        }
    }
}*/

void SequentialScene::query_subscene(int json_index, double transition_fraction) {
    bool is_transition = json_index % 2 == 0;
    if(is_transition)render_transition(pix, json_index_to_transition_index(json_index), transition_fraction);
    else render_non_transition(pix, json_index_to_content_index(json_index));
}

Pixels SequentialScene::query(bool& done_scene) {
    done_scene = true;
    vector<json> sequence_json = contents["sequence"];
    int frames_spent = time;
    for (int i = 0; i < sequence_json.size(); i++) {
        if(i > index_in_sequence){
            add_subscene_audio(i, contents["sequence"][i]);
            index_in_sequence = i;
        }
        json element_json = sequence_json[i];

        int frames_in_this_subscene = subsequence_durations_in_frames[i];
        if (frames_spent < frames_in_this_subscene) {
            query_subscene(i, static_cast<double>(frames_spent)/frames_in_this_subscene);
            done_scene = false;
            break;
        }
        frames_spent -= frames_in_this_subscene;
    }
    time++;
    return pix;
}
