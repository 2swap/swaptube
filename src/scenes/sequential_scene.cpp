#pragma once

#include "scene.h"
#include "subscenes/Subscene.cpp"
using json = nlohmann::json;

class SequentialScene : public Scene {
public:
    SequentialScene(const json& config, const json& contents, MovieWriter* writer);
    Pixels query(bool& done_scene) override;
    void query_subscene(int content_index, double transition_fraction);
    void add_subscene_audio(int i, const json& element_json);
    // Pure virtual methods for rendering
    virtual Subscene* interpolate(Subscene* s1, Subscene* s2, double weight) = 0;
    void render_non_transition(int content_index);
    void render_transition(int transition_index, double weight);
    ~SequentialScene() {
        // Delete the objects pointed to by the pointers in the subscenes vector.
        for (Subscene* subscene : subscenes) {
            delete subscene;
        }
        subscenes.clear();
    }

private:
    MovieWriter* audio_writer = nullptr;
    vector<double> subsequence_durations_in_frames;
    int index_in_sequence = -1;
    int whole_sequence_duration_in_frames = 0;
    void append_duration(double duration_seconds);

protected:
    vector<Subscene*> subscenes;
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

void SequentialScene::query_subscene(int json_index, double transition_fraction) {
    bool is_transition = json_index % 2 == 0;
    if(is_transition)render_transition(json_index_to_transition_index(json_index), transition_fraction);
    else render_non_transition(json_index_to_content_index(json_index));
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

void SequentialScene::render_non_transition(int index) {
    pix = subscenes[index]->get();
}

void SequentialScene::render_transition(int transition_index, double weight) {
    int curr_board_index = transition_index-1;
    int next_board_index = transition_index;

    if(next_board_index == 0) {
        pix.fill(BLACK);
        render_non_transition(next_board_index);
        pix.mult_color(weight);
    }
    else if(curr_board_index == subscenes.size() - 1) {
        pix.fill(BLACK);
        render_non_transition(curr_board_index);
        pix.mult_color(1-weight);
    }
    else {
        pix = interpolate(subscenes[curr_board_index], subscenes[next_board_index], weight)->get();
    }
}
