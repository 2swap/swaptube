#pragma once

#include "scene.h"
using json = nlohmann::json;

int content_index_to_json_index(int b){
    return b*2+1;
}
int json_index_to_transition_index(int b){
    return b/2;
}
int json_index_to_content_index(int b){
    return b/2;
}

template <typename T>
class SequentialScene : public Scene {
public:
    Scene* createScene(const int width, const int height, const json& scene) override {
        return new SequentialScene<T>(width, height, scene);
    }

    SequentialScene(const int width, const int height, const json& contents) : Scene(width, height, contents) {
        vector<json> sequence_json = contents["sequence"];
        for (int i = 1; i < sequence_json.size()-1; i+=2) {
            cout << "constructing sequential subscene " << i << endl;
            json curr = sequence_json[i];

            subscenes.push_back(T(width, height, curr));
        }
    }

    void add_subscene_audio(int i, const json& element_json) {
        if (element_json.find("audio") != element_json.end()) {
            string audio_path = element_json["audio"].get<string>();
            double duration_seconds = WRITER->add_audio_get_length(audio_path);
            append_duration(duration_seconds);
            WRITER->add_subtitle(duration_seconds, element_json["script"]);
        } else {
            double duration = element_json["duration_seconds"].get<double>();
            WRITER->add_silence(duration);
            append_duration(duration);
        }
    }

    void query_subscene(int json_index, double transition_fraction) {
        bool is_transition = json_index % 2 == 0;
        if(is_transition)render_transition(json_index_to_transition_index(json_index), transition_fraction);
        else render_non_transition(json_index_to_content_index(json_index));
    }

    const Pixels& query(bool& done_scene) {
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

    void render_non_transition(int index) {
        bool dont_care_if_done;
        pix = subscenes[index].query(dont_care_if_done);
    }

    void render_transition(int transition_index, double weight) {
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
            T subscene(subscenes[curr_board_index], subscenes[next_board_index], weight);
            bool dont_care_if_done;
            pix = subscene.query(dont_care_if_done);
        }
    }

private:
    vector<double> subsequence_durations_in_frames;
    int index_in_sequence = -1;
    int whole_sequence_duration_in_frames = 0;
    void append_duration(double duration_seconds){
        int duration_frames = duration_seconds * VIDEO_FRAMERATE;
        subsequence_durations_in_frames.push_back(duration_frames);
        whole_sequence_duration_in_frames += duration_frames;
    }

protected:
    vector<T> subscenes;
};

