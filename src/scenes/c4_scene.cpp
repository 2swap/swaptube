#pragma once

#include "scene.h"
#include "sequential_scene.cpp"
#include "Connect4/c4.h"
using json = nlohmann::json;

inline int C4_RED           = 0xffdc267f;
inline int C4_YELLOW        = 0xffffb000;
inline int C4_EMPTY         = 0xff222222;

inline int IBM_ORANGE = 0xFFFE6100;
inline int IBM_PURPLE = 0xFF785EF0;
inline int IBM_BLUE   = 0xFF648FFF;
inline int IBM_GREEN  = 0xFF5EB134;

class C4Scene : public SequentialScene {
public:
    C4Scene(const json& config, const json& contents, MovieWriter* writer);
    void render_non_transition(Pixels& p, int content_index);
    void render_transition(Pixels& p, int transition_index, double weight);
    Scene* createScene(const json& config, const json& scene, MovieWriter* writer) override {
        return new C4Scene(config, scene, writer);
    }

private:
    int time_in_this_block = 0;
    vector<Board> boards;
    vector<string> names;
};

C4Scene::C4Scene(const json& config, const json& contents, MovieWriter* writer) : SequentialScene(config, contents, writer) {
    vector<json> sequence_json = contents["sequence"];
    for (int i = 1; i < sequence_json.size()-1; i+=2) {
        cout << "constructing board " << i << endl;
        json board = sequence_json[i];

        // Concatenate annotations into a single string
        string concatenatedAnnotations = "";
        if (board.contains("annotations")) {
            vector<string> annotations = board["annotations"];
            for (const auto& annotation : annotations) {
                concatenatedAnnotations += annotation;
            }
        }

        // Concatenate highlights into a single string
        string concatenatedHighlights = "";
        if (board.contains("highlight")) {
            vector<string> highlights = board["highlight"];
            for (const auto& highlight : highlights) {
                concatenatedHighlights += highlight;
            }
        }

        boards.push_back(Board(board["representation"], concatenatedAnnotations, concatenatedHighlights));
    }
}

void C4Scene::render_non_transition(Pixels& p, int board_index) {
}

void C4Scene::render_transition(Pixels& p, int transition_index, double weight) {
    int curr_board_index = transition_index-1;
    int next_board_index = transition_index;
    int curr_board_json_index = content_index_to_json_index(curr_board_index);
    int next_board_json_index = content_index_to_json_index(next_board_index);
    time_in_this_block = 0;
    p.fill(BLACK);

    if(next_board_index == 0) {
        Pixels x = p;
        render_non_transition(x, next_board_index);
        pix.copy(x, 0, 0, 1-weight);
        return;
    }
    if(curr_board_index == boards.size() - 1) {
        Pixels x = p;
        render_non_transition(x, curr_board_index);
        pix.copy(x, 0, 0, weight);
        return;
    }

    Board transition = c4lerp((curr_board_index >= 0) ? boards[curr_board_index] : Board(""), (next_board_index < boards.size()) ? boards[next_board_index] : Board(""), weight);

    json sequence_json = contents["sequence"];
    int cap = sequence_json.size();
    double curr_spread         = (curr_board_json_index < 0    || !sequence_json[curr_board_json_index].value   ("spread", false)) ? 0 : 1;
    double next_spread         = (next_board_json_index >= cap || !sequence_json[next_board_json_index].value   ("spread", false)) ? 0 : 1;
    double curr_threat_diagram = (curr_board_json_index < 0    || !sequence_json[curr_board_json_index].contains("reduction"    )) ? 0 : 1;
    double next_threat_diagram = (next_board_json_index >= cap || !sequence_json[next_board_json_index].contains("reduction"    )) ? 0 : 1;

    double spread         = lerp(curr_spread        , next_spread        , smoother2(weight));
    double threat_diagram = lerp(curr_threat_diagram, next_threat_diagram, smoother2(weight));
    render_c4_board(pix, transition, 0, spread, threat_diagram);
    if(curr_threat_diagram == 1 && next_threat_diagram == 1){
        int diff_index = -1;
        for(int i = 0; i < sequence_json[curr_board_json_index]["reduction"].size(); i++)
            if(sequence_json[curr_board_json_index]["reduction"][i] != sequence_json[next_board_json_index]["reduction"][i]){
                diff_index = i;
                break;
            }
        render_threat_diagram(pix, sequence_json[curr_board_json_index]["reduction"], sequence_json[curr_board_json_index]["reduction_colors"], threat_diagram, diff_index, smoother2(weight), spread);
    }
    if(curr_threat_diagram == 1 && next_threat_diagram == 0){
        render_threat_diagram(pix, sequence_json[curr_board_json_index]["reduction"], sequence_json[curr_board_json_index]["reduction_colors"], threat_diagram, -1, 0, spread);
    }
    if(curr_threat_diagram == 0 && next_threat_diagram == 1){
        render_threat_diagram(pix, sequence_json[next_board_json_index]["reduction"], sequence_json[next_board_json_index]["reduction_colors"], threat_diagram, -1, 0, spread);
    }
}
