#pragma once

#include "../../Scene.h"
#include "../../../Host_Device_Shared/TuringMachine.h"
#include <vector>

class BeaverIndividualScene : public Scene {
public:
    BeaverIndividualScene(const TuringMachine& tm, uint32_t* icons, ivec4& icons_whnm, int& icons_len, const vec2& dimension = vec2(1, 1));

    uint32_t transitions_to_show = 0;
    void reset(std::vector<uint32_t> start_tape, uint32_t start_state, int start_pos);
    void set_tm(TuringMachine new_tm);
    void default_everything(bool cur_tape = false, bool spacetime = false, bool table = false, bool sleep_cycle = false, bool unbind_parent_controls = false);

private:
    TuringMachine tm;

    int tape_length;
    vector<uint32_t> grid;
    uint32_t* icons;
    ivec4 icons_whnm;
    int icons_len;

    int steps = 0;

    vector<uint32_t> tape;
    int head_position;
    vector<uint32_t> head_position_history;
    uint32_t current_state = 0;
    vector<uint32_t> used_transition_history = {0};

    void draw() override;
};
