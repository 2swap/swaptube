#include "../../Scene.h"
#include "../../../Host_Device_Shared/TuringMachine.h"
#include <vector>

class BeaverIndividualScene : public Scene {
public:
    BeaverIndividualScene(const TuringMachine& tm, uint32_t* icons, ivec2& icons_wh, int& icons_len, const vec2& dimension = vec2(1, 1));

    uint32_t transitions_to_show = 0;

private:
    int last_iter;
    const TuringMachine tm;

    const int tape_length;
    vector<uint32_t> grid;
    uint32_t* icons;
    ivec2 icons_wh;
    int icons_len;

    int steps = 0;

    vector<uint32_t> tape;
    int head_position;
    vector<uint32_t> head_position_history;
    uint32_t current_state = 0;
    vector<uint32_t> used_transition_history = {0};

    void draw() override;
};
