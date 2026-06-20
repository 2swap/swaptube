#include "../../Common/CoordinateScene.h"
#include "../../../Host_Device_Shared/TuringMachine.h"

void parse_tm_from_string(char* s, int num_states, int num_symbols, TuringMachine& tm);

class TuringMachineScene : public CoordinateScene {
public:
    TuringMachineScene(const TuringMachine& tm, const vec2& dimension = vec2(1, 1));

private:
    int last_iter;
    const TuringMachine tm;

    const int tape_length;
    vector<uint32_t> grid;

    int steps = 0;

    vector<uint32_t> tape;
    int head_position;
    uint32_t current_state = 0;

    void draw() override;

    const StateQuery populate_state_query() const;
};
