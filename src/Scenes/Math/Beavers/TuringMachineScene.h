#include "../../Common/CoordinateScene.h"
#include "../../../Host_Device_Shared/TuringMachine.h"

class TuringMachineScene : public CoordinateScene {
public:
    TuringMachineScene(const TuringMachine& tm, const vec2& dimension = vec2(1, 1));

private:
    int last_iter;
    const TuringMachine tm;

    const int tape_length;
    vector<uint32_t> grid;

    int steps = 0;

    vector<uint8_t> tape;
    int head_position;
    uint32_t current_state = 0;

    void draw() override;
};
