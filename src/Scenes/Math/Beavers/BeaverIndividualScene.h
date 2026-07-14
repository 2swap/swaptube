#include "../../Scene.h"
#include "../../../Host_Device_Shared/TuringMachine.h"
#include <vector>

class BeaverIndividualScene : public Scene {
public:
    BeaverIndividualScene(const TuringMachine& tm, uint32_t* icons, ivec2& icons_wh, int& icons_len, const vec2& dimension = vec2(1, 1));

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
    uint32_t current_state = 0;
    uint32_t last_state = 0;

    void draw() override;

    const StateQuery populate_state_query() const;

    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;
};

