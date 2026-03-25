#include "../../Common/CoordinateScene.h"
#include "../../../Host_Device_Shared/TuringMachine.h"

void parse_tm_from_string(char* s, int num_states, int num_symbols, TuringMachine& tm);

class TuringMachineScene : public CoordinateScene {
public:
    TuringMachineScene(const TuringMachine& tm, const vec2& dimension = vec2(1, 1));

private:
    int last_iter;
    TuringMachine tm;

    void draw() override;

    const StateQuery populate_state_query() const;

    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;
};
