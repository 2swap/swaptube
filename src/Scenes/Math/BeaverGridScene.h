#include "../Common/CoordinateScene.h"

class BeaverGridScene : public CoordinateScene {
public:
    BeaverGridScene(const int num_states, const int num_symbols, const vec2& dimension = vec2(1, 1));

private:
    int num_states;
    int num_symbols;

    void draw() override;

    const StateQuery populate_state_query() const;

    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;

};
