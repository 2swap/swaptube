#include "../../Common/CoordinateScene.h"

class BeaverGridScene : public CoordinateScene {
public:
    BeaverGridScene(const int num_states, const int num_symbols, const vec2& dimension = vec2(1, 1));

private:
    void draw() override;

    const StateQuery populate_state_query() const;

};
