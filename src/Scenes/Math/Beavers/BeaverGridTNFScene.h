#include "../../Common/CoordinateScene.h"

class BeaverGridTNFScene : public CoordinateScene {
public:
    BeaverGridTNFScene(const vec2& dimension = vec2(1, 1));

private:
    void draw() override;

    const StateQuery populate_state_query() const;

    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;

};
