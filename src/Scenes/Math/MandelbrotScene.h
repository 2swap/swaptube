#pragma once

#include "../Common/CoordinateScene.h"

class MandelbrotScene : public CoordinateScene {
public:
    MandelbrotScene(const double width = 1, const double height = 1);
    const StateQuery populate_state_query() const override;
    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;
    void draw() override;
};
