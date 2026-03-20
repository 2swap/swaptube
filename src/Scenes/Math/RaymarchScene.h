#pragma once

#include "../Scene.h"
#include "../../Host_Device_Shared/vec.h"
#include <complex>

class RaymarchScene : public Scene {
public:
    RaymarchScene(const vec2& dimensions = vec2(1,1));
    const StateQuery populate_state_query() const override;
    bool check_if_data_changed() const override;
    void draw() override;
    void change_data() override;
    void mark_data_unchanged() override;
};