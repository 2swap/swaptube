#pragma once

#include "../Scene.h"

class PauseScene : public Scene {
public:
    PauseScene(const vec2& dimensions);

    void draw() override;

    const StateQuery populate_state_query() const override;
    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;
};
