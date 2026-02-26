#pragma once

#include <string>
#include "../Scene.h"

using std::string;

class PngScene : public Scene {
public:
    PngScene(string pn, const double width = 1, const double height = 1);

    bool check_if_data_changed() const override;
    void mark_data_unchanged() override;
    void change_data() override;

    void draw() override;

    const StateQuery populate_state_query() const override;

private:
    string picture_name;
};
