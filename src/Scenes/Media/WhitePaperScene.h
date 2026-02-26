#pragma once
#include "../../IO/VisualMedia.h"
#include "../Scene.h"
#include <vector>
#include <string>

class WhitePaperScene : public Scene {
public:
    WhitePaperScene(const string& prefix, const string& author, const vector<int>& page_numbers, const double width = 1, const double height = 1);

    bool check_if_data_changed() const override;
    void mark_data_unchanged() override;
    void change_data() override;

    void draw() override;

    const StateQuery populate_state_query() const override;

private:
    const string prefix;
    const string author;
    const vector<int> page_numbers;
};
