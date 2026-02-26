#pragma once
#include "../Common/CompositeScene.h"
#include "../Media/LatexScene.h"
#include <memory>
#include <vector>
#include <string>

class BarChartScene : public CompositeScene {
public:
    BarChartScene(
        const string& title,
        const vector<string>& labels,
        const double width = 1,
        const double height = 1
    );

    void change_title(const TransitionType tt, const string& new_title);

    void add_bar(const TransitionType tt, const string& label, bool should_reposition_bars = true);

    void reposition_bars(const TransitionType tt);

    void draw() override;

    const StateQuery populate_state_query() const override;

private:
    shared_ptr<LatexScene> title_scene;
    vector<shared_ptr<LatexScene>> bar_scenes;
};
