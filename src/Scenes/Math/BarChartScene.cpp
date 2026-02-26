#include "BarChartScene.h"
#include "../../Core/State/StateManager.h"

BarChartScene::BarChartScene(
    const string& title,
    const vector<string>& labels,
    const double width,
    const double height
) : CompositeScene() {
    const string latex = "\\text{" + title + "}";
    title_scene = make_shared<LatexScene>(latex, .5, 1, 0.2);
    add_scene(title_scene, "title", .5, .1);
    manager.set({
        {"bar0", "0"},
        {"bar1", "0"},
        {"bar2", "0"},
        {"bar3", "0"},
        {"bar4", "0"},
    });
    for (const string& label : labels) {
        add_bar(MICRO, label, false);
    }
    for(int i = 0; i < bar_scenes.size(); ++i) {
        manager.set("bar" + to_string(i) + ".x", to_string(0.5 + (i - (bar_scenes.size() - 1) / 2.0) * 0.15));
    }
}

void BarChartScene::change_title(const TransitionType tt, const string& new_title) {
    title_scene->begin_latex_transition(tt, "\\text{" + new_title + "}");
}

void BarChartScene::add_bar(const TransitionType tt, const string& label, bool should_reposition_bars) {
    const string bar_latex = "\\text{" + label + "}";
    shared_ptr<LatexScene> bar_scene = make_shared<LatexScene>(bar_latex, 1, 0.5, 0.08);
    bar_scenes.push_back(bar_scene);
    add_scene_fade_in(tt, bar_scene, "bar" + to_string(bar_scenes.size() - 1), 0.15 * bar_scenes.size(), .9);
    if (should_reposition_bars) {
        reposition_bars(tt);
    }
}

void BarChartScene::reposition_bars(const TransitionType tt) {
    for(int i = 0; i < bar_scenes.size(); ++i) {
        manager.transition(tt, "bar" + to_string(i) + ".x", to_string(0.5 + (i - (bar_scenes.size() - 1) / 2.0) * 0.15));
    }
}

void BarChartScene::draw() {
    CompositeScene::draw();
    for (int i = 0; i < bar_scenes.size(); ++i) {
        double bar_height = state["bar" + to_string(i)] * pix.h * .6;
        double bar_width = 0.1 * pix.w;
        double bar_x = state["bar" + to_string(i) + ".x"] * pix.w - bar_width / 2;
        double bar_y = pix.h * .8 - bar_height;
        int alpha = state["bar" + to_string(i) + ".opacity"] * 255;
        int color = alpha << 24 | 0xffffff;
        pix.fill_rect(bar_x, bar_y, bar_width, bar_height, color);
    }
}

const StateQuery BarChartScene::populate_state_query() const {
    StateQuery sq = CompositeScene::populate_state_query();
    state_query_insert_multiple(sq, {
        "bar0", "bar1", "bar2", "bar3", "bar4",
    });
    return sq;
}
