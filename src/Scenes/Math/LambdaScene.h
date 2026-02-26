#pragma once
#include "../../DataObjects/LambdaCalculus/LambdaExpression.h"
#include "../Scene.h"
#include "../../IO/VisualMedia.h"
#include <memory>
#include <string>
#include <utility>

class LambdaScene : public Scene {
public:
    LambdaScene(const std::shared_ptr<const LambdaExpression> lambda, const double width = 1, const double height = 1);

    const StateQuery populate_state_query() const override;

    void reduce();

    void set_expression(std::shared_ptr<LambdaExpression> lambda);

    void render_diagrams();

    void set_title(std::string t);

    std::shared_ptr<LambdaExpression> get_clone();

    void on_end_transition_extra_behavior(const TransitionType tt) override;
    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;

    float get_scale(std::shared_ptr<const LambdaExpression> expr);

    void draw() override;

private:
    std::shared_ptr<LambdaExpression> le;
    std::shared_ptr<LambdaExpression> last_le;
    Pixels le_pix;
    int last_le_w = 0;
    int last_le_h = 0;
    int tick = 0;
    std::string title;
};
