#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Common/TwoswapScene.h"
#include "../Scenes/Physics/MovingPendulumGridScene.h"
#include "../Scenes/Math/LambdaScene.h"
#include "../Scenes/Math/RootFractalScene.h"
#include "../IO/Writer.h"

void render_video() {
    CompositeScene cs;

    shared_ptr<Scene> ms = make_shared<MovingPendulumGridScene>();
    cs.add_scene(ms, "ms");
    stage_macroblock(SilenceBlock(2.5), 5);
    ms->manager.set({
        {"mode", "3"},
        {"zoom", "-.8"},
        {"physics_multiplier", "300"},
        {"rk4_step_size", ".4 30 /"},
        {"theta_or_momentum", "0"},
    });
    ms->manager.transition(MACRO, {
        {"zoom", "-1.5"},
        {"theta_or_momentum", "1.2"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    string term_string = "((\\n. (\\f. (((n (\\f. (\\n. (n (f (\\f. (\\x. ((n f) (f x))))))))) (\\x. f)) (\\x. x)))) (\\a. (\\b. (a (a (a b))))))";
    shared_ptr<LambdaExpression> term = parse_lambda_from_string(term_string);
    shared_ptr<LambdaScene> ls = make_shared<LambdaScene>(term);
    cs.add_scene_fade_in(MICRO, ls, "ls");
    cs.fade_subscene(MICRO, "ms", 0);
    ls->reduce();
    cs.render_microblock();
    cs.remove_subscene("ms");
    stage_macroblock(SilenceBlock(1.5), 6);
    while(remaining_microblocks_in_macroblock) {
        ls->reduce();
        cs.render_microblock();
    }

    stage_macroblock(SilenceBlock(.5), 2);
    shared_ptr<RootFractalScene> rfs = make_shared<RootFractalScene>();
    cs.add_scene_fade_in(MICRO, rfs, "rfs");
    rfs->manager.begin_timer("fractal_timer");
    rfs->manager.set({
        {"coefficient0_r", "2.2 <fractal_timer> - 3 * 7 - 8 / sin"},
        {"coefficient0_i", "2.2 <fractal_timer> - 3 * 7 - 9 / sin"},
        {"center_x", "-.3"},
        {"center_y", ".6"},
        {"zoom", "2.5"},
        {"terms", "20"},
        {"coefficients_opacity", "0"},
        {"ticks_opacity", "0"},
    });
    cs.fade_subscene(MACRO, "ls", 0);
    ls->reduce();
    cs.render_microblock();
    ls->reduce();
    cs.render_microblock();
    cs.remove_subscene("ls");
    stage_macroblock(SilenceBlock(1.5), 1);
    cs.render_microblock();

    shared_ptr<TwoswapScene> ts = make_shared<TwoswapScene>();
    cs.fade_subscene(MICRO, "rfs", 0);
    cs.add_scene_fade_in(MICRO, ts, "ts");
    stage_macroblock(SilenceBlock(.5), 1);
    cs.render_microblock();
    cs.remove_subscene("rfs");
    stage_macroblock(SilenceBlock(.5), 2);
    ts->manager.transition(MACRO, {
        {"2swap_effect_completion", "1"}
    });
    cs.render_microblock();
    cs.render_microblock();
    stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();
}
