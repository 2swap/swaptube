#include "../Scenes/Math/ManifoldScene.cpp"

void render_video() {
    ManifoldScene ms;

    ms.stage_macroblock(SilenceBlock(13), 6);
    ms.state_manager.set({
        {"d", "10"},
        {"v_min", "-3.14"},
        {"v_max", "3.14"},
        {"u_min", "0"},
        {"u_max", "3"},
    });
    ms.state_manager.transition(MICRO, {
        {"qi", ".2"},
        {"qj", "{t} 2 / sin"},
        {"qk", ".1"},
        {"manifold_x", "(u)"},
        {"manifold_y", "0"},
        {"manifold_z", "(v)"},
    });
    ms.render_microblock();
    ms.render_microblock();

    ms.state_manager.transition(MICRO, {
        {"manifold_x", "(u) (v) sin *"},
        {"manifold_y", "(v) 2 / sin (u) .5 ^ *"},
        {"manifold_z", "(u) (v) cos *"},
    });
    ms.render_microblock();
    ms.render_microblock();

    ms.state_manager.transition(MICRO, {
        {"v_min", "-6.28"},
        {"v_max", "6.28"},
    });
    ms.render_microblock();
    ms.render_microblock();
}
