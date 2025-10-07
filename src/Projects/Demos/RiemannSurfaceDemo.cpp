#include "../Scenes/Math/ManifoldScene.cpp"

void render_video() {
    ManifoldScene ms;

    ms.stage_macroblock(SilenceBlock(13), 8);
    ms.state_manager.set({
        {"d", "20"},
        {"u_min", "0"},
        {"u_max", "6"},
        {"v_min", "-3.14"},
        {"v_max", "3.14"},
        // u is radius, v is angle
        {"manifold_x", "(u) (v) cos *"},
        {"manifold_y", "0"},
        {"manifold_z", "(u) (v) sin * -1 *"},
        {"color_r"   , "(v) 2 / cos (u) .5 ^ *"}, // real component is cos(angle/2) * sqrt(radius)
        {"color_i"   , "(v) 2 / sin (u) .5 ^ *"}, // imaginary component is sin(angle/2) * sqrt(radius)
    });
    ms.state_manager.transition(MICRO, {
        {"q1", "1"},
        {"qi", ".5"},
        {"qj", "0"},
        {"qk", "0"},
    });
    ms.render_microblock();
    ms.render_microblock();

    ms.state_manager.transition(MICRO, {
        {"q1", "1"},
        {"qi", ".5 .1 {t} sin 4 / * +"},
        {"qj", "0"},
        {"qk", ".1 {t} cos 4 / *"},
    });
    ms.render_microblock();
    ms.render_microblock();

    ms.state_manager.transition(MICRO, {
        {"manifold_y", "(v) 2 / sin (u) .5 ^ *"},
    });
    ms.render_microblock();
    ms.render_microblock();

    ms.state_manager.transition(MICRO, {
        {"v_min", "-6.28"},
        {"v_max", "6.28"},
        {"v_segs", "6000"},
    });
    ms.render_microblock();
    ms.render_microblock();
}
