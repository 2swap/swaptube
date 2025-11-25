#include "../Scenes/Math/ManifoldScene.cpp"

void render_video() {
    ManifoldScene ms;
    ms.add_manifold("surface",
        "(a)", "(b)", "0", "(a)", "(b)",
        "-3.14", "3.14", "3000",
        "-3.14", "3.14", "3000"
    );

    ms.stage_macroblock(SilenceBlock(13), 8);
    ms.manager.set({
        {"d", "20"},
        {"manifoldsurface_u_min", "0"},
        {"manifoldsurface_u_max", "6"},
        {"manifoldsurface_v_min", "-3.14"},
        {"manifoldsurface_v_max", "3.14"},
        // u is radius, v is angle
        {"manifoldsurface_x", "(a) (b) cos *"},
        {"manifoldsurface_y", "0"},
        {"manifoldsurface_z", "(a) (b) sin * -1 *"},
        {"manifoldsurface_r", "(b) 2 / cos (a) .5 ^ *"}, // real component is cos(angle/2) * sqrt(radius)
        {"manifoldsurface_i", "(b) 2 / sin (a) .5 ^ *"}, // imaginary component is sin(angle/2) * sqrt(radius)
    });
    ms.manager.transition(MICRO, {
        {"q1", "1"},
        {"qi", ".5"},
        {"qj", "0"},
        {"qk", "0"},
    });
    ms.render_microblock();
    ms.render_microblock();

    ms.manager.transition(MICRO, {
        {"q1", "1"},
        {"qi", ".5 .1 {t} sin 4 / * +"},
        {"qj", "0"},
        {"qk", ".1 {t} cos 4 / *"},
    });
    ms.render_microblock();
    ms.render_microblock();

    ms.manager.transition(MICRO, {
        {"manifoldsurface_y", "(b) 2 / sin (a) .5 ^ *"},
    });
    ms.render_microblock();
    ms.render_microblock();

    ms.manager.transition(MICRO, {
        {"manifoldsurface_v_min", "-6.28"},
        {"manifoldsurface_v_max", "6.28"},
        {"manifoldsurface_v_steps", "6000"},
    });
    ms.render_microblock();
    ms.render_microblock();
}
