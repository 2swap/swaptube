#include "../Scenes/Math/ManifoldScene.cpp"

void render_video() {
    ManifoldScene ms;

    ms.add_manifold("",
        "(a)", "0", "(b)",
        "(a)", "(b)",
        "-1", "1", "3000",
        "-1", "1", "3000"
    );

    ms.stage_macroblock(SilenceBlock(13), 12);
    ms.manager.set({
        {"d", "10"},
    });
    ms.manager.transition(MICRO, {
        {"qi", ".2"},
        {"qj", "{t} 2 / sin"},
        {"qk", ".1"},
    });
    ms.render_microblock();
    ms.render_microblock();

    ms.manager.set({
        {"u_wave", "3"},
        {"v_wave", "9"},
        {"mult", "2"},
    });
    ms.manager.transition(MICRO, {
        {"manifold_x", "(a) -.1 +"},
        {"manifold_y", "(a) <u_wave> * sin (b) <v_wave> * sin + <mult> /"},
        {"manifold_z", "(b)"},
    });
    ms.render_microblock();
    ms.render_microblock();

    // Plane in 3D space
    ms.manager.transition(MICRO, {
        {"manifold_x", "(a)"},
        {"manifold_y", "0"},
        {"manifold_z", "(b)"},
    });
    ms.render_microblock();
    ms.render_microblock();

    //Parameterize a sphere with spherical coordinates
    ms.manager.transition(MICRO, {
        {"manifold_x", "(a) cos (b) sin *"},
        {"manifold_y", "(a) sin (b) sin *"},
        {"manifold_z", "(b) cos"},
        {"manifold_u_min", "-1.57"},
        {"manifold_u_max", "1.57"},
        {"manifold_v_min", "-3.14"},
        {"manifold_v_max", "3.14"},
    });
    ms.render_microblock();
    ms.render_microblock();

    // Back to a plane
    ms.manager.transition(MICRO, {
        {"manifold_x", "(a)"},
        {"manifold_y", "0"},
        {"manifold_z", "(b)"},
        {"manifold_u_min", "-3.14"},
        {"manifold_u_max", "3.14"},
        {"manifold_v_min", "-3.14"},
        {"manifold_v_max", "3.14"},
    });
    ms.render_microblock();
    ms.render_microblock();

    // Torus
    ms.manager.transition(MICRO, {
        {"manifold_x", ".5 (a) cos * 2 + (b) cos *"},
        {"manifold_y", ".5 (a) cos * 2 + (b) sin *"},
        {"manifold_z", ".5 (a) sin *"},
        {"manifold_u_min", "-3.14"},
        {"manifold_u_max", "3.14"},
        {"manifold_v_min", "-3.14"},
        {"manifold_v_max", "3.14"},
    });
    ms.render_microblock();
    ms.render_microblock();
}
