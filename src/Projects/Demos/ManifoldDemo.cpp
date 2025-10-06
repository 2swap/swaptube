#include "../Scenes/Math/ManifoldScene.cpp"

void render_video() {
    ManifoldScene ms;

    ms.stage_macroblock(SilenceBlock(13), 12);
    ms.state_manager.set({
        {"d", "10"},
    });
    ms.state_manager.transition(MICRO, {
        {"qi", ".2"},
        {"qj", "{t} 2 / sin"},
        {"qk", ".1"},
    });
    ms.render_microblock();
    ms.render_microblock();

    ms.state_manager.set({
        {"u_wave", "5"},
        {"v_wave", "5"},
        {"mult", "5"},
        {"manifold_x", "(u)"},
        {"manifold_y", "(u) <u_wave> * sin (v) <v_wave> * sin + <mult> /"},
        {"manifold_z", "(v)"},
    });
    ms.state_manager.transition(MICRO, {
        {"u_wave", "3"},
        {"v_wave", "9"},
        {"mult", "2"},
    });
    ms.render_microblock();
    ms.render_microblock();

    // Plane in 3D space
    ms.state_manager.transition(MICRO, {
        {"manifold_x", "(u)"},
        {"manifold_y", "0"},
        {"manifold_z", "(v)"},
    });
    ms.render_microblock();
    ms.render_microblock();

    //Parameterize a sphere with spherical coordinates
    ms.state_manager.transition(MICRO, {
        {"manifold_x", "(u) cos (v) sin *"},
        {"manifold_y", "(u) sin (v) sin *"},
        {"manifold_z", "(v) cos"},
        {"u_min", "-1.57"},
        {"u_max", "1.57"},
        {"v_min", "-3.14"},
        {"v_max", "3.14"},
    });
    ms.render_microblock();
    ms.render_microblock();

    // Back to a plane
    ms.state_manager.transition(MICRO, {
        {"manifold_x", "(u)"},
        {"manifold_y", "0"},
        {"manifold_z", "(v)"},
        {"u_min", "-3.14"},
        {"u_max", "3.14"},
        {"v_min", "-3.14"},
        {"v_max", "3.14"},
    });
    ms.render_microblock();
    ms.render_microblock();

    // Torus
    ms.state_manager.transition(MICRO, {
        {"manifold_x", ".5 (u) cos * 2 + (v) cos *"},
        {"manifold_y", ".5 (u) cos * 2 + (v) sin *"},
        {"manifold_z", ".5 (u) sin *"},
        {"u_min", "-3.14"},
        {"u_max", "3.14"},
        {"v_min", "-3.14"},
        {"v_max", "3.14"},
    });
    ms.render_microblock();
    ms.render_microblock();
}
