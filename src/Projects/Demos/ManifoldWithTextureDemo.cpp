#include "../Scenes/Math/ManifoldScene.h"

void render_video() {
    ManifoldScene ms;

    ms.add_manifold("",
        "(a)", "(b) -1 *", "0",
        "(a)", "(b)",
        "-1", "1", "5000",
        "-1", "1", "5000"
    );

    stage_macroblock(SilenceBlock(15), 6);
    ms.manager.set({
        {"d", "4"},
    });

    Pixels pix;
    png_to_pix(pix, "map");
    ms.set_texture(pix);

    //Parameterize a sphere with spherical coordinates
    ms.manager.transition(MICRO, {
        {"manifold_x", "(a) cos (b) sin *"},
        {"manifold_y", "(b) cos"},
        {"manifold_z", "(a) sin (b) sin *"},
        {"manifold_a_min", "0"},
        {"manifold_a_max", "pi 2 *"},
        {"manifold_b_min", "0"},
        {"manifold_b_max", "pi"},
    });
    ms.render_microblock();
    ms.render_microblock();

    ms.manager.transition(MICRO, {
        {"q1", "{t} 2 / sin"},
        {"qi", "{t} 2 * sin .6 *"},
        {"qj", "{t} 2 / cos"},
        {"qk", "0"},
    });
    ms.render_microblock();
    ms.render_microblock();
    ms.render_microblock();
    ms.render_microblock();
}
