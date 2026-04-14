#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Math/MandelbrotScene.h"
#include "../Scenes/Media/LatexScene.h"

void render_video() {
    shared_ptr<LatexScene> ts = make_shared<LatexScene>("\\text{2swap}", .3);
    shared_ptr<MandelbrotScene> ms = make_shared<MandelbrotScene>();
    CompositeScene cs;
    cs.add_scene(ms, "ms");
    cs.add_scene(ts, "ts");
    ts->manager.set(unordered_map<string,string>{
        {"swaptube_opacity", "0"},
    });
    ms->manager.set({
        {"zoom", "13.073"},
        {"max_iterations", "180"},
        {"pixel_param_z", "0"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "1"},
        {"side_panel", "0"},
        {"point_path_r", "0"},
        {"point_path_i", "0"},
        {"point_path_length", "0"},
        {"internal_shade", "0"},
        {"gradation", "1"},
        {"seed_z_r", "0"},
        {"seed_z_i", "0"},
        {"seed_c_r", "-0.1535638"},
        {"seed_c_i", "-1.0304198"},
        {"breath", "75"},
        {"seed_x_r", "2"},
        {"seed_x_i", "0"},
    });
    stage_macroblock(SilenceBlock(2), 1);
    cs.render_microblock();

    Pixels *pix;
    ms->query(pix);
    pix_to_png(*pix, "mandelbrot");
    cs.query(pix);
    pix_to_png(*pix, "2swap");
}
