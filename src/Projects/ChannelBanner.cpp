#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"

void render_video() {
    TwoswapScene ts(.4, .4);
    MandelbrotScene ms;
    CompositeScene cs;
    cs.add_scene(&ts, "ts");
    cs.add_scene(&ms, "ms");
    ts.state_manager.set(unordered_map<string,string>{
        {"swaptube_opacity", "0"},
    });
    ms.state_manager.set(unordered_map<string,string>{
        {"zoom_r", ".0000021"},
        {"zoom_i", "0"},
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
    cs.inject_audio_and_render(SilenceSegment(2));
    /*
    Pixels *pix;
    ms.query(pix);
    pix_to_png(*pix, "mandelbrot");
    cs.query(pix);
    pix_to_png(*pix, "2swap");
    */
}
