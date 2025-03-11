#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"

void render_video() {
    TwoswapScene ts(.5, .5);
    MandelbrotScene ms;
    CompositeScene cs;
    cs.add_scene(&ms, "ms");
    cs.add_scene(&ts, "ts");
    ts.state_manager.set(unordered_map<string,string>{
        {"swaptube_opacity", "0"},
    });
    ms.state_manager.set(unordered_map<string,string>{
        {"zoom_r", ".0"},
        {"zoom_i", "0.002"},
        {"max_iterations", "100"},
        {"pixel_param_z", "1"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "0"},
        {"side_panel", "0"},
        {"point_path_r", "0"},
        {"point_path_i", "0"},
        {"point_path_length", "0"},
        {"internal_shade", "0"},
        {"gradation", "1"},
        {"seed_z_r", "0.005"},
        {"seed_z_i", ".018"},
        {"seed_c_r", "-1.8601766"},
        {"seed_c_i", ".0004338"},
        {"breath", "-25.7 3 *"},
        {"seed_x_r", "2"},
        {"seed_x_i", "0"},
    });
    ms.inject_audio_and_render(SilenceSegment(2));
}
