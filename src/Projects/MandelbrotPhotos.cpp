#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/StateSliderScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"
#include "../Scenes/Common/ExposedPixelsScene.cpp"

void render_video() {
    MandelbrotScene ms;
    ms.state_manager.set(unordered_map<string,string>{
        {"zoom_r", "1"},
        {"zoom_i", "0"},
        {"max_iterations", "6"},
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
        {"seed_c_r", "-.5"},
        {"seed_c_i", "0"},
        {"breath", "0"},
        {"seed_x_r", "2"},
        {"seed_x_i", "0"},
    });
    ms.stage_macroblock(SilenceBlock(2), 3);
    ms.render_microblock();
    ms.export_png("1");
    ms.state_manager.set("max_iterations", "15");
    ms.render_microblock();
    ms.export_png("2");
    ms.state_manager.set("max_iterations", "1000");
    ms.render_microblock();
    ms.export_png("3");
}
