#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/StateSliderScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"
#include "../Scenes/Common/ExposedPixelsScene.cpp"

void render_video() {
    CompositeScene cs;
    shared_ptr<MandelbrotScene> ms = make_shared<MandelbrotScene>();
    cs.add_scene(ms, "ms");
    unordered_map<string,string> init = {
        {"zoom_r", "2 <zoom_exp> ^"},
        {"zoom_exp", "0"},
        {"zoom_i", "0"},
        {"max_iterations", "150 2 <zoom_exp> -3 / ^ *"},
        {"seed_z_r", "0"},
        {"seed_z_i", "0"},
        {"seed_x_r", "2"},
        {"seed_x_i", "0"},
        {"seed_c_r", "0"},
        {"seed_c_i", "0"},
        {"pixel_param_z", "0"},
        {"pixel_param_x", "1"},
        {"pixel_param_c", "0"},
        {"gradation", "0"},
        {"side_panel", "1"},
        {"point_path_r", "0"},
        {"point_path_i", "0"},
        {"point_path_length", "0"},
        {"internal_shade", "0"},
    };
    cs.state_manager.set(init);
    ms->state_manager.set(unordered_map<string,string>{
        {"zoom_r", "[zoom_r]"},
        {"zoom_i", "[zoom_i]"},
        {"max_iterations", "[max_iterations]"},
        {"seed_z_r", "[seed_z_r]"},
        {"seed_z_i", "[seed_z_i]"},
        {"seed_x_r", "[seed_x_r]"},
        {"seed_x_i", "[seed_x_i]"},
        {"seed_c_r", "[seed_c_r]"},
        {"seed_c_i", "[seed_c_i]"},
        {"pixel_param_z", "[pixel_param_z]"},
        {"pixel_param_x", "[pixel_param_x]"},
        {"pixel_param_c", "[pixel_param_c]"},
        {"gradation", "[gradation]"},
        {"side_panel", "[side_panel]"},
        {"point_path_length", "[point_path_length]"},
        {"point_path_r", "[point_path_r]"},
        {"point_path_i", "[point_path_i]"},
        {"internal_shade", "[internal_shade]"},
        {"breath", "{t} 3 / sin 2 / "},
    });
    cs.state_manager.transition(MICRO, unordered_map<string,string>{
        {"max_iterations", "200"},
        {"seed_c_r", "{t} 4.1 / sin 2 *"},
        {"seed_c_i", "{t} 5.6 / cos 2 *"},
        {"seed_x_r", "2"},
        {"seed_x_i", "0"},
        {"seed_z_r", "{t} 5.2 / cos 2 *"},
        {"seed_z_i", "{t} 2.9 / cos 2 *"},
        {"zoom_exp", "2"},
    });
    cs.stage_macroblock(FileBlock("Here's a tour of the X-Set, by moving the origin around in 6-space. Enjoy!"), 1);
    cs.render_microblock();
    cs.stage_macroblock(SilenceBlock(35), 1);
    cs.render_microblock();
}
