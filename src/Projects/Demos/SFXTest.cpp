#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../io/SFX.cpp"

void render_video() {
    MandelbrotScene ms;
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

    vector<float> left; vector<float> right;
    FourierSound fs1;
    FourierSound fs2;
    fs1.load("A");
    fs2.load("E");
    generate_audio(.5, left, right, fs1, fs2);
    ms.inject_audio_and_render(GeneratedSegment(left, right));
    ms.inject_audio_and_render(GeneratedSegment(left, right));
    ms.inject_audio_and_render(GeneratedSegment(left, right));
}
