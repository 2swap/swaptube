#include "../Scenes/MusicScene.cpp"

void render_video() {
    MusicScene ms;
    vector<float> left; vector<float> right;
    ms.generate_audio(12, left, right);
    ms.stage_macroblock_and_render(GeneratedSegment(left, right));
}
