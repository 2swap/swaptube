#include "../Scenes/Common/CoordinateScene.cpp"

void render_video() {
    CoordinateScene cs;
    cs.state_manager.set({
        {"center_x", "0.5"},
        {"center_y", "0.5"},
        {"zoom", "10 <microblock_fraction> 2 * ^"},
    });
    cs.inject_audio_and_render(SilenceSegment(3));
    cs.state_manager.set({
        //{"center_x", "<t> sin 1 <zoom> / * 0.5 +"},
        //{"center_y", "<t> cos 1 <zoom> / * 0.5 +"},
        {"zoom", "100.01"},
    });
    cs.inject_audio_and_render(SilenceSegment(3));
}
