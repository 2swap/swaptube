#include "../Scenes/Common/CoordinateScene.cpp"

void render_video() {
    CoordinateScene cs;
    cs.state_manager.set({
        {"center_x", "0.5"},
        {"center_y", "0.5"},
        {"zoom", "10 <microblock_fraction> 2 * ^"},
    });
    cs.inject_audio_and_render(SilenceSegment(6));
    cs.state_manager.microblock_transition({
        {"center_x", "<t> sin <zoom> * 0.5 +"},
        {"center_y", "<t> cos <zoom> * 0.5 +"},
    });
    cs.inject_audio_and_render(SilenceSegment(3));
}
