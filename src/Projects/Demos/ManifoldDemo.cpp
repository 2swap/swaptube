#include "../Scenes/Math/ManifoldScene.cpp"

void render_video() {
    ManifoldScene ms;

    ms.stage_macroblock(SilenceBlock(4), 1);
    ms.state_manager.set({
        {"d", "5"},
    });
    ms.state_manager.transition(MICRO, {
        {"d", "10"},
    });
    ms.render_microblock();

    ms.stage_macroblock(SilenceBlock(4), 1);
    ms.state_manager.transition(MICRO, {
        {"qi", ".2"},
        {"qj", ".4"},
    });
    ms.render_microblock();
}
