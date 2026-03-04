#include "../Scenes/Common/TwoswapScene.h"
#include "../IO/Writer.h"

void render_video() {
    TwoswapScene ts;

    float aspect_ratio = get_video_aspect_ratio();
    bool mobile = aspect_ratio == 9.0f / 16.0f;
    bool desktop = aspect_ratio == 16.0f / 9.0f;
    if(mobile) {
        ts.manager.set("swaptube_opacity", "0");
        ts.manager.set("zoom", "2.1");
    } else if (!desktop) {
        throw std::runtime_error("Unsupported platform. Aspect ratio of " + std::to_string(aspect_ratio) + " is neither 16:9 nor 9:16.");
    }

    stage_macroblock(SilenceBlock(5), 5);

    ts.render_microblock();
    ts.manager.transition(MICRO, {
        {"2swap_effect_completion", "1"}
    });
    ts.render_microblock();
    ts.manager.transition(MICRO, {
        {"swaptube_effect_completion", "1"}
    });
    ts.render_microblock();
    ts.manager.transition(MICRO, {
        {"6884_effect_completion", "1"}
    });
    ts.render_microblock();
    ts.render_microblock();
}
