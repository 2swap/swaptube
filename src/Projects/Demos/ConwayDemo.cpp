#include "../Scenes/Math/ConwayScene.h"
#include "../IO/SVG.h"

void render_video() {
    ScalingParams sp(ivec2(1000, 1000));
    Pixels env = latex_to_pix("\\text{No, I just} \\\\\\\\ \\text{manually typed} \\\\\\\\ \\text{every bit in the} \\\\\\\\ \\text{encoded mp4 output.}", sp);
    ConwayScene cs(ivec2(20000,20000), env);

    cs.manager.set("zoom", "-2");
    cs.manager.set("center_x", "22000");// {t} .2 * cos 200 * +");
    cs.manager.set("center_y", "-9000");// {t} .2 * sin 200 * +");
    cs.manager.set("ticks_opacity", "1");

    stage_macroblock(SilenceBlock(4), 5);
    cs.render_microblock();
    cs.manager.transition(MICRO, "zoom", "-1");
    cs.render_microblock();
    cs.render_microblock();
    cs.manager.transition(MICRO, "zoom", "-2");
    cs.render_microblock();
    cs.render_microblock();
}
