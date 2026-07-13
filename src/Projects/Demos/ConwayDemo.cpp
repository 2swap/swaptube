#include "../Scenes/Math/ConwayScene.h"
#include "../IO/SVG.h"

void render_video() {
    ScalingParams sp(ivec2(1000, 1000));
    Pixels env = latex_to_pix("\\text{I really} \\\\\\\\ \\text{loved your} \\\\\\\\ \\text{crochet talk!}", sp);
    ConwayScene cs(ivec2(20000,20000), env);

    cs.manager.set("zoom", "-2");
    cs.manager.set("center_x", "-1000");// {t} .2 * cos 200 * +");
    cs.manager.set("center_y", "-24000");// {t} .2 * sin 200 * +");
    cs.manager.set("ticks_opacity", "0");

    stage_macroblock(SilenceBlock(10), 5);
    cs.render_microblock();
    cs.manager.transition(MICRO, "zoom", "-11");
    cs.render_microblock();
    cs.render_microblock();
    cs.manager.transition(MICRO, "zoom", "-2");
    cs.render_microblock();
    cs.render_microblock();
}
