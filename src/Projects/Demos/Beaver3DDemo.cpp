#include "../Scenes/Math/Beavers/BeaverTNF3DScene.h"
#include <string>

void render_video() {
    const std::string x = "0.798901905827";
    const std::string y = "0.520142615817";
    const std::string z = "0.475562333469";
    const std::string q1 = "0.991979231542";
    const std::string qi = "0.0573778858507";
    const std::string qj = "0.110815533039";
    const std::string qk = "-0.0201221281438";

    BeaverTNF3DScene bs;
    bs.manager.set({
        {"target_x", x},
        {"target_y", y},
        {"target_z", z},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"camera_distance_exp", ".5"},
        {"camera_distance", "<camera_distance_exp> exp"},
        {"fov", ".5"},
        {"max_steps", "50"},
        {"max_tnf_depth", "15"},
    });

    stage_macroblock(SilenceBlock(5), 1);
    bs.manager.transition(MICRO, {{"q1", ".5"}, {"qi", "{t} cos"}, {"qj", "{t} sin 1 +"}, {"qk", "{t} .7 * sin"}});
    bs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    bs.manager.transition(MICRO, {{"q1", "1"}, {"qi", "0"}, {"qj", "0"}, {"qk", "0"}});
    bs.render_microblock();

    stage_macroblock(SilenceBlock(4), 1);
    bs.manager.transition(MICRO, "camera_distance_exp", "-1");
    bs.render_microblock();

    stage_macroblock(SilenceBlock(.3), 1);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    bs.manager.transition(MICRO, {{"q1", q1}, {"qi", qi}, {"qj", qj}, {"qk", qk}});
    bs.render_microblock();

    stage_macroblock(SilenceBlock(.3), 1);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(4), 1);
    bs.manager.transition(MICRO, "camera_distance_exp", "-5");
    bs.render_microblock();
}
