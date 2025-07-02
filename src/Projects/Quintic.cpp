#include "../Scenes/Math/ComplexPlotScene.cpp"

void render_video(){
    ComplexPlotScene cps;
    StateSet woo{
        {"root0_r", "1"},
        {"root0_i", "<t> .2 * cos"},
        {"root1_r", "1"},
        {"root1_i", "<t> .2 * cos"},
    };
    cps.state_manager.transition(MICRO, woo);
    cps.stage_macroblock(SilenceBlock(3), 1);
    cps.render_microblock();
}
