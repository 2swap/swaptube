#include "../Scenes/Math/ComplexArbitraryFunctionScene.cpp"

void render_video() {
    ComplexArbitraryFunctionScene scene;
    scene.stage_macroblock(SilenceBlock(10), 7);
    scene.render_microblock();
    scene.state_manager.transition(MICRO, {{"sqrt_coef", "0"}, {"sin_coef", "1"}});
    scene.render_microblock();
    scene.render_microblock();
    scene.state_manager.transition(MICRO, {{"sin_coef", "0"}, {"cos_coef", "1"}});
    scene.render_microblock();
    scene.render_microblock();
    scene.state_manager.transition(MICRO, {{"cos_coef", "0"}, {"exp_coef", "1"}});
    scene.render_microblock();
    scene.render_microblock();
}
