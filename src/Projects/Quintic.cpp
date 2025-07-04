#include "../Scenes/Math/ComplexPlotScene.cpp"

void render_video(){
    ComplexPlotScene cps(6);


    cps.stage_macroblock(SilenceBlock(8), 5);
    cps.state_manager.set({
        {"roots_or_coefficients_control", "1"},
    });
    cps.state_manager.transition(MICRO, {
        {"coefficient0_r", "8"},
    });
    cps.render_microblock();
    cps.state_manager_coefficients_to_roots();
    cps.state_manager.set({
        {"roots_or_coefficients_control", "0"},
    });

    cps.stage_swap_roots_when_in_root_mode(MICRO, "0","2");
    cps.render_microblock();
    cps.stage_swap_roots_when_in_root_mode(MICRO, "1","0");
    cps.render_microblock();
    cps.stage_swap_roots_when_in_root_mode(MICRO, "2","1");
    cps.render_microblock();
    cps.stage_swap_roots_when_in_root_mode(MICRO, "0","2");
    cps.render_microblock();
}
