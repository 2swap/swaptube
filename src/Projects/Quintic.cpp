#include "../Scenes/Math/ComplexPlotScene.cpp"

void render_video(){
    ComplexPlotScene cps(6);


    cps.stage_macroblock(SilenceBlock(15), 5);
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

    string r0 = cps.state_manager.get_equation("root0_r");
    string i0 = cps.state_manager.get_equation("root0_i");
    string r1 = cps.state_manager.get_equation("root1_r");
    string i1 = cps.state_manager.get_equation("root1_i");
    cps.state_manager.transition(MICRO, {
        {"root0_r", r1},
        {"root0_i", i1},
        {"root1_r", r0},
        {"root1_i", i0},
    });
    cps.render_microblock();
    cps.state_manager.transition(MICRO, {
        {"coefficient0_r", "0"},
    });
    cps.render_microblock();
    cps.render_microblock();
    cps.render_microblock();
}
