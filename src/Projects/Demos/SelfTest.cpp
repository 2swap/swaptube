void render_complex() {
    PRINT_TO_TERMINAL = false;
    ComplexPlotScene coefficients(VIDEO_WIDTH/2, VIDEO_HEIGHT);
    ComplexPlotScene roots(VIDEO_WIDTH/2, VIDEO_HEIGHT);
    coefficients.set_mode(COEFFICIENTS);
    roots.set_mode(ROOTS);
    MouseScene mouse;

    CompositeScene composite;
    composite.add_scene(&coefficients, 0, 0, .5, 1);
    composite.add_scene(&roots, .5, 0, .5, 1);
    composite.add_scene(&mouse, 0, 0, 1, 1);

    state_manager.add_equation("mouse_x", "-50");
    state_manager.add_equation("mouse_y", "100");

    state_manager.add_equation("root_r0", "<t> sin");
    state_manager.add_equation("root_i0", "<t> cos");
    state_manager.add_equation("root_r1", "<t> 2 * sin");
    state_manager.add_equation("root_i1", "<t> 3 * cos");
    state_manager.add_equation("root_r2", "<t> 4 * sin");
    state_manager.add_equation("root_i2", "<t> 5 * cos");
    composite.stage_macroblock_and_render(AudioSegment(2));

    state_manager.add_transition("root_r0", "0");
    state_manager.add_transition("root_i0", ".5");
    state_manager.add_transition("root_r1", "1");
    state_manager.add_transition("root_i1", "-1.5");
    state_manager.add_transition("root_r2", ".5");
    state_manager.add_transition("root_i2", ".7");
    composite.stage_macroblock_and_render(AudioSegment(2));

    for(int i = 0; i < 3; i++){
        coefficients.state_manager_roots_to_coefficients();
        state_manager.add_transition("mouse_x", "<coefficient_r2_pixel>");
        state_manager.add_transition("mouse_y", "<coefficient_i2_pixel>");
        composite.stage_macroblock_and_render(AudioSegment(1));
        state_manager.add_transition("coefficient_i2", "1.2");
        state_manager.add_transition("coefficient_r2", "1.2");
        composite.stage_macroblock_and_render(AudioSegment(2));

        coefficients.state_manager_coefficients_to_roots();
        state_manager.add_transition("mouse_x", "<root_r2_pixel> "+to_string(VIDEO_WIDTH/2)+" +");
        state_manager.add_transition("mouse_y", "<root_i2_pixel>");
        composite.stage_macroblock_and_render(AudioSegment(1));
        state_manager.add_transition("root_i2", "-1.2");
        state_manager.add_transition("root_r2", "1.6");
        composite.stage_macroblock_and_render(AudioSegment(2));
    }
}

void render_3d(){
    ThreeDimensionScene tds;
    for(int i = -7; i <= 7; i+=2)
    for(int j = -7; j <= 7; j+=2)
    for(int k = -7; k <= 7; k+=2)
        tds.add_point(Point(""+i+j+k, glm::dvec3(i, j, k), OPAQUE_WHITE));

    glm::dquat q(1,0,0,0);
    state_manager.add_equation("x", "0");
    state_manager.add_equation("y", "0");
    state_manager.add_equation("z", "0");
    state_manager.add_equation("d", "0");
    state_manager.add_equation("surfaces_opacity", "1");
    state_manager.add_equation("points_opacity", "1");
    state_manager.add_equation("lines_opacity", "1");
    state_manager.add_equation("q1", std::to_string(q.w));
    state_manager.add_equation("qi", std::to_string(q.x));
    state_manager.add_equation("qj", std::to_string(q.y));
    state_manager.add_equation("qk", std::to_string(q.z));
    tds.stage_macroblock_and_render(AudioSegment(2));

    glm::dquat quats[6] = {PITCH_DOWN,PITCH_UP,YAW_RIGHT,YAW_LEFT,ROLL_CW,ROLL_CCW};
    for(glm::dquat mult : quats){
        q *= mult;
        state_manager.add_transition("q1", std::to_string(q.w));
        state_manager.add_transition("qi", std::to_string(q.x));
        state_manager.add_transition("qj", std::to_string(q.y));
        state_manager.add_transition("qk", std::to_string(q.z));
        tds.stage_macroblock_and_render(AudioSegment(2));
    }
}

void render_video() {
    render_complex();
}
