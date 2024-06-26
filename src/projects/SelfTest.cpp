void render_latex(){
    LatexScene latex("a=b");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_transition("a=b=c");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.inject_audio_and_render(AudioSegment(1));
}

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

    dag.add_equation("mouse_x", "-50");
    dag.add_equation("mouse_y", "100");

    dag.add_equation("root_r0", "<t> sin");
    dag.add_equation("root_i0", "<t> cos");
    dag.add_equation("root_r1", "<t> 2 * sin");
    dag.add_equation("root_i1", "<t> 3 * cos");
    dag.add_equation("root_r2", "<t> 4 * sin");
    dag.add_equation("root_i2", "<t> 5 * cos");
    composite.inject_audio_and_render(AudioSegment(2));

    dag.add_transition("root_r0", "0");
    dag.add_transition("root_i0", ".5");
    dag.add_transition("root_r1", "1");
    dag.add_transition("root_i1", "-1.5");
    dag.add_transition("root_r2", ".5");
    dag.add_transition("root_i2", ".7");
    composite.inject_audio_and_render(AudioSegment(2));

    for(int i = 0; i < 3; i++){
        coefficients.dag_roots_to_coefficients();
        dag.add_transition("mouse_x", "<coefficient_r2_pixel>");
        dag.add_transition("mouse_y", "<coefficient_i2_pixel>");
        composite.inject_audio_and_render(AudioSegment(1));
        dag.add_transition("coefficient_i2", "1.2");
        dag.add_transition("coefficient_r2", "1.2");
        composite.inject_audio_and_render(AudioSegment(2));

        coefficients.dag_coefficients_to_roots();
        dag.add_transition("mouse_x", "<root_r2_pixel> "+to_string(VIDEO_WIDTH/2)+" +");
        dag.add_transition("mouse_y", "<root_i2_pixel>");
        composite.inject_audio_and_render(AudioSegment(1));
        dag.add_transition("root_i2", "-1.2");
        dag.add_transition("root_r2", "1.6");
        composite.inject_audio_and_render(AudioSegment(2));
    }
}

void render_3d(){
    ThreeDimensionScene tds;
    for(int i = -7; i <= 7; i+=2)
    for(int j = -7; j <= 7; j+=2)
    for(int k = -7; k <= 7; k+=2)
        tds.add_point(Point(""+i+j+k, glm::vec3(i, j, k), WHITE));

    glm::quat q(1,0,0,0);
    dag.add_equation("x", "0");
    dag.add_equation("y", "0");
    dag.add_equation("z", "0");
    dag.add_equation("d", "0");
    dag.add_equation("surfaces_opacity", "1");
    dag.add_equation("points_opacity", "1");
    dag.add_equation("lines_opacity", "1");
    dag.add_equation("q1", std::to_string(q.w));
    dag.add_equation("qi", std::to_string(q.x));
    dag.add_equation("qj", std::to_string(q.y));
    dag.add_equation("qk", std::to_string(q.z));
    tds.inject_audio_and_render(AudioSegment(2));

    glm::quat quats[6] = {PITCH_DOWN,PITCH_UP,YAW_RIGHT,YAW_LEFT,ROLL_CW,ROLL_CCW};
    for(glm::quat mult : quats){
        q *= mult;
        dag.add_transition("q1", std::to_string(q.w));
        dag.add_transition("qi", std::to_string(q.x));
        dag.add_transition("qj", std::to_string(q.y));
        dag.add_transition("qk", std::to_string(q.z));
        tds.inject_audio_and_render(AudioSegment(2));
    }
}

void render_video() {
    //render_complex();
    render_latex();
}
