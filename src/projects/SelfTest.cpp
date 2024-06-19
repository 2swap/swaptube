void render_video() {
    ComplexPlotScene coefficients(VIDEO_WIDTH/2, VIDEO_HEIGHT);
    ComplexPlotScene roots(VIDEO_WIDTH/2, VIDEO_HEIGHT);
    coefficients.add_point(0,0);
    coefficients.add_point(0,0);
    coefficients.add_point(0,0);
    coefficients.set_mode(COEFFICIENTS);
    roots.add_point(0,0);
    roots.add_point(0,0);
    roots.add_point(0,0);
    roots.set_mode(ROOTS);

    CompositeScene composite;
    composite.add_scene(&coefficients, 0, 0, .5, 1);
    composite.add_scene(&roots, .5, 0, .5, 1);

    dag.add_equation("r0", "<t> sin");
    dag.add_equation("i0", "<t> cos");
    dag.add_equation("r1", "<t> 2 * sin");
    dag.add_equation("i1", "<t> 3 * cos");
    dag.add_equation("r2", "<t> 4 * sin");
    dag.add_equation("i2", "<t> 5 * cos");
    composite.inject_audio_and_render(AudioSegment(6.28));




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
