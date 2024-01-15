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
    composite.add_scene(&coefficients, 0, 0, 1);
    composite.add_scene(&roots, .5, 0, 1);

    VariableScene v(&composite);
    v.insert_variable("r0", "t sin");
    v.insert_variable("i0", "t cos");
    v.insert_variable("r1", "t 2 * sin");
    v.insert_variable("i1", "t 3 * cos");
    v.insert_variable("r2", "t 4 * sin");
    v.insert_variable("i2", "t 5 * cos");
    v.inject_audio_and_render(AudioSegment(3.14));

    v.stage_transition(std::unordered_map<std::string, std::string>{
        {"r0", "1"},
        {"i0", "1"},
        {"r1", "1"},
        {"i1", "1"},
        {"r2", "1"},
        {"i2", "1"}
    });
    v.inject_audio_and_render(AudioSegment(3.14));
    v.inject_audio_and_render(AudioSegment(3.14));
}