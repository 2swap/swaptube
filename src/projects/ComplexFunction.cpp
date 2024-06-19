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
}
