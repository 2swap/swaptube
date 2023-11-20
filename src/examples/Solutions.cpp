void render_video() {
    Graph<C4Board> g;
    g.decay = 0.1;
    g.repel_force = 5;
    g.gravity_strength = 2;
    //g.dimensions = 2;
    g.lock_root_at_origin = true;
    C4GraphScene gs(&g, "444", MANUAL);
    //g.add_node(new C4Board("444"));
    g.sanitize_for_closure();
    gs.physics_multiplier = 3;
    gs.set_original_camera_pos(glm::vec3(0,0,-20));

    VariableScene v(&gs);
    v.set_variables(std::unordered_map<std::string, std::string>{
        {"q1", "t .1 * cos"},
        {"qi", "0"},
        {"qj", "t .1 * sin"},
        {"qk", "0"}
    });
    v.inject_audio_and_render(AudioSegment(1));
    g.add_node(new C4Board("4444"));
    g.sanitize_for_closure();
    v.inject_audio_and_render(AudioSegment(1));
    g.add_node(new C4Board("4445"));
    g.sanitize_for_closure();
    v.inject_audio_and_render(AudioSegment(1));
    g.add_node(new C4Board("4446"));
    g.sanitize_for_closure();
    v.inject_audio_and_render(AudioSegment(1));
    g.add_node(new C4Board("4447"));
    g.sanitize_for_closure();
    v.inject_audio_and_render(AudioSegment(1));
    cout << g.size() << endl;
}