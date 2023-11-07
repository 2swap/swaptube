void render_video() {
    Graph<C4Board> g;
    C4GraphScene gs(&g, "444", MANUAL);
    g.add_node(new C4Board("444"));
    g.add_node(new C4Board("4444"));
    gs.construct_surfaces();
    g.sanitize_for_closure();
    gs.physics_multiplier = 3;
    gs.set_camera_pos(glm::vec3(0,0,-15));

    VariableScene v(&gs);
    v.set_variables(std::unordered_map<std::string, std::string>{
        {"q1", "t .1 * cos"},
        {"qi", "0"},
        {"qj", "t .1 * sin"},
        {"qk", "0"}
    });
    v.inject_audio_and_render(AudioSegment(4));
}