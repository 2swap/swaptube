void render_video() {
    Graph<C4Board> g;
    C4GraphScene gs(&g, "436675555353311111133222", SIMPLE_WEAK);
    gs.physics_multiplier = 3;
//    gs.surfaces_on = false;
//    gs.set_camera_pos(glm::vec3(0,0,-150));

    VariableScene v(&gs);
    v.set_variables(std::unordered_map<std::string, std::string>{
        {"q1", "t .1 * cos"},
        {"qi", "0"},
        {"qj", "t .1 * sin"},
        {"qk", "0"}
    });
    v.inject_audio_and_render(AudioSegment(5));
}
