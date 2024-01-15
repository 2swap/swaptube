void render_video() {
    ThreeDimensionScene tds;

    C4Scene c40("43334444343773");

    tds.add_surface(Surface(glm::vec3(0,0,-1),glm::vec3(-8,2,8),glm::vec3(-2,-9,0),&c40));

    VariableScene v(&tds);
    v.set_variables(std::unordered_map<std::string, std::string>{
        {"x", "t sin 30 *"},
        {"y", "t 3 * cos"},
        {"z", "t cos 30 *"},
        {"q1", "t 4 / cos"},
        {"q2", "0"},
        {"q3", "t -4 / sin"},
        {"q4", "0"}
    });
    v.inject_audio_and_render(AudioSegment(3));
}