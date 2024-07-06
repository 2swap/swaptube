void render_video() {
    ThreeDimensionScene tds;

    C4Scene c40("43334444343773");

    tds.add_surface(Surface(glm::vec3(0,0,-1),glm::vec3(-8,2,8),glm::vec3(-2,-9,0),&c40));

    dag.add_equations(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "1"},
        {"lines_opacity", "0"},
        {"points_opacity", "0"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "20"},
        {"q1", "<t> 4 / cos"},
        {"qi", "0"},
        {"qj", "<t> -4 / sin"},
        {"qk", "0"}
    });
    tds.inject_audio_and_render(AudioSegment(3));
}