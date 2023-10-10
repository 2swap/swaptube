void render_video() {
    ThreeDimensionScene tds;
    for(int i = -7; i <= 7; i+=2)
    for(int j = -7; j <= 7; j+=2)
    for(int k = -7; k <= 7; k+=2)
        tds.add_point(Point(glm::vec3(i, j, k), WHITE));

    VariableScene v(&tds);
    glm::quat q = tds.get_quat();
    v.set_variables(std::unordered_map<std::string, std::string>{
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"q1", std::to_string(q.w)},
        {"q2", std::to_string(q.x)},
        {"q3", std::to_string(q.y)},
        {"q4", std::to_string(q.z)}
    });
    v.inject_audio_and_render(AudioSegment(2));

    glm::quat quats[6] = {PITCH_DOWN,PITCH_UP,YAW_RIGHT,YAW_LEFT,ROLL_CW,ROLL_CCW};
    for(glm::quat mult : quats){
        q *= mult;
        std::cout << q.w << q.x << q.y << q.z << std::endl;
        v.stage_transition(std::unordered_map<std::string, std::string>{
            {"x", "0"},
            {"y", "0"},
            {"z", "0"},
            {"q1", std::to_string(q.w)},
            {"q2", std::to_string(q.x)},
            {"q3", std::to_string(q.y)},
            {"q4", std::to_string(q.z)}
        });
        v.inject_audio_and_render(AudioSegment(2));
    }
}