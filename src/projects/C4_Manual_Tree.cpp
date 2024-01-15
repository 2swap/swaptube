void render_video() {
    Graph<C4Board> g;
    g.decay = 0.1;
    g.repel_force = 0.5;
    g.gravity_strength = 0;
    g.dimensions = 2;
    g.sqrty = true;
    g.lock_root_at_origin = true;
    C4GraphScene gs(&g, "444", MANUAL);
    g.sanitize_for_closure();
    gs.physics_multiplier = 3;

    VariableScene v(&gs);
    v.set_variables(std::unordered_map<std::string, std::string>{
        {"q1", "t 4 / cos"},
        {"qi", "0"},
        {"qj", "t -4 / sin"},
        {"qk", "0"},
        {"d", "2"}
    });
    v.inject_audio_and_render(AudioSegment(.1));
    v.stage_transition(std::unordered_map<std::string, std::string>{
        {"q1", "t 4 / cos"},
        {"qi", "0"},
        {"qj", "t -4 / sin"},
        {"qk", "0"},
        {"d", "20"}
    });
    v.inject_audio_and_render(AudioSegment(1));

    v.inject_audio_and_render(AudioSegment(1));
    for(int i = 1; i <= 7; i++){
        g.add_node(new C4Board("444" + to_string(i)));
        g.sanitize_for_closure();
        v.inject_audio_and_render(AudioSegment(.1));
    }
    v.inject_audio_and_render(AudioSegment(.1));
    g.dimensions = 3;
    for(int i = 1; i <= 7; i++){
        for(int j = 1; j <= 7; j++){
            if(j==4) continue;
            g.add_node(new C4Board("444" + to_string(i) + to_string(j)));
            g.sanitize_for_closure();
        }
        v.inject_audio_and_render(AudioSegment(.1));
    }
    v.inject_audio_and_render(AudioSegment(.1));


    cout << g.size() << endl;
}