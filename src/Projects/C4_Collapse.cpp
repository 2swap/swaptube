#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Connect4/Connect4GraphScene.cpp"

void render_video() {
    PRINT_TO_TERMINAL = false;

    vector<string> variations{
        /*
    "4151",
    "4152",
    "4153",
    "4154",
    "4155",
    "4156",
    "4157",
    */
    "4221",
    "4222",
    "4223",
    "4224",
    "4225",
    "4226",
    "4227",
    "4361",
    "4362",
    "4363",
    "4364",
    "4365",
    "4366",
    "4367",
    /*
    "4441",
    "4442",
    "4443",
    "444441",
    "444442",
    "444443",
    */
    "444444",
    };
    for(const string& variation : variations){
        Graph<C4Board> g;
        C4GraphScene gs(&g, variation, TRIM_STEADY_STATES);

        g.dimensions = 3;

        StateSet state{
            {"q1", "<t> .1 * cos"},
            {"qi", "0"},
            {"qj", "<t> .1 * sin"},
            {"qk", "0"},
            {"d", "400"},
            {"surfaces_opacity", "0"},
            {"points_opacity", "0"},
            {"physics_multiplier", "10"},
        };
        gs.state_manager.set(state);

        gs.inject_audio_and_render(AudioSegment(2));
        //cout << g.size() << " <- SIZE COMPARISON -> " << g2.size() << endl;
        //g.render_json(variation + ".json");
    }
}

