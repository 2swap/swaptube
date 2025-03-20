#include "../Scenes/Math/GraphScene.cpp"
#include "../DataObjects/PacmanPackage.cpp"

unordered_set<string> get_explicit_packages() {
    unordered_set<string> packages;
    FILE* pipe = popen("pacman -Q", "r");
    if (!pipe) {
        throw runtime_error("Failed to run pacman command.");
    }

    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        istringstream line(buffer);
        string package_name;
        line >> package_name; // Extract only the first word (package name)
        if (!package_name.empty()) {
            packages.insert(package_name);
        }
    }

    pclose(pipe);
    return packages;
}

void render_video() {
    //PRINT_TO_TERMINAL = false;
    SAVE_FRAME_PNGS = false;

    Graph g;
    g.dimensions = 3;
    GraphScene gs(&g);

    gs.state_manager.set({
        {"q1", "<t> .1 * cos"},
        {"qi", "0"},
        {"qj", "<t> .1 * sin"},
        {"qk", "0"},
        {"surfaces_opacity", "1"},
        {"physics_multiplier", "3"},
        {"attract", "1"},
        {"repel", ".2"},
    });
    const unordered_set<string> expl = get_explicit_packages();
    for(const string& s : expl){
        g.add_to_stack(new PacmanPackage(s));
    }
    g.expand_graph_completely();
    cout << g.size() << endl;
    gs.inject_audio_and_render(SilenceSegment(5));
}
