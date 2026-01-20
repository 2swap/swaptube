#include "../Scenes/Connect4/C4GraphScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"

void render_video() {
    shared_ptr<Graph> g = make_shared<Graph>();
    string variation = "436675555353311111133222";
    shared_ptr<C4GraphScene> gs = make_shared<C4GraphScene>(g, false, variation, SIMPLE_WEAK);
    shared_ptr<C4Scene> c4s = make_shared<C4Scene>(variation, .5, .5);
    CompositeScene cs;
    cs.add_scene(gs, "gs");
    cs.add_scene(c4s, "c4s", .25, .25);

    gs->manager.set(unordered_map<string, string>{
        {"q1", "{t} .1 * cos"},
        {"qi", "0"},
        {"qj", "{t} .1 * sin"},
        {"qk", "0"},
        {"d", "1"},
        {"points_opacity", "0"},
        {"physics_multiplier", "10"},
    });
    stage_macroblock(SilenceBlock(10), 1);
    cs.render_microblock();
}

