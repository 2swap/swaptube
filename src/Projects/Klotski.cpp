#include "../Scenes/KlotskiScene.cpp"
#include "../DataObjects/KlotskiBoard.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Math/GraphScene.cpp"
#include "../Scenes/Common/ExposedPixelsScene.cpp"

void render_video(){
    //FOR_REAL = false;
    PRINT_TO_TERMINAL = false;
    string begin_latex = "a=bc";
    string end_latex = "\\frac{a}{c}=b";

    CompositeScene cs;
    //KlotskiScene ks(6, 6, ".affo..aiko.bbiko.ecghh.ecgm..eddm..", true);
    //KlotskiScene ks(6, 6, "..afff..a..cbba..c.dd..e.....e..hhhe", true);
    KlotskiScene ks(4, 5, "abbcabbc.dd.efgheijh", false);
    /*cs.add_scene(&ks, "ks");

    cs.inject_audio(SilenceSegment(5), 10);
    cs.render();
    while(cs.microblocks_remaining()) {
        ks.stage_random_move();
        cs.render();
    }
    ks.stage_move('b', 0, 10);
    cs.inject_audio_and_render(SilenceSegment(.5));
    ks.state_manager.pop_it(true);
    cs.inject_audio_and_render(SilenceSegment(.5));
    cs.inject_audio_and_render(SilenceSegment(.5));
    */

    Graph g;
    g.add_to_stack(new KlotskiBoard(4, 5, "abbcabbc.dd.efgheijh", false));
    GraphScene gs(&g);
    gs.state_manager.set({
        {"physics_multiplier","1"},
        {"attract","1"},
        {"repel","1"},
        {"decay",".4"},
    });
    cs.add_scene(&gs, "gs");
    cs.inject_audio(SilenceSegment(5), 10);
    while(cs.microblocks_remaining()) {
        g.expand_once();
        cs.render();
    }
}
