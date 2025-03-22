#include "../Scenes/KlotskiScene.cpp"
#include "../DataObjects/KlotskiBoard.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Math/GraphScene.cpp"
#include "../Scenes/Common/ExposedPixelsScene.cpp"

void render_video(){
    //FOR_REAL = false;
    PRINT_TO_TERMINAL = false;

    KlotskiScene advanced    (6, 6, ".affo..aiko.bbiko.ecghh.ecgm..eddm..", true, 0.5, 0.5);
    KlotskiScene intermediate(6, 6, "..afff..a..cbba..c.dd..e.....e..hhhe", true, 0.5, 0.5);
    KlotskiScene beginner    (6, 6, "...a.....a..bb.a..cddd..c.....c.....", true, 0.5, 0.5);
    KlotskiScene expert      (6, 6, "bbb..i.....i..ajji..aghhe.cgffe.cddd", true, 0.5, 0.5);
    KlotskiScene reddit      (6, 6, ".abbbm.ac..mddc.ikffhhike.g.lle.gjjj", true, 0.5, 0.5);
    KlotskiScene guh3        (6, 6, "akjjhhaknnigaklligcffeigc..eppbbdd..", true, 0.5, 0.5);
    KlotskiScene guh4        (6, 6, "bbbaff...ahhddca.eg.cppeg.nnikjjllik", true, 0.5, 0.5);
    KlotskiScene video       (6, 6, "acbbe.acdde.ffkg....kgm...igm...i...", true, 0.5, 0.5);
    KlotskiScene thinkfun1   (6, 6, "a.cbbba.c....dde...i.e...ifffg.....g", true, 0.5, 0.5);
    KlotskiScene thinkfun2   (6, 6, ".a.ffe.a...e.a.dde...c.....cbb......", true, 0.5, 0.5);
    KlotskiScene thinkfun3   (6, 6, "..a.....a...bba....dd....ff..chhh..c", true, 0.5, 0.5);
    KlotskiScene sun(4, 5, "abbcabbc.dd.efgheijh", false);
    for(KlotskiScene ks : {beginner, intermediate, advanced, expert, reddit, guh3, guh4, video, thinkfun1, thinkfun2, thinkfun3}){
        CompositeScene cs;
        cs.add_scene(&ks, "ks", 0.14, 0.25);

        /*cs.inject_audio(SilenceSegment(5), 10);
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

        Graph gt;
        gt.add_to_stack(new KlotskiBoard(ks.copy_board()));
        gt.expand_completely();
        int size = gt.size();
        cout << endl << size;

        Graph g;
        g.add_to_stack(new KlotskiBoard(ks.copy_board()));
        GraphScene gs(&g, .8, 1);
        gs.state_manager.set({
            {"q1", "<t> .1 * cos"},
            {"qi", "0"},
            {"qj", "<t> .1 * sin"},
            {"qk", "0"},
            {"physics_multiplier","3"},
            {"attract","1"},
            {"repel","1"},
            {"decay",".8"},
            {"surfaces_opacity","0"},
        });
        cs.add_scene(&gs, "gs", .6, .5);
        cs.inject_audio(SilenceSegment(8), size*2);
        while(cs.microblocks_remaining()) {
            g.expand_once();
            cs.render();
        }
    }
}
