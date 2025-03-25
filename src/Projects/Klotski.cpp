#include "../Scenes/KlotskiScene.cpp"
#include "../DataObjects/KlotskiBoard.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Math/GraphScene.cpp"
#include "../Scenes/Common/ExposedPixelsScene.cpp"

void render_video(){
    //FOR_REAL = false;
    //PRINT_TO_TERMINAL = false;

    //rushhours
    KlotskiScene advanced      (6, 6, ".affo..aiko.bbiko.ecghh.ecgm..eddm..", true , 0.5, 0.5);
    KlotskiScene intermediate  (6, 6, "..afff..a..cbba..c.dd..e.....e..hhhe", true , 0.5, 0.5);
    KlotskiScene beginner      (6, 6, "...a.....a..bb.a..cddd..c.....c.....", true , 0.5, 0.5);
    KlotskiScene expert        (6, 6, "bbb..i.....i..ajji..aghhe.cgffe.cddd", true , 0.5, 0.5);
    KlotskiScene reddit        (6, 6, ".abbbm.ac..mddc.ikffhhike.g.lle.gjjj", true , 0.5, 0.5);
    KlotskiScene guh3          (6, 6, "akjjhhaknnigaklligcffeigc..eppbbdd..", true , 0.5, 0.5);
    KlotskiScene guh4          (6, 6, "bbbaff...ahhddca.eg.cppeg.nnikjjllik", true , 0.5, 0.5);
    KlotskiScene video         (6, 6, "acbbe.acdde.ffkg....kgm...igm...i...", true , 0.5, 0.5);
    KlotskiScene thinkfun1     (6, 6, "a.cbbba.c....dde...i.e...ifffg.....g", true , 0.5, 0.5);
    KlotskiScene thinkfun2     (6, 6, ".a.ffe.a...e.a.dde...c.....cbb......", true , 0.5, 0.5);
    KlotskiScene thinkfun3     (6, 6, "..a.....a...bba....dd....ff..chhh..c", true , 0.5, 0.5);

    //big
    KlotskiScene sun           (4, 5, "abbcabbc.dd.efgheijh"                , false, 0.5, 0.5);

    //other
    KlotskiScene apk           (4, 4, "abccefii.gii.ghh"                    , false, 0.5, 0.5);
    KlotskiScene mathgames_12_13_04(4, 4, "..a.bbaa.cd..cdd"                    , false, 0.5, 0.5);
    KlotskiScene euler766_easy (4, 3, "aab.acd.efgh"                        , false, 0.5, 0.5);

    //gpt
    KlotskiScene gpt2          (4, 4, "abcdabcd..ff..fg"                    , false, 0.5, 0.5);
    KlotskiScene gpt3          (5, 5, "aaa..bbb...gb...gcffggcff"           , false, 0.5, 0.5);
    KlotskiScene weird1        (8, 8, ".b.aaa...bbba....b..a.d.cccccdd....eefddeeee.ff...gg.fhhgggggf.h"                    , false, 0.5, 0.5);
    KlotskiScene weird2        (6, 4, ".dcc..dd.c...e.eff.eeef."            , false, 0.5, 0.5);
    KlotskiScene weird3        (8, 8, "....bbb.ccccccb...d..eb...d..e....d.eef..gddefffgg.d.fhhgg.....h", false, 0.5, 0.5);

    //suns
    KlotskiScene fatsun        (4, 5, "abbcabbc.bb.e..he..h"                , false, 0.5, 0.5);
    KlotskiScene sun_no_minis  (4, 5, "abbcabbc.dd.e..he..h"                , false, 0.5, 0.5);
    KlotskiScene truncatedsun  (4, 4, "abbcabbc.dd.efgh"                    , false, 0.5, 0.5);

    //geometry
    KlotskiScene jam3x3        (6, 6, "cea...cea...cea......bbb...ddd...fff", true , 0.5, 0.5);
    KlotskiScene cube_4d       (5, 5, "c.a....a..aaaaa..a....a.b"           , false, 0.5, 0.5);
    KlotskiScene cube_6d       (5, 5, "c.a.d..a..aaaaa..a....a.b"           , false, 0.5, 0.5);
    KlotskiScene big_block     (5, 5, "cea..cea..ceabb...dd...ff"           , true , 0.5, 0.5);
    KlotskiScene diamond       (6, 6, "a.a.........a.a...b.bc.c......b.bc.c", false, 0.5, 0.5);
    KlotskiScene doublering    (6, 6, "..ac....ac......bb....dd............", true , 0.5, 0.5);
    KlotskiScene outer_ring    (6, 6, "bbbb.a.....ac....ac....ac.....c.dddd", true , 0.5, 0.5);
    KlotskiScene plus_3_corners(5, 5, "cc.ddc.a.d.aaa...a.b...bb"           , false, 0.5, 0.5);
    KlotskiScene plus_4_corners(5, 5, "cc.ddcca...aaa.eea..ee..."           , false, 0.5, 0.5);
    KlotskiScene ring          (5, 5, "..a....a..bb............."           , true , 0.5, 0.5);
    KlotskiScene ring_big      (9, 9, "....a........a......................bb...........................................", true , 0.5, 0.5);
    KlotskiScene rows          (6, 6, "aaa.........aaa...bbb.........bbb...", true , 0.5, 0.5);
    KlotskiScene small_block   (4, 4, "ceadcead..bb.fbb"                    , false, 0.5, 0.5);
    KlotskiScene t_shapes      (6, 6, "aa...b.a.bbb.a..bbd.....ddd.ccd..cc.", false, 0.5, 0.5);
    KlotskiScene triangles     (6, 6, "c.f...ccff..b.e.h.bbeehha.d.i.aaddii", false, 0.5, 0.5);
    KlotskiScene triangles2    (6, 6, "ccffggc..f.gbbb..hb..hhha.d..iaaddii", false, 0.5, 0.5);

    auto gpt = {gpt2, gpt3, weird1, weird2, weird3};
    auto other = {apk, mathgames_12_13_04, euler766_easy};
    auto suns = {fatsun, sun_no_minis, truncatedsun};
    auto rushhours = {beginner, intermediate, advanced, expert, reddit, guh3, guh4, video, thinkfun1, thinkfun2, thinkfun3};
    auto big = {sun};
    auto geometry = {jam3x3, cube_4d, cube_6d, big_block, diamond, doublering, outer_ring, plus_3_corners, plus_4_corners, ring, ring_big, rows, small_block, t_shapes, triangles, triangles2};
    for(KlotskiScene ks : geometry){
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
