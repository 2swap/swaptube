#pragma once
#include <vector>
#include <set>
#include <unordered_set>
#include <string>
#include <memory>

#include "GenericBoard.h"

#define EMPTY_SPACE '.'

struct KlotskiMove {
    char piece = '.';
    int dx = 0;
    int dy = 0;
};

bool in_bounds(int min, int val, int max);

class KlotskiBoard : public GenericBoard {
public:
    int h, w;
    unordered_set<char> letters;
    string blurb;
    bool rushhour = true;
    string game_name = "klotski";

    KlotskiBoard(int w, int h, const string& rep, const bool rush);
    ~KlotskiBoard();

    shared_ptr<Scene> make_scene() const override;
    bool is_solution() override;
    double type_specific_hash() override;
    double type_specific_reverse_hash() override;
    double type_specific_reverse_hash_2() override;

    void print();
    void compute_letters();
    bool can_move_piece(const KlotskiMove& km);
    KlotskiBoard move_piece(const KlotskiMove& km);
    unordered_set<GenericBoard*> get_children();
    void get_random_move(KlotskiMove& km);
    KlotskiMove move_required_to_reach(const KlotskiBoard& kb);
};

// Common Boards
//rushhours
inline KlotskiBoard advanced      (6, 6, ".affo..aiko.bbiko.ecghh.ecgm..eddm..", true );
inline KlotskiBoard intermediate  (6, 6, "..afff..a..cbba..c.dd..e.....e..hhhe", true );
inline KlotskiBoard beginner      (6, 6, "...a.....a..bb.a..cddd..c.....c.....", true );
inline KlotskiBoard expert        (6, 6, "jjj..i.....i..abbi..aghhe.cgffe.cddd", true );
inline KlotskiBoard reddit        (6, 6, ".adddm.ac..mbbc.ikffhhike.g.lle.gjjj", true );
inline KlotskiBoard guh3          (6, 6, "akjjhhaknnigakbbigcffeigc..epplldd..", true );
inline KlotskiBoard guh4          (6, 6, "dddaff...ahhbbca.eg.cppeg.nnikjjllik", true );
inline KlotskiBoard thinkfun1     (6, 6, "a.cddda.c....bbe...i.e...ifffg.....g", true );
inline KlotskiBoard thinkfun2     (6, 6, ".a.ffe.a...e.a.bbe...c.....cdd......", true );
inline KlotskiBoard thinkfun3     (6, 6, "..a.....a...bba....dd....ff..chhh..c", true );

//big
inline KlotskiBoard sun_mine      (4, 5, "abbcabbc.dd.efgheijh"                , false);
inline KlotskiBoard sun           (4, 5, "abbcabbceddhefghi..j"                , false);
inline KlotskiBoard sun_no_b      (4, 5, "a..ca..ceddhefghi..j"                , false);
inline KlotskiBoard klotski_flip  (4, 5, "j..ihgfehddecbbacbba"                , false);
inline KlotskiBoard klotski_bonus (4, 5, "..afbbahbbchddcejgie"                , false);
inline KlotskiBoard klotski_bonus2(4, 5, "jbbcgbbcddhi.eha.efa"                , false);
inline KlotskiBoard klotski_necklace(4, 5, "afgcabbc.bb.eddheijh"                , false);
inline KlotskiBoard klotski_necklace_2(4, 5, "acfgac.e.bbeibbhddjh"                , false);
inline KlotskiBoard klotski_whisker(4, 5, ".bb.abbfaceghcejhidd"                , false);
inline KlotskiBoard klotski_earring(4, 5, "..fgbbijbbddacehaceh"                , false);
inline KlotskiBoard sun_pit       (4, 5, "fbbagbbace.icejhdd.h"                , false);
inline KlotskiBoard klotski_solution_slow(4, 5, "fgacehacehdd.bbi.bbj"                , false);
inline KlotskiBoard klotski_solution     (4, 5, "acehacehfgdd.bbi.bbj"                , false);

//other
//KlotskiBoard apk           (4, 4, "abccefii.gii.ghh"                    , false);
inline KlotskiBoard apk           (4, 4, "cc.agiibgiiefhh."                    , false);
inline KlotskiBoard mathgames_12_13_04(4, 4, "..a.bbaa.cd..cdd"                    , false);
inline KlotskiBoard mathgames_12_13_04_nook(6, 6, "eeeeeee..a.eebbaaee.cd.ee.cddeeee.ee", false);
inline KlotskiBoard euler766_easy (4, 3, "aab.acd.efgh"                        , false);

//gpt
inline KlotskiBoard gpt2          (4, 4, "abcdabcd..ff..fg"                    , false);
inline KlotskiBoard gpt3          (5, 5, "aaa..bbb...gb...gcffggcff"           , false);
inline KlotskiBoard weird1        (8, 8, ".b.aaa...bbba....b..a.d.cccccdd....eefddeeee.ff...gg.fhhgggggf.h", false);
inline KlotskiBoard weird2        (6, 4, ".dcc..dd.c...e.eff.eeef."            , false);
inline KlotskiBoard weird3        (8, 8, "....bbb.ccccccb...d..eb...d..e....d.eef..gddefffgg.d.fhhgg.....h", false);

//suns
inline KlotskiBoard fatsun        (4, 5, "abbcabbc.bb.e..he..h"                , false);
inline KlotskiBoard sun_no_minis  (4, 5, "abbcabbc.dd.e..he..h"                , false);
inline KlotskiBoard sun_small     (4, 5, ".bb..bb..dd.e..ee..e"                , false);
inline KlotskiBoard truncatedsun  (4, 4, "abbcabbc.dd.efgh"                    , false);

//geometry
inline KlotskiBoard jam3x3        (6, 6, "cea...cea...cea......bbb...ddd...fff", true );
inline KlotskiBoard full_15_puzzle(4, 4, "abcdefghijklmno."                    , false);
inline KlotskiBoard manifold_1d   (7, 7, "a......a.........................................", true );
inline KlotskiBoard manifold_2d   (7, 7, "ac.....ac........................................", true );
inline KlotskiBoard manifold_3d   (7, 7, "ace....ace.......................................", true );
inline KlotskiBoard manifold_4d   (7, 7, "aceg...aceg......................................", true );
inline KlotskiBoard triangle      (9, 9, "....a........a.....................................................c........c....", true );
inline KlotskiBoard triangle_inv  (9, 9, "....cc.......c...................................a........a......................", true );
inline KlotskiBoard ring_7x7      (7, 7, "...a......a..........bb..........................", true );
inline KlotskiBoard iblock        (7, 7, "...........a..bb..a..............dd..............", true );
inline KlotskiBoard cube_3d       (5, 5, "caa...aa..aaaaa..a....a.b"           , false);
inline KlotskiBoard cube_4d       (5, 5, "c.a....a..aaaaa..a....a.b"           , false);
inline KlotskiBoard cube_6d       (5, 5, "c.a.d..a..aaaaa..a....a.b"           , false);
inline KlotskiBoard big_block     (5, 5, "cea..cea..ceabb...dd...ff"           , true );
inline KlotskiBoard diamond       (6, 6, "a.a.........a.a...b.bc.c......b.bc.c", false);
inline KlotskiBoard doublering    (6, 6, "..ac....ac......bb....dd............", true );
inline KlotskiBoard outer_ring    (6, 6, "bbbb.a.....ac....ac....ac.....c.dddd", true );
inline KlotskiBoard plus_3_corners(5, 5, "cc.ddc.a.d.aaa...a.b...bb"           , false);
inline KlotskiBoard plus_4_corners(5, 5, "cc.ddcca...aaa.eea..ee..."           , false);
inline KlotskiBoard ring          (5, 5, "..a....a..bb............."           , true );
inline KlotskiBoard ring_big      (9, 9, "....a........a......................bb...........................................", true );
inline KlotskiBoard rows          (6, 6, "aaa.........aaa...bbb.........bbb...", false);
inline KlotskiBoard small_block   (4, 4, "ceadcead..bb.fbb"                    , false);
inline KlotskiBoard t_shapes      (6, 6, "aa...b.a.bbb.a..bbd.....ddd.ccd..cc.", false);
inline KlotskiBoard triangles     (6, 6, "c.f...ccff..b.e.h.bbeehha.d.i.aaddii", false);
inline KlotskiBoard triangles2    (6, 6, "ccffggc..f.gbbb..hb..hhha.d..iaaddii", false);
