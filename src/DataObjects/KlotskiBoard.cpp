#include "KlotskiBoard.h"
#include "../Scenes/KlotskiScene.cpp"
#include <cmath>
#include <cstdlib>

using std::string;
using std::shared_ptr;
using std::make_shared;
using std::unordered_set;
using std::set;

bool in_bounds(int min, int val, int max) {
    return min <= val && val < max;
}

KlotskiBoard::KlotskiBoard(int w, int h, const string& rep, const bool rush)
    : GenericBoard(rep), h(h), w(w), rushhour(rush) {
    if(rep.size() != w*h) throw runtime_error("Rushhour board: invalid string length");
}

KlotskiBoard::~KlotskiBoard() { }

shared_ptr<Scene> KlotskiBoard::make_scene() const {
    return make_shared<KlotskiScene>(KlotskiBoard(*this));
}

void KlotskiBoard::print() {
    cout << endl;
    for(int y = 0; y < h; y++) {
        for(int x = 0; x < w; x++)
            cout << representation[y*w+x];
        cout << endl;
    }
    cout << endl;
}

bool KlotskiBoard::is_solution() {
    if (rushhour)
        return representation[17] == representation[16] && representation[16] != EMPTY_SPACE;
    else {
        if(h==5 && w==4)
            return representation[18]==representation[17] &&
                   representation[14]==representation[13] &&
                   representation[14]==representation[17];
        else
            return false;
    }
}

void KlotskiBoard::compute_letters() {
    if(letters.empty()) {
        for(int i = 0; i < h*w; i++) {
            letters.insert(representation[i]);
        }
        letters.erase(EMPTY_SPACE);
    }
}

double KlotskiBoard::type_specific_reverse_hash_2() {
    compute_letters();
    double hash_in_progress = 0;
    set<double> s;

    for (const char& letter: letters) {
        double sum = 0;
        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
                if(representation[(h-1-y)*w + x] == letter) {
                    int i = y*w + x;
                    sum += sin((i+1)*cbrt(i+2));
                }
        s.insert(cbrt(sum));
    }

    for(double d : s)
        hash_in_progress += d;

    return hash_in_progress;
}

double KlotskiBoard::type_specific_reverse_hash() {
    compute_letters();
    double hash_in_progress = 0;
    set<double> s;

    for (const char& letter: letters) {
        double sum = 0;
        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
                if(representation[y*w + (w-1-x)] == letter) {
                    int i = y*w + x;
                    sum += sin((i+1)*cbrt(i+2));
                }
        s.insert(cbrt(sum));
    }

    for(double d : s)
        hash_in_progress += d;

    return hash_in_progress;
}

double KlotskiBoard::type_specific_hash() {
    compute_letters();
    double hash_in_progress = 0;
    set<double> s;

    for (const char& letter: letters) {
        double sum = 0;
        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
                if(representation[y*w + x] == letter) {
                    int i = y*w + x;
                    sum += sin((i+1)*cbrt(i+2));
                }
        s.insert(cbrt(sum));
    }

    for(double d : s)
        hash_in_progress += d;

    return hash_in_progress;
}

bool KlotskiBoard::can_move_piece(const KlotskiMove& km) {
    for(int y = 0; y < h; y++)
        for(int x = 0; x < w; x++) {
            if(representation[y*w + x] == km.piece) {
                bool inside = in_bounds(0, y+km.dy, h) && in_bounds(0, x+km.dx, w);
                if(!inside) return false;
                char target = representation[(y+km.dy)*w + (x+km.dx)];
                if(target != EMPTY_SPACE && target != km.piece)
                    return false;
            }
        }
    return true;
}

KlotskiBoard KlotskiBoard::move_piece(const KlotskiMove& km) {
    string rep(h*w, EMPTY_SPACE);
    for(int y = 0; y < h; y++)
        for(int x = 0; x < w; x++) {
            int pos = y*w + x;
            char letter_here = representation[pos];
            if(letter_here == km.piece) {
                bool inside = in_bounds(0, y+km.dy, h) && in_bounds(0, x+km.dx, w);
                if(inside) {
                    int target = (y+km.dy)*w + (x+km.dx);
                    rep[target] = km.piece;
                }
            } else if(letter_here != EMPTY_SPACE)
                rep[pos] = letter_here;
        }

    return KlotskiBoard(w, h, rep, rushhour);
}

unordered_set<GenericBoard*> KlotskiBoard::get_children() {
    unordered_set<GenericBoard*> children;

    compute_letters();

    for (const char& letter: letters) {
        for(int dy = -1; dy <= 1; dy++)
            for(int dx = -1; dx <= 1; dx++) {
                if((dx+dy)%2 == 0) continue;
                if(rushhour && (letter - 'a' + dy)%2 == 0) continue;

                KlotskiMove cmove{letter, dx, dy};
                if(can_move_piece(cmove))
                    children.insert(new KlotskiBoard(move_piece(cmove)));
            }
    }

    return children;
}

void KlotskiBoard::get_random_move(KlotskiMove& km) {
    int idx = 0;
    compute_letters();

    for(int count_dir = 0; count_dir < 2; count_dir++) {
        for (const char& letter: letters) {
            for(int dy = -1; dy <= 1; dy++) {
                for(int dx = -1; dx <= 1; dx++) {
                    if((dx+dy)%2 == 0) continue;
                    if(rushhour && (letter - 'a' + dy)%2 == 0) continue;

                    if(can_move_piece({letter, dx, dy})) {
                        if(count_dir == 0)
                            idx++;
                        else {
                            if(idx == 0) {
                                km = {letter, dx, dy};
                                return;
                            }
                            idx--;
                        }
                    }
                }
            }
        }
        idx = rand() % idx;
    }
}

KlotskiMove KlotskiBoard::move_required_to_reach(const KlotskiBoard& kb) {
    compute_letters();
    double kb_hash = KlotskiBoard(kb).get_hash();
    for (const char& letter: letters) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if ((dx + dy) % 2 == 0)
                    continue;
                if (rushhour && (letter - 'a' + dy) % 2 == 0)
                    continue;
                KlotskiMove cmove{letter, dx, dy};
                if (can_move_piece(cmove)) {
                    KlotskiBoard newBoard = move_piece(cmove);
                    if (newBoard.get_hash() == kb_hash)
                        return cmove;
                }
            }
        }
    }
    return KlotskiMove{EMPTY_SPACE, 0, 0};
}

// Common Boards
//rushhours
KlotskiBoard advanced      (6, 6, ".affo..aiko.bbiko.ecghh.ecgm..eddm..", true );
KlotskiBoard intermediate  (6, 6, "..afff..a..cbba..c.dd..e.....e..hhhe", true );
KlotskiBoard beginner      (6, 6, "...a.....a..bb.a..cddd..c.....c.....", true );
KlotskiBoard expert        (6, 6, "jjj..i.....i..abbi..aghhe.cgffe.cddd", true );
KlotskiBoard reddit        (6, 6, ".adddm.ac..mbbc.ikffhhike.g.lle.gjjj", true );
KlotskiBoard guh3          (6, 6, "akjjhhaknnigakbbigcffeigc..epplldd..", true );
KlotskiBoard guh4          (6, 6, "dddaff...ahhbbca.eg.cppeg.nnikjjllik", true );
KlotskiBoard thinkfun1     (6, 6, "a.cddda.c....bbe...i.e...ifffg.....g", true );
KlotskiBoard thinkfun2     (6, 6, ".a.ffe.a...e.a.bbe...c.....cdd......", true );
KlotskiBoard thinkfun3     (6, 6, "..a.....a...bba....dd....ff..chhh..c", true );

//big
KlotskiBoard sun_mine      (4, 5, "abbcabbc.dd.efgheijh"                , false);
KlotskiBoard sun           (4, 5, "abbcabbceddhefghi..j"                , false);
KlotskiBoard sun_no_b      (4, 5, "a..ca..ceddhefghi..j"                , false);
KlotskiBoard klotski_flip  (4, 5, "j..ihgfehddecbbacbba"                , false);
KlotskiBoard klotski_bonus (4, 5, "jbbaibba.fddcehgceh."                , false);
KlotskiBoard sun_pit       (4, 5, "abbfabbcg.ecijeh.ddh"                , false);
KlotskiBoard klotski_solution_slow(4, 5, "fgacehacehdd.bbi.bbj"                , false);
KlotskiBoard klotski_solution     (4, 5, "acehacehfgdd.bbi.bbj"                , false);

//other
//KlotskiBoard apk           (4, 4, "abccefii.gii.ghh"                    , false);
KlotskiBoard apk           (4, 4, "cc.agiibgiiefhh."                    , false);
KlotskiBoard mathgames_12_13_04(4, 4, "..a.bbaa.cd..cdd"                    , false);
KlotskiBoard euler766_easy (4, 3, "aab.acd.efgh"                        , false);

//gpt
KlotskiBoard gpt2          (4, 4, "abcdabcd..ff..fg"                    , false);
KlotskiBoard gpt3          (5, 5, "aaa..bbb...gb...gcffggcff"           , false);
KlotskiBoard weird1        (8, 8, ".b.aaa...bbba....b..a.d.cccccdd....eefddeeee.ff...gg.fhhgggggf.h", false);
KlotskiBoard weird2        (6, 4, ".dcc..dd.c...e.eff.eeef."            , false);
KlotskiBoard weird3        (8, 8, "....bbb.ccccccb...d..eb...d..e....d.eef..gddefffgg.d.fhhgg.....h", false);

//suns
KlotskiBoard fatsun        (4, 5, "abbcabbc.bb.e..he..h"                , false);
KlotskiBoard sun_no_minis  (4, 5, "abbcabbc.dd.e..he..h"                , false);
KlotskiBoard sun_small     (4, 5, ".bb..bb..dd.e..ee..e"                , false);
KlotskiBoard truncatedsun  (4, 4, "abbcabbc.dd.efgh"                    , false);

//geometry
KlotskiBoard jam3x3        (6, 6, "cea...cea...cea......bbb...ddd...fff", true );
KlotskiBoard full_15_puzzle(4, 4, "abcdefghijklmno."                    , false);
KlotskiBoard manifold_1d   (7, 7, "a......a.........................................", true );
KlotskiBoard manifold_2d   (7, 7, "ac.....ac........................................", true );
KlotskiBoard manifold_3d   (7, 7, "ace....ace.......................................", true );
KlotskiBoard manifold_4d   (7, 7, "aceg...aceg......................................", true );
KlotskiBoard triangle      (9, 9, "....a........a.....................................................c........c....", true );
KlotskiBoard triangle_inv  (9, 9, "....cc.......c...................................a........a......................", true );
KlotskiBoard ring_7x7      (7, 7, "...a......a..........bb..........................", true );
KlotskiBoard iblock        (7, 7, "...........a..bb..a..............dd..............", true );
KlotskiBoard cube_3d       (5, 5, "caa...aa..aaaaa..a....a.b"           , false);
KlotskiBoard cube_4d       (5, 5, "c.a....a..aaaaa..a....a.b"           , false);
KlotskiBoard cube_6d       (5, 5, "c.a.d..a..aaaaa..a....a.b"           , false);
KlotskiBoard big_block     (5, 5, "cea..cea..ceabb...dd...ff"           , true );
KlotskiBoard diamond       (6, 6, "a.a.........a.a...b.bc.c......b.bc.c", false);
KlotskiBoard doublering    (6, 6, "..ac....ac......bb....dd............", true );
KlotskiBoard outer_ring    (6, 6, "bbbb.a.....ac....ac....ac.....c.dddd", true );
KlotskiBoard plus_3_corners(5, 5, "cc.ddc.a.d.aaa...a.b...bb"           , false);
KlotskiBoard plus_4_corners(5, 5, "cc.ddcca...aaa.eea..ee..."           , false);
KlotskiBoard ring          (5, 5, "..a....a..bb............."           , true );
KlotskiBoard ring_big      (9, 9, "....a........a......................bb...........................................", true );
KlotskiBoard rows          (6, 6, "aaa.........aaa...bbb.........bbb...", false);
KlotskiBoard small_block   (4, 4, "ceadcead..bb.fbb"                    , false);
KlotskiBoard t_shapes      (6, 6, "aa...b.a.bbb.a..bbd.....ddd.ccd..cc.", false);
KlotskiBoard triangles     (6, 6, "c.f...ccff..b.e.h.bbeehha.d.i.aaddii", false);
KlotskiBoard triangles2    (6, 6, "ccffggc..f.gbbb..hb..hhha.d..iaaddii", false);
