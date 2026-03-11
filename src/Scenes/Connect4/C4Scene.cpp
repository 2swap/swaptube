#include "C4Scene.h"
#include "../../DataObjects/Connect4/SteadyState.h"
#include <iostream>
using namespace std;

C4Scene::C4Scene(const std::string& rep, const vec2& dimensions)
    :Scene(dimensions), board(7, 6), annotations(empty_annotations), representation(rep){
        board.append_to_queue(rep);
        manager.set("highlight", "0");
        manager.set("annotations_opacity", "0");
}

void C4Scene::undo(int steps) {
    if(steps > representation.size()) steps = representation.size();
    int new_size = representation.size() - steps;
    board.undo(steps);
    representation = representation.substr(0, new_size);
    cout << "Undo " << steps << " steps. New representation: " << representation << endl;
}

void C4Scene::use_up_queue() {
    board.use_up_queue();
}

void C4Scene::flush_queue_undo_all() {
    board.flush_queue();
    undo(representation.size());
}

void C4Scene::play(const std::string& rep){
    board.append_to_queue(rep);
    representation += rep;
}

void C4Scene::set_annotations(const TransitionType tt, const std::string& s) {
    annotations = s;
    manager.transition(tt, "annotations_opacity", "1");
}

void C4Scene::set_annotations_from_steadystate(const TransitionType tt) {
    shared_ptr<SteadyState> ss = find_steady_state(representation, nullptr);
    annotations = ss->to_string();
    cout << "Annotations from steady state: " << annotations << endl;
    // Replace all '2's and '1's with spaces
    replace(annotations.begin(), annotations.end(), '2', ' ');
    replace(annotations.begin(), annotations.end(), '1', ' ');
    manager.transition(tt, "annotations_opacity", "1");
}

std::string C4Scene::get_annotations(){return annotations;}

void C4Scene::clear_annotations(const TransitionType tt) {
    manager.transition(tt, "annotations_opacity", "0");
}

char C4Scene::get_annotation(int x, int y){
    return annotations[x+(board.h-1-y)*board.w];
}

void C4Scene::set_fast_mode(bool fast){
    board.fast_mode = fast;
}

void C4Scene::draw_empty_board(){
    vec2 ell_dim = vec2(1,1) * get_stone_width() * .3;
    for(int x=0; x<board.w; x++){
        for(int y=0; y<board.h; y++){
            vec2 disc;
            get_disc_screen_coordinates(vec2(x,y), disc);
            pix.fill_ellipse(disc, ell_dim, C4_EMPTY);
        }
    }
}

void C4Scene::highlight_winning_discs(){
    double highlight = state["highlight"];
    if(highlight <= 0.1) return;

    C4Board b(FULL, representation);
    Bitboard winning_discs = b.winning_discs();
    vec2 ell_dim = vec2(1,1) * get_stone_width() * .2 * highlight;
    for (int x=0; x<board.w; x++) {
        for (int y=0; y<board.h; y++) {
            if(!bitboard_at(winning_discs, x, y)) continue;
            vec2 disc;
            get_disc_screen_coordinates(vec2(x, C4_HEIGHT - 1 - y), disc);
            bool is_red = bitboard_at(b.red_bitboard, x, y);
            int color = is_red ? C4_RED : C4_YELLOW;
            pix.fill_ellipse(disc, ell_dim, color);
        }
    }
}

void C4Scene::draw_c4_disc(int disc_x, double disc_y, bool is_red) {
    vec2 disc;
    get_disc_screen_coordinates(vec2(disc_x, disc_y), disc);
    int col = is_red ? C4_RED : C4_YELLOW;
    int darkcol = is_red ? C4_RED_DARK : C4_YELLOW_DARK;
    double clamped_w = clamp((disc_y - board.h + 1) / 2, 0.0, 1.0);
    col = colorlerp(col, col & 0x00ffffff, clamped_w);
    darkcol = colorlerp(darkcol, darkcol & 0x00ffffff, clamped_w);

    double stone_width = get_stone_width();
    double piece_fill_radius = stone_width*.35;
    double piece_stroke_radius = stone_width*.47;
    pix.fill_circle(disc, piece_stroke_radius, col);
    pix.fill_circle(disc, piece_fill_radius  , darkcol);
}

void C4Scene::draw_annotations(){
    if(state["annotations_opacity"] <= 0.01) return;
    for(int x=0; x<board.w; x++){
        for(int y=0; y<board.h; y++){
            char annotation = get_annotation(x, y);
            if(annotation != ' ') {
                vec2 disc;
                get_disc_screen_coordinates(vec2(x, y), disc);
                ScalingParams sp(vec2(get_stone_width()*1.3, get_stone_width()*1.3));
                std::string annotation_str(1, annotation);
                if(annotation != '@') annotation_str = "\\text{" + annotation_str + "}";
                else annotation_str = "\\@";
                Pixels latex = latex_to_pix(annotation_str, sp);
                pix.overlay(latex, disc-latex.size/2, state["annotations_opacity"]);
            }
        }
    }
}

void C4Scene::get_disc_screen_coordinates(const vec2& stone, vec2& disc) {
    disc = (stone.x - vec2(board.w, board.h)/2.+.5) * get_stone_width() + pix.size/2;
    disc.y *= -1;
}

const StateQuery C4Scene::populate_state_query() const {
    return StateQuery{"highlight", "annotations_opacity"};
}

void C4Scene::mark_data_unchanged() { board.mark_unchanged(); }
void C4Scene::change_data() { board.iterate_physics(); }
bool C4Scene::check_if_data_changed() const { return board.has_been_updated_since_last_scene_query(); }

double C4Scene::get_stone_width() const {
    return min(get_width(), get_height())/10;
}

void C4Scene::draw(){
    draw_empty_board();
    draw_annotations();
    for(const Disc& disc : board.discs){
        draw_c4_disc(disc.x, disc.py, disc.index%2==0);
    }
    highlight_winning_discs();
}
