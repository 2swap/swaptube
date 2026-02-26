#include "ConvolutionScene.h"
#include <stdexcept>
#include <string>
#include <cmath>

namespace {
inline double transparency_profile(double x){return x<.5 ? cube(x/.5) : 1;}
}

ConvolutionScene::ConvolutionScene(const double width, const double height)
: Scene(width, height) {
    manager.set({
        {"transparency_profile", "<in_transition_state> {microblock_fraction} *"},
        {"in_transition_state", "0"},
    });
}

std::pair<int, int> ConvolutionScene::get_coords_from_pixels(const Pixels& p){
    return std::make_pair((get_width()-p.w)/2, (get_height()-p.h)/2);
}

void ConvolutionScene::begin_transition(const TransitionType tt, const Pixels& p) {
    if(!state.contains("in_transition_state"))
        throw std::runtime_error("ConvolutionScene must be rendered once before calling begin_transition!");
    if(state["in_transition_state"]) end_transition();
    p2 = p;

    transition_coords = get_coords_from_pixels(p);

    intersections = find_intersections(p1, p2);
    manager.set({
        {"transparency_profile", "<in_transition_state> {m" + std::string(tt == MICRO?"i":"a") + "croblock_fraction} *"},
        {"in_transition_state", "1"},
    });
    current_transition_type = tt;
}

void ConvolutionScene::jump(const Pixels& p) {
    manager.set("in_transition_state", "0");
    p1 = p;
    coords = get_coords_from_pixels(p);
    jumped = true;
}

void ConvolutionScene::on_end_transition_extra_behavior(const TransitionType tt) {
    update_state();
    if((MICRO == current_transition_type || MACRO == tt) && state["in_transition_state"] == 1) end_transition();
}

void ConvolutionScene::end_transition(){
    if(state["in_transition_state"] != 1) throw std::runtime_error("End Transition called on a ConvolutionScene not in transition!");
    p1 = p2;
    coords = transition_coords;
    manager.set("in_transition_state", "0");
}

void ConvolutionScene::draw(){
    // TODO This has a rather large bug:
    // transitions cannot occur at the same time as a LatexScene changes in width and height.
    // This currently causes visual bugs, and the rescaling doesn't happen and the image is cropped.
    // This is because we only once compute the convolution parameters relating the two latex statements.
    // Fixing this would mean running that very slow algorithm every frame.
    // One solution could be just downscaling on the fly... quality could be lost though
    if(state["in_transition_state"]==1) {
        double tp = transparency_profile(state["transparency_profile"]);
        double tp1 = transparency_profile(1-state["transparency_profile"]);
        double smooth = smoother2(state["transparency_profile"]);

        double top_vx = 0;
        double top_vy = 0;

        for (int i = 0; i < intersections.size(); i++) {
            const StepResult& step = intersections[i];
            int x = round(lerp(coords.first , transition_coords.first -step.max_x, smooth));
            int y = round(lerp(coords.second, transition_coords.second-step.max_y, smooth));
            // Render the intersection at the interpolated position
            pix.overlay(step.induced1, x, y, tp1);
            pix.overlay(step.induced2, x+step.max_x, y+step.max_y, tp );

            if(i == 0 && false){
                top_vx += transition_coords.first  - step.max_x - coords.first ;
                top_vy += transition_coords.second - step.max_y - coords.second;
            }
        }

        int dx = round(lerp(-top_vx, 0, smooth));
        int dy = round(lerp(-top_vy, 0, smooth));

        StepResult last_intersection = intersections[intersections.size()-1];
        pix.overlay(last_intersection.current_p1, dx +            coords.first, dy +            coords.second, tp1*tp1);
        pix.overlay(last_intersection.current_p2, dx + transition_coords.first, dy + transition_coords.second, tp *tp );
    } else {
        p1 = get_p1();
        coords = get_coords_from_pixels(p1);
        pix.overwrite(p1, coords.first, coords.second);
    }
}

const StateQuery ConvolutionScene::populate_state_query() const {
    return StateQuery{"transparency_profile", "in_transition_state"};
}
void ConvolutionScene::mark_data_unchanged() { jumped = false; }
void ConvolutionScene::change_data() { }
bool ConvolutionScene::check_if_data_changed() const {
    double its = state["in_transition_state"];
    return (its == 1) || jumped;
} // No DataObjects, but we treat transitioning as changing data
