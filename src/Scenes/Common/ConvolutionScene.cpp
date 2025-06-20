#pragma once

#include "../Scene.cpp"
#include "../../misc/Convolution.cpp"

// Special function which ensures no temporary transparency while performing transitions
inline double transparency_profile(double x){return x<.5 ? cube(x/.5) : 1;}

class ConvolutionScene : public Scene {
public:
    ConvolutionScene(const Pixels& p, const double width = 1, const double height = 1)
    : Scene(width, height), p1(p), coords(get_coords_from_pixels(p)) {
        state_manager.set("transparency_profile", "<microblock_fraction>");
    }

    pair<int, int> get_coords_from_pixels(const Pixels& p){
        return make_pair((get_width()-p.w)/2, (get_height()-p.h)/2);
    }

    void begin_transition(const Pixels& p) {
        if(in_transition_state) end_transition();
        p2 = p;

        transition_coords = get_coords_from_pixels(p);

        intersections = find_intersections(p1, p2);
        in_transition_state = true;
    }

    void jump(const Pixels& p) {
        in_transition_state = false;
        p1 = p;
        coords = get_coords_from_pixels(p);
    }

    void on_end_transition_extra_behavior(bool is_macroblock){
        if(in_transition_state && !override_transition_end) end_transition();
    }

    void end_transition(){
        assert(in_transition_state);
        p1 = p2;
        coords = transition_coords;
        in_transition_state = false;
    }

    void draw() override{
        if(in_transition_state) {
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
        }
        else {
            pix.overwrite(p1, coords.first, coords.second);
        }
    }

    const StateQuery populate_state_query() const override {
        return in_transition_state ?
               StateQuery{"transparency_profile"}
               : StateQuery{};
    }
    void mark_data_unchanged() override { }
    void change_data() override { }
    bool check_if_data_changed() const override { return in_transition_state; } // No DataObjects, but we treat transitioning as changing data
    bool override_transition_end = false;
    Pixels get_copy_p1() const {
        return p1;
    }

protected:
    bool in_transition_state = false;

    // Things used for non-transition states
    Pixels p1;
    pair<int, int> coords;
    vector<StepResult> intersections;
    Pixels p2;
    pair<int, int> transition_coords;
};
