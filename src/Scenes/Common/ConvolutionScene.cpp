#pragma once

#include "../Scene.cpp"
#include "../../misc/Convolution.cpp"

// Special function which ensures no temporary transparency while performing transitions
inline double transparency_profile(double x){return x<.5 ? cube(x/.5) : 1;}

class ConvolutionScene : public Scene {
public:
    ConvolutionScene(const Pixels& p, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
    : Scene(width, height), p1(p), coords(get_coords_from_pixels(p)) {
        pix.fill(TRANSPARENT_BLACK);
        pix.overwrite(p1, coords.first, coords.second);
    }

    pair<int, int> get_coords_from_pixels(const Pixels& p){
        return make_pair((pix.w-p.w)/2, (pix.h-p.h)/2);
    }

    void begin_transition(const Pixels& p) {
        if(in_transition_state) end_transition();
        p2 = p;

        transition_coords = get_coords_from_pixels(p);

        intersections = find_intersections(p1, p2);
        in_transition_state = true;
    }

    void on_end_transition(){
        if(in_transition_state) end_transition();
    }

    void end_transition(){
        cout << "Ending transition" << endl;
        assert(in_transition_state);
        p1 = p2;
        coords = transition_coords;
        pix.overwrite(p1, coords.first, coords.second);
        in_transition_state = false;
    }

    void draw() override{
        if(in_transition_state) {
            pix.fill(TRANSPARENT_BLACK);
            double tp = transparency_profile(state["subscene_transition_fraction"]);
            double tp1 = transparency_profile(1-state["subscene_transition_fraction"]);
            double smooth = smoother2(state["subscene_transition_fraction"]);

            double top_vx = 0;
            double top_vy = 0;

            for (int i = 0; i < intersections.size(); i++) {
                const StepResult& step = intersections[i];
                int x = round(lerp(coords.first , transition_coords.first -step.max_x, smooth));
                int y = round(lerp(coords.second, transition_coords.second-step.max_y, smooth));
                if(i == 0){
                    top_vx += transition_coords.first  - step.max_x - coords.first ;
                    top_vy += transition_coords.second - step.max_y - coords.second;
                }

                // Render the intersection at the interpolated position
                pix.overlay(step.induced1, x-step.current_p1.w, y-step.current_p1.h, tp1);
                pix.overlay(step.induced2, x-step.current_p2.w, y-step.current_p2.h, tp );

                //pix.overlay(step.intersection, 0  , i*191, 1);
                //pix.overlay(step.map         , 500, i*191, 1);
            }

            int dx = round(lerp(-top_vx, 0, smooth));
            int dy = round(lerp(-top_vy, 0, smooth));

            StepResult last_intersection = intersections[intersections.size()-1];
            pix.overlay(last_intersection.current_p1, dx +            coords.first, dy +            coords.second, tp1*tp1);
            pix.overlay(last_intersection.current_p2, dx + transition_coords.first, dy + transition_coords.second, tp *tp );
        }
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"subscene_transition_fraction"};
    }
    void mark_data_unchanged() override { }
    void change_data() override { }
    bool check_if_data_changed() const override { return false; } // No DataObjects

private:
    bool in_transition_state = false;

    // Things used for non-transition states
    Pixels p1;
    pair<int, int> coords;
    vector<StepResult> intersections;
    Pixels p2;
    pair<int, int> transition_coords;
};
