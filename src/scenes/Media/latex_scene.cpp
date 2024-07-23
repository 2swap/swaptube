#pragma once

#include "../../io/visual_media.cpp"
#include "../scene.cpp"

// Special function which ensures no temporary transparency while performing latex transitions
inline double transparency_profile(double x){return x<.5 ? cube(x/.5) : 1;}

class LatexScene : public Scene {
public:
    LatexScene(double extra_scale, const string& eqn, const int width, const int height)
    : Scene(width, height), equation_string(eqn) {
        cout << "rendering latex: " << equation_string << endl;
        ScalingParams sp(pix.w * extra_scale, pix.h * extra_scale);
        equation_pixels = eqn_to_pix(equation_string, sp);
        scale_factor = sp.scale_factor;
        coords = make_pair((pix.w-equation_pixels.w)/2, (pix.h-equation_pixels.h)/2);
        pix.fill(TRANSPARENT_BLACK);
        pix.overwrite(equation_pixels, coords.first, coords.second);
    }

    void append_transition(string eqn) {
        if(in_transition_state) end_transition();
        begin_transition(equation_string + eqn);
    }

    void begin_transition(string eqn) {
        if(in_transition_state) end_transition();
        transition_equation_string = eqn;

        cout << "rendering latex: " << transition_equation_string << endl;
        ScalingParams sp(scale_factor);
        transition_equation_pixels = eqn_to_pix(transition_equation_string, sp);
        transition_coords = make_pair((pix.w-transition_equation_pixels.w)/2, (pix.h-transition_equation_pixels.h)/2);

        cout << equation_string << " <- Finding Intersections -> " << eqn << endl;
        intersections = find_intersections(equation_pixels, transition_equation_pixels);
        in_transition_state = true;
        transition_audio_segment = dag["audio_segment_number"];
    }

    void end_transition(){
        cout << "Ending transition" << endl;
        assert(in_transition_state);
        equation_pixels = transition_equation_pixels;
        coords = transition_coords;
        equation_string = transition_equation_string;
        pix.overwrite(equation_pixels, coords.first, coords.second);
        in_transition_state = false;
    }

    void query(Pixels*& p) override {
        if(in_transition_state && dag["audio_segment_number"] != transition_audio_segment)
            end_transition();

        if(!in_transition_state){
            p = &pix;
        } else { // in a transition
            pix.fill(TRANSPARENT_BLACK);
            double tp = transparency_profile(dag["transition_fraction"]);
            double tp1 = transparency_profile(1-dag["transition_fraction"]);
            double smooth = smoother2(dag["transition_fraction"]);

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

            p = &pix;
        }
    }

private:
    double transition_audio_segment = -1;
    bool in_transition_state = false;
    double scale_factor;

    // Things used for non-transition states
    string equation_string;
    Pixels equation_pixels;
    pair<int, int> coords;

    // Things used for transitions
    string transition_equation_string;
    vector<StepResult> intersections;
    Pixels transition_equation_pixels;
    pair<int, int> transition_coords;
};
