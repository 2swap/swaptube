#pragma once

#include "scene.cpp"

class LatexScene : public Scene {
public:
    LatexScene(const int width, const int height, string eqn) : Scene(width, height), equation_string(eqn) {init_latex_scene();}
    LatexScene(string eqn) : Scene(VIDEO_WIDTH, VIDEO_HEIGHT), equation_string(eqn) {init_latex_scene();}

    void init_latex_scene(){
        cout << "rendering latex: " << equation_string << endl;
        equation_pixels = eqn_to_pix(equation_string, pix.w / 640 + 1);
        coords = make_pair((pix.w-equation_pixels.w)/2, (pix.h-equation_pixels.h)/2);
    }

    void append_transition(string eqn) {
        if(in_transition_state) end_transition();
        begin_transition(equation_string + eqn);
    }

    void begin_transition(string eqn) {
        if(in_transition_state) end_transition();
        transition_equation_string = eqn;

        cout << "rendering latex: " << transition_equation_string << endl;
        transition_equation_pixels = eqn_to_pix(transition_equation_string, pix.w / 640 + 1);
        transition_coords = make_pair((pix.w-transition_equation_pixels.w)/2, (pix.h-transition_equation_pixels.h)/2);

        cout << equation_string << " <- Finding Intersections -> " << eqn << endl;
        intersections = find_intersections(equation_pixels, transition_equation_pixels);
        cout << "Number of intersections found: " << intersections.size() << endl;
        in_transition_state = true;
        transition_fraction = -1;
    }

    void end_transition(){
        equation_pixels = transition_equation_pixels;
        coords = transition_coords;
        equation_string = transition_equation_string;
        in_transition_state = false;
    }

    void query(bool& done_scene, Pixels*& p) override {
        pix.fill(BLACK);
        double weight = dag["transition_fraction"];
        // Define end of transition as falling edge of transition_fraction
        if(weight < transition_fraction) end_transition();
        transition_fraction = weight;
        if(!in_transition_state){
            pix.copy(equation_pixels, coords.first, coords.second, 1);
            p = &pix;
        } else { // in a transition
            double tp = transparency_profile(weight);
            double tp1 = transparency_profile(1-weight);
            double smooth = smoother2(weight);

            for (int i = 0; i < intersections.size(); i++) {
                const StepResult& step = intersections[i];
                int x = round(lerp(coords.first , transition_coords.first -step.max_x, smooth));
                int y = round(lerp(coords.second, transition_coords.second-step.max_y, smooth));

                // Render the intersection at the interpolated position
                pix.copy(step.induced1, x-step.current_p1.w, y-step.current_p1.h, tp1);
                pix.copy(step.induced2, x-step.current_p2.w, y-step.current_p2.h, tp);

                //pix.copy(step.intersection, 0  , i*191, 1);
                //pix.copy(step.map         , 500, i*191, 1);
            }

            StepResult last_intersection = intersections[intersections.size()-1];
            pix.copy(last_intersection.current_p1, coords.first, coords.second, tp1*tp1);
            pix.copy(last_intersection.current_p2, transition_coords.first, transition_coords.second, tp*tp);

            p = &pix;
        }
    }

private:
    double transition_fraction = -1;
    bool in_transition_state;

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
