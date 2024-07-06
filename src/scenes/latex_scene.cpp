#pragma once

#include "../misc/visual_media.cpp"
#include "scene.cpp"

class LatexScene : public Scene {
public:
    LatexScene(const int width, const int height, string eqn) : Scene(width, height), equation_string(eqn) {init_latex_scene();}
    LatexScene(string eqn) : Scene(VIDEO_WIDTH, VIDEO_HEIGHT), equation_string(eqn) {init_latex_scene();}

    void init_latex_scene(){
        cout << "rendering latex: " << equation_string << endl;
        ScalingParams sp(pix.w, pix.h);
        equation_pixels = eqn_to_pix(equation_string, sp);
        scale_factor = sp.scale_factor;
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
        ScalingParams sp(scale_factor);
        transition_equation_pixels = eqn_to_pix(transition_equation_string, sp);
        transition_coords = make_pair((pix.w-transition_equation_pixels.w)/2, (pix.h-transition_equation_pixels.h)/2);

        cout << equation_string << " <- Finding Intersections -> " << eqn << endl;
        intersections = find_intersections(equation_pixels, transition_equation_pixels);
        cout << "Number of intersections found: " << intersections.size() << endl;
        in_transition_state = true;
    }

    void end_transition(){
        equation_pixels = transition_equation_pixels;
        coords = transition_coords;
        equation_string = transition_equation_string;
        in_transition_state = false;
    }

    void query(bool& done_scene, Pixels*& p) override {
        pix.fill(TRANSPARENT_BLACK);

        if(in_transition_state){
            // On rising edge of audio segment, we end transition
            if(dag["audio_segment_number"] > last_audio_segment_number) end_transition();
        }
        
        if(!in_transition_state){
            pix.overwrite(equation_pixels, coords.first, coords.second);
            p = &pix;
        } else { // in a transition
            double tp = transparency_profile(dag["transition_fraction"]);
            double tp1 = transparency_profile(1-dag["transition_fraction"]);
            double smooth = smoother2(dag["transition_fraction"]);

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
        last_audio_segment_number = dag["audio_segment_number"];
    }

private:
    double last_audio_segment_number = 0;
    bool in_transition_state;
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
