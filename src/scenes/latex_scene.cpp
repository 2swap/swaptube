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
        pix.copy(equation_pixels, coords.first, coords.second, 1);
    }

    void query(bool& done_scene, Pixels*& p) override {
        done_scene = scene_duration_frames <= time;
        time++;
        p = &pix;
    }

    string equation_string;
    Pixels equation_pixels;
    pair<int, int> coords;
};

class LatexTransitionScene : public Scene {
public:
    //interpolative constructor
    LatexTransitionScene(const LatexScene& subscene1, const LatexScene& subscene2):Scene(subscene1.w, subscene1.h) {
        assert(subscene1.w == subscene2.w);
        assert(subscene1.h == subscene2.h);
        cout << subscene1.equation_string << " <- Finding Intersections -> " << subscene2.equation_string << endl;
        intersections = find_intersections(subscene1.equation_pixels, subscene2.equation_pixels);
        cout << "Number of intersections found: " << intersections.size() << endl;
        coords1 = subscene1.coords;
        coords2 = subscene2.coords;
    }

    void query(bool& done_scene, Pixels*& p) override {
        pix.fill(BLACK);
        double weight = static_cast<double>(time)/scene_duration_frames;
        done_scene = scene_duration_frames <= time;

        double tp = transparency_profile(weight);
        double tp1 = transparency_profile(1-weight);
        double smooth = smoother2(weight);

        for (int i = 0; i < intersections.size(); i++) {
            const StepResult& step = intersections[i];
            int x = round(lerp(coords1.first , coords2.first -step.max_x, smooth));
            int y = round(lerp(coords1.second, coords2.second-step.max_y, smooth));

            // Render the intersection at the interpolated position
            pix.copy(step.induced1, x-step.current_p1.w, y-step.current_p1.h, tp1);
            pix.copy(step.induced2, x-step.current_p2.w, y-step.current_p2.h, tp);

            //pix.copy(step.intersection, 0  , i*191, 1);
            //pix.copy(step.map         , 500, i*191, 1);
        }

        StepResult last_intersection = intersections[intersections.size()-1];
        pix.copy(last_intersection.current_p1, coords1.first, coords1.second, tp1*tp1);
        pix.copy(last_intersection.current_p2, coords2.first, coords2.second, tp*tp);

        time++;

        p = &pix;
    }

private:
    vector<StepResult> intersections;
    pair<int, int> coords1;
    pair<int, int> coords2;
};
