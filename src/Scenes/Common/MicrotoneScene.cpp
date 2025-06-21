#pragma once

#include "CoordinateScene.cpp"
#include "../../io/SFX.cpp"
#include <vector>

class MicrotoneScene : public CoordinateScene {
public:
    MicrotoneScene(const double width = 1, const double height = 1)
        : CoordinateScene(width, height) { }

    void add_sound(const FourierSound& fs) {
        scape.add(fs);
        circles_to_render++;
    }

    void draw() override {
        CoordinateScene::draw();
        generate_tone();
    }

    void generate_tone(){
        if(elapsed == 0) elapsed = state["t"]*SAMPLERATE;
        new_freqs.clear();
        for (int i = 0; i < circles_to_render; i++) {
            new_freqs.push_back(pow(5./4, state["circle" + to_string(i) + "_x"]) * pow(3./2, state["circle" + to_string(i) + "_y"]));
        }
        int total_samples = SAMPLERATE/FRAMERATE;
        vector<float> left;
        vector<float> right;
        scape.generate_audio(total_samples, left, right, new_freqs);
        WRITER.add_sfx(left, right, elapsed);
        elapsed += left.size();
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        sq.insert("t");
        return sq;
    }

private:
    vector<double> new_freqs;
    Soundscape scape;
    int elapsed = 0;
};

