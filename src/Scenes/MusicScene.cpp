#pragma once
#include "Scene.cpp"

class MusicScene : public Scene {
public:
    MusicScene(const double width = 1, const double height = 1) : Scene(width, height) { }

    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }

    void mark_data_unchanged() override { }
    void change_data() override { }
    bool check_if_data_changed() const override { return false;}

    void draw() override { }

    void generate_audio(double duration, vector<float>& left, vector<float>& right){
        int total_samples = duration*44100;
        vector<double> notes{12, 5, 17, 22, 21, 15, 17, 9, 12, 5, 17, 12, 22, 17, 29, 34, 33, 17, 25, 17, 27, 19, 29, 24, 21, 22, 15, 17, 27, 26, 22, 10, 19, 17, 12, 5, 15, 14, 10, 22, 24, 17, 20, 22, 29, 24, 22, 24};
        vector<double> bass{5, 5, 5, -100, -100, 5, 3, 3, 3, -100, -100, 3, 2, 2, 2, -100, -100, 2, 1, 1, 1, 3, 3, 3, 5, 5, 5, -100, -100, 5, 3, 3, 3, -100, -100, 3, 2, 2, 2, -100, -100, 2, 8, 8, 10, 3, 3, 3, };
        int first = notes.size();
        for(int i = 0; i < first; i++) {
            notes.push_back(notes[i]+2);
            bass.push_back(bass[i]+2);
        };
        int samples_per_note = total_samples/notes.size();
        for(int note = 0; note < notes.size(); note++){
            for(int i = 0; i < samples_per_note; i++){
                double pct_complete = i/static_cast<double>(samples_per_note);
                float val = .03*sin(i*pow(2, notes[note]/12.)*2050./44100.);
                val *= pow(.5, 4*pct_complete);
                     val += .04*sin(i*pow(2, bass[note]/12.-1)*2050./44100.);
                left.push_back(val);
                right.push_back(val);
            }
        }
    }
};
