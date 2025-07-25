#include "../Scenes/Common/CompositeScene.cpp"

void generate_audio(double duration, vector<float>& left, vector<float>& right){
    int total_samples = duration*SAMPLERATE;
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
            float val = .03*sin(i*pow(2, notes[note]/12.)*2050./SAMPLERATE);
            val *= pow(.5, 4*pct_complete);
                 val += .04*sin(i*pow(2, bass[note]/12.-1)*2050./SAMPLERATE);
            left.push_back(val);
            right.push_back(val);
        }
    }
}

void render_video() {
    CompositeScene cs;
    vector<float> left; vector<float> right;
    generate_audio(12, left, right);
    cs.stage_macroblock(GeneratedBlock(left, right), 1);
    cs.render_microblock();
}
