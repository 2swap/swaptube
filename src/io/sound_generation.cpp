
    void generate_beep(double duration){
        vector<float> left;
        vector<float> right;
        int total_samples = duration*44100;
        double note = state["tone"];
        for(int i = 0; i < total_samples; i++){
            double pct_complete = i/static_cast<double>(total_samples);
            float val = .03*sin(i*note*2200./44100.)/note;
            val *= pow(.5, 4*pct_complete);
            left.push_back(val);
            right.push_back(val);
        }
        WRITER.add_sfx(left, right, state["t"]*44100);
    }

