#pragma once

#include <unordered_map>
#include <cassert>

using namespace std;

class Scene {
public:
    Scene(const int width, const int height) : w(width), h(height), pix(width, height){};
    virtual Pixels* query(bool& done_scene) = 0;
    virtual void update_variables(const unordered_map<string, double>& variables) {};

    void render(const AudioSegment& audio){
        scene_duration_frames = WRITER->add_audio_segment(audio) * VIDEO_FRAMERATE;
        cout << "Rendering a scene" << endl;
        cout << "Frame Count:" << scene_duration_frames << endl;

        bool done_scene = false;
        while (!done_scene) {
            WRITER->set_audiotime(video_time_s);
            Pixels* p = query(done_scene);
            assert(p->w == VIDEO_WIDTH && p->h == VIDEO_HEIGHT);
            video_time_s += 1./VIDEO_FRAMERATE;
            if((video_num_frames++)%15 == 0) p->print_to_terminal();
            WRITER->addFrame(*p);
        }
    }

    int w = 0;
    int h = 0;
  
protected:
    Pixels pix;
    int time = 0;
    int scene_duration_frames = 0;
};

//#include "mandelbrot_scene.cpp"
#include "latex_scene.cpp"
#include "header_scene.cpp"
#include "c4_scene.cpp"
#include "twoswap_scene.cpp"
#include "composite_scene.cpp"
#include "variable_scene.cpp"
