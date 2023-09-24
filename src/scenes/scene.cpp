#pragma once

#include <unordered_map>
#include <cassert>

using namespace std;

class Scene {
public:
    Scene(const int width, const int height) : w(width), h(height), pix(width, height){};
    virtual Scene* createScene(const int width, const int height) = 0;
    virtual const Pixels& query(bool& done_scene) = 0;
    virtual void update_variables(const unordered_map<string, double>& variables) {};

    void render(const AudioSegment& audio){
        scene_duration_frames = WRITER->add_audio_segment(audio) * VIDEO_FRAMERATE;

        bool done_scene = false;
        while (!done_scene) {
            WRITER->set_audiotime(video_time_s);
            const Pixels& p = query(done_scene);
            video_time_s += 1./VIDEO_FRAMERATE;
            if((video_num_frames++)%15 == 0) p.print_to_terminal();
            WRITER->set_audiotime(0.0);
            WRITER->addFrame(p);
        }
    }
  
protected:
    Pixels pix;
    int w = 0;
    int h = 0;
    int time = 0;
    int scene_duration_frames = 0;
};

//#include "mandelbrot_scene.cpp"
//#include "latex_scene.cpp"
#include "header_scene.cpp"
#include "c4_scene.cpp"
#include "twoswap_scene.cpp"
#include "composite_scene.cpp"
#include "variable_scene.cpp"

static Scene* create_scene_determine_type(const int width, const int height, const string& scene_type) {
    cout << endl << "Creating a " << scene_type << " scene" << endl;
    if (scene_type == "c4") {
        return new C4Scene(width, height);
    }
    //else if (scene_type == "latex") {
    //    return new LatexScene(width, height);
    //}
    //else if (scene_type == "mandelbrot") {
    //    return new MandelbrotScene(width, height);
    //}
    else if (scene_type == "header") {
        return new HeaderScene(width, height);
    }
    else if (scene_type == "2swap") {
        return new TwoswapScene(width, height);
    }
    else if (scene_type == "composite") {
        return new CompositeScene(width, height);
    }
    else if (scene_type == "variable") {
        return new VariableScene(width, height);
    }
    else {
        cerr << "Unknown scene type: " << scene_type << endl;
        exit(1);
    }
}