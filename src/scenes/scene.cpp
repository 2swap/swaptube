#pragma once

#include <unordered_map>
#include <cassert>
#include "dagger.cpp"

using namespace std;

int video_sessions_left = 0;

class Scene {
public:
    Scene(const int width, const int height) : w(width), h(height), pix(width, height){};
    Scene() : w(VIDEO_WIDTH), h(VIDEO_HEIGHT), pix(VIDEO_WIDTH, VIDEO_HEIGHT){};
    virtual void query(bool& done_scene, Pixels*& p) = 0;

    void query(Pixels*& p){
        bool b = false;
        query(b, p);
    }

    /*
    void set_variable(double& d, const string& var, const unordered_map<string, double>& variables){
        rendered = false;
        if(variables.find(var) != variables.end())
            d = variables.at(var);
    }
    */

    void resize(int width, int height){
        if(w == width && h == height) return;
        w = width;
        h = height;
        pix = Pixels(w, h);
        rendered = false;
    }

    void inject_audio_and_render(const AudioSegment& audio){
        inject_audio(audio, 1);
        render();
    }

    void inject_audio(const AudioSegment& audio, int expected_video_sessions){
        if(!FOR_REAL)
            return;
        if (video_sessions_left != 0) {
            failout("ERROR: Attempted to add audio twice in a row, without rendering video!\nYou probably forgot to use render()!");
        }

        superscene_frames_left = FOR_REAL ? WRITER->add_audio_segment(audio) * VIDEO_FRAMERATE : 0;
        cout << "Scene should last " << superscene_frames_left << " frames, with " << expected_video_sessions << " sessions.";
        video_sessions_left = expected_video_sessions;
    }

    void render(){
        if(!FOR_REAL)
            return;
        scene_duration_frames = superscene_frames_left / video_sessions_left;
        cout << "Rendering a scene. Frame Count:" << scene_duration_frames << endl;

        time = 0;
        bool done_scene = false;
        while (!done_scene) {
            done_scene = render_one_frame();
        }
        dag.close_all_transitions();
    }

    int w = 0;
    int h = 0;
  
private:
    bool render_one_frame(){
        if(!FOR_REAL)
            return true;
        if (video_sessions_left == 0) {
            failout("ERROR: Attempted to render video, without having added audio first!\nYou probably forgot to inject_audio() or inject_audio_and_render()!");
        }

        dag.set_special("t", video_time_s);
        dag.set_special("transition_fraction", 1 - static_cast<double>(superscene_frames_left) / scene_duration_frames);
        dag.evaluate_all();
        bool done_scene = false;
        Pixels* p = nullptr;
        superscene_frames_left--;
        WRITER->set_time(video_time_s);
        query(done_scene, p);
        assert(p->w == VIDEO_WIDTH && p->h == VIDEO_HEIGHT);
        video_time_s += 1./VIDEO_FRAMERATE;
        if(PRINT_TO_TERMINAL && ((video_num_frames++)%5 == 0)) p->print_to_terminal();
        WRITER->add_frame(*p);
        if(done_scene) video_sessions_left--;
        return done_scene;
    }

protected:
    bool rendered = false;
    bool is_transition = false;
    Pixels pix;
    int time = 0;
    int scene_duration_frames = 0;
    int superscene_frames_left = 0;
};

//#include "mandelbrot_scene.cpp"
#include "latex_scene.cpp"
#include "header_scene.cpp"
#include "c4_scene.cpp"
#include "composite_scene.cpp"
#include "complex_plot_scene.cpp"
#include "3d_scene.cpp"
#include "graph_scene.cpp"
#include "c4_graph_scene.cpp"
#include "mouse_scene.cpp"
