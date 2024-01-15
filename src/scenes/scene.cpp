#pragma once

#include <unordered_map>
#include <cassert>

using namespace std;

enum VIDEO_WRITING_STATUS {
    READY_FOR_AUDIO,
    READY_FOR_VIDEO
} write_status = READY_FOR_AUDIO;

class Scene {
public:
    Scene(const int width, const int height) : w(width), h(height), pix(width, height){};
    Scene() : w(VIDEO_WIDTH), h(VIDEO_HEIGHT), pix(VIDEO_WIDTH, VIDEO_HEIGHT){};
    virtual void query(bool& done_scene, Pixels*& p) = 0;
    virtual void update_variables(const unordered_map<string, double>& variables) {};
    virtual unordered_map<string, string> get_default_variables() {return unordered_map<string, string>();};

    void query(Pixels*& p){
        bool b = false;
        query(b, p);
    }

    void set_variable(double& d, const string& var, const unordered_map<string, double>& variables){
        rendered = false;
        if(variables.find(var) != variables.end())
            d = variables.at(var);
    }

    void resize(int width, int height){
        if(w == width && h == height) return;
        w = width;
        h = height;
        pix = Pixels(w, h);
        rendered = false;
    }

    void inject_audio_and_render(const AudioSegment& audio){
        inject_audio(audio);
        render();
    }

    void inject_audio(const AudioSegment& audio){
        if (write_status != READY_FOR_AUDIO) {
            cerr << "======================================================\n";
            cerr << "ERROR: Attempted to add audio twice in a row, without rendering video!\n";
            cerr << "You probably forgot to use render()!\n";
            cerr << "Exiting the program...\n";
            cerr << "======================================================\n";
            exit(EXIT_FAILURE);
        }

        time = 0;
        scene_duration_frames = WRITER->add_audio_segment(audio) * VIDEO_FRAMERATE;
        write_status = READY_FOR_VIDEO;
    }

    void render(){
        if (write_status != READY_FOR_VIDEO) {
            cerr << "======================================================\n";
            cerr << "ERROR: Attempted to render video, without having added audio first!\n";
            cerr << "You probably forgot to inject_audio() or inject_audio_and_render()!\n";
            cerr << "Exiting the program...\n";
            cerr << "======================================================\n";
            exit(EXIT_FAILURE);
        }

        cout << "Rendering a scene" << endl;
        cout << "Frame Count:" << scene_duration_frames << endl;

        bool done_scene = false;
        Pixels* p = nullptr;
        while (!done_scene) {
            WRITER->set_time(video_time_s);
            query(done_scene, p);
            assert(p->w == VIDEO_WIDTH && p->h == VIDEO_HEIGHT);
            video_time_s += 1./VIDEO_FRAMERATE;
            if((video_num_frames++)%5 == 0) p->print_to_terminal();
            WRITER->add_frame(*p);
        }
        write_status = READY_FOR_AUDIO;
    }

    int w = 0;
    int h = 0;
  
protected:
    bool rendered = false;
    bool is_transition = false;
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
#include "complex_plot_scene.cpp"
#include "3d_scene.cpp"
#include "graph_scene.cpp"
#include "c4_graph_scene.cpp"
