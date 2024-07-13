#pragma once

#include <unordered_map>
#include <cassert>
#include "dagger.cpp"

using namespace std;

class Scene {
public:
    Scene(const int width, const int height) : w(width), h(height), pix(width, height){};
    Scene() : w(VIDEO_WIDTH), h(VIDEO_HEIGHT), pix(VIDEO_WIDTH, VIDEO_HEIGHT){};
    virtual void query(bool& done_scene, Pixels*& p) = 0;

    void query(Pixels*& p){
        bool b = false;
        query(b, p);
    }

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
        cout << "Scene says: " << audio.get_subtitle_text() << endl;
        if (video_sessions_left != 0) {
            failout("ERROR: Attempted to add audio without having finished rendering video!\nYou probably forgot to use render()!\n"
                    "This superscene was created with " + to_string(video_sessions_total) + " total video sessions, "
                    "but render() was only called " + to_string(video_sessions_total-video_sessions_left) + "times.");
        }

        superscene_frames_total = superscene_frames_left = FOR_REAL ? WRITER->add_audio_segment(audio) * VIDEO_FRAMERATE : 0;
        video_sessions_total = video_sessions_left = expected_video_sessions;
        cout << "Scene should last " << superscene_frames_left << " frames, with " << expected_video_sessions << " sessions.";
    }

    void render(){
        if(!FOR_REAL){
            dag.close_all_transitions();
            return;
        }

        int video_sessions_done = video_sessions_total - video_sessions_left;
        int superscene_frames_done = superscene_frames_total - superscene_frames_left;
        double num_frames_per_session = static_cast<double>(superscene_frames_total) / video_sessions_total;
        int num_frames_to_be_done_after_this_time = round(num_frames_per_session * (video_sessions_done + 1));
        scene_duration_frames = num_frames_to_be_done_after_this_time - superscene_frames_done;
        cout << "Rendering a scene. Frame Count: " << scene_duration_frames << " (sessions left: " << video_sessions_left << ", " << superscene_frames_left << " frames total)" << endl;

        time = 0;
        for (int frame = 0; frame < scene_duration_frames; frame++) {
            render_one_frame(frame);
        }
        video_sessions_left--;
        if(video_sessions_left == 0){
            dag.close_all_transitions();
            dag.set_special("audio_segment_number", dag["audio_segment_number"] + 1);
        }
    }

    Pixels* expose_pixels() {
        return &pix;
    }

    int w = 0;
    int h = 0;
  
private:
    void render_one_frame(int subscene_frame){
        dag.set_special("frame_number", dag["frame_number"] + 1);
        dag.set_special("t", dag["frame_number"] / VIDEO_FRAMERATE);
        dag.set_special("transition_fraction", 1 - static_cast<double>(superscene_frames_left) / superscene_frames_total);
        dag.set_special("subscene_transition_fraction", static_cast<double>(subscene_frame) / scene_duration_frames);
        dag.evaluate_all();

        if (video_sessions_left == 0) {
            failout("ERROR: Attempted to render video, without having added audio first!\nYou probably forgot to inject_audio() or inject_audio_and_render()!");
        }

        bool unused = false;
        Pixels* p = nullptr;
        WRITER->set_time(dag["t"]);
        query(unused, p);
        assert(p->w == VIDEO_WIDTH && p->h == VIDEO_HEIGHT);
        if(PRINT_TO_TERMINAL && (int(dag["frame_number"]) % 5 == 0)) p->print_to_terminal();
        WRITER->add_frame(*p);
        superscene_frames_left--;
    }

protected:
    bool rendered = false;
    bool is_transition = false;
    Pixels pix;
    int time = 0;
    int video_sessions_left = 0;
    int video_sessions_total = 0;
    int scene_duration_frames = 0;
    int superscene_frames_left = 0;
    int superscene_frames_total = 0;
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
#include "png_scene.cpp"
#include "exposed_pixels_scene.cpp"
