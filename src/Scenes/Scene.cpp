#pragma once

#include <unordered_map>
#include <chrono>
#include <cassert>
#include "../misc/StateManager.cpp"
#include "../io/DebugPlot.h"
#include "../misc/pixels.h"
#include "../io/AudioSegment.cpp"

using namespace std;

static int frame_number;
static bool FOR_REAL = true; // Whether we should actually be writing any AV output
static bool PRINT_TO_TERMINAL = true;

class Scene {
public:
    virtual void draw() = 0;
    Scene(const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : w(width), h(height), state_manager(make_shared<StateManager>()), pix(width, height) {
        state_manager->set_special("frame_number", 0);
        state_manager->set_special("audio_segment_number", 0);
        state_manager->set_special("transition_fraction", 0);
        state_manager->set_special("subscene_transition_fraction", 0);
        state_manager->add_equation("t", "<frame_number> " + to_string(VIDEO_FRAMERATE) + " /");
    }

    // Scenes which contain other scenes use this to populate the StateQuery
    virtual const StateQuery populate_state_query() const = 0;
    virtual bool update_data_objects_check_if_changed() = 0;
    virtual bool has_subscene_state_changed() const {return false;}
    void query(Pixels*& p) {
        update_state();
        if(state != last_state || has_subscene_state_changed() || update_data_objects_check_if_changed())
            draw();
        p=&pix;
    }

    void resize(int width, int height){
        if(w == width && h == height) return;
        w = width;
        h = height;
        pix = Pixels(w, h);
    }

    void inject_audio_and_render(const AudioSegment& audio){
        inject_audio(audio, 1);
        render();
    }

    void inject_audio(const AudioSegment& audio, int expected_video_sessions){
        WRITER.add_shtooka(audio);
        if(!FOR_REAL)
            return;
        cout << "Scene says: " << audio.get_subtitle_text() << endl;
        if (video_sessions_left != 0) {
            failout("ERROR: Attempted to add audio without having finished rendering video!\nYou probably forgot to use render()!\n"
                    "This superscene was created with " + to_string(video_sessions_total) + " total video sessions, "
                    "but render() was only called " + to_string(video_sessions_total-video_sessions_left) + " times.");
        }

        superscene_frames_total = superscene_frames_left = WRITER.add_audio_segment(audio) * VIDEO_FRAMERATE;
        video_sessions_total = video_sessions_left = expected_video_sessions;
        cout << "Scene should last " << superscene_frames_left << " frames, with " << expected_video_sessions << " sessions.";
    }

    void render(){
        if(!FOR_REAL){
            state_manager->close_all_transitions();
            return;
        }

        int video_sessions_done = video_sessions_total - video_sessions_left;
        int superscene_frames_done = superscene_frames_total - superscene_frames_left;
        double num_frames_per_session = static_cast<double>(superscene_frames_total) / video_sessions_total;
        int num_frames_to_be_done_after_this_time = round(num_frames_per_session * (video_sessions_done + 1));
        scene_duration_frames = num_frames_to_be_done_after_this_time - superscene_frames_done;
        cout << "Rendering a scene. Frame Count: " << scene_duration_frames << " (sessions left: " << video_sessions_left << ", " << superscene_frames_left << " frames total)" << endl;

        for (int frame = 0; frame < scene_duration_frames; frame++) {
            render_one_frame(frame);
        }
        cout << endl;
        video_sessions_left--;
        if(video_sessions_left == 0){
            state_manager->close_all_transitions();
            state_manager->set_special("audio_segment_number", (*state_manager)["audio_segment_number"] + 1);
        }
    }

    void update_state() {
        last_state = state;
        state = state_manager->get_state(populate_state_query());
    }

    Pixels* expose_pixels() {
        return &pix;
    }

    int w = 0;
    int h = 0;
    shared_ptr<StateManager> state_manager;

private:
    void render_one_frame(int subscene_frame){
        auto start_time = chrono::high_resolution_clock::now(); // Start timing

        state_manager->set_special("frame_number", frame_number);
        frame_number++;
        state_manager->set_special("transition_fraction", 1 - static_cast<double>(superscene_frames_left) / superscene_frames_total);
        state_manager->set_special("subscene_transition_fraction", static_cast<double>(subscene_frame) / scene_duration_frames);

        state_manager->evaluate_all();
        state_manager_time_plot.add_datapoint(vector<double>{(*state_manager)["t"], (*state_manager)["transition_fraction"], (*state_manager)["subscene_transition_fraction"]});

        if (video_sessions_left == 0) {
            failout("ERROR: Attempted to render video, without having added audio first!\nYou probably forgot to inject_audio() or inject_audio_and_render()!");
        }

        Pixels* p = nullptr;
        WRITER.set_time((*state_manager)["t"]);
        query(p);
        if(PRINT_TO_TERMINAL && (int((*state_manager)["frame_number"]) % 5 == 0)) p->print_to_terminal();
        WRITER.add_frame(*p);
        superscene_frames_left--;

        auto end_time = chrono::high_resolution_clock::now(); // End timing
        chrono::duration<double, milli> frame_duration = end_time - start_time; // Calculate duration in milliseconds
        time_per_frame_plot.add_datapoint(frame_duration.count()); // Add the time to DebugPlot
        memutil_plot.add_datapoint(get_free_memory()); // Add the time to DebugPlot
        cout << "#";
        fflush(stdout);
    }

protected:
    bool rendered = false;
    bool is_transition = false;
    Pixels pix;
    int video_sessions_left = 0;
    int video_sessions_total = 0;
    int scene_duration_frames = 0;
    int superscene_frames_left = 0;
    int superscene_frames_total = 0;
    State state = unordered_map<string, double>{{"THIS IS A SENTINEL STATE. Do not remove. Without me, stateless scenes will never render.", 0.0}};
    State last_state;
};
