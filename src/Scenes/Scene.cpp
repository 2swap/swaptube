#pragma once

#include <unordered_map>
#include <chrono>
#include <cassert>
#include "../misc/StateManager.cpp"
#include "../io/DebugPlot.h"
#include "../misc/pixels.h"
#include "../io/AudioSegment.cpp"
#include "../io/VisualMedia.cpp"

using namespace std;

static int frame_number;
static bool FOR_REAL = true; // Whether we should actually be writing any AV output
static bool PRINT_TO_TERMINAL = true;
static bool SAVE_FRAME_PNGS = true;

class Scene {
public:
    Scene(const double width = 1, const double height = 1)
        : state_manager(), pix(width*VIDEO_WIDTH, height*VIDEO_HEIGHT) {
        state_manager.add_equation("t", "<frame_number> " + to_string(VIDEO_FRAMERATE) + " /");
        state_manager.add_equation("w", to_string(pix.w));
        state_manager.add_equation("h", to_string(pix.h));
    }

    virtual const StateQuery populate_state_query() const = 0;
    virtual bool check_if_data_changed() const = 0;
    virtual void draw() = 0;
    virtual void change_data() = 0;
    virtual void mark_data_unchanged() = 0;
    virtual void on_end_transition() = 0;
    void update() {
        has_updated_since_last_query = true;
        update_state();
        change_data();
    }
    virtual bool needs_redraw() const {
        bool state_change = check_if_state_changed();
        bool data_change = check_if_data_changed();
        return !has_ever_rendered || state_change || data_change;
    }
    bool check_if_state_changed() const {
        return state != last_state;
    }
    void query(Pixels*& p) {
        State temp_state = state;
        if(!has_updated_since_last_query){
            update();
            has_updated_since_last_query = false;
        }
        update_state(); // We already called update, but it is possible that state was changed in a parent statemanager's values
        if(needs_redraw()){
            pix = Pixels(get_width(), get_height());
            has_ever_rendered = true;
            draw();
        }
        last_state = temp_state;
        mark_data_unchanged();
        p=&pix;
    }

    void inject_audio_and_render(const AudioSegment& audio){
        inject_audio(audio, 1);
        render();
    }

    void inject_audio(const AudioSegment& audio, int expected_microblocks){
        WRITER.add_shtooka(audio);
        if(!FOR_REAL)
            return;
        cout << "Scene says: " << audio.get_subtitle_text() << endl;
        if (remaining_microblocks != 0) {
            throw runtime_error("ERROR: Attempted to add audio without having finished rendering video!\nYou probably forgot to use render()!\n"
                    "This macroblock was created with " + to_string(total_microblocks) + " total microblocks, "
                    "but render() was only called " + to_string(total_microblocks - remaining_microblocks) + " times.");
        }

        total_macroblock_frames = remaining_macroblock_frames = WRITER.add_audio_segment(audio) * VIDEO_FRAMERATE;
        total_microblocks = remaining_microblocks = expected_microblocks;
        cout << "Macroblock should last " << remaining_macroblock_frames << " frames, with " << expected_microblocks << " microblocks.";
    }

    void render(){
        if(!FOR_REAL){
            state_manager.close_microblock_transitions();
            state_manager.close_macroblock_transitions();
            on_end_transition();
            state_manager.evaluate_all();
            return;
        }

        int complete_microblocks = total_microblocks - remaining_microblocks;
        int complete_macroblock_frames = total_macroblock_frames - remaining_macroblock_frames;
        double num_frames_per_session = static_cast<double>(total_macroblock_frames) / total_microblocks;
        int num_frames_to_be_done_after_this_time = round(num_frames_per_session * (complete_microblocks + 1));
        scene_duration_frames = num_frames_to_be_done_after_this_time - complete_macroblock_frames;
        cout << "Rendering a scene. Frame Count: " << scene_duration_frames << " (microblocks left: " << remaining_microblocks << ", " << remaining_macroblock_frames << " frames total)" << endl;

        for (int frame = 0; frame < scene_duration_frames; frame++) {
            render_one_frame(frame);
        }
        cout << endl;
        remaining_microblocks--;
        if(remaining_microblocks == 0){
            global_state["audio_segment_number"]++;
            state_manager.close_macroblock_transitions();
        }
        state_manager.close_microblock_transitions();
        on_end_transition();
    }

    void update_state() {
        state_manager.evaluate_all();
        StateQuery sq = populate_state_query();
        sq.insert("w");
        sq.insert("h");
        state = state_manager.get_state(sq);
    }

    int get_width() const{
        return state_manager.get_state({"w"})["w"];
    }

    int get_height() const{
        return state_manager.get_state({"h"})["h"];
    }

    StateManager state_manager;

private:
    void render_one_frame(int microblock_frame_number){
        auto start_time = chrono::high_resolution_clock::now(); // Start timing

        global_state["frame_number"]++;
        global_state["macroblock_fraction"] = 1 - static_cast<double>(remaining_macroblock_frames) / total_macroblock_frames;
        global_state["microblock_fraction"] = static_cast<double>(microblock_frame_number) / scene_duration_frames;

        state_manager_time_plot.add_datapoint(vector<double>{global_state["macroblock_fraction"], global_state["microblock_fraction"]});

        if (remaining_microblocks == 0) {
            throw runtime_error("ERROR: Attempted to render video, without having added audio first!\nYou probably forgot to inject_audio() or inject_audio_and_render()!");
        }

        Pixels* p = nullptr;
        WRITER.set_time(global_state["frame_number"] / VIDEO_FRAMERATE);
        query(p);
        if(PRINT_TO_TERMINAL && (int(global_state["frame_number"]) % 5 == 0)) p->print_to_terminal();
        if(SAVE_FRAME_PNGS && (int(global_state["frame_number"]) % VIDEO_FRAMERATE == 0)) {
            int roundedFrameNumber = round(global_state["frame_number"]);
            ostringstream stream;
            stream << setw(6) << setfill('0') << roundedFrameNumber;
            ensure_dir_exists(PATH_MANAGER.this_run_output_dir + "frames");
            pix_to_png(p->naive_scale_down(5), "frames/frame_"+stream.str());
        }
        WRITER.add_frame(*p);
        remaining_macroblock_frames--;

        auto end_time = chrono::high_resolution_clock::now(); // End timing
        chrono::duration<double, milli> frame_duration = end_time - start_time; // Calculate duration in milliseconds
        time_per_frame_plot.add_datapoint(frame_duration.count()); // Add the time to DebugPlot
        memutil_plot.add_datapoint(get_free_memory()); // Add the time to DebugPlot
        cout << "#";
        fflush(stdout);
    }

    bool has_updated_since_last_query = false;
    int remaining_microblocks = 0;
    int total_microblocks = 0;
    int scene_duration_frames = 0;
    int remaining_macroblock_frames = 0;
    int total_macroblock_frames = 0;
    State last_state;

protected:
    Pixels pix;
    State state;
    bool has_ever_rendered = false;
};
