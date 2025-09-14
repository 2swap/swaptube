#pragma once

#include <unordered_map>
#include <chrono>
#include <cassert>
#include <glm/glm.hpp>
#include "../misc/StateManager.cpp"
#include "../io/VisualMedia.cpp"
#include "../io/DebugPlot.h"
#include "../misc/pixels.h"
#include "../io/Macroblock.cpp"

class Scene {
public:
    Scene(const double width = 1, const double height = 1)
        : state_manager() {
        state_manager.set("w", to_string(width));
        state_manager.set("h", to_string(height));
    }

    virtual const StateQuery populate_state_query() const = 0;
    virtual bool check_if_data_changed() const = 0;
    virtual void draw() = 0;
    virtual void change_data() = 0;
    virtual void mark_data_unchanged() = 0;

    virtual void on_end_transition_extra_behavior(const TransitionType tt){};
    void on_end_transition(const TransitionType tt) {
        if(tt == MACRO) state_manager.close_transitions(tt);
                        state_manager.close_transitions(MICRO);
        on_end_transition_extra_behavior(tt);
    }

    void update() {
        has_updated_since_last_query = true;

        // Data and state can be co-dependent, so update state before and after since state changes are idempotent.
        update_state();
        change_data();
        update_state();
    }

    virtual bool needs_redraw() const {
        bool state_change = check_if_state_changed();
        bool data_change = check_if_data_changed();
        cout << (state_change ? "S" : ".") << (data_change ? "D" : ".") << flush;
        return !has_ever_rendered || state_change || data_change;
    }

    bool check_if_state_changed() const {
        return state != last_state;
    }

    void query(Pixels*& p) {
        cout << "(" << flush;
        if(!has_updated_since_last_query){
            last_state = state;
            update();
        }
        if(needs_redraw()){
            pix = Pixels(get_width(), get_height());
            cout << "|" << flush;
            has_ever_rendered = true;
            draw();
        }
        mark_data_unchanged();
        has_updated_since_last_query = false;
        p=&pix;
        cout << ")" << flush;
    }

    int microblocks_remaining() {
        return remaining_microblocks;
    }

    void stage_macroblock(const Macroblock& audio, int expected_microblocks){
        if (expected_microblocks <= 0){
            throw runtime_error("ERROR: Staged a macroblock with non-positive microblock count. (" + to_string(expected_microblocks) + " microblocks)");
        }
        if (remaining_microblocks != 0) {
            throw runtime_error("ERROR: Attempted to add audio without having finished rendering video!\nYou probably forgot to use render_microblock()!\n"
                    "This macroblock had " + to_string(total_microblocks) + " microblocks, "
                    "but render_microblock() was only called " + to_string(total_microblocks - remaining_microblocks) + " times.");
        }

        total_microblocks = expected_microblocks;
        remaining_microblocks = expected_microblocks;
        cout << endl << audio.blurb() << " staged to last " << to_string(expected_microblocks) << " microblock(s)." << endl;
        audio.write_shtooka();
        if(FOR_REAL) {
            double macroblock_length_seconds = audio.invoke_get_macroblock_length_seconds();
            total_macroblock_frames = remaining_macroblock_frames = macroblock_length_seconds * FRAMERATE;

            // Add blips for audio synchronization
            double time = get_global_state("t");
            //AUDIO_WRITER.add_blip(time * SAMPLERATE, false);
            double microblock_length_seconds = macroblock_length_seconds / expected_microblocks;
            for(int i = 0; i < expected_microblocks; i++) {
                //AUDIO_WRITER.add_blip((time + i * microblock_length_seconds) * SAMPLERATE, true);
            }
        }
    }

    int render_microblock(){
        if (remaining_microblocks == 0) {
            throw runtime_error("ERROR: Attempted to render video, without having added audio first!\nYou probably forgot to stage_macroblock()!\nOr perhaps you staged too few microblocks- " + to_string(total_microblocks) + " were staged, but there should have been more.");
        }

        int complete_microblocks = total_microblocks - remaining_microblocks;
        int complete_macroblock_frames = total_macroblock_frames - remaining_macroblock_frames;
        double num_frames_per_session = static_cast<double>(total_macroblock_frames) / total_microblocks;
        int num_frames_to_be_done_after_this_time = round(num_frames_per_session * (complete_microblocks + 1));
        scene_duration_frames = num_frames_to_be_done_after_this_time - complete_macroblock_frames;
        if(total_microblocks < 5)
            cout << "Rendering a microblock. Frame Count: " << scene_duration_frames <<
                " (microblocks left: " << remaining_microblocks << ", " <<
                remaining_macroblock_frames << " frames total)" << endl;

        for (int frame = 0; frame < scene_duration_frames; frame++) {
            render_one_frame(frame);
        }
        remaining_microblocks--;
        bool done_macroblock = remaining_microblocks == 0;
                            global_state["microblock_number"]++;
        if (done_macroblock) {
            global_state["macroblock_number"]++;
            if (SAVE_FRAME_PNGS) {
                int roundedFrameNumber = round(global_state["frame_number"]);
                ostringstream stream;
                stream << setw(6) << setfill('0') << roundedFrameNumber;
                export_frame(stream.str(), 4);
            }
        }
        on_end_transition(done_macroblock ? MACRO : MICRO);

        // return the total number of frames rendered here
        return scene_duration_frames;
    }

    void update_state() {
        state_manager.evaluate_all();
        StateQuery sq = populate_state_query();
        sq.insert("w");
        sq.insert("h");
        state = state_manager.get_state(sq);
        if(global_publisher_key) publish_global(stage_publish_to_global());
    }

    int get_width() const{
        return VIDEO_WIDTH * state_manager.get_state({"w"})["w"];
    }

    int get_height() const{
        return VIDEO_HEIGHT * state_manager.get_state({"h"})["h"];
    }

    void export_frame(const string& filename, int scaledown = 1) const {
        ensure_dir_exists(PATH_MANAGER.this_run_output_dir + "frames");
        pix_to_png(pix.naive_scale_down(scaledown), "frames/frame_"+filename);
    }

    StateManager state_manager;
    bool global_publisher_key = false; // Scenes can publish to global state only if this is manually set to true in the project
    string global_identifier = ""; // This is prefixed before the published global state elements to uniquely identify this scene if necessary. Not used (empty) by default.

protected:
    Pixels pix;
    State state;
    bool has_ever_rendered = false;

    glm::vec2 get_width_height() const{
        return glm::vec2(get_width(), get_height());
    }

    double get_geom_mean_size() const{ return geom_mean(get_width(),get_height()); }

private:
    State last_state;
    bool has_updated_since_last_query = false;
    int remaining_microblocks = 0;
    int total_microblocks = 0;
    int scene_duration_frames = 0;
    int remaining_macroblock_frames = 0;
    int total_macroblock_frames = 0;

    virtual unordered_map<string, double> stage_publish_to_global() const { return unordered_map<string, double>(); }
    void publish_global(const unordered_map<string, double>& s) const {
        for(const auto& p : s) {
            global_state[global_identifier + p.first] = p.second;
        }
    }

    void render_one_frame(int microblock_frame_number){
        auto start_time = chrono::high_resolution_clock::now(); // Start timing
        cout << "[" << flush;

        global_state["macroblock_fraction"] = 1 - static_cast<double>(remaining_macroblock_frames) / total_macroblock_frames;
        global_state["microblock_fraction"] = static_cast<double>(microblock_frame_number) / scene_duration_frames;
        global_state["t"] = global_state["frame_number"] / FRAMERATE;

        if(FOR_REAL) {
            state_manager_time_plot.add_datapoint(vector<double>{global_state["macroblock_fraction"], global_state["microblock_fraction"], smoother2(global_state["macroblock_fraction"]), smoother2(global_state["microblock_fraction"])});
            SUBTITLE_WRITER.set_substime(global_state["frame_number"] / FRAMERATE);
            Pixels* p = nullptr;
            query(p);
            if(int(global_state["frame_number"]) % 5 == 0 && PRINT_TO_TERMINAL) p->print_to_terminal();
            VIDEO_WRITER.add_frame(*p);

            auto end_time = chrono::high_resolution_clock::now(); // End timing
            chrono::duration<double, milli> frame_duration = end_time - start_time; // Calculate duration in milliseconds
            time_per_frame_plot.add_datapoint(frame_duration.count());
            cumulative_time_plot.add_datapoint(std::chrono::duration_cast<std::chrono::nanoseconds>(start_time.time_since_epoch()).count() / 1000000000.0);
            memutil_plot.add_datapoint(get_free_memory());
        }

        remaining_macroblock_frames--;
        global_state["frame_number"]++;
        //GUI.timeline_window.update(global_state["frame_number"], global_state["t"], total_microblocks - remaining_microblocks, total_microblocks, microblock_frame_number, scene_duration_frames);
        cout << "]" << flush;
    }
};
