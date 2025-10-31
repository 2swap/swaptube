#pragma once

#include <unordered_map>
#include <chrono>
#include <glm/glm.hpp>
#include "../misc/StateManager.cpp"
#include "../io/VisualMedia.cpp"
#include "../io/DebugPlot.h"
#include "../misc/pixels.h"
#include "../io/Macroblock.cpp"
#include "../misc/GetFreeMemory.h"

static int remaining_microblocks_in_macroblock = 0;
static int remaining_frames_in_macroblock = 0;
static int total_microblocks_in_macroblock = 0;
static int total_frames_in_macroblock = 0;

static vector<string> keys_to_record;

class Scene {
public:
    Scene(const double width = 1, const double height = 1)
        : state() {
        state.set("w", to_string(width));
        state.set("h", to_string(height));
    }

    virtual const StateQuery populate_state_query() const = 0;
    virtual bool check_if_data_changed() const = 0;
    virtual void draw() = 0;
    virtual void change_data() = 0;
    virtual void mark_data_unchanged() = 0;

    virtual void on_end_transition_extra_behavior(const TransitionType tt){};
    void on_end_transition(const TransitionType tt) {
        if(tt == MACRO) state.close_transitions(tt);
                        state.close_transitions(MICRO);
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
        return current_state != last_state;
    }

    void query(Pixels*& p) {
        cout << "(" << flush;
        if(!has_updated_since_last_query){
            last_state = current_state;
            update();
        }
        // The only time we skip render entirely is when the project flags to skip a section.
        if(needs_redraw() && FOR_REAL) {
            has_ever_rendered = true;
            pix = Pixels(get_width(), get_height());
            cout << "|" << flush;
            draw();
        }
        mark_data_unchanged();
        has_updated_since_last_query = false;
        p=&pix;
        cout << ")" << flush;
    }

    // TODO can this be made a global static function instead of a member function?
    void stage_macroblock(const Macroblock& macroblock, int expected_microblocks_in_macroblock){
        if (expected_microblocks_in_macroblock <= 0) {
            throw runtime_error("ERROR: Staged a macroblock with non-positive microblock count. (" + to_string(expected_microblocks_in_macroblock) + " microblocks)");
        }
        if (remaining_microblocks_in_macroblock != 0) {
            throw runtime_error("ERROR: Attempted to add audio without having finished rendering video!\nYou probably forgot to use render_microblock()!\n"
                    "This macroblock had " + to_string(total_microblocks_in_macroblock) + " microblocks, "
                    "but render_microblock() was only called " + to_string(total_microblocks_in_macroblock - remaining_microblocks_in_macroblock) + " times.");
        }

        AUDIO_WRITER.encode_buffers();

        total_microblocks_in_macroblock = remaining_microblocks_in_macroblock = expected_microblocks_in_macroblock;
        macroblock.write_shtooka();

        total_frames_in_macroblock = macroblock.invoke_get_macroblock_length_frames();
        if (!rendering_on()) total_frames_in_macroblock = min(500, total_microblocks_in_macroblock); // Don't do too many simmed microblocks in smoketest
        remaining_frames_in_macroblock = total_frames_in_macroblock;

        cout << endl << macroblock.blurb() << " staged to last " << to_string(expected_microblocks_in_macroblock) << " microblock(s), " << to_string(total_frames_in_macroblock) << " frame(s)." << endl;

        double macroblock_length_seconds = static_cast<double>(total_frames_in_macroblock) / FRAMERATE;

        { // Add hints for audio synchronization
            double time = get_global_state("t");
            double microblock_length_seconds = macroblock_length_seconds / expected_microblocks_in_macroblock;
            int macroblock_length_samples = round(macroblock_length_seconds * SAMPLERATE);
            int microblock_length_samples = round(microblock_length_seconds * SAMPLERATE);
            AUDIO_WRITER.add_blip(round(time * SAMPLERATE), MACRO, macroblock_length_samples, microblock_length_samples);
            for(int i = 0; i < expected_microblocks_in_macroblock; i++) {
                AUDIO_WRITER.add_blip(round((time + i * microblock_length_seconds) * SAMPLERATE), MICRO, macroblock_length_samples, microblock_length_samples);
            }
        } // Audio hints
    }

    void render_microblock(){
        if (remaining_microblocks_in_macroblock == 0) {
            throw runtime_error("ERROR: Attempted to render video, without having added audio first!\nYou probably forgot to stage_macroblock()!\nOr perhaps you staged too few microblocks- " + to_string(total_microblocks_in_macroblock) + " were staged, but there should have been more.");
        }

        int complete_microblocks = total_microblocks_in_macroblock - remaining_microblocks_in_macroblock;
        int complete_macroblock_frames = total_frames_in_macroblock - remaining_frames_in_macroblock;
        double num_frames_per_session = static_cast<double>(total_frames_in_macroblock) / total_microblocks_in_macroblock;
        int num_frames_to_be_done_after_this_time = round(num_frames_per_session * (complete_microblocks + 1));
        int scene_duration_frames = num_frames_to_be_done_after_this_time - complete_macroblock_frames;
        if(total_microblocks_in_macroblock < 10)
            cout << "Rendering a microblock. Frame Count: " << scene_duration_frames <<
                " (microblocks left: " << remaining_microblocks_in_macroblock << ", " <<
                remaining_frames_in_macroblock << " frames total)" << endl;

        for (int frame = 0; frame < scene_duration_frames; frame++) {
            render_one_frame(frame, scene_duration_frames);
        }
        remaining_microblocks_in_macroblock--;
        bool done_macroblock = remaining_microblocks_in_macroblock == 0;
                            global_state["microblock_number"]++;
        if (done_macroblock) {
            global_state["macroblock_number"]++;
            if (SAVE_FRAME_PNGS && rendering_on()) {
                int roundedFrameNumber = round(global_state["frame_number"]);
                ostringstream stream;
                stream << setw(6) << setfill('0') << roundedFrameNumber;
                export_frame(stream.str(), 3);
            }
        }
        on_end_transition(done_macroblock ? MACRO : MICRO);
    }

    void update_state() {
        state.evaluate_all();
        StateQuery sq = populate_state_query();
        sq.insert("w");
        sq.insert("h");
        current_state = state.get_state(sq);
        if(global_identifier.size() > 0) publish_global(stage_publish_to_global());
    }

    int get_width() const{
        return VIDEO_WIDTH * state.get_state({"w"})["w"];
    }

    int get_height() const{
        return VIDEO_HEIGHT * state.get_state({"h"})["h"];
    }

    void export_frame(const string& filename, int scaledown = 1) const {
        ensure_dir_exists(PATH_MANAGER.this_run_output_dir + "frames");
        pix_to_png(pix.naive_scale_down(scaledown), "frames/frame_"+filename);
    }

    StateManager state;
    string global_identifier = ""; // This is prefixed before the published global state elements
                                   // to uniquely identify this scene if necessary.
                                   // Empty by default, meaning no state is published.

protected:
    Pixels pix;
    State current_state;
    bool has_ever_rendered = false;

    glm::vec2 get_width_height() const{
        return glm::vec2(get_width(), get_height());
    }

    double get_geom_mean_size() const{ return geom_mean(get_width(),get_height()); }

private:
    State last_state;
    bool has_updated_since_last_query = false;

    virtual unordered_map<string, double> stage_publish_to_global() const { return unordered_map<string, double>(); }
    void publish_global(const unordered_map<string, double>& s) const {
        for(const auto& p : s) {
            global_state[global_identifier + "." + p.first] = p.second;
        }
    }

    void write_one_frame_to_data_plots(const chrono::high_resolution_clock::duration& frame_duration) {
        state_time_plot.add_datapoint(vector<double>{global_state["macroblock_fraction"], global_state["microblock_fraction"], smoother2(global_state["macroblock_fraction"]), smoother2(global_state["microblock_fraction"])});
        time_per_frame_plot.add_datapoint(frame_duration.count());
        memutil_plot.add_datapoint(get_free_memory());

        vector<double> global_values;
        if (globals_plot == nullptr) globals_plot = make_shared<DebugPlot>("Global Recorder", keys_to_record);
        for(const auto& key : keys_to_record) {
            if (global_state.find(key) == global_state.end())
                global_values.push_back(0);
            else global_values.push_back(global_state[key]);
        }
        globals_plot->add_datapoint(global_values);
    }

    void render_one_frame(int microblock_frame_number, int scene_duration_frames) {
        auto start_time = chrono::high_resolution_clock::now();
        cout << "[" << flush;

        global_state["macroblock_fraction"] = 1 - static_cast<double>(remaining_frames_in_macroblock) / total_frames_in_macroblock;
        global_state["microblock_fraction"] = static_cast<double>(microblock_frame_number) / scene_duration_frames;

        Pixels* p = nullptr;
        query(p);

        bool fifth_frame = int(global_state["frame_number"]) % 5 == 0;
        if((!rendering_on() || fifth_frame) && PRINT_TO_TERMINAL) p->print_to_terminal();

        if (rendering_on()) { // Do not encode during smoketest
            VIDEO_WRITER.add_frame(*p);
        }

        auto end_time = chrono::high_resolution_clock::now();
        write_one_frame_to_data_plots(end_time - start_time);

        remaining_frames_in_macroblock--;
        global_state["frame_number"]++;
        global_state["t"] = global_state["frame_number"] / FRAMERATE;
        cout << "]" << flush;
    }
};
