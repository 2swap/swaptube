#include "Scene.h"
#include "../Core/Smoketest.h"
#include "../IO/Writer.h"

int remaining_microblocks_in_macroblock = 0;
int remaining_frames_in_macroblock = 0;
int total_microblocks_in_macroblock = 0;
int total_frames_in_macroblock = 0;

void stage_macroblock(const Macroblock& macroblock, int expected_microblocks_in_macroblock){
    if (expected_microblocks_in_macroblock <= 0) {
        throw runtime_error("ERROR: Staged a macroblock with non-positive microblock count. (" + to_string(expected_microblocks_in_macroblock) + " microblocks)");
    }
    if (remaining_microblocks_in_macroblock != 0) {
        throw runtime_error("ERROR: Attempted to add audio without having finished rendering video!\nYou probably forgot to use render_microblock()!\n"
                "This macroblock had " + to_string(total_microblocks_in_macroblock) + " microblocks, "
                "but render_microblock() was only called " + to_string(total_microblocks_in_macroblock - remaining_microblocks_in_macroblock) + " times.");
    }

    get_writer().audio->encode_buffers();

    total_microblocks_in_macroblock = remaining_microblocks_in_macroblock = expected_microblocks_in_macroblock;
    cout << "Set remaining microblocks in macroblock to " << to_string(remaining_microblocks_in_macroblock) << endl;
    macroblock.write_shtooka();

    total_frames_in_macroblock = macroblock.write_and_get_duration_frames();
    if (!rendering_on()) total_frames_in_macroblock = min(10, total_microblocks_in_macroblock); // Don't do too many simmed microblocks in smoketest
    cout << "Set total frames in macroblock to " << to_string(total_frames_in_macroblock) << ". We are " << (rendering_on() ? "rendering" : "smoketesting") << "." << endl;
    remaining_frames_in_macroblock = total_frames_in_macroblock;

    cout << endl << macroblock.blurb() << " staged to last " << to_string(expected_microblocks_in_macroblock) << " microblock(s), " << to_string(total_frames_in_macroblock) << " frame(s)." << endl;

    double macroblock_length_seconds = static_cast<double>(total_frames_in_macroblock) / get_video_framerate_fps();

    if (AUDIO_HINTS) { // Add hints for audio synchronization
        double time = get_global_state("t");
        double microblock_length_seconds = macroblock_length_seconds / expected_microblocks_in_macroblock;
        int macroblock_length_samples = round(macroblock_length_seconds * get_audio_samplerate_hz());
        int microblock_length_samples = round(microblock_length_seconds * get_audio_samplerate_hz());
        get_writer().audio->add_blip(round(time * get_audio_samplerate_hz()), MACRO, macroblock_length_samples, microblock_length_samples);
        for(int i = 0; i < expected_microblocks_in_macroblock; i++) {
            get_writer().audio->add_blip(round((time + i * microblock_length_seconds) * get_audio_samplerate_hz()), MICRO, macroblock_length_samples, microblock_length_samples);
        }
    } // Audio hints
}

Scene::Scene(const double width, const double height)
    : state() {
    manager.set({
        {"w", to_string(width)},
        {"h", to_string(height)},
    });
}

void Scene::on_end_transition(const TransitionType tt) {
    if(tt == MACRO) manager.close_transitions(tt);
                    manager.close_transitions(MICRO);
    on_end_transition_extra_behavior(tt);
}

void Scene::update() {
    has_updated_since_last_query = true;

    // Data and state can be co-dependent, so update state before and after since state changes are idempotent.
    last_state = state;
    update_state();
    change_data();
    update_state();
}

bool Scene::needs_redraw() const {
    bool state_change = check_if_state_changed();
    bool data_change = check_if_data_changed();
    cout << (state_change ? "S" : ".") << (data_change ? "D" : ".") << flush;
    return !has_ever_rendered || state_change || data_change;
}

bool Scene::check_if_state_changed() const {
    return state != last_state;
}

void Scene::query(Pixels*& p) {
    cout << "(" << flush;
    if(!has_updated_since_last_query) update();

    // The only time we skip render entirely is when the project flags to skip a section.
    if(needs_redraw() && is_for_real()) {
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

void Scene::render_microblock(){
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
    set_global_state("microblock_number", get_global_state("microblock_number") + 1);
    if (done_macroblock) {
        set_global_state("macroblock_number", get_global_state("macroblock_number") + 1);
        if (rendering_on()) {
            int roundedFrameNumber = round(get_global_state("frame_number"));
            ostringstream stream;
            stream << setw(6) << setfill('0') << roundedFrameNumber;
            export_frame(stream.str(), 1);
        }
    }
    on_end_transition(done_macroblock ? MACRO : MICRO);
}

void Scene::update_state() {
    manager.evaluate_all();
    StateQuery sq = populate_state_query();
    sq.insert("w");
    sq.insert("h");
    state = manager.respond_to_query(sq);
    if(global_identifier.size() > 0) publish_global();
}

int Scene::get_width() const{
    // TODO shouldn't this really be the container/parent size, not the video?
    return get_video_width_pixels() * manager.respond_to_query({"w"})["w"];
}

int Scene::get_height() const{
    return get_video_height_pixels() * manager.respond_to_query({"h"})["h"];
}

void Scene::export_frame(const string& filename, int scaledown) const {
    pix_to_png(pix.naive_scale_down(scaledown), "frames/frame_"+filename);
}

void Scene::set_global_identifier(const string& id){
    // What we are actually here to do
    global_identifier = id;

    // We also need to publish it immediately, or else it may not be present on the first frame
    // of something trying to read from global, since global ordering is not guaranteed.
    // Update_state does this for us.
    update_state();
}

vec2 Scene::get_width_height() const{
    return vec2(get_width(), get_height());
}

double Scene::get_geom_mean_size() const{ return geom_mean(get_width(),get_height()); }

void Scene::publish_global() {
    const unordered_map<string, double>& s = stage_publish_to_global();
    for(const auto& p : s) {
        set_global_state(global_identifier + "." + p.first, p.second);
    }
}

void Scene::render_one_frame(int microblock_frame_number, int scene_duration_frames) {
    cout << "[" << flush;

    set_global_state("macroblock_fraction", 1 - static_cast<double>(remaining_frames_in_macroblock) / total_frames_in_macroblock);
    set_global_state("microblock_fraction", static_cast<double>(microblock_frame_number) / scene_duration_frames);

    Pixels* p = nullptr;
    query(p);

    bool fifth_frame = int(get_global_state("frame_number")) % 5 == 0;
    if(!rendering_on() || fifth_frame) p->print_to_terminal();

    if (rendering_on()) { // Do not encode during smoketest
        get_writer().video->add_frame(*p);
    }

    remaining_frames_in_macroblock--;
    set_global_state("frame_number", get_global_state("frame_number") + 1);
    set_global_state("t", get_global_state("frame_number") / get_video_framerate_fps());
    cout << "]" << flush;
}
