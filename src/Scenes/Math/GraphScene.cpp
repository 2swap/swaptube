#include "GraphScene.h"
#include "../Media/LatexScene.h"
#include "../../DataObjects/Graph.h"
#include "../../IO/Writer.h"
#include "../../Core/Smoketest.h"
#include "../../Host_Device_Shared/vec.h"

vector<int> tones = {0,4,7};
int tone_incr = 0;
void node_pop(double subdiv, bool added_not_deleted) {
    int tone_number = added_not_deleted?tones[tone_incr%tones.size()]:-6;
    double tone = pow(2,tone_number/12.);
    tone_incr++;
    int samplerate = get_audio_samplerate_hz();
    int num_samples = samplerate * .1;
    vector<sample_t> left;
    vector<sample_t> right;
     left.reserve(num_samples);
    right.reserve(num_samples);
    for(int i = 0; i < num_samples; i++){
        float val_f = .07 * pow(.5,i*80./samplerate) * sin(tone*i*6.283*440/samplerate);
        // convert float to sample_t, which is 32-bit signed integer
        sample_t val = static_cast<sample_t>(val_f * 2147483648); // scale to 32-bit signed integer range
             if(val_f < -1.0) val = -2147483648; // clamp to -1.0
        else if(val_f >  1.0) val =  2147483647; // clamp to 1.0
         left.push_back(val);
        right.push_back(val);
    }
    double time = get_global_state("t");
    get_writer().audio->add_sfx(left, right, (time+subdiv/get_video_framerate_fps())*get_audio_samplerate_hz());
}

GraphScene::GraphScene(shared_ptr<Graph> g, bool surfaces_on, const double width, const double height) : ThreeDimensionScene(width, height), surfaces_override_unsafe(!surfaces_on), graph(g) {
    curr_hash = 0;
    next_hash = 0;
    color_scheme = {0xff0079ff, 0xff00dfa2, 0xfff6fa70, 0xffff0060};
    manager.begin_timer("time_since_graph_init");
    manager.set({
        {"repel", "1"},
        {"attract", "1"},
        {"decay", ".95"},
        {"physics_multiplier", "1"},
        {"centering_strength", ".1"},
        {"dimensions", "3"},
        {"mirror_force", "0"},
        {"highlight_point_opacity", "1"},
        {"flip_by_symmetry", "0"},
        {"q1", "1 {t} 12 / sin <dimensions> 2 - lerp"},
        {"qi", "0"},
        {"qj", "{t} 12 / cos <dimensions> 2 - *"},
        {"qk", "0"},
        {"desired_nodes", "<growth_rate> 1.5 <time_since_graph_init> ^ 1 - * 1000000 min"},
        {"growth_rate", "100"},
    });
    last_node_count = -1;
}

void GraphScene::graph_to_3d(){
    cout << "<";
    clear_lines();
    clear_points();

    vec3 curr_pos;
    vec3 next_pos;
    bool curr_found = false;
    bool next_found = false;
    // TODO Perhaps we should merge the graph and TDS point/line datatypes so that this translation becomes unnecessary
    for(pair<double, Node> p : graph->nodes){
        Node node = p.second;
        vec3 node_pos(node.position.x, node.position.y, node.position.z); // convert 4d to 3d
        if(p.first == curr_hash) { curr_pos = node_pos; curr_found = true; }
        if(p.first == next_hash) { next_pos = node_pos; next_found = true; }
        if(node.draw_point) add_point(Point(node_pos, node.color, 1, node.radius()));
        double so = node.splash_opacity();
        int color = color_scheme[static_cast<int>(abs(p.first)*4)%4];
        if(so>0) add_point(Point(node_pos, color, so, node.splash_radius()));

        for(const Edge& neighbor_edge : node.neighbors){
            double neighbor_id = neighbor_edge.to;
            Node neighbor = graph->nodes.find(neighbor_id)->second;
            vec3 neighbor_pos(neighbor.position.x, neighbor.position.y, neighbor.position.z);
            add_line(Line(node_pos, neighbor_pos, get_edge_color(node, neighbor), neighbor_edge.opacity));
        }
    }

    float opa = 0;
    vec3 pos_to_render(0,0,0);
    if(curr_found || next_found){
        double smooth_interp = smoother2(state["microblock_fraction"]);
        if     (!curr_found) pos_to_render = next_pos;
        else if(!next_found) pos_to_render = curr_pos;
        else                 pos_to_render = veclerp(curr_pos, next_pos, cbrt(smooth_interp)); // TODO remove the cbrt here.
        opa = lerp(curr_found?1:0, next_found?1:0, smooth_interp);
        double hpo = state["highlight_point_opacity"];
        if(hpo > 0.001)
            add_point(Point(vec3{pos_to_render.x, pos_to_render.y, pos_to_render.z}, 0xffff0000, hpo*opa, 3*opa));
    }

    // automagical camera distancing
    auto_distance = lerp(auto_distance, graph->af_dist(), 0.1);
    auto_camera = veclerp(auto_camera, pos_to_render * opa, 0.1);
    // Looks jarring when puzzle moves if we simply do: //auto_camera = pos_to_render * opa;
    cout << ">";
}

int GraphScene::get_edge_color(const Node& node, const Node& neighbor){
    return OPAQUE_WHITE;
}

const StateQuery GraphScene::populate_state_query() const {
    StateQuery s = ThreeDimensionScene::populate_state_query();
    state_query_insert_multiple(s, {"desired_nodes", "physics_multiplier", "repel", "attract", "decay", "microblock_fraction", "centering_strength", "dimensions", "mirror_force", "highlight_point_opacity", "flip_by_symmetry"});
    return s;
}

void GraphScene::mark_data_unchanged() { graph->mark_unchanged(); }
void GraphScene::change_data() {
    int nodes_to_add = state["desired_nodes"] - graph->size();
    if(nodes_to_add > 0) {
        graph->expand(nodes_to_add);
        graph->make_bidirectional();
    }
    if(last_node_count > -1){
        int diff = graph->size() - last_node_count;
        for(int i = 0; i < abs(diff); i++) {
            node_pop(static_cast<double>(i)/abs(diff), diff>0);
        }
    }
    last_node_count = graph->size();
    int amount_to_iterate = state["physics_multiplier"];
    if(!rendering_on()) amount_to_iterate = min(amount_to_iterate, 1); // No need to spread graphs out in smoketest
    graph->iterate_physics(amount_to_iterate, state["repel"], state["attract"], state["decay"], state["centering_strength"], state["dimensions"], state["mirror_force"], state["flip_by_symmetry"]>0);
    if(graph->has_been_updated_since_last_scene_query()) {
        graph_to_3d();
        clear_surfaces();
        update_surfaces();
    }
}
bool GraphScene::check_if_data_changed() const {
    return ThreeDimensionScene::check_if_data_changed() || graph->has_been_updated_since_last_scene_query();
}

void GraphScene::on_end_transition_extra_behavior(const TransitionType tt) {
    curr_hash = next_hash;
}

void GraphScene::update_surfaces(){
    if(surfaces_override_unsafe) {
        graph_surface_map.clear(); return;
    }
    unordered_set<string> updated_ids;

    for(pair<double, Node> p : graph->nodes){
        Node& node = p.second;
        if(node.data->get_highlight_type() == 1) continue;
        string rep = node.data->representation;

        auto it = graph_surface_map.find(rep);
        if(it != graph_surface_map.end()) {
            it->second.first.center = vec3(node.position.x, node.position.y, node.position.z);
        } else {
            graph_surface_map.emplace(rep, make_pair(make_surface(node), node.data->make_scene()));
        }

        // Add this id to the set of updated or created surfaces
        updated_ids.insert(rep);
    }

    // Remove any surfaces from graph_surface_map that were not updated or created
    for (auto it = graph_surface_map.begin(); it != graph_surface_map.end(); ) {
        if (updated_ids.find(it->first) == updated_ids.end()) {
            it = graph_surface_map.erase(it);
        } else {
            add_surface(it->second.first, it->second.second);
            ++it;
        }
    }
}

Surface GraphScene::make_surface(Node node) const {
    return Surface(vec3(node.position.x, node.position.y, node.position.z),
                   vec3(1,0,0),
                   vec3(0,static_cast<float>(get_video_height_pixels())/get_video_width_pixels(), 0),
                   node.data->representation);
}

// Override the default surface render routine to make all graph surfaces point at the camera
void GraphScene::render_surface(const Surface& surface) {
    //make all the boards face the camera
    quat cam2 = camera_direction * camera_direction;

    // Rotate pos_x_dir vector
    quat left_as_quat(0.0f, surface.pos_x_dir.x, surface.pos_x_dir.y, surface.pos_x_dir.z);
    quat rotated_left_quat = cam2 * left_as_quat;

    // Rotate pos_y_dir vector
    quat up_as_quat(0.0f, surface.pos_y_dir.x, surface.pos_y_dir.y, surface.pos_y_dir.z);
    quat rotated_up_quat = cam2 * up_as_quat;

    Surface surface_rotated(
        surface.center,
        vec3(rotated_left_quat.i, rotated_left_quat.j, rotated_left_quat.k),
        vec3(rotated_up_quat.i, rotated_up_quat.j, rotated_up_quat.k),
        surface.name
    );

    ThreeDimensionScene::render_surface(surface_rotated);
}
