#include "GraphScene.h"
#include "../Media/LatexScene.h"
#include "../../DataObjects/Graph.h"
#include "../../IO/Writer.h"
#include "../../IO/SFX.h"
#include "../../Core/Smoketest.h"
#include "../../Host_Device_Shared/vec.h"

string to_string_with_precision(const double a_value, const int n){
    ostringstream out;
    out.precision(n);
    out << fixed << a_value;
    string s = out.str();
    if(s.find('.') != string::npos){
        // trim trailing zeros
        s.erase(s.find_last_not_of('0') + 1, string::npos);
        // if there is a decimal point with nothing after it, trim that too
        if(s.back() == '.') s.pop_back();
    }
    // Remove leading zero for numbers between -1 and 1 (e.g. 0.5 -> .5, -0.5 -> -.5)
    if(s[0] == '0' && s[1] == '.') s.erase(0, 1);
    else if(s[0] == '-' && s[1] == '0' && s[2] == '.') s.erase(1, 1);
    return s;
}

GraphScene::GraphScene(const vec2& dimensions)
    : ThreeDimensionScene(dimensions) {
    curr_hash = 0;
    next_hash = 0;
    color_scheme = {0xff0079ff, 0xff00dfa2, 0xfff6fa70, 0xffff0060};
    manager.begin_timer("time_since_graph_init");
    manager.set({
        {"repel", "1"},
        {"attract", "1"},
        {"decay", ".95"},
        {"physics_multiplier", "1"},
        {"dimensions", "3"},
        {"edge_weights_size", "0"},
        {"node_labels_size", "1"},
        {"midpoint_multiplier", "1"},
        {"q1", "1 {t} 12 / sin <dimensions> 2 - lerp"},
        {"qi", "0"},
        {"qj", "{t} 12 / cos <dimensions> 2 - *"},
        {"qk", "0"},
    });

    config = new GraphDrawingConfig;
    graph = new Graph;
    add_data_object(config);
    add_data_object(graph);
}

void GraphScene::transition_node_position(const TransitionType tt, const double hash, const vec4& new_position){
    vec4 old_position = graph->nodes.find(hash)->second.position;
    pair<vec4, vec4> transition_pair = make_pair(old_position, new_position);
    if(tt == TransitionType::MICRO) nodes_in_micro_transition[hash] = transition_pair;
    else                            nodes_in_macro_transition[hash] = transition_pair;
}

void GraphScene::on_end_transition_extra_behavior(const TransitionType tt){
    if(tt == MACRO) {
        for(pair<double, pair<vec4, vec4>> p : nodes_in_macro_transition){
            double hash = p.first;
            vec4 end = p.second.second;
            graph->move_node(hash, end);
        }
        nodes_in_macro_transition.clear();
    }
    config->step_transition(tt);
    for(pair<double, pair<vec4, vec4>> p : nodes_in_micro_transition){
        double hash = p.first;
        vec4 end = p.second.second;
        graph->move_node(hash, end);
    }
    nodes_in_micro_transition.clear();
    curr_hash = next_hash;
}

void GraphScene::draw(){
    set_camera_direction();
    // TODO we only need to do this if data changed, not if state changed.
    float micro = state["microblock_fraction"];
    float macro = state["macroblock_fraction"];
    for(pair<double, pair<vec4, vec4>> p : nodes_in_micro_transition){
        double hash = p.first;
        vec4 start = p.second.first;
        vec4 end = p.second.second;
        vec4 interp_pos = veclerp(start, end, smoother2(micro));
        graph->move_node(hash, interp_pos);
    }
    for(pair<double, pair<vec4, vec4>> p : nodes_in_macro_transition){
        double hash = p.first;
        vec4 start = p.second.first;
        vec4 end = p.second.second;
        vec4 interp_pos = veclerp(start, end, smoother2(macro));
        graph->move_node(hash, interp_pos);
    }

    for(pair<double, Node> p : graph->nodes){
        config->add_node_if_missing(p.first);
        for(const Edge& neighbor_edge : p.second.neighbors){
            double neighbor_id = neighbor_edge.to;
            config->add_edge_if_missing(p.first, neighbor_id);
        }
    }

    clear_lines();
    clear_points();

    vec3 curr_pos;
    vec3 next_pos;
    bool curr_found = false;
    bool next_found = false;

    float midpoint_thickness = .35 * state["midpoint_multiplier"];

    //Pixels labels(pix.wh);

    // TODO Perhaps we should merge the graph and TDS point/line datatypes so that this translation becomes unnecessary
    // I can't think of a good pattern though.
    for(pair<double, Node> p : graph->nodes){
        double hash = p.first;
        Node node = p.second;
        vec3 node_pos(node.position);
        const NodeRenderData nrd = config->get_node_render_data(hash, macro, micro);
        if(hash == curr_hash) { curr_pos = node_pos; curr_found = true; }
        if(hash == next_hash) { next_pos = node_pos; next_found = true; }
        if (nrd.radius > 0) {
            add_point(Point(node_pos, nrd.color, 1, nrd.radius));
            if (nrd.splash_opacity > 0 && nrd.splash_radius > 0) {
                add_point(Point(node_pos, nrd.color, nrd.splash_opacity, nrd.splash_radius + nrd.radius));
            }
        }
        if (nrd.label_size > 0.1 && nrd.label != "") {
            bool behind_camera = false;
            vec2 pos = coordinate_to_pixel(node.position, behind_camera) + label_offset * get_width_height();
            vec2 dim = label_size * get_width_height() * nrd.label_size * state["node_labels_size"];
            //write_text(labels, latex_color(label_color, nrd.label), pos, dim, 1);
        }

        for(const Edge& neighbor_edge : node.neighbors){
            // Don't duplicate edges
            if (hash > neighbor_edge.to) continue;
            double neighbor_id = neighbor_edge.to;
            Node neighbor = graph->nodes.find(neighbor_id)->second;
            const EdgeRenderData erd = config->get_edge_render_data(hash, neighbor_id, macro, micro);
            vec3 neighbor_pos(neighbor.position.x, neighbor.position.y, neighbor.position.z);
            if(erd.post_color == erd.pre_color){ // Fade or no-change
                add_line(Line(node_pos, neighbor_pos, erd.post_color, 1, erd.is_dashed));
            } else { // Directed transition
                vec3 pos_pre = erd.direction ? neighbor_pos : node_pos;
                vec3 pos_post = erd.direction ? node_pos : neighbor_pos;
                uint32_t pre_color = erd.direction ? erd.post_color : erd.pre_color;
                uint32_t post_color = erd.direction ? erd.pre_color : erd.post_color;
                vec3 midpoint = veclerp(pos_pre, pos_post, erd.midpoint_fraction);
                add_line(Line(node_pos, midpoint, post_color, 1, erd.is_dashed));
                add_line(Line(neighbor_pos, midpoint, pre_color, 1, erd.is_dashed));
                add_point(Point(midpoint, erd.post_color, 1, midpoint_thickness));
            }

            if(erd.label != "" && erd.label_size > 0.1) {
                bool behind_camera = false;
                vec2 node_screen_pos = coordinate_to_pixel(node_pos, behind_camera);
                vec2 neighbor_screen_pos = coordinate_to_pixel(neighbor_pos, behind_camera);
                if(behind_camera) continue;
                float angle = atan2(neighbor_screen_pos.y - node_screen_pos.y, neighbor_screen_pos.x - node_screen_pos.x);
                // Make the angle fit into -pi/2, pi/2.
                float text_rotation_angle = angle;
                while (text_rotation_angle > M_PI/2) text_rotation_angle -= M_PI;
                while (text_rotation_angle < -M_PI/2) text_rotation_angle += M_PI;
                angle += M_PI / 2;
                vec2 offset = edge_label_offset * vec2(cos(angle), sin(angle)) * get_geom_mean_size();
                vec2 midpoint = (node_screen_pos + neighbor_screen_pos) / 2;
                vec2 pos = midpoint + offset;
                vec2 dim = vec2(0.4, 0.06) * get_width_height() * erd.label_size;
                if (erd.label.size() <= 2) { // Simple edge weights (2 digit numbers) dont need rotation
                    text_rotation_angle = 0;
                }
                //write_text(labels, erd.label, pos, dim, 1, text_rotation_angle);
            }
        }
    }

    float opa = 0;
    vec3 pos_to_render(0,0,0);
    if(curr_found || next_found){
        double smooth_interp = smoother2(micro);
        if     (!curr_found) pos_to_render = next_pos;
        else if(!next_found) pos_to_render = curr_pos;
        else                 pos_to_render = veclerp(curr_pos, next_pos, smooth_interp);
        opa = lerp(curr_found?1:0, next_found?1:0, smooth_interp);
    }

    ThreeDimensionScene::draw();

    //pix.overlay_gpu(labels, vec2(0,0), 1);
}

const StateQuery GraphScene::populate_state_query() const {
    StateQuery s = ThreeDimensionScene::populate_state_query();
    state_query_insert_multiple(s, {"physics_multiplier", "repel", "attract", "decay", "microblock_fraction", "macroblock_fraction", "dimensions", "edge_weights_size", "midpoint_multiplier", "node_labels_size"});
    return s;
}
