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

GraphScene::GraphScene(shared_ptr<Graph> g, const vec2& dimensions)
    : ThreeDimensionScene(dimensions), graph(g) {
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
        {"mirror_force", "0"},
        {"edge_weights_size", "0"},
        {"q1", "1 {t} 12 / sin <dimensions> 2 - lerp"},
        {"qi", "0"},
        {"qj", "{t} 12 / cos <dimensions> 2 - *"},
        {"qk", "0"},
    });

    config = make_shared<GraphDrawingConfig>();
    add_data_object(&(*config));
    add_data_object(&(*graph));
}

void GraphScene::transition_node_position(const TransitionType tt, const double hash, const vec4& new_position){
    vec4 old_position = graph->nodes.find(hash)->second.position;
    pair<vec4, vec4> transition_pair = make_pair(old_position, new_position);
    if(tt == TransitionType::MICRO) nodes_in_micro_transition[hash] = transition_pair;
    else                            nodes_in_macro_transition[hash] = transition_pair;
}

void GraphScene::on_end_transition_extra_behavior(const TransitionType tt){
    if(tt == MACRO) nodes_in_macro_transition.clear();
    config->step_transition(tt);
    nodes_in_micro_transition.clear();
    curr_hash = next_hash;
}

void GraphScene::draw(){
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

    float midpoint_thickness = get_geom_mean_size() / 1920.0;

    // TODO Perhaps we should merge the graph and TDS point/line datatypes so that this translation becomes unnecessary
    for(pair<double, Node> p : graph->nodes){
        double hash = p.first;
        Node node = p.second;
        vec3 node_pos(node.position);
        if(hash == curr_hash) { curr_pos = node_pos; curr_found = true; }
        if(hash == next_hash) { next_pos = node_pos; next_found = true; }
        uint32_t color = config->get_node_color(hash, macro, micro);
        float node_radius = config->get_node_radius(hash, macro, micro);
        add_point(Point(node_pos, color, 1, node_radius));
        float splash_opacity = config->get_node_splash_opacity(hash, macro, micro);
        float splash_radius = config->get_node_splash_radius(hash, macro, micro);
        if (splash_opacity > 0 && splash_radius > 0) {
            add_point(Point(node_pos, color, splash_opacity, splash_radius + node_radius));
        }

        for(const Edge& neighbor_edge : node.neighbors){
            // Don't duplicate edges
            if (hash > neighbor_edge.to) continue;
            double neighbor_id = neighbor_edge.to;
            Node neighbor = graph->nodes.find(neighbor_id)->second;
            vec3 neighbor_pos(neighbor.position.x, neighbor.position.y, neighbor.position.z);
            uint32_t edge_color_1 = config->get_edge_color(hash, neighbor_id, macro, micro);
            uint32_t edge_color_fade = config->get_edge_fade_color(hash, neighbor_id, macro, micro);
            uint32_t edge_color_2 = config->get_edge_target_color(hash, neighbor_id, macro, micro);
            if(edge_color_1 == edge_color_2) {
                add_line(Line(node_pos, neighbor_pos, edge_color_1));
            } else if (edge_color_fade != edge_color_1 && edge_color_fade != edge_color_2) {
                add_line(Line(node_pos, neighbor_pos, edge_color_fade));
            } else {
                float midpoint_fraction = config->get_edge_midpoint_fraction(hash, neighbor_id, macro, micro);
                uint32_t midpoint_color = edge_color_2;
                if(config->get_edge_direction(hash, neighbor_id)){
                    midpoint_fraction = 1 - midpoint_fraction;
                    uint32_t temp = edge_color_1;
                    edge_color_1 = edge_color_2;
                    edge_color_2 = temp;
                    midpoint_color = edge_color_1;
                }
                vec3 midpoint = veclerp(node_pos, neighbor_pos, midpoint_fraction);
                add_line(Line(midpoint, neighbor_pos, edge_color_1));
                add_point(Point(midpoint, midpoint_color, 1, midpoint_thickness));
                add_line(Line(node_pos, midpoint, edge_color_2));
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

    vec2 node_label_downshift = vec2(0, 0.03) * get_width_height();
    for(pair<double, Node> p : graph->nodes){
        Node node = p.second;
        string label = config->get_node_label(p.first, macro, micro);
        float label_size = config->get_node_label_size(p.first, macro, micro);
        if(label != "" && label_size > 0.1){
            bool behind_camera = false;
            vec2 pos = coordinate_to_pixel(node.position, behind_camera);
            pos += node_label_downshift; // shift down a bit so it doesn't overlap with the point
            vec2 half_dim = vec2(0.1, 0.02) * get_width_height() * label_size;
            vec2 top_left = pos - half_dim;
            vec2 bottom_right = pos + half_dim;
            write_text(pix, label, top_left, bottom_right, 1);
        }
    }

    // Draw edge weights if option enabled
    double edge_weights_size = state["edge_weights_size"];
    if (edge_weights_size > 0.1){
        for(pair<double, Node> p : graph->nodes){
            Node node = p.second;
            vec3 node_pos(node.position);
            for(const Edge& neighbor_edge : node.neighbors){
                // Skip one direction of each edge since we'll draw the weight from both nodes
                if (p.first > neighbor_edge.to) continue;
                double neighbor_id = neighbor_edge.to;
                Node neighbor = graph->nodes.find(neighbor_id)->second;
                vec3 neighbor_pos(neighbor.position.x, neighbor.position.y, neighbor.position.z);
                vec3 midpoint = (node_pos + neighbor_pos) / 2.0f;
                bool behind_camera = false;
                vec2 node_screen_pos = coordinate_to_pixel(node_pos, behind_camera);
                vec2 neighbor_screen_pos = coordinate_to_pixel(neighbor_pos, behind_camera);
                float angle = atan2(neighbor_screen_pos.y - node_screen_pos.y, neighbor_screen_pos.x - node_screen_pos.x);
                // Make the angle fit into -pi/2, pi/2.
                float modulo_angle = angle;
                while (modulo_angle > M_PI/4) modulo_angle -= M_PI/2;
                while (modulo_angle < -M_PI/4) modulo_angle += M_PI/2;
                angle += M_PI / 2;
                while (angle < -M_PI) angle += M_PI;
                while (angle > 0) angle -= M_PI;
                vec2 offset = 0.01 * vec2(cos(angle), sin(angle)) * get_width_height();
                vec2 pos = coordinate_to_pixel(midpoint, behind_camera) + offset;
                // Distance in 3D space is weight
                double weight = length(neighbor_pos - node_pos);
                string weight_label = to_string_with_precision(weight, 1);
                vec2 half_dim = vec2(0.035, 0.025) * get_width_height() * edge_weights_size;
                vec2 top_left = pos - half_dim;
                vec2 bottom_right = pos + half_dim;
                write_text(pix, weight_label, top_left, bottom_right, 1, modulo_angle);
            }
        }
    }
}

const StateQuery GraphScene::populate_state_query() const {
    StateQuery s = ThreeDimensionScene::populate_state_query();
    state_query_insert_multiple(s, {"physics_multiplier", "repel", "attract", "decay", "microblock_fraction", "macroblock_fraction", "dimensions", "mirror_force", "edge_weights_size"});
    return s;
}
