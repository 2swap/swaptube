#include "GraphScene.h"
#include "../Media/LatexScene.h"
#include "../../DataObjects/Graph.h"
#include "../../IO/Writer.h"
#include "../../IO/SFX.h"
#include "../../Core/Smoketest.h"
#include "../../Host_Device_Shared/vec.h"

GraphScene::GraphScene(shared_ptr<Graph> g, const vec2& dimensions)
    : ThreeDimensionScene(dimensions), graph(g) {
    curr_hash = 0;
    next_hash = 0;
    color_scheme = {0xff0079ff, 0xff00dfa2, 0xfff6fa70, 0xffff0060};
    manager.begin_timer("time_since_graph_init");
    manager.set({
        {"label_size", "1"},
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

    config = make_shared<GraphDrawingConfig>();
    add_data_object(&(*config));
    add_data_object(&(*graph));
}

void GraphScene::transition_node_position(const TransitionType tt, const double hash, const vec4& shift){
    vec4 old_position = graph->nodes.find(hash)->second.position;
    pair<vec4, vec4> transition_pair = make_pair(old_position, shift);
    if(tt == TransitionType::MICRO) nodes_in_micro_transition[hash] = transition_pair;
    else                            nodes_in_macro_transition[hash] = transition_pair;
}

void GraphScene::on_end_transition_extra_behavior(const TransitionType tt){
    if(tt == MACRO) nodes_in_macro_transition.clear();
    nodes_in_micro_transition.clear();
    curr_hash = next_hash;
}

void GraphScene::draw(){
    // TODO we only need to do this if data changed, not if state changed.
    for(pair<double, pair<vec4, vec4>> p : nodes_in_micro_transition){
        double hash = p.first;
        vec4 start = p.second.first;
        vec4 end = p.second.second;
        vec4 interp_pos = veclerp(start, end, smoother2(state["microblock_fraction"]));
        graph->move_node(hash, interp_pos);
    }
    for(pair<double, pair<vec4, vec4>> p : nodes_in_macro_transition){
        double hash = p.first;
        vec4 start = p.second.first;
        vec4 end = p.second.second;
        vec4 interp_pos = veclerp(start, end, smoother2(state["macroblock_fraction"]));
        graph->move_node(hash, interp_pos);
    }

    clear_lines();
    clear_points();

    vec3 curr_pos;
    vec3 next_pos;
    bool curr_found = false;
    bool next_found = false;

    // TODO Perhaps we should merge the graph and TDS point/line datatypes so that this translation becomes unnecessary
    for(pair<double, Node> p : graph->nodes){
        double hash = p.first;
        Node node = p.second;
        vec3 node_pos(node.position);
        if(hash == curr_hash) { curr_pos = node_pos; curr_found = true; }
        if(hash == next_hash) { next_pos = node_pos; next_found = true; }
        add_point(Point(node_pos, config->get_node_color(hash), 1, config->get_node_radius(hash)));
        //double so = node.splash_opacity();
        int color = color_scheme[static_cast<int>(abs(hash)*100)%4];
        //if(so>0) add_point(Point(node_pos, color, so, node.splash_radius()));

        for(const Edge& neighbor_edge : node.neighbors){
            double neighbor_id = neighbor_edge.to;
            Node neighbor = graph->nodes.find(neighbor_id)->second;
            vec3 neighbor_pos(neighbor.position.x, neighbor.position.y, neighbor.position.z);
            add_line(Line(node_pos, neighbor_pos, neighbor_edge.color, neighbor_edge.opacity));
        }
    }

    float opa = 0;
    vec3 pos_to_render(0,0,0);
    if(curr_found || next_found){
        double smooth_interp = smoother2(state["microblock_fraction"]);
        if     (!curr_found) pos_to_render = next_pos;
        else if(!next_found) pos_to_render = curr_pos;
        else                 pos_to_render = veclerp(curr_pos, next_pos, smooth_interp);
        opa = lerp(curr_found?1:0, next_found?1:0, smooth_interp);
        double hpo = state["highlight_point_opacity"];
        if(hpo > 0.001)
            add_point(Point(pos_to_render, 0xffff0000, hpo*opa, 3*opa));
    }

    auto_camera = veclerp(auto_camera, pos_to_render * opa, 0.1);
    // Looks jarring when puzzle moves if we simply do: //auto_camera = pos_to_render * opa;

    ThreeDimensionScene::draw();

    float label_size = state["label_size"];
    if(label_size > 0.02) {
        for(pair<double, Node> p : graph->nodes){
            Node node = p.second;
            if(node.label != "") {
                bool behind_camera = false;
                vec2 pos = coordinate_to_pixel(node.position, behind_camera);
                vec2 downshift = vec2(0, 0.03) * get_width_height();
                pos += downshift; // shift down a bit so it doesn't overlap with the point
                vec2 half_dim = vec2(0.05, 0.02) * get_width_height() * label_size;
                vec2 top_left = pos - half_dim;
                vec2 bottom_right = pos + half_dim;
                write_text(pix, "\\text{" + node.label + "}", top_left, bottom_right, 1);
            }
        }
    }
}

const StateQuery GraphScene::populate_state_query() const {
    StateQuery s = ThreeDimensionScene::populate_state_query();
    state_query_insert_multiple(s, {"label_size", "desired_nodes", "physics_multiplier", "repel", "attract", "decay", "microblock_fraction", "centering_strength", "dimensions", "mirror_force", "highlight_point_opacity", "flip_by_symmetry"});
    return s;
}
