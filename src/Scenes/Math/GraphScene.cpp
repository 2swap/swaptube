#pragma once

#include "../Scene.cpp"
#include "../Common/ThreeDimensionScene.cpp"
#include "../Media/LatexScene.cpp"
#include "../../DataObjects/Graph.cpp"

double age_to_size(double x){
    return ((3*x - 1) * exp(-.5*x)) + 1;
}

vector<int> tones = {0,4,7};
int tone_incr = 0;
void node_pop(double subdiv, bool added_not_deleted) {
    int tone_number = added_not_deleted?tones[tone_incr%tones.size()]:-6;
    double tone = pow(2,tone_number/12.);
    tone_incr++;
    int num_samples = 44100*.1;
    vector<float> left;
    vector<float> right;
     left.reserve(num_samples);
    right.reserve(num_samples);
    for(int i = 0; i < num_samples; i++){
        float val = .07 * pow(.5,i*80/44100.) * sin(tone*i*6.283*440/44100.);
         left.push_back(val);
        right.push_back(val);
    }
    WRITER.add_sfx(left, right, (get_global_state("t")+subdiv/VIDEO_FRAMERATE)*44100);
}

class GraphScene : public ThreeDimensionScene {
public:
    double curr_hash = 0;
    double next_hash = 0;
    GraphScene(Graph* g, const double width = 1, const double height = 1) : ThreeDimensionScene(width, height), graph(g) {
        state_manager.set(unordered_map<string, string>{
            {"repel", "1"},
            {"attract", "1"},
            {"decay", ".95"},
            {"physics_multiplier", "1"},
            {"centering_strength", "1"},
        });
    }

    void graph_to_3d(){
        clear_lines();
        clear_points();

        glm::dvec3 curr_pos;
        glm::dvec3 next_pos;
        bool curr_found = false;
        bool next_found = false;
        for(pair<double, Node> p : graph->nodes){
            Node node = p.second;
            glm::dvec3 node_pos = glm::dvec3(node.position);
            if(p.first == curr_hash) { curr_pos = node_pos; curr_found = true; }
            if(p.first == next_hash) { next_pos = node_pos; next_found = true; }
            NodeHighlightType highlight = (node.data->get_highlight_type() == 0) ? NORMAL : RING;
            add_point(Point(node_pos, node.color, highlight, 1, age_to_size(node.age)));

            for(const Edge& neighbor_edge : node.neighbors){
                double neighbor_id = neighbor_edge.to;
                Node neighbor = graph->nodes.find(neighbor_id)->second;
                glm::dvec3 neighbor_pos = glm::dvec3(neighbor.position);
                add_line(Line(node_pos, neighbor_pos, get_edge_color(node, neighbor), neighbor_edge.opacity));
            }
        }

        double opa = 0;
        glm::dvec3 pos_to_render;
        if(curr_found || next_found){
            double smooth_interp = smoother2(state["microblock_fraction"]);
            if     (!curr_found) pos_to_render = next_pos;
            else if(!next_found) pos_to_render = curr_pos;
            else                 pos_to_render = veclerp(curr_pos, next_pos, smooth_interp);
            opa = lerp(curr_found?1:0, next_found?1:0, smooth_interp);
            add_point(Point(pos_to_render, 0xffff0000, BULLSEYE, opa, 1.1*opa));
        }

        // automagical camera distancing
        auto_distance = lerp(auto_distance, graph->af_dist(), 0.1);
        auto_camera = veclerp(auto_camera, pos_to_render * opa, 0.02);
    }

    virtual int get_edge_color(const Node& node, const Node& neighbor){
        return OPAQUE_WHITE;
    }

    const StateQuery populate_state_query() const override {
        StateQuery s = ThreeDimensionScene::populate_state_query();
        s.insert("physics_multiplier");
        s.insert("repel");
        s.insert("attract");
        s.insert("decay");
        s.insert("microblock_fraction");
        s.insert("centering_strength");
        return s;
    }

    void mark_data_unchanged() override { graph->mark_unchanged(); }
    void change_data() override {
        if(last_node_count > -1){
            int diff = graph->size() - last_node_count;
            for(int i = 0; i < abs(diff); i++) node_pop(static_cast<double>(i)/diff, diff>0);
        }
        last_node_count = graph->size();
        graph->iterate_physics(state["physics_multiplier"], state["repel"], state["attract"], state["decay"], state["centering_strength"]);
        graph_to_3d();
        clear_surfaces();
        update_surfaces();
    }
    bool check_if_data_changed() const override {
        return graph->has_been_updated_since_last_scene_query();
    }

    void on_end_transition(bool is_macroblock) {
        curr_hash = next_hash;
    }

    void update_surfaces(){
        if(surfaces_override_unsafe) {
            graph_surface_map.clear(); return;
        }
        unordered_set<string> updated_ids;

        for(pair<double, Node> p : graph->nodes){
            Node& node = p.second;
            if(node.data->get_highlight_type() == 1) continue;
            string rep = node.data->representation;

            auto it = graph_surface_map.find(rep);
            if(graph_surface_map.find(rep) != graph_surface_map.end()) {
                it->second.opacity = node.opacity;
                it->second.center = glm::dvec3(node.position);
            } else {
                graph_surface_map.emplace(rep, make_surface(node));
            }

            // Add this id to the set of updated or created surfaces
            updated_ids.insert(rep);
        }

        // Remove any surfaces from graph_surface_map that were not updated or created
        for (auto it = graph_surface_map.begin(); it != graph_surface_map.end(); ) {
            if (updated_ids.find(it->first) == updated_ids.end()) {
                it = graph_surface_map.erase(it);
            } else {
                add_surface(it->second);
                ++it;
            }
        }
    }

    virtual Surface make_surface(Node node) const {
        return Surface(glm::dvec3(node.position),glm::dvec3(1,0,0),glm::dvec3(0,static_cast<double>(VIDEO_HEIGHT)/VIDEO_WIDTH,0), node.data->make_scene(), node.data->representation, node.opacity);
    }

    // Override the default surface render routine to make all graph surfaces point at the camera
    void render_surface(const Surface& surface) override {
        //make all the boards face the camera
        glm::dquat conj2 = glm::conjugate(camera_direction) * glm::conjugate(camera_direction);
        glm::dquat cam2 = camera_direction * camera_direction;

        // Rotate pos_x_dir vector
        glm::dquat left_as_quat(0.0f, surface.pos_x_dir.x, surface.pos_x_dir.y, surface.pos_x_dir.z);
        glm::dquat rotated_left_quat = conj2 * left_as_quat * cam2;

        // Rotate pos_y_dir vector
        glm::dquat up_as_quat(0.0f, surface.pos_y_dir.x, surface.pos_y_dir.y, surface.pos_y_dir.z);
        glm::dquat rotated_up_quat = conj2 * up_as_quat * cam2;

        Surface surface_rotated(
            surface.center,
            glm::dvec3(rotated_left_quat.x, rotated_left_quat.y, rotated_left_quat.z),
            glm::dvec3(rotated_up_quat.x, rotated_up_quat.y, rotated_up_quat.z),
            surface.scenePointer,
            surface.name,
            surface.opacity
        );

        ThreeDimensionScene::render_surface(surface_rotated);
    }

    void draw() override{
        ThreeDimensionScene::draw();
    }

    bool surfaces_override_unsafe = false; // For really big graphs, you can permanently turn off node stuff. Careful.

protected:
    Graph* graph;
    unordered_map<string, Surface> graph_surface_map;

private:
    int last_node_count = -1;
};

