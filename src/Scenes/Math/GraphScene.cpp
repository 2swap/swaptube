#pragma once

#include "../Scene.cpp"
#include "../Common/ThreeDimensionScene.cpp"
#include "../Media/LatexScene.cpp"
#include "../../DataObjects/Graph.cpp"

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
    double time = get_global_state("t");
    AUDIO_WRITER.add_sfx(left, right, (time+subdiv/VIDEO_FRAMERATE)*44100);
}

class GraphScene : public ThreeDimensionScene {
public:
    double curr_hash = 0;
    double next_hash = 0;
    GraphScene(Graph* g, bool surfaces_on, const double width = 1, const double height = 1) : ThreeDimensionScene(width, height), surfaces_override_unsafe(!surfaces_on), graph(g) {
        state_manager.set(unordered_map<string, string>{
            {"repel", "1"},
            {"attract", "1"},
            {"decay", ".95"},
            {"physics_multiplier", "1"},
            {"centering_strength", "1"},
            {"dimensions", "3"},
            {"mirror_force", "0"},
        });
    }

    void graph_to_3d(){
        cout << "<" << endl;
        clear_lines();
        clear_points();

        glm::vec3 curr_pos;
        glm::vec3 next_pos;
        bool curr_found = false;
        bool next_found = false;
        // TODO Although this loop is quite naive, it turns out to be very slow, and is often the bottleneck.
        // A solution would be to somehow merge the graph and TDS point/line datatypes so that this translation becomes unnecessary
        // Also, it can be naively skipped whenever the graph has not had its data updated.
        for(pair<double, Node> p : graph->nodes){
            Node node = p.second;
            glm::vec3 node_pos = glm::vec3(node.position.x, node.position.y, node.position.z);
            if(p.first == curr_hash) { curr_pos = node_pos; curr_found = true; }
            if(p.first == next_hash) { next_pos = node_pos; next_found = true; }
            add_point(Point(node_pos, node.color, 1, node.radius()));

            for(const Edge& neighbor_edge : node.neighbors){
                double neighbor_id = neighbor_edge.to;
                Node neighbor = graph->nodes.find(neighbor_id)->second;
                glm::vec3 neighbor_pos = glm::vec3(neighbor.position);
                add_line(Line(node_pos, neighbor_pos, get_edge_color(node, neighbor), neighbor_edge.opacity));
            }
        }

        float opa = 0;
        glm::vec3 pos_to_render(0.f);
        if(curr_found || next_found){
            double smooth_interp = smoother2(state["microblock_fraction"]);
            if     (!curr_found) pos_to_render = next_pos;
            else if(!next_found) pos_to_render = curr_pos;
            else                 pos_to_render = veclerp(curr_pos, next_pos, smooth_interp);
            opa = lerp(curr_found?1:0, next_found?1:0, smooth_interp);
            add_point(Point(pos_to_render, 0xffff0000, opa, 3*opa));
        }

        // automagical camera distancing
        auto_distance = lerp(auto_distance, graph->af_dist(), 0.1);
        auto_camera = veclerp(auto_camera, pos_to_render * opa, 0.02);
        cout << ">" << endl;
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
        s.insert("dimensions");
        s.insert("mirror_force");
        return s;
    }

    void mark_data_unchanged() override { graph->mark_unchanged(); }
    void change_data() override {
        if(last_node_count > -1){
            int diff = graph->size() - last_node_count;
            for(int i = 0; i < abs(diff); i++) node_pop(static_cast<double>(i)/abs(diff), diff>0);
        }
        last_node_count = graph->size();
        graph->iterate_physics(state["physics_multiplier"], state["repel"], state["attract"], state["decay"], state["centering_strength"], state["dimensions"], state["mirror_force"]);
        if(graph->has_been_updated_since_last_scene_query()) {
            graph_to_3d();
            clear_surfaces();
            update_surfaces();
        }
    }
    bool check_if_data_changed() const override {
        return ThreeDimensionScene::check_if_data_changed() || graph->has_been_updated_since_last_scene_query();
    }

    void on_end_transition_extra_behavior(const TransitionType tt) override {
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
            if(it != graph_surface_map.end()) {
                it->second.first.opacity = node.opacity;
                it->second.first.center = glm::vec3(node.position.x, node.position.y, node.position.z);
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

    Graph* expose_graph_ptr() { return graph; }

    virtual Surface make_surface(Node node) const {
        return Surface(glm::vec3(node.position.x, node.position.y, node.position.z),
                       glm::vec3(1,0,0),
                       glm::vec3(0,static_cast<float>(VIDEO_HEIGHT)/VIDEO_WIDTH,0),
                       node.data->representation, node.opacity);
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
            glm::vec3(rotated_left_quat.x, rotated_left_quat.y, rotated_left_quat.z),
            glm::vec3(rotated_up_quat.x, rotated_up_quat.y, rotated_up_quat.z),
            surface.name,
            surface.opacity
        );

        ThreeDimensionScene::render_surface(surface_rotated);
    }

    void draw() override{
        ThreeDimensionScene::draw();
    }

    bool surfaces_override_unsafe = false; // For really big graphs, you can permanently turn off node stuff. This happens in the constructor, but careful when handling manually.

protected:
    Graph* graph;
    unordered_map<string, pair<Surface, shared_ptr<Scene>>> graph_surface_map;

private:
    int last_node_count = -1;
};

