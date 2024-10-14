#pragma once

#include "../Scene.cpp"
#include "../Common/ThreeDimensionScene.cpp"
#include "../../DataObjects/Graph.cpp"

template <typename T>
class GraphScene : public ThreeDimensionScene {
public:
    GraphScene(Graph<T>* g, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : ThreeDimensionScene(width, height), graph(g) {}

    void graph_to_3d(){
        clear_lines();
        clear_points();

        for(pair<double, Node<T>> p : graph->nodes){
            Node<T> node = p.second;
            glm::vec3 node_pos = glm::vec3(node.position);
            NodeHighlightType highlight = NORMAL;//(node.data->get_hash() == graph->root_node_hash) ? BULLSEYE : (node.highlight? RING : NORMAL);
            add_point(Point(node_pos, node.color, highlight, 1));

            for(const Edge& neighbor_edge : node.neighbors){
                double neighbor_id = neighbor_edge.to;
                Node<T> neighbor = graph->nodes.find(neighbor_id)->second;
                glm::vec3 neighbor_pos = glm::vec3(neighbor.position);
                add_line(Line(node_pos, neighbor_pos, get_edge_color(node, neighbor), neighbor_edge.opacity));
            }
        }
    }

    virtual int get_edge_color(const Node<T>& node, const Node<T>& neighbor){
        return OPAQUE_WHITE;
    }

    const StateQuery populate_state_query() const override {
        StateQuery s = ThreeDimensionScene::populate_state_query();
        s.insert("physics_multiplier");
        return s;
    }

    void mark_data_unchanged() override { graph->mark_unchanged(); }
    void change_data() override {
        graph->iterate_physics(state["physics_multiplier"]);
        graph_to_3d();
        clear_surfaces();
        update_surfaces();
    }
    bool check_if_data_changed() const override {
        return graph->has_been_updated_since_last_scene_query();
    }

    void update_surfaces(){
        unordered_set<string> updated_ids;

        for(pair<double, Node<C4Board>> p : graph->nodes){
            Node<C4Board>& node = p.second;
            if(graph_surface_map.find(node.id) != graph_surface_map.end()) {
                graph_surface_map[node.id].opacity = node.opacity;
                graph_surface_map[node.id].center = glm::vec3(node.position);
            } else {
                graph_surface_map[node.id] = make_surface(node);
            }

            // Add this id to the set of updated or created surfaces
            updated_ids.insert(node.id);
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

    virtual Surface make_surface(Node<T> node) const = 0;

    // Override the default surface render routine to make all graph surfaces point at the camera
    void render_surface(const Surface& surface) override {
        //make all the boards face the camera
        glm::quat conj2 = glm::conjugate(camera_direction) * glm::conjugate(camera_direction);
        glm::quat cam2 = camera_direction * camera_direction;

        // Rotate pos_x_dir vector
        glm::quat left_as_quat(0.0f, surface.pos_x_dir.x, surface.pos_x_dir.y, surface.pos_x_dir.z);
        glm::quat rotated_left_quat = conj2 * left_as_quat * cam2;

        // Rotate pos_y_dir vector
        glm::quat up_as_quat(0.0f, surface.pos_y_dir.x, surface.pos_y_dir.y, surface.pos_y_dir.z);
        glm::quat rotated_up_quat = conj2 * up_as_quat * cam2;

        Surface surface_rotated(
            surface.center,
            glm::vec3(rotated_left_quat.x, rotated_left_quat.y, rotated_left_quat.z),
            glm::vec3(rotated_up_quat.x, rotated_up_quat.y, rotated_up_quat.z),
            surface.scenePointer,
            surface.opacity
        );

        ThreeDimensionScene::render_surface(surface_rotated);
    }

    void draw() override{
        ThreeDimensionScene::draw();
    }

protected:
    Graph<T>* graph;
    unordered_map<string, Surface> graph_surface_map;
};
