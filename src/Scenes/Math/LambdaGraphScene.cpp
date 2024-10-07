#pragma once

#include "GraphScene.cpp"
#include "LambdaScene.cpp"
#include "../../DataObjects/HashableString.cpp"

class LambdaGraphScene : public GraphScene<HashableString> {
public:
    LambdaGraphScene(Graph<HashableString>* g, string rep, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
    : GraphScene(g, width, height), root_node_representation(rep) {
        HashableString* reppy = new HashableString(rep);
        graph->add_to_stack(reppy);
        state_manager.set(unordered_map<string, string>{
            {"physics_multiplier", "1"},
        });
    }

    void update_surfaces(){
        clear_surfaces();
        for(pair<double, Node<HashableString>> p : graph->nodes){
            Node<HashableString>& node = p.second;
            if(node.opacity < 0.1) continue;
            glm::vec3 node_pos = glm::vec3(node.position);
            shared_ptr<LambdaExpression> lambda = parse_lambda_from_string(node.data->representation);
            lambda->set_color_recursive(node.color);
            add_surface(Surface(node_pos,glm::vec3(1,0,0),glm::vec3(0,1,0), make_shared<LambdaScene>(lambda, 600, 600), node.opacity));
        }
    }

    void inheritable_preprocessing() override{
        update_surfaces();
    }

    void render_surface(const Surface& surface) {
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

private:
    string root_node_representation;
};
