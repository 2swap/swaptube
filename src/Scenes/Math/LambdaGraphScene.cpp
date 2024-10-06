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
        for(pair<double, shared_ptr<LambdaScene>> p : scene_map){
            if(!graph->node_exists(p.first))
                remove_surface(p.second);
        }
        for(pair<double, Node<HashableString>> p : graph->nodes){
            Node<HashableString> node = p.second;
            glm::vec3 node_pos = glm::vec3(node.position);
            if(node.opacity < 0.01) continue;
            shared_ptr<LambdaExpression> lambda = parse_lambda_from_string(node.data->representation);
            lambda->set_color_recursive(node.color);
            double id = p.first;
            auto found = scene_map.find(id);
            if(found == scene_map.end())
                scene_map[id] = make_shared<LambdaScene>(lambda, 600, 600);
            shared_ptr<LambdaScene> search = scene_map.find(id)->second;
            bool should_add = true;
            for (auto it = surfaces.begin(); it != surfaces.end(); ++it){
                if (it->scenePointer == search){
                    should_add = false;
                    break;
                }
            }
            if(should_add)
                add_surface(Surface(node_pos,glm::vec3(1,0,0),glm::vec3(0,1,0), search, node.opacity));
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
    unordered_map<double, shared_ptr<LambdaScene>> scene_map;
    string root_node_representation;
};
