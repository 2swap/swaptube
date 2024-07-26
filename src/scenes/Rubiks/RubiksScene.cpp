#pragma once

#include "../Scene.cpp"
#include "../Common/3DScene.cpp"

class RubiksScene : public ThreeDimensionScene {
public:
    RubiksScene(const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : ThreeDimensionScene(width, height) {}

    void update_surfaces(){
        for(glm::quat q : symmetries) {
            for(glm::
            Node<C4Board> node = p.se cond;
            glm::vec3 node_pos = glm::vec3(node.position);
            glm::vec3 x_dir = glm::vec3(1,0,0);
            glm::vec3 y_dir = glm::vec3(0,1,0);
            int color = get_color_as_function_of_normal(glm::cross(x_dir, y_dir);
            surfaces.push_back(Surface(node_pos, x_dir, y_dir, color));
        }
        rendered = false;
    }

    int get_color_as_function_of_normal(const glm::vec3& normal){
        return 0xffffffff;
    }

    void clear_surfaces(){
        for(Surface s : surfaces){
            delete s.scenePointer;
        }
        surfaces.clear();
        rendered = false;
    }

    ~C4GraphScene(){
        for (Surface surface : surfaces){
            delete surface.scenePointer;
        }
    }

private:
    string root_node_representation;
};
