#pragma once

#include "../Scene.cpp"
#include "../Common/3DScene.cpp"

class RubiksScene : public ThreeDimensionScene {
public:
    RubiksScene(const double width = 1, const double height = 1) : ThreeDimensionScene(width, height) {}

    void update_surfaces(){
        for(glm::dquat q : symmetries) {
            for(glm::
            Node<C4Board> node = p.second;
            glm::dvec3 node_pos = glm::dvec3(node.position);
            glm::dvec3 x_dir = glm::dvec3(1,0,0);
            glm::dvec3 y_dir = glm::dvec3(0,1,0);
            int color = get_color_as_function_of_normal(glm::cross(x_dir, y_dir);
            surfaces.push_back(Surface(node_pos, x_dir, y_dir, color));
        }
    }

    int get_color_as_function_of_normal(const glm::dvec3& normal){
        return 0xffffffff;
    }

    void clear_surfaces(){
        for(Surface s : surfaces){
            delete s.scenePointer;
        }
        surfaces.clear();
    }

    ~C4GraphScene(){
        for (Surface surface : surfaces){
            delete surface.scenePointer;
        }
    }

private:
    string root_node_representation;
};
