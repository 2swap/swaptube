#include "../../Scene.h"

class BeaverTNF3DScene : public Scene {
public:
    BeaverTNF3DScene(const vec2& dimension = vec2(1, 1));

private:
    void draw() override;
};
