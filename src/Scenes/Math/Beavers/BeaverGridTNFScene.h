#include "../../Common/CoordinateScene.h"

class BeaverGridTNFScene : public CoordinateScene {
public:
    BeaverGridTNFScene(const vec2& dimension = vec2(1, 1));

private:
    void draw() override;
};
