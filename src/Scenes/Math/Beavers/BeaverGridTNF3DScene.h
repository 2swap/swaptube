#include "../../Scene.h"

class BeaverGridTNF3DScene : public Scene {
public:
    BeaverGridTNF3DScene(const vec2& dimension = vec2(1, 1));

private:
    void draw() override;

    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;

};
