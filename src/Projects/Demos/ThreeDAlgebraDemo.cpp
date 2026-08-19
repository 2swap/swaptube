#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Math/ThreeDAlgebraScene.h"
#include "../Core/Smoketest.h"

void render_video() {
    CompositeScene cs;

    StateSet parentcontrol = {
        {"yz_x", "[yz_x]"}, {"yz_y", "[yz_y]"}, {"yz_z", "[yz_z]"},
        {"zz_x", "[zz_x]"}, {"zz_y", "[zz_y]"}, {"zz_z", "[zz_z]"},
        {"a_x", "[a_x]"}, {"a_y", "[a_y]"}, {"a_z", "[a_z]"},
        {"b_x", "[b_x]"}, {"b_y", "[b_y]"}, {"b_z", "[b_z]"},
        {"d", "[d]"},
    };

    cs.manager.set({
        {"yz_x", ".6"}, {"yz_y", ".6"}, {"yz_z", "0"},
        {"zz_x", "0"}, {"zz_y", ".6"}, {"zz_z", ".6"},
        {"a_x", "1"}, {"a_y", "0"}, {"a_z", "0"},
        {"b_x", "1"}, {"b_y", "0"}, {"b_z", "0"},
        {"d", "10"},
    });

    StateSet ab = {
        {"a_x", "0"}, {"a_y", "1"}, {"a_z", "0"},
        {"b_x", "0"}, {"b_y", "0"}, {"b_z", "1"},
    };

    shared_ptr<ThreeDAlgebraScene> left = make_shared<ThreeDAlgebraScene>(vec2(.5, 1));
    left->manager.set({{"associativity", "0"}});
    left->manager.set(parentcontrol);
    cs.add_scene(left, "left", vec2(.25, .5));

    shared_ptr<ThreeDAlgebraScene> right = make_shared<ThreeDAlgebraScene>(vec2(.5, 1));
    right->manager.set({{"associativity", "1"}});
    right->manager.set(parentcontrol);
    cs.add_scene(right, "right", vec2(.75, .5));

    stage_macroblock(SilenceBlock(5));
    cs.render_microblock();
    cs.manager.transition(MICRO, ab);
    cs.render_microblock();
    cs.render_microblock();
}
