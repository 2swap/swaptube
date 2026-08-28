#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Common/ThreeDimensionScene.h"
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

    shared_ptr<ThreeDAlgebraScene> left = make_shared<ThreeDAlgebraScene>(vec2(1./3, 1));
    left->manager.set({{"associativity", "0"}});
    left->manager.set(parentcontrol);
    cs.add_scene(left, "left", vec2(1./6, .5));

    shared_ptr<ThreeDimensionScene> basis = make_shared<ThreeDimensionScene>(vec2(.6, 1));
    basis->manager.set(parentcontrol);
    basis->manager.begin_timer("tx");
    basis->manager.set({
        {"point0.x", "1"},  {"point0.y", "0"},  {"point0.z", "0"},   // xx
        {"point1.x", "0"},  {"point1.y", "1"},  {"point1.z", "0"},   // xy
        {"point2.x", "0"},  {"point2.y", "0"},  {"point2.z", "1"},   // xz
        {"point3.x", "-1"}, {"point3.y", "0"},  {"point3.z", "0"},   // yy
        {"point4.x", "<yz_x>"}, {"point4.y", "<yz_y>"}, {"point4.z", "<yz_z>"},   // yz
        {"point5.x", "<zz_x>"}, {"point5.y", "<zz_y>"}, {"point5.z", "<zz_z>"},   // zz
        {"points_radius_multiplier", "2"},
        {"slow", "<tx> .2 *"},
        {"d", "4"}, {"q1", "<slow> sin"}, {"qi", "<slow> sin .2 *"}, {"qj", "<slow> cos"},
    });
    basis->point_names = {
        {0, "xx"}, {1, "xy"}, {2, "xz"}, {3, "yy"}, {4, "yz"}, {5, "zz"},
    };
    // Add axes
    float line_distance = 1.2;
    basis->add_line(Line(vec3(-line_distance, 0, 0), vec3(line_distance, 0, 0), 0xffff0000, 1.0f, false)); // x-axis
    basis->add_line(Line(vec3(0, -line_distance, 0), vec3(0, line_distance, 0), 0xff00ff00, 1.0f, false)); // y-axis
    basis->add_line(Line(vec3(0, 0, -line_distance), vec3(0, 0, line_distance), 0xff0000ff, 1.0f, false)); // z-axis
    cs.add_scene(basis, "basis", vec2(.5, .5));

    shared_ptr<ThreeDAlgebraScene> right = make_shared<ThreeDAlgebraScene>(vec2(1./3, 1));
    right->manager.set({{"associativity", "1"}});
    right->manager.set(parentcontrol);
    cs.add_scene(right, "right", vec2(5./6, .5));

    stage_macroblock(SilenceBlock(8));
    cs.render_microblock();
    cs.manager.transition(MICRO, ab);
    cs.render_microblock();
    cs.render_microblock();
    cs.manager.transition(MICRO, {
        {"yz_x", "0"}, {"yz_y", "1"}, {"yz_z", "0"},
        {"zz_x", "0"}, {"zz_y", "0"}, {"zz_z", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();
}
