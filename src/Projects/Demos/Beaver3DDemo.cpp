#include "../Scenes/Math/Beavers/BeaverTNF3DScene.h"
#include "../Scenes/Common/ThreeDimensionScene.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Core/State/StateTester.h"
#include "../Core/Smoketest.h"
#include <memory>
#include <string>

static void add_grid_lines(ThreeDimensionScene& ts, const std::vector<std::vector<double>> cuts, const std::vector<std::vector<double>> skip, int color, float opacity) {
    auto contains = [](const std::vector<double>& v, double x) {
        for (double y : v) if (y == x) return true;
        return false;
    };
    for (int axis = 0; axis < 3; axis++) {
        const int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
        for (double c1 : cuts[a1]) for (double c2 : cuts[a2]) {
            if (contains(skip[a1], c1) && contains(skip[a2], c2)) continue;
            double s[3], e[3];
            s[axis] = 0; e[axis] = 1;
            s[a1] = e[a1] = c1;
            s[a2] = e[a2] = c2;
            ts.add_line(Line(vec3(s[0], s[1], s[2]), vec3(e[0], e[1], e[2]), color, opacity, false));
        }
    }
}

void render_video() {
    // Alternative target
    /*
    const std::string x = "0.938773719835";
    const std::string y = "0.316870220152";
    const std::string z = "0.991666579364";
    const std::string q1 = "0.252669356126";
    const std::string qi = "0.568644342848";
    const std::string qj = "-0.74655452518";
    const std::string qk = "-0.235495538714";
    */

    const std::string x = "0.49995335936557356";
    const std::string y = "0.31268900632874175";
    const std::string z = "0.74691659212574901";
    const std::string q1 = "-0.295876521928";
    const std::string qi = "0.632884003983";
    const std::string qj = "-0.325099928934";
    const std::string qk = "-0.637357793928";

    // The camera lives in the composite's state; the fractal and the wireframe
    // overlay both read it from there, so they stay in register.
    CompositeScene cs;
    auto bs = std::make_shared<BeaverTNF3DScene>();
    auto ts = std::make_shared<ThreeDimensionScene>();
    cs.add_scene(bs, "bs");
    cs.add_scene(ts, "ts");   // added last: drawn over the fractal

    cs.manager.set({
        {"target_x", x},
        {"target_y", y},
        {"target_z", z},
        {"q1", "0"},
        {"qi", "1"},
        {"qj", "0"},
        {"qk", "0"},
        {"camera_distance_exp", ".5"},
        {"camera_distance", "<camera_distance_exp> exp 0.000000105791925107 +"},
        {"fov", ".5"},
    });
    bs->manager.set({
        {"target_x", "[target_x]"},
        {"target_y", "[target_y]"},
        {"target_z", "[target_z]"},
        {"q1", "[q1]"},
        {"qi", "[qi]"},
        {"qj", "[qj]"},
        {"qk", "[qk]"},
        {"camera_distance", "[camera_distance]"},
        {"fov", "[fov]"},
        {"max_steps", "100"},
        {"max_tnf_depth", "28"},
    });
    ts->manager.set({
        {"x", "[target_x]"},
        {"y", "[target_y]"},
        {"z", "[target_z]"},
        {"q1", "[q1]"},
        {"qi", "[qi] -1 *"},
        {"qj", "[qj] -1 *"},
        {"qk", "[qk] -1 *"},
        {"d", "[camera_distance]"},
        {"fov", "[fov]"},
    });

    // root cube wireframe
    add_grid_lines(*ts, {{0, 1}, {0, 1}, {0, 1}}, {{}, {}, {}}, 0xffffffff, 1);

    stage_macroblock(SilenceBlock(4), 1);
    cs.manager.transition(MICRO, {{"q1", "-.5"}, {"qi", "{t} cos"}, {"qj", "{t} sin"}});
    cs.render_microblock();

    // depth 1 wireframe
    add_grid_lines(*ts, {{0, .5, 1}, {0, .5, 1}, {0, 1. / 3, 2. / 3, 1}}, {{0, 1}, {0, 1}, {0, 1}}, 0xff888888, 1);

    stage_macroblock(SilenceBlock(1), 1);
    cs.manager.transition(MICRO, {{"q1", q1}, {"qi", qi}, {"qj", qj}, {"qk", qk}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.1), 1);
    cs.render_microblock();

    stage_macroblock(SilenceBlock(14), 1);
    cs.manager.transition(MICRO, "camera_distance_exp", "-26");
    cs.render_microblock();
}
