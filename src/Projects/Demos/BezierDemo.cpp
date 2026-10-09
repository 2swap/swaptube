#include "../Scenes/Math/MandelbrotScene.h"
#include "../Core/State/BezierStateCurve.h"
#include "../Core/State/StateTester.h"

void render_video() {
    MandelbrotScene ms;
    vector<StateSet> waypoints;
    waypoints.push_back({
        {"seed_x_r","3"},
        {"seed_c_r","1"}
    });
    waypoints.push_back({
        {"seed_x_r","2"},
        {"seed_c_r","1"}
    });
    waypoints.push_back({
        {"seed_x_r","2"},
        {"seed_c_r","2"}
    });
    BezierStateCurve bsc(waypoints);
    stage_macroblock(SilenceBlock(5));

    bsc.run_curve(ms);
}
