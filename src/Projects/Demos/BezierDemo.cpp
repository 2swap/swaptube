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
    stage_macroblock(SilenceBlock(5), 2);
    StateSet ss1 = bsc.pop_next_state_set();
    for(const auto& p: ss1) {
        cout << p.first << ": " << p.second << endl;
    }
    StateSet ss2 = bsc.pop_next_state_set();
    cout << "SS2" << endl;
    for(const auto& p: ss2) {
        cout << p.first << ": " << p.second << endl;
    }




    ms.manager.set(ss1);
    ms.render_microblock();
    ms.manager.set(ss2);
    ms.render_microblock();
}
