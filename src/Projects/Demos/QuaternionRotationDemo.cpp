#include "../Scenes/Common/ThreeDimensionScene.h"

double sqrt2over2 = 0.7071067811865476;
quat PITCH_DOWN(sqrt2over2, sqrt2over2, 0, 0);
quat PITCH_UP(sqrt2over2, -sqrt2over2, 0, 0);
quat YAW_RIGHT(sqrt2over2, 0, -sqrt2over2, 0);
quat YAW_LEFT(sqrt2over2, 0, sqrt2over2, 0);
quat ROLL_CW(sqrt2over2, 0, 0, -sqrt2over2);
quat ROLL_CCW(sqrt2over2, 0, 0, sqrt2over2);

void render_3d(){
    ThreeDimensionScene tds;
    for(int i = -7; i <= 7; i+=2)
    for(int j = -7; j <= 7; j+=2)
    for(int k = -7; k <= 7; k+=2)
        tds.add_point(Point(vec3(i, j, k), OPAQUE_WHITE));

    quat q(1,0,0,0);
    tds.manager.set("x", "0");
    tds.manager.set("y", "0");
    tds.manager.set("z", "0");
    tds.manager.set("d", "0");
    tds.manager.set("surfaces_opacity", "1");
    tds.manager.set("points_opacity", "1");
    tds.manager.set("lines_opacity", "1");
    tds.manager.set("q1", std::to_string(q.w));
    tds.manager.set("qi", std::to_string(q.x));
    tds.manager.set("qj", std::to_string(q.y));
    tds.manager.set("qk", std::to_string(q.z));
    stage_macroblock(SilenceBlock(2), 1);
    tds.render_microblock();

    quat quats[6] = {PITCH_DOWN,PITCH_UP,YAW_RIGHT,YAW_LEFT,ROLL_CW,ROLL_CCW};
    for(quat mult : quats){
        q *= mult;
        tds.manager.transition(MICRO, {{"q1", std::to_string(q.w)},
                                             {"qi", std::to_string(q.x)},
                                             {"qj", std::to_string(q.y)},
                                             {"qk", std::to_string(q.z)}
        });
        stage_macroblock(SilenceBlock(2), 1);
        tds.render_microblock();
    }
}

void render_video() {
    render_3d();
}
