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
    tds.enable_globe("earth_tiny");
    for(int i = -1; i <= 1; i+=1)
    for(int j = -1; j <= 1; j+=1)
    for(int k = -1; k <= 1; k+=1)
        tds.add_point(Point(vec3(i, j, k), OPAQUE_WHITE));

    tds.add_line(Line(vec3(0, 0, 0), vec3(2, 0, 0), 0xffff0000));
    tds.add_line(Line(vec3(0, 0, 0), vec3(0, 2, 0), 0xff00ff00));
    tds.add_line(Line(vec3(0, 0, 0), vec3(0, 0, 2), 0xff0000ff));

    tds.manager.set("x", "0");
    tds.manager.set("y", "0");
    tds.manager.set("z", "0");
    tds.manager.set("d", "5");
    tds.manager.set("surfaces_opacity", "1");
    tds.manager.set("points_opacity", "1");
    tds.manager.set("lines_opacity", "1");
    tds.manager.set("q1", "1");
    tds.manager.set("qi", ".1");
    tds.manager.set("qj", ".1");
    tds.manager.set("qk", "0");

    stage_macroblock(SilenceBlock(2), 1);
    quat q(1,0,0,0);
    tds.manager.transition(MICRO, "q1", std::to_string(q.u));
    tds.manager.transition(MICRO, "qi", std::to_string(q.i));
    tds.manager.transition(MICRO, "qj", std::to_string(q.j));
    tds.manager.transition(MICRO, "qk", std::to_string(q.k));
    tds.render_microblock();

    stage_macroblock(SilenceBlock(2), 1);
    tds.render_microblock();

    quat quats[6] = {PITCH_DOWN,PITCH_UP,YAW_RIGHT,YAW_LEFT,ROLL_CW,ROLL_CCW};
    for(quat mult : quats){
        q *= mult;
        tds.manager.transition(MICRO, {{"q1", std::to_string(q.u)},
                                       {"qi", std::to_string(q.i)},
                                       {"qj", std::to_string(q.j)},
                                       {"qk", std::to_string(q.k)}
        });
        stage_macroblock(SilenceBlock(2), 1);
        tds.render_microblock();
    }
}

void render_video() {
    render_3d();
}
