#include "../Scenes/Common/ThreeDimensionScene.cpp"

double sqrt2over2 = 0.7071067811865476;
glm::quat PITCH_DOWN(sqrt2over2, sqrt2over2, 0, 0);
glm::quat PITCH_UP(sqrt2over2, -sqrt2over2, 0, 0);
glm::quat YAW_RIGHT(sqrt2over2, 0, -sqrt2over2, 0);
glm::quat YAW_LEFT(sqrt2over2, 0, sqrt2over2, 0);
glm::quat ROLL_CW(sqrt2over2, 0, 0, -sqrt2over2);
glm::quat ROLL_CCW(sqrt2over2, 0, 0, sqrt2over2);

void render_3d(){
    ThreeDimensionScene tds;
    for(int i = -7; i <= 7; i+=2)
    for(int j = -7; j <= 7; j+=2)
    for(int k = -7; k <= 7; k+=2)
        tds.add_point(Point(glm::vec3(i, j, k), OPAQUE_WHITE));

    glm::quat q(1,0,0,0);
    tds.state.set("x", "0");
    tds.state.set("y", "0");
    tds.state.set("z", "0");
    tds.state.set("d", "0");
    tds.state.set("surfaces_opacity", "1");
    tds.state.set("points_opacity", "1");
    tds.state.set("lines_opacity", "1");
    tds.state.set("q1", std::to_string(q.w));
    tds.state.set("qi", std::to_string(q.x));
    tds.state.set("qj", std::to_string(q.y));
    tds.state.set("qk", std::to_string(q.z));
    tds.stage_macroblock(SilenceBlock(2), 1);
    tds.render_microblock();

    glm::quat quats[6] = {PITCH_DOWN,PITCH_UP,YAW_RIGHT,YAW_LEFT,ROLL_CW,ROLL_CCW};
    for(glm::quat mult : quats){
        q *= mult;
        tds.state.transition(MICRO, {{"q1", std::to_string(q.w)},
                                             {"qi", std::to_string(q.x)},
                                             {"qj", std::to_string(q.y)},
                                             {"qk", std::to_string(q.z)}
        });
        tds.stage_macroblock(SilenceBlock(2), 1);
        tds.render_microblock();
    }
}

void render_video() {
    render_3d();
}
