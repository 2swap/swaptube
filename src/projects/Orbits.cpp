using namespace std;
#include <string>
const string project_name = "Orbits";
#include "../io/PathManager.cpp"
const int width_base = 640;
const int height_base = 360;
const int mult = 3;

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;

#include "../io/writer.cpp"

#include "../scenes/Physics/OrbitScene.cpp"
#include "../misc/Timer.cpp"

void render_video() {
    OrbitSim sim;
    OrbitScene scene(&sim);
    scene.physics_multiplier = 2;

    if(true){
        sim.add_fixed_object(0xffff0000, 1, "planet1");
        scene.dag.add_equations(unordered_map<string, string>{
            {"planet1.x", "-.3"},
            {"planet1.y", "-.3"},
            {"planet1.z", "0"}
        });

        sim.add_fixed_object(0xff00ff00, 1, "planet2");
        scene.dag.add_equations(unordered_map<string, string>{
            {"planet2.x", "0.3"},
            {"planet2.y", "0.3"},
            {"planet2.z", "0"}
        });

        sim.add_fixed_object(0xff0000ff, 1, "planet3");
        scene.dag.add_equations(unordered_map<string, string>{
            {"planet3.x", "-.4"},
            {"planet3.y", "0.3"},
            {"planet3.z", "0"}
        });
    }
    if(false){
        scene.dag.add_equations(unordered_map<string, string>{
            {"q1", "<t> 4 / cos"},
            {"qi", "0"},
            {"qj", "<t> -4 / sin"},
            {"qk", "0"},
        });
    }
    sim.mobile_interactions = false;

    scene.inject_audio_and_render(AudioSegment(.1));
    for(double x = -1; x < 1; x+=.01) for(double y = -1; y < 1; y+=.01){
        glm::vec3 pos(x,y,0);
        int color = sim.predict_fate_of_object(pos, scene.dag);
        sim.add_mobile_object(pos, color, 1);
    }

    scene.inject_audio_and_render(AudioSegment(10));
}
int main() {
    Timer timer;
    render_video();
    timer.stop_timer();
    return 0;
}
