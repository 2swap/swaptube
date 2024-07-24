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
    scene.physics_multiplier = 1;

    if(false){
        sim.add_fixed_object(1, OPAQUE_WHITE, 1, "planet1");
        scene.dag.add_equations(unordered_map<string, string>{
            {"planet1.x", "-.3"},
            {"planet1.y", "-.3"},
            {"planet1.z", "0"}
        });

        sim.add_fixed_object(1, OPAQUE_WHITE, 1, "planet2");
        scene.dag.add_equations(unordered_map<string, string>{
            {"planet2.x", "0.3"},
            {"planet2.y", "0.3"},
            {"planet2.z", "0"}
        });

        sim.add_fixed_object(1, OPAQUE_WHITE, 1, "planet3");
        scene.dag.add_equations(unordered_map<string, string>{
            {"planet3.x", "-.3"},
            {"planet3.y", "0.3"},
            {"planet3.z", "0"}
        });
    }
    if(true){
        scene.dag.add_equations(unordered_map<string, string>{
            {"q1", "<t> 4 / cos"},
            {"qi", "0"},
            {"qj", "<t> -4 / sin"},
            {"qk", "0"},
            {"d", "1"},
        });
    }
    scene.inject_audio_and_render(AudioSegment(1));
    sim.add_mobile_object(glm::vec3(-.2,-.1,.1), 1, 0xffff0000, 1);
    sim.add_mobile_object(glm::vec3(-.4,-.2,-.2), 1, 0xffff0000, 1);
    sim.add_mobile_object(glm::vec3(-.2,.2,.3), 1, 0xffff0000, 1);
    sim.add_mobile_object(glm::vec3(.2,-.1,-.2), 1, 0xffff0000, 1);
    scene.inject_audio_and_render(AudioSegment(10));
}
int main() {
    Timer timer;
    render_video();
    timer.stop_timer();
    return 0;
}
