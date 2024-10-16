using namespace std;
#include <string>
const string project_name = "OrbitsColorSchemeTest";
#include "../io/PathManager.cpp"
const int width_base = 640;
const int height_base = 360;
const int mult = 1;

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;

#include "../io/writer.cpp"

#include "../Scenes/Physics/OrbitScene2D.cpp"
#include "../Scenes/Media/StateSliderScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../misc/Timer.cpp"
#include "../misc/ColorScheme.cpp"

extern void run_cuda_test();

void render_video() {
    ColorScheme cs("02152603346e6eacdae2e2b6");
    OrbitSim sim;
    OrbitScene2D scene(&sim);
    CompositeScene comp;
    comp.add_scene(&scene, "orbit2d_s", 0, 0, 1, 1, true); 

    sim.add_fixed_object(cs.get_color(), 1, "planet1");
    scene.state_manager.set(unordered_map<string, string>{
        {"planet1.x", "-.3"},
        {"planet1.y", "-.3"},
        {"planet1.z", "0"}
    });
    sim.add_fixed_object(cs.get_color(), 1, "planet2");
    scene.state_manager.set(unordered_map<string, string>{
        {"planet2.x", "0.3171"},
        {"planet2.y", "0.3"},
        {"planet2.z", "0"}
    });
    sim.add_fixed_object(cs.get_color(), 1, "planet3");
    scene.state_manager.set(unordered_map<string, string>{
        {"planet3.x", "-.4"},
        {"planet3.y", "0.3"},
        {"planet3.z", "0"}
    });

    scene.state_manager.set(unordered_map<string, string>{
        {"force_constant", "0.000001"},
        {"collision_threshold", "0.055"},
        {"drag", "0.9997"},
        {"drag_display", "1 <drag> -"},
        {"zoom", "0.5"},
        {"screen_center_x", "0"},
        {"screen_center_y", "0"},
        {"screen_center_z", "0"},
    });

    int latex_col = cs.get_color();
    StateSliderScene fc  ("force_constant"     , "Force" , latex_col);
    StateSliderScene drag("drag_display"       , "Drag"  , latex_col);
    comp.add_scene(&fc,     "fc_s", 0, .9, 1, .1, true); 
    comp.add_scene(&drag, "drag_s", 0, .8, 1, .1, true); 
    scene.physics_multiplier = 1;

    comp.inject_audio_and_render(AudioSegment(2));
}
int main() {
    Timer timer;
    render_video();
    timer.stop_timer();
    return 0;
}
