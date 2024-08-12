using namespace std;
#include <string>
const string project_name = "Orbits";
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

void render_video() {
    ColorScheme cs = get_color_schemes()[6];
    OrbitSim sim;
    OrbitScene2D scene(&sim);
    CompositeScene comp;
    comp.add_scene(&scene, "orbit2d_s", 0, 0, 1, 1, true); 
    sim.add_fixed_object(cs.get_color(), "planet1");
    comp.state_manager.set(unordered_map<string, string>{
        {"planet1.opacity", "1"},
        {"planet1.x", "-.2"},
        {"planet1.y", "-.1"},
        {"planet1.z", "0"},
    });
    sim.add_fixed_object(cs.get_color(), "planet2");
    comp.state_manager.set(unordered_map<string, string>{
        {"planet2.opacity", "1"},
        {"planet2.x", "0.2171"},
        {"planet2.y", "0.1"},
        {"planet2.z", "0"}
    });

    comp.state_manager.set(unordered_map<string, string>{
        {"collision_threshold", "0.05"},
        {"drag", "0.97"},
        {"drag_display", "1 <drag> -"},
        {"zoom", "0.5"},
        {"screen_center_x", "0"},
        {"screen_center_y", "0"},
        {"screen_center_z", "0"},
        {"physics_multiplier", "1"},
        {"predictions_opacity", "1"},
        {"point_path.opacity", "0"},
        {"point_path.x", "1"},
        {"point_path.y", "1"},
        {"eps", "0.003"},
        {"tick_duration", "0.002"}
    });

    StateSliderScene drag("eps", "\\epsilon", 0, 1);
    comp.add_scene(&drag, "eps_s", 0, .9, 1, .1, true); 

    scene.inject_audio_and_render(AudioSegment(3));
    scene.state_manager.set(unordered_map<string, string>{
        {"eps", "0.0001"},
    });
    scene.inject_audio_and_render(AudioSegment(3));
    scene.state_manager.set(unordered_map<string, string>{
        {"eps", "0.00001"},
    });
    scene.inject_audio_and_render(AudioSegment(3));
}
int main() {
    Timer timer;
    render_video();
    return 0;
}
