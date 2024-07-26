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

#include "../scenes/Physics/OrbitScene2D.cpp"
#include "../scenes/Media/DagLatexScene.cpp"
#include "../scenes/Common/CompositeScene.cpp"
#include "../misc/Timer.cpp"

void render_video() {
    OrbitSim sim;
    OrbitScene2D scene(&sim);
    DagLatexScene fc("force_constant");
    DagLatexScene drag("drag");
    DagLatexScene ct("collision_threshold");
    CompositeScene comp;
    comp.add_scene(&scene, "orbit2d_s", 0, 0, 1, 1, true); 
    comp.add_scene(&fc,     "fc_s", 0, .9, 1, .1, true); 
    comp.add_scene(&drag, "drag_s", 0, .8, 1, .1, true); 
    comp.add_scene(&ct,     "ct_s", 0, .7, 1, .1, true); 
    scene.physics_multiplier = 1;

    if(true){
        sim.add_fixed_object(0xffff0000, 1, "planet1");
        scene.dag->add_equations(unordered_map<string, string>{
            {"planet1.x", "-.3"},
            {"planet1.y", "-.3"},
            {"planet1.z", "0"}
        });

        sim.add_fixed_object(0xff00ff00, 1, "planet2");
        scene.dag->add_equations(unordered_map<string, string>{
            {"planet2.x", "0.3171"},
            {"planet2.y", "0.3"},
            {"planet2.z", "0"}
        });

        sim.add_fixed_object(0xff0000ff, 1, "planet3");
        scene.dag->add_equations(unordered_map<string, string>{
            {"planet3.x", "-.4"},
            {"planet3.y", "0.3"},
            {"planet3.z", "0"}
        });
    }

    scene.dag->add_equations(unordered_map<string, string>{
        {"force_constant", "0.0001"},
        {"collision_threshold", "0.01"},
        {"drag", "0.99"}
    });

    /*
    scene.inject_audio_and_render(AudioSegment(.1));
    sim.mobile_interactions = false;
    double delta = 0.02;
    int bounds = 2;
    for(double x = -bounds; x < bounds; x+=delta) for(double y = -bounds; y < bounds; y+=delta){
        glm::vec3 pos(x,y,0);
        int color = sim.predict_fate_of_object(pos, scene.dag);
        sim.add_mobile_object(pos, color, 1);
    }
    */
    /*
    scene.dag.add_transitions(unordered_map<string, string>{
        {"planet2.x", "0.3"},
        {"planet2.y", "-.3"},
        {"planet2.z", "0"}
    });
    */
    scene.dag->add_transitions(unordered_map<string, string>{
        {"drag", "0.9999"}
    });
    comp.inject_audio_and_render(AudioSegment(3));
}
int main() {
    Timer timer;
    render_video();
    timer.stop_timer();
    return 0;
}
