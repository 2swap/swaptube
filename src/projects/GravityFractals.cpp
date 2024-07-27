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
#include "../misc/ColorScheme.cpp"

extern void run_cuda_test();

void render_video() {
    comp.inject_audio_and_render(AudioSegment("Weird things happen when you drop an object into a complex gravitational field."));
    comp.inject_audio_and_render(AudioSegment("Can you predict ahead of time which of the three planets this object will crash into? The red one, the blue one, or the green one?"));
    comp.inject_audio_and_render(AudioSegment("If you said the _ one, you're right!"));
    comp.inject_audio_and_render(AudioSegment("But that's probably entirely by chance, there's no way you foresaw this complicated trajectory."));
    comp.inject_audio_and_render(AudioSegment("Let's drop more objects!"));
    comp.inject_audio_and_render(AudioSegment("Naturally, a whole bunch of them crash into each of the 3 planets."));
    comp.inject_audio_and_render(AudioSegment("We know for sure that the ones which start sufficiently close to each planet will indeed crash into it."));
    comp.inject_audio_and_render(AudioSegment("But that reasoning only goes so far. After all, the point we saw at the beginning landed at the single furthest planet."));
    comp.inject_audio_and_render(AudioSegment("Now, I won't lead you on- this is not a problem with a clean solution. There's no simple equation here to tell you the answer."));
    comp.inject_audio_and_render(AudioSegment("I'll just go ahead and spoil the pattern."));
    comp.inject_audio_and_render(AudioSegment("This plot shows, if an object is dropped at each point, the planet which it would crash into."));
    comp.inject_audio_and_render(AudioSegment("There's a lot of emergent complexity going on here."));
    comp.inject_audio_and_render(AudioSegment("Let's watch an object at every point fall."));
    comp.inject_audio_and_render(AudioSegment("Wow."));
    comp.inject_audio_and_render(AudioSegment("We can also move around the planets and see how the plot changes."));
    comp.inject_audio_and_render(AudioSegment("If their shape is symmetric, so is the plot."));
    comp.inject_audio_and_render(AudioSegment("Let's drop a fourth one in there too and see what happens."));
    comp.inject_audio_and_render(AudioSegment("Now, one thing that I found super interesting is the presence of disjoint areas of the same color."));
    comp.inject_audio_and_render(AudioSegment("What would cause two distinct areas which don't contact each other to drop to the same point?"));
    comp.inject_audio_and_render(AudioSegment("Well, no need to philosophize, let's drop some points and watch."));
    comp.inject_audio_and_render(AudioSegment("???????"));
    comp.inject_audio_and_render(AudioSegment("Now, those who know a thing or two about physics could probably tell you that, if these gravitational attractors are actually points, then in reality, you shouldn't ever actually crash."));
    comp.inject_audio_and_render(AudioSegment("You might fall into a stable orbit or spin around chaotically forever, but your kinetic energy wouldn't dissipate enough to cause your orbit to decay into a crash."));
    comp.inject_audio_and_render(AudioSegment("And that's true! I am slightly cheating here- I am applying a constant force, a 'drag', per se, which constantly slows the particle down."));
    comp.inject_audio_and_render(AudioSegment("Now, if your instinct matches mine, the first thing you want to do is mess around with this number and see what happens...!"));
    comp.inject_audio_and_render(AudioSegment("Check that out. Perhaps unsurprisingly, the greater the drag, the less complicated the orbital patterns are, because each particle will just slowly march to the nearest attractor."));
    comp.inject_audio_and_render(AudioSegment("As drag decreases, the more time the object spends winding in complicated orbits before its fate is decided, and thus, the more scrambled the colors get."));
    comp.inject_audio_and_render(AudioSegment("We can do all sorts of things, such as "));
    comp.inject_audio_and_render(AudioSegment(""));
    comp.inject_audio_and_render(AudioSegment(""));
    comp.inject_audio_and_render(AudioSegment(""));
    // Some example animations:
    ColorScheme cs = get_color_schemes()[6];
    OrbitSim sim;
    OrbitScene2D scene(&sim);
    CompositeScene comp;
    comp.add_scene(&scene, "orbit2d_s", 0, 0, 1, 1, true); 

    sim.add_fixed_object(cs.get_color(), 1, "planet1");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet1.x", "-.3"},
        {"planet1.y", "-.3"},
        {"planet1.z", "0"}
    });
    sim.add_fixed_object(cs.get_color(), 1, "planet2");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet2.x", "0.3171"},
        {"planet2.y", "0.3"},
        {"planet2.z", "0"}
    });
    sim.add_fixed_object(cs.get_color(), 1, "planet3");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet3.x", "-.4"},
        {"planet3.y", "0.3"},
        {"planet3.z", "0"}
    });

    scene.dag->add_equations(unordered_map<string, string>{
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
    DagLatexScene fc  ("force_constant"     , "Force" , latex_col);
    DagLatexScene drag("drag_display"       , "Drag"  , latex_col);
    comp.add_scene(&fc,     "fc_s", 0, .9, 1, .1, true); 
    comp.add_scene(&drag, "drag_s", 0, .8, 1, .1, true); 
    scene.physics_multiplier = 1;

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
        {"drag", "0.9998"},
        {"screen_center_y", ".6"},
    });
    comp.inject_audio_and_render(AudioSegment(3));
    scene.dag->add_transitions(unordered_map<string, string>{
        {"screen_center_x", "<t> cos 40 /"},
        {"screen_center_y", "<t> sin 40 / .6 +"},
        {"zoom", "20"}
    });
    comp.inject_audio_and_render(AudioSegment(3));
    comp.inject_audio_and_render(AudioSegment(3));
}
int main() {
    Timer timer;
    render_video();
    timer.stop_timer();
    return 0;
}
