using namespace std;
#include <string>
const string project_name = "GravityFractals";
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
    ColorScheme cs("0079ff00dfa2f6fa70ff0060333333");
    OrbitSim sim;
    OrbitScene2D scene(&sim);
    CompositeScene comp;
    comp.add_scene(&scene, "orbit2d_s", 0, 0, 1, 1, true);

    unsigned int planet1_color = cs.get_color();
    unsigned int planet2_color = cs.get_color();
    unsigned int planet3_color = cs.get_color();
    unsigned int planet4_color = cs.get_color();
    unsigned int planet5_color = cs.get_color();
    sim.add_fixed_object(planet1_color, 1, "planet1");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet1.x", "-.3"},
        {"planet1.y", "-.3"},
        {"planet1.z", "0"}
    });
    sim.add_fixed_object(planet2_color, 1, "planet2");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet2.x", "0.3171"},
        {"planet2.y", "0.35"},
        {"planet2.z", "0"}
    });
    sim.add_fixed_object(planet3_color, 1, "planet3");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet3.x", "-.4"},
        {"planet3.y", "0.3"},
        {"planet3.z", "0"}
    });

    scene.dag->add_equations(unordered_map<string, string>{
        {"tick_duration", ".1"},
        {"collision_threshold", "0.055"},
        {"drag", "0.99"},
        {"zoom", "0.5"},
        {"screen_center_x", "0"},
        {"screen_center_y", "0"},
        {"screen_center_z", "0"},
        {"predictions_opacity", "0"},
    });
    scene.physics_multiplier = 10;



    // Drop an object
    sim.add_mobile_object(glm::vec3(.6, -.4, 0), OPAQUE_WHITE, 1);
    comp.inject_audio_and_render(AudioSegment("Weird things happen when you drop an object into a complex gravitational field."));



    // Do nothing, let the object keep falling
    comp.inject_audio_and_render(AudioSegment("Can you predict ahead of time which of the three planets this object will crash into? The red one, the blue one, or the green one?"));
    comp.inject_audio_and_render(AudioSegment("If you said the green one, you're right!"));
    comp.inject_audio_and_render(AudioSegment("But that's probably entirely by chance, there's no way you foresaw this complicated trajectory."));



    // Drop a field of random points
    double delta = 0.08;
    double bounds = 0.7;
    scene.physics_multiplier = 20;
    if(FOR_REAL)for(double x = -bounds; x < bounds; x+=delta) for(double y = -bounds; y < bounds; y+=delta){
        glm::vec3 pos(x,y,0);
        sim.add_mobile_object(pos, OPAQUE_WHITE, 1);
    }
    comp.inject_audio_and_render(AudioSegment("Let's drop more objects!"));
    comp.inject_audio_and_render(AudioSegment("Naturally, a whole bunch of them crash into each of the 3 planets."));



    // Make a blob of points around each planet which will fall in, and color them the same color as the planet which they will fall into.
    scene.physics_multiplier = 10;
    if(FOR_REAL)for (int i = 0; i < 100; i++) {
        sim.add_mobile_object(glm::vec3((*scene.dag)["planet1.x"] + (rand() / double(RAND_MAX) - 0.5) * 0.2, (*scene.dag)["planet1.y"] + (rand() / double(RAND_MAX) - 0.5) * 0.2, 0), planet1_color, 1);
        sim.add_mobile_object(glm::vec3((*scene.dag)["planet2.x"] + (rand() / double(RAND_MAX) - 0.5) * 0.2, (*scene.dag)["planet2.y"] + (rand() / double(RAND_MAX) - 0.5) * 0.2, 0), planet2_color, 1);
        sim.add_mobile_object(glm::vec3((*scene.dag)["planet3.x"] + (rand() / double(RAND_MAX) - 0.5) * 0.2, (*scene.dag)["planet3.y"] + (rand() / double(RAND_MAX) - 0.5) * 0.2, 0), planet3_color, 1);
    }
    comp.inject_audio_and_render(AudioSegment("We know for sure that the ones which start sufficiently close to each planet will indeed crash into it,"));
    comp.inject_audio_and_render(AudioSegment("due to a lack of energy to escape that planet's gravity well."));



    comp.inject_audio_and_render(AudioSegment("But that reasoning only goes so far."));
    comp.inject_audio_and_render(AudioSegment("Now, I won't lead you on- this is not a problem with a clean solution. There's no simple equation here to tell you."));



    // Turn on the predictions
    scene.dag->add_transitions(unordered_map<string, string>{
        {"tick_duration", "0.1"},
        {"predictions_opacity", "1"},
    });
    comp.inject_audio_and_render(AudioSegment("I'll just go ahead and spoil the pattern."));
    comp.inject_audio_and_render(AudioSegment("This plot shows, if an object is dropped at each point, the planet which it would crash into."));



    // Pan around by setting screen_center_x/y to some moving formula, and slightly increase zoom.
    scene.dag->add_transitions(unordered_map<string, string>{
        {"screen_center_x", "<t> cos"},
        {"screen_center_y", "<t> sin"},
        {"zoom", "1.5"}
    });
    comp.inject_audio_and_render(AudioSegment("There's a lot of emergent complexity going on here."));
    comp.inject_audio_and_render(AudioSegment(3));



    // Zoom out and restore the position
    // Drop a field of points
    delta = 0.02;
    scene.physics_multiplier = 0;
    for(double x = -bounds; x < bounds; x+=delta) for(double y = -bounds; y < bounds; y+=delta){
        glm::vec3 pos(x,y,0);
        int color = sim.predict_fate_of_object(pos, *(scene.dag));
        sim.add_mobile_object(pos, color, 1);
    }
    scene.dag->add_transitions(unordered_map<string, string>{
        {"screen_center_x", "0"},
        {"screen_center_y", "0"},
        {"predictions_opacity", "0.1"},
        {"zoom", "0.5"}
    });
    comp.inject_audio_and_render(AudioSegment("Let's watch an object at every point fall."));
    scene.physics_multiplier = 10;



    comp.inject_audio_and_render(AudioSegment(3)); // Leave some time (3s) to appreciate the simulation
    comp.inject_audio_and_render(AudioSegment("Wow."));
    comp.inject_audio_and_render(AudioSegment(3)); // Leave some time (3s) to appreciate the simulation



    // Move one of the planets
    scene.dag->add_transitions(unordered_map<string, string>{
        {"predictions_opacity", "1"},
        {"planet2.x", "0.3"},
        {"planet2.y", "-.3"},
        {"planet2.z", "0"}
    });
    comp.inject_audio_and_render(AudioSegment("We can also move around the planets and see how the plot changes."));



    // Transition the planets to an equilateral triangle
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "0"},
        {"planet1.y", "-0.57735 .75 *"},
        {"planet2.x", "0.5 .75 *"},
        {"planet2.y", "0.288675 .75 *"},
        {"planet3.x", "-0.5 .75 *"},
        {"planet3.y", "0.288675 .75 *"}
    });
    comp.inject_audio_and_render(AudioSegment("If their shape is symmetric, so is the plot."));



    // Transition the planets to an isosceles triangle, with a vertical axis of symmetry.
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "0"},
        {"planet1.y", "-0.3"},
        {"planet2.x", "0.3"},
        {"planet2.y", "0.2"},
        {"planet3.x", "-0.3"},
        {"planet3.y", "0.2"}
    });
    comp.inject_audio_and_render(AudioSegment("We can play around with some different shapes and watch what happens to the design."));



    // Transition the planets to a collinear configuration, with one planet at (0,0)
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "-0.5"},
        {"planet1.y", "0"},
        {"planet2.x", "0.5"},
        {"planet2.y", "0"},
        {"planet3.x", "0"},
        {"planet3.y", "0"}
    });
    comp.inject_audio_and_render(AudioSegment("Observe the effects when the planets are aligned in a straight line."));



    // Move one planet far away from the other two (to about (0.8, 0))
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "-0.8"},
        {"planet1.y", "0.5"}
    });
    comp.inject_audio_and_render(AudioSegment("Let's move one of the planets far away from the other two."));



    // Move the planets back to where they were
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "-.33"},
        {"planet1.y", "-.3"},
        {"planet2.x", "0.3171"},
        {"planet2.y", "0.35"},
        {"planet3.x", "-.4"},
        {"planet3.y", "0.3"}
    });



    comp.inject_audio_and_render(AudioSegment("Let's drop a fourth one in there too and see what happens."));
    // Add fourth planet far in the distance, and using the transition mechanics, slide it in from far away.
    sim.add_fixed_object(planet4_color, 1, "planet4");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet4.x", "5.0"},
        {"planet4.y", "5.0"},
        {"planet4.z", "0"}
    });
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet4.x", "0.4"},
        {"planet4.y", "-.4"}
    });



    comp.inject_audio_and_render(AudioSegment("And here's a fifth."));
    // Add fifth planet far in the distance, but using the transition mechanics, slide it in from far away.
    sim.add_fixed_object(planet5_color, 1, "planet5");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet5.x", "-5.0"},
        {"planet5.y", "-5.0"},
        {"planet5.z", "0"}
    });



    // Move the planets around a little bit
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "-0.5"},
        {"planet1.y", "0"},
        {"planet2.x", "0.5"},
        {"planet2.y", "0"},
        {"planet3.x", "0"},
        {"planet3.y", "0.5"},
        {"planet4.x", "0"},
        {"planet4.y", "-0.5"},
        {"planet5.x", "0"},
        {"planet5.y", "0"}
    });
    comp.inject_audio_and_render(AudioSegment(3)); // Leave some time (3s) to appreciate the simulation



    // Move the planets into a regular pentagon
    {
        double radius = 0.5;
        unordered_map<string, string> pentagon_transitions;
        for (int i = 0; i < 5; ++i) {
            double angle = i * 2 * M_PI / 5;
            double x = radius * cos(angle);
            double y = radius * sin(angle);
            pentagon_transitions["planet" + to_string(i + 1) + ".x"] = to_string(x);
            pentagon_transitions["planet" + to_string(i + 1) + ".y"] = to_string(y);
        }
        scene.dag->add_transitions(pentagon_transitions);
    }
    comp.inject_audio_and_render(AudioSegment(3)); // Leave some time (3s) to appreciate the simulation



    // Move the planets into arbitrarily selected spots in the half-unit sphere
    {
        unordered_map<string, string> half_unit_sphere_transitions;
        half_unit_sphere_transitions["planet1.x"] = to_string(0.3);
        half_unit_sphere_transitions["planet1.y"] = to_string(0.4);
        half_unit_sphere_transitions["planet2.x"] = to_string(-0.3);
        half_unit_sphere_transitions["planet2.y"] = to_string(0.3);
        half_unit_sphere_transitions["planet3.x"] = to_string(0.2);
        half_unit_sphere_transitions["planet3.y"] = to_string(-0.2);
        half_unit_sphere_transitions["planet4.x"] = to_string(-0.4);
        half_unit_sphere_transitions["planet4.y"] = to_string(-0.1);
        half_unit_sphere_transitions["planet5.x"] = to_string(0.1);
        half_unit_sphere_transitions["planet5.y"] = to_string(0.2);
        scene.dag->add_transitions(half_unit_sphere_transitions);
    }
    comp.inject_audio_and_render(AudioSegment(3)); // Leave some time (3s) to appreciate the simulation



    // Get planets 3 4 5 far away before deleting
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet3.x", "7"},
        {"planet3.y", "7"},
        {"planet4.x", "-7"},
        {"planet4.y", "7"},
        {"planet5.x", "0"},
        {"planet5.y", "-10"}
    });



    comp.inject_audio_and_render(AudioSegment("You might also be surprised to the amount of complexity observed from merely 2 planets."));
    // Remove all but 2 planets by calling remove_fixed_object(dag_name)
    sim.remove_fixed_object("planet3");
    sim.remove_fixed_object("planet4");
    sim.remove_fixed_object("planet5");
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "0.1"},
        {"planet1.y", "0.2"},
        {"planet2.x", "-.1"},
        {"planet2.y", "-.2"},
    });



    // Add a third planet back
    sim.add_fixed_object(planet3_color, 1, "planet3");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet3.x", "-0.4"},
        {"planet3.y", "0.3"},
        {"planet3.z", "0"}
    });
    // Set the planets to be all black with the exception of planet 1, by accessing their data in the sim.
    comp.inject_audio_and_render(AudioSegment("Now, one thing that I found super interesting is the presence of disjoint areas of the same color."));
    comp.inject_audio_and_render(AudioSegment("What would cause two distinct areas which don't contact each other to drop to the same point?"));



    comp.inject_audio_and_render(AudioSegment("Well, no need to philosophize, let's drop some points and watch."));
    // Drop another field of points, but only including ones which are planet 1's color, and only those which are further left than x=-2 or further right than x=3.
    for(double x = -2; x < -2; x+=delta) for(double y = -2; y < 2; y+=delta){
        glm::vec3 pos(x, y, 0);
        if (sim.predict_fate_of_object(pos, *(scene.dag)) == planet1_color) {
            sim.add_mobile_object(pos, planet1_color, 1);
        }
    }
    for(double x = 4; x < 4; x+=delta) for(double y = -2; y < 2; y+=delta){
        glm::vec3 pos(x, y, 0);
        if (sim.predict_fate_of_object(pos, *(scene.dag)) == planet1_color) {
            sim.add_mobile_object(pos, planet1_color, 1);
        }
    }
    comp.inject_audio_and_render(AudioSegment(3));



    // Set drag to zero and disable the predictive drawing since it will not converge
    scene.dag->add_equations(unordered_map<string, string>{
        {"drag", "1"},
        {"predictions_opacity", "0"},
    });
    comp.inject_audio_and_render(AudioSegment("Now, those who know a thing or two about physics could probably tell you that, if these gravitational attractors are actually points, then in reality, you shouldn't ever actually crash."));



    // Drop an object, which since drag is now zero, should spin around forever.
    sim.add_mobile_object(glm::vec3(0.5, -0.5, 0), OPAQUE_WHITE, 1);
    comp.inject_audio_and_render(AudioSegment("You might fall into a stable orbit or spin around chaotically forever, but your kinetic energy wouldn't dissipate enough to cause your orbit to decay into a crash."));



    // Start increasing drag back to what it was
    scene.dag->add_transitions(unordered_map<string, string>{
        {"drag", "0.99"},
    });
    comp.inject_audio_and_render(AudioSegment("And that's true! I am slightly cheating here- I am applying a constant force, a 'drag', per se, which slows the particle down."));



    // Display the parameters at play
    int latex_col = 0xffeeeeee;
    DagLatexScene drag("drag"       , "Drag"  , latex_col);
    comp.add_scene(&drag, "drag_s", .04, .86, 1, .1, true); 
    scene.dag->add_transitions(unordered_map<string, string>{
        {"predictions_opacity", "1"},
    });
    comp.inject_audio_and_render(AudioSegment("Let's mess around with this parameter and see what happens...!"));



    // Increase drag
    scene.dag->add_transitions(unordered_map<string, string>{
        {"drag", "0.95"},
    });
    comp.inject_audio_and_render(AudioSegment("Check that out. Perhaps unsurprisingly, the greater the drag, the less complicated the orbital patterns are, because each particle will just slowly march to the nearest attractor."));



    // Decrease drag
    scene.dag->add_transitions(unordered_map<string, string>{
        {"drag", "0.998"},
    });
    comp.inject_audio_and_render(AudioSegment("As drag decreases, the more time the object spends winding in complicated orbits before its fate is decided, and thus, the more scrambled the colors get."));



    // Comment on the similarity between this and the three body problem, and add to the script.
    // Remove all of the fixed objects, add 3 mobile objects, and turn the mobile object interactions on for the simulator
    sim.fixed_objects.clear();
    sim.add_mobile_object(glm::vec3(-0.5, 0, 0), cs.get_color(), 1);
    sim.add_mobile_object(glm::vec3(0.5, 0, 0), cs.get_color(), 1);
    sim.add_mobile_object(glm::vec3(0, 0.5, 0), cs.get_color(), 1);
    sim.mobile_interactions = true;
    comp.inject_audio_and_render(AudioSegment("This phenomenon is reminiscent of the three-body problem, where predicting the exact trajectory of each body becomes incredibly complex."));


    //TODO path rendering
    //TODO 3d fractals
    //TODO convergence time
    //TODO show ripples and gravity wells
}

int main() {
    Timer timer;
    render_video();
    timer.stop_timer();
    return 0;
}

