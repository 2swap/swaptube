using namespace std;
#include <string>
const string project_name = "GravityFractals";
#include "../io/PathManager.cpp"
const int width_base = 640;
const int height_base = 360;
const int mult = 2;

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;

#include "../io/writer.cpp"

#include "../scenes/Physics/OrbitScene2D.cpp"
#include "../scenes/Physics/OrbitScene3D.cpp"
#include "../scenes/Media/DagLatexScene.cpp"
#include "../scenes/Common/CompositeScene.cpp"
#include "../misc/Timer.cpp"
#include "../misc/ColorScheme.cpp"

void render_2d() {
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
        {"collision_threshold", "0.005"},
        {"drag", "0.98"},
        {"drag_slider", "1 <drag> -"},
        {"zoom", "0.53"},
        {"point_path.x", "0"},
        {"point_path.y", "0"},
        {"point_path.opacity", "0"},
        {"screen_center_x", ".6"},
        {"screen_center_y", "-.4"},
        {"screen_center_z", "0"},
        {"predictions_opacity", "0"},
        {"physics_multiplier", "1"},
    });



    // Drop an object
    if(FOR_REAL) sim.add_mobile_object(glm::vec3(.6, -.4, 0), OPAQUE_WHITE, 1);
    comp.inject_audio_and_render(AudioSegment("Weird things happen when you drop an object into a complex gravitational field."));
    scene.dag->add_transitions(unordered_map<string, string>{
        {"physics_multiplier", "3"},
        {"screen_center_x", "0"},
        {"screen_center_y", "0"},
        {"zoom", "0.5"},
    });



    // Do nothing, let the object keep falling
    comp.inject_audio_and_render(AudioSegment("Can you predict ahead of time which of the three planets this object will crash into?"));
    scene.dag->add_transitions(unordered_map<string, string>{
        {"physics_multiplier", "10"},
        {"zoom", "0.5"},
    });
    comp.inject_audio_and_render(AudioSegment("If you said the green one, you're right!"));
    comp.inject_audio_and_render(AudioSegment("But, it's hard to foresee this trajectory."));
    scene.dag->add_equations(unordered_map<string, string>{
        {"point_path.x", ".6"},
        {"point_path.y", "-.4"},
    });
    scene.dag->add_transitions(unordered_map<string, string>{
        {"point_path.opacity", "1"},
    });
    comp.inject_audio_and_render(AudioSegment(3));
    scene.dag->add_transitions(unordered_map<string, string>{
        {"point_path.x", "0.6 <t> 2 / sin 10 / +"},
        {"point_path.y", "-.4 <t> 2 / cos 10 / +"},
    });
    comp.inject_audio_and_render(AudioSegment("Moving the starting point only a little bit causes a huge change in behavior."));
    comp.inject_audio_and_render(AudioSegment(3));



    scene.dag->add_equations(unordered_map<string, string>{
        {"physics_multiplier", "0"},
        {"physics_multiplier", "1"},
    });
    scene.dag->add_transitions(unordered_map<string, string>{
        {"point_path.opacity", "0"},
    });
    // Make a blob of points around each planet which will fall in, and color them the same color as the planet which they will fall into.
    if(FOR_REAL)for (int i = 0; i < 500; i++) {
        float theta  = (rand() / double(RAND_MAX)) * 6.283;
        float radius = (rand() / double(RAND_MAX)) * 0.12;
        float dx = radius * cos(theta);
        float dy = radius * sin(theta);
        sim.add_mobile_object(glm::vec3((*scene.dag)["planet1.x"] + dx, (*scene.dag)["planet1.y"] + dy, 0), planet1_color, 1);
        sim.add_mobile_object(glm::vec3((*scene.dag)["planet2.x"] + dx, (*scene.dag)["planet2.y"] + dy, 0), planet2_color, 1);
        sim.add_mobile_object(glm::vec3((*scene.dag)["planet3.x"] + dx, (*scene.dag)["planet3.y"] + dy, 0), planet3_color, 1);
    }
    comp.inject_audio_and_render(AudioSegment("We know for sure that the ones which start sufficiently close to each planet will indeed crash into it,"));
    scene.dag->add_transitions(unordered_map<string, string>{
        {"physics_multiplier", "10"},
    });
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
        {"screen_center_x", "<t> 5 / cos"},
        {"screen_center_y", "<t> 5 / sin"},
        {"zoom", "1.5"}
    });
    comp.inject_audio_and_render(AudioSegment("There's a lot of emergent complexity going on here."));
    scene.dag->add_transitions(unordered_map<string, string>{
        {"zoom", "3"}
    });
    comp.inject_audio_and_render(AudioSegment(6));
    comp.inject_audio_and_render(AudioSegment("But, it seems like the complexity is finite... more about that later."));



    // Zoom out and restore the position
    // Drop a field of points
    float delta = 0.03;
    int bounds = 1;
    scene.dag->add_equations(unordered_map<string, string>{
        {"physics_multiplier", "0"},
    });
    if(FOR_REAL) for(float x = -bounds; x < bounds; x+=delta) for(float y = -bounds; y < bounds; y+=delta){
        glm::vec3 pos(x,y,0);
        int color = sim.predict_fate_of_object(pos, *(scene.dag));
        sim.add_mobile_object(pos, color, 1);
    }
    scene.dag->add_transitions(unordered_map<string, string>{
        {"screen_center_x", "0"},
        {"screen_center_y", "0"},
        {"predictions_opacity", "0"},
        {"zoom", "0.5"}
    });
    comp.inject_audio_and_render(AudioSegment("Let's watch an object at every point fall."));
    scene.dag->add_equations(unordered_map<string, string>{
        {"physics_multiplier", "10"},
    });



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
    comp.inject_audio_and_render(AudioSegment(3));



    // Move one planet far away from the other two (to about (0.8, 0))
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "-0.8"},
        {"planet1.y", "0.5"}
    });
    comp.inject_audio_and_render(AudioSegment(3));



    // Move the planets back to where they were
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "-.43"},
        {"planet1.y", "-.4"},
        {"planet2.x", "0.4171"},
        {"planet2.y", "0.45"},
        {"planet3.x", "-.5"},
        {"planet3.y", "0.4"}
    });



    comp.inject_audio_and_render(AudioSegment("Let's drop a fourth one in there too and see what happens."));
    // Add fourth planet far in the distance, and using the transition mechanics, slide it in from far away.
    sim.add_fixed_object(planet4_color, 1, "planet4");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet4.x", "9.0"},
        {"planet4.y", "5.0"},
        {"planet4.z", "0"}
    });
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet4.x", "0.5"},
        {"planet4.y", "-.5"}
    });
    comp.inject_audio_and_render(AudioSegment(3)); // Leave some time (3s) to appreciate the simulation



    comp.inject_audio_and_render(AudioSegment("And here's a fifth."));
    // Add fifth planet far in the distance, but using the transition mechanics, slide it in from far away.
    sim.add_fixed_object(planet5_color, 1, "planet5");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet5.x", "-5.0"},
        {"planet5.y", "-8.0"},
        {"planet5.z", "0"}
    });
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet5.x", "0"},
        {"planet5.y", "0"}
    });
    comp.inject_audio_and_render(AudioSegment(3)); // Leave some time (3s) to appreciate the simulation



    // Move the planets around a little bit
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "-.5"},
        {"planet1.y", "-.5"},
        {"planet2.x", "0.5"},
        {"planet2.y", "-.5"},
        {"planet3.x", "0.5"},
        {"planet3.y", "0.5"},
        {"planet4.x", "-.5"},
        {"planet4.y", "0.5"},
    });
    comp.inject_audio_and_render(AudioSegment(5)); // Leave some time (3s) to appreciate the simulation
    comp.inject_audio_and_render(AudioSegment(1)); // Leave some time (3s) to appreciate the simulation
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "-.5"},
        {"planet1.y", "-.5"},
        {"planet2.x", "0.8"},
        {"planet2.y", "-.5"},
        {"planet3.x", "0.5"},
        {"planet3.y", "0.5"},
        {"planet4.x", "-.8"},
        {"planet4.y", "0.5"},
    });
    comp.inject_audio_and_render(AudioSegment(5)); // Leave some time (3s) to appreciate the simulation
    comp.inject_audio_and_render(AudioSegment(1)); // Leave some time (3s) to appreciate the simulation



    // Move the planets into a regular pentagon
    {
        double radius = 0.5;
        unordered_map<string, string> pentagon_transitions;
        for (int i = 0; i < 5; ++i) {
            double angle = i * 2 * M_PI / 5;
            double y = radius * cos(angle);
            double x = radius * sin(angle);
            pentagon_transitions["planet" + to_string(i + 1) + ".x"] = to_string(x);
            pentagon_transitions["planet" + to_string(i + 1) + ".y"] = to_string(y);
        }
        scene.dag->add_transitions(pentagon_transitions);
    }
    comp.inject_audio_and_render(AudioSegment(4)); // Leave some time (3s) to appreciate the simulation
    comp.inject_audio_and_render(AudioSegment(2)); // Leave some time (3s) to appreciate the simulation



    // Move the planets into arbitrarily selected spots in the half-unit sphere
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "0.3"},
        {"planet1.y", "0.4"},
        {"planet2.x", "-.3"},
        {"planet2.y", "0.3"},
        {"planet3.x", "0.2"},
        {"planet3.y", "-.2"},
        {"planet4.x", "-.4"},
        {"planet4.y", "-.1"},
        {"planet5.x", "0.1"},
        {"planet5.y", "0.2"},
    });
    comp.inject_audio_and_render(AudioSegment(4)); // Leave some time (3s) to appreciate the simulation
    comp.inject_audio_and_render(AudioSegment(2)); // Leave some time (3s) to appreciate the simulation



    // Get planets 3 4 5 far away before deleting
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet3.x", "9"},
        {"planet3.y", "9"},
        {"planet4.x", "-9"},
        {"planet4.y", "9"},
        {"planet5.x", "0"},
        {"planet5.y", "-13"}
    });



    comp.inject_audio_and_render(AudioSegment("You might also be surprised to the amount of complexity observed from merely 2 planets."));
    // Remove all but 2 planets by calling remove_fixed_object(dag_name)
    sim.remove_fixed_object("planet3");
    sim.remove_fixed_object("planet4");
    sim.remove_fixed_object("planet5");
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "0.1"},
        {"planet1.y", "0.1"},
        {"planet2.x", "-.1"},
        {"planet2.y", "-.1"},
    });
    comp.inject_audio_and_render(AudioSegment(4)); // Leave some time (3s) to appreciate the simulation
    comp.inject_audio_and_render(AudioSegment(2)); // Leave some time (3s) to appreciate the simulation
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "0.2"},
        {"planet1.y", "0.2"},
        {"planet2.x", "-.2"},
        {"planet2.y", "-.2"},
    });
    comp.inject_audio_and_render(AudioSegment(4)); // Leave some time (3s) to appreciate the simulation
    comp.inject_audio_and_render(AudioSegment(2)); // Leave some time (3s) to appreciate the simulation



    // Let's drop a field of points again just for fun.
    scene.dag->add_equations(unordered_map<string, string>{
        {"physics_multiplier", "0"},
    });
    scene.dag->add_transitions(unordered_map<string, string>{
        {"predictions_opacity", "0"},
    });
    if(FOR_REAL) for(float x = -bounds; x < bounds; x+=delta) for(float y = -bounds; y < bounds; y+=delta){
        glm::vec3 pos(x,y,0);
        int color = sim.predict_fate_of_object(pos, *(scene.dag));
        sim.add_mobile_object(pos, color, 1);
    }
    comp.inject_audio_and_render(AudioSegment(2)); // Leave some time (3s) to appreciate the simulation
    scene.dag->add_equations(unordered_map<string, string>{
        {"physics_multiplier", "10"},
    });
    comp.inject_audio_and_render(AudioSegment(6)); // Leave some time (3s) to appreciate the simulation



    // Add a third planet back
    sim.add_fixed_object(planet3_color, 1, "planet3");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet3.x", "-9.0"},
        {"planet3.y", "6.0"},
        {"planet3.z", "0"}
    });
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet3.x", "-0.4"},
        {"planet3.y", "0.3"},
        {"predictions_opacity", "1"},
        {"planet3.z", "0"}
    });
    // Set the planets to be all black with the exception of planet 1, by accessing their data in the sim.
    comp.inject_audio_and_render(AudioSegment("Now, one thing that I found super interesting is the presence of disjoint areas of the same color."));
    comp.inject_audio_and_render(AudioSegment("What would cause two distinct areas which don't contact each other to drop to the same point?"));



    scene.dag->add_transitions(unordered_map<string, string>{
        {"point_path.x", "<t> 3 / cos"},
        {"point_path.y", "<t> 3 / sin"},
        {"point_path.opacity", "1"},
    });
    comp.inject_audio_and_render(AudioSegment("We can visualize this by plotting the path that an object at a given point will fall."));
    //Disable predictions cause we are about to set drag to 0, so there is no convergence
    scene.dag->add_transitions(unordered_map<string, string>{
        {"predictions_opacity", "0"},
        {"point_path.opacity", "0"},
    });
    comp.inject_audio_and_render(AudioSegment(3));



    // Set drag to zero
    scene.dag->add_equations(unordered_map<string, string>{
        {"drag", "1"},
    });
    comp.inject_audio_and_render(AudioSegment("Now, those who know a thing or two about physics could probably tell you that, if these gravitational attractors are actually points, then in reality, you shouldn't ever actually crash."));



    // Drop an object, which since drag is now zero, should spin around forever.
    if(FOR_REAL) sim.add_mobile_object(glm::vec3(0.5, -0.5, 0), OPAQUE_WHITE, 1);
    comp.inject_audio_and_render(AudioSegment("You might fall into a stable orbit or spin around chaotically forever, but your kinetic energy wouldn't dissipate enough to cause your orbit to decay into a crash."));



    // Start increasing drag back to what it was
    DagLatexScene drag("drag_slider", latex_text("Drag"), OPAQUE_WHITE, 0, .06, VIDEO_WIDTH*.4, VIDEO_HEIGHT*.1);
    comp.add_scene(&drag, "drag_s", .05, .85, .4, .1, true); 
    comp.inject_audio_and_render(AudioSegment("And that's true! I am slightly cheating here- I am applying a constant force, a 'drag', per se, which slows the particle down."));



    scene.dag->add_transitions(unordered_map<string, string>{
        {"drag", "0.94"},
    });
    comp.inject_audio_and_render(AudioSegment(3));



    // Display the parameters at play
    scene.dag->add_equations(unordered_map<string, string>{
        {"drag", "0.98"},
    });
    scene.dag->add_transitions(unordered_map<string, string>{
        {"predictions_opacity", "1"},
    });
    comp.inject_audio_and_render(AudioSegment("Let's mess around with this parameter and see what happens...!"));



    // Increase drag
    scene.dag->add_transitions(unordered_map<string, string>{
        {"drag", "0.94"},
    });
    comp.inject_audio_and_render(AudioSegment("Check that out. Perhaps unsurprisingly, the greater the drag, the less complicated the orbital patterns are, because each particle will just slowly march to the nearest attractor."));



    // Decrease drag
    scene.dag->add_transitions(unordered_map<string, string>{
        {"drag", "0.995"},
    });
    comp.inject_audio_and_render(AudioSegment("As drag decreases, the more time the object spends winding in complicated orbits before its fate is decided, and thus, the more scrambled the colors get."));



    // Restore drag
    scene.dag->add_transitions(unordered_map<string, string>{
        {"drag", "0.98"},
    });
    comp.inject_audio_and_render(AudioSegment(3));



    // Display the parameters at play
    DagLatexScene tick_duration_scene("tick_duration", "\\Delta_t", OPAQUE_WHITE, 0, 2, VIDEO_WIDTH*.4, VIDEO_HEIGHT*.1);
    comp.add_scene(&tick_duration_scene, "tick_s", .55, .85, .4, .1, true); 
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "<t> 20 / sin .6 *"},
        {"planet1.y", "<t> 20 / cos .6 *"},
        {"planet2.x", "<t> 16 / sin .4 *"},
        {"planet2.y", "<t> 16 / cos .4 *"},
        {"planet3.x", "<t> 24 / sin .5 *"},
        {"planet3.y", "<t> 24 / cos .5 *"},
    });
    comp.inject_audio_and_render(AudioSegment("Let me show you one other parameter which affects the output of the plot."));
    scene.dag->add_transitions(unordered_map<string, string>{
        {"tick_duration", "1.3"},
    });
    comp.inject_audio_and_render(AudioSegment("This represents the resolution of time between which we recompute the forces and velocities on the falling objects."));
    scene.dag->add_transitions(unordered_map<string, string>{
        {"tick_duration", ".1"},
    });
    comp.inject_audio_and_render(AudioSegment("As it decreases, the lines between areas become perfectly flat."));
    scene.dag->add_transitions(unordered_map<string, string>{
        {"tick_duration", ".6"},
    });
    comp.inject_audio_and_render(AudioSegment("As it increases- that is, as our simulation loses precision, these ripple-like artifacts are produced."));
    scene.dag->add_transitions(unordered_map<string, string>{
        {"tick_duration", "2"},
    });
    comp.inject_audio_and_render(AudioSegment("This isn't representative of any physical property, but it was a hurdle for me to overcome while rendering these animations."));
    scene.dag->add_transitions(unordered_map<string, string>{
        {"tick_duration", ".4"},
    });
    comp.inject_audio_and_render(AudioSegment("At first I wasn't certain whether they were an artifact of imprecise computations, or actually part of the graph."));
    scene.dag->add_transitions(unordered_map<string, string>{
        {"tick_duration", ".1"},
    });
    comp.inject_audio_and_render(AudioSegment("But it seems that they are indeed a mirage!"));



    // Remove all of the fixed objects, add 3 mobile objects, and turn the mobile object interactions on for the simulator
    comp.remove_scene(&tick_duration_scene); 
    comp.remove_scene(&drag); 
    scene.dag->add_equations(unordered_map<string, string>{
        {"drag", "1"},
        {"physics_multiplier", "3"},
    });
    sim.fixed_objects.clear();
    if(FOR_REAL) {
        sim.add_mobile_object(glm::vec3(-0.5, 0, 0), cs.get_color(), 1);
        sim.add_mobile_object(glm::vec3(0.3, 0, 0), cs.get_color(), 1);
        sim.add_mobile_object(glm::vec3(0, 0.5, 0), cs.get_color(), 1);
        sim.mobile_interactions = true;
    }
    comp.inject_audio_and_render(AudioSegment("This phenomenon is reminiscent of the three-body problem, where predicting the exact trajectory of each body becomes incredibly complex."));
    comp.inject_audio_and_render(AudioSegment(3));
}

void render_3d(){
    ColorScheme cs("0079ff00dfa2f6fa70ff0060333333");
    OrbitSim sim;
    OrbitScene3D scene(&sim);

    unsigned int planet1_color = cs.get_color();
    unsigned int planet2_color = cs.get_color();
    unsigned int planet3_color = cs.get_color();
    unsigned int planet4_color = cs.get_color();
    unsigned int planet5_color = cs.get_color();
    sim.add_fixed_object(planet1_color, 1, "planet1");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet1.x", "-.3"},
        {"planet1.y", "-.3"},
        {"planet1.z", "-.3"}
    });
    sim.add_fixed_object(planet2_color, 1, "planet2");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet2.x", "0.3"},
        {"planet2.y", "0.3"},
        {"planet2.z", "-.3"}
    });
    sim.add_fixed_object(planet3_color, 1, "planet3");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet3.x", "-.3"},
        {"planet3.y", "0.3"},
        {"planet3.z", "0.3"}
    });
    sim.add_fixed_object(planet4_color, 1, "planet4");
    scene.dag->add_equations(unordered_map<string, string>{
        {"planet4.x", "0.3"},
        {"planet4.y", "-.3"},
        {"planet4.z", "0.3"}
    });
    scene.dag->add_equations(unordered_map<string, string>{
        {"tick_duration", ".1"},
        {"collision_threshold", "0.005"},
        {"drag", "0.95"},
        {"drag_slider", "1 <drag> -"},
        {"physics_multiplier", "1"},
    });
    scene.dag->add_equations(unordered_map<string, string>{
        {"q1", "<t> 4 / cos"},
        {"qi", "0"},
        {"qj", "<t> -4 / sin"},
        {"qk", "0"},
        {"d", "1.5"},
    });

    scene.inject_audio_and_render(AudioSegment(3));
    scene.dag->add_transitions(unordered_map<string, string>{
        {"planet1.x", "-.4"},
        {"planet1.y", "-.3"},
        {"planet1.z", "-.4"},
        {"planet2.x", "0.3"},
        {"planet2.y", "0.2"},
        {"planet2.z", "-.4"},
        {"planet3.x", "-.3"},
        {"planet3.y", "0.4"},
        {"planet3.z", "0.3"},
        {"planet4.x", "0.5"},
        {"planet4.y", "-.3"},
        {"planet4.z", "0.3"}
    });
    /*
    scene.dag->add_transitions(unordered_map<string, string>{
        {"drag", "0.98"},
    });
    */
    scene.inject_audio_and_render(AudioSegment(7));
    scene.inject_audio_and_render(AudioSegment(3));
}

int main() {
    Timer timer;
    FOR_REAL = false;
    render_2d();
    FOR_REAL = true;
    render_3d();
    timer.stop_timer();
    return 0;
}

