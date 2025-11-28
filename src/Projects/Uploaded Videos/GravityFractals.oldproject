using namespace std;
#include <string>
const string project_name = "GravityFractals";
#include "../io/PathManager.cpp"
const int width_base = 640;
const int height_base = 360;
const int mult = 3;

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;

#include "../io/writer.cpp"

#include "../Scenes/Physics/OrbitScene2D.cpp"
#include "../Scenes/Physics/OrbitScene3D.cpp"
#include "../Scenes/Media/StateSliderScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
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
    scene.state.set(unordered_map<string, string>{
        {"planet1.opacity", "1"},
        {"planet2.opacity", "1"},
        {"planet3.opacity", "1"},
        {"planet4.opacity", "1"},
        {"planet5.opacity", "1"},
    });
    sim.add_fixed_object(planet1_color, "planet1");
    scene.state.set(unordered_map<string, string>{
        {"planet1.x", "-.3"},
        {"planet1.y", "-.3"},
        {"planet1.z", "0"},
    });
    sim.add_fixed_object(planet2_color, "planet2");
    scene.state.set(unordered_map<string, string>{
        {"planet2.x", "0.3171"},
        {"planet2.y", "0.35"},
        {"planet2.z", "0"},
    });
    sim.add_fixed_object(planet3_color, "planet3");
    scene.state.set(unordered_map<string, string>{
        {"planet3.x", "-.4"},
        {"planet3.y", "0.3"},
        {"planet3.z", "0"},
    });

    scene.state.set(unordered_map<string, string>{
        {"tick_duration", ".05"},
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
        {"physics_multiplier", "3"},
    });



    // Drop an object
    if(FOR_REAL) sim.add_mobile_object(glm::dvec3(.6, -.4, 0), OPAQUE_WHITE);
    comp.stage_macroblock_and_render(AudioSegment("Which planet is this falling particle going to crash into?"));
    scene.state.transition(unordered_map<string, string>{
        {"physics_multiplier", "6"},
        {"screen_center_x", "0"},
        {"screen_center_y", "0"},
        {"zoom", "0.5"},
    });



    // Do nothing, let the object keep falling
    comp.stage_macroblock_and_render(AudioSegment("It's not an easy question."));
    scene.state.transition(unordered_map<string, string>{
        {"physics_multiplier", "10"},
    });
    comp.stage_macroblock_and_render(AudioSegment("If you said the green one, you're right!"));
    scene.state.set(unordered_map<string, string>{
        {"point_path.x", ".6"},
        {"point_path.y", "-.4"},
    });
    scene.state.transition(unordered_map<string, string>{
        {"point_path.opacity", "1"},
    });
    comp.stage_macroblock_and_render(AudioSegment("But, it's hard to foresee this trajectory."));
    scene.state.transition(unordered_map<string, string>{
        {"point_path.x", "0.6 {t} 2 / sin 10 / +"},
        {"point_path.y", "-.4 {t} 2 / cos 10 / +"},
    });
    comp.stage_macroblock_and_render(AudioSegment("Moving the starting point only a little bit causes a huge change in behavior."));
    scene.state.transition(unordered_map<string, string>{
        {"point_path.opacity", "0"},
    });
    comp.stage_macroblock_and_render(AudioSegment(3));



    scene.state.set(unordered_map<string, string>{
        {"physics_multiplier", "0"},
        {"physics_multiplier", "1"},
    });
    // Make a blob of points around each planet which will fall in, and color them the same color as the planet which they will fall into.
    if(FOR_REAL)for (int i = 0; i < 500; i++) {
        float theta  = (rand() / double(RAND_MAX)) * 6.283;
        float radius = (rand() / double(RAND_MAX)) * 0.12;
        float dx = radius * cos(theta);
        float dy = radius * sin(theta);
        sim.add_mobile_object(glm::dvec3((*scene.state["planet1.x"] + dx, (*scene.state["planet1.y"] + dy, 0), planet1_color);
        sim.add_mobile_object(glm::dvec3((*scene.state["planet2.x"] + dx, (*scene.state["planet2.y"] + dy, 0), planet2_color);
        sim.add_mobile_object(glm::dvec3((*scene.state["planet3.x"] + dx, (*scene.state["planet3.y"] + dy, 0), planet3_color);
    }
    comp.stage_macroblock_and_render(AudioSegment("We know for sure that the ones which start sufficiently close to each planet will indeed crash into it,"));
    scene.state.transition(unordered_map<string, string>{
        {"physics_multiplier", "10"},
    });
    comp.stage_macroblock_and_render(AudioSegment("due to a lack of energy to escape that planet's gravity well."));



    scene.state.set(unordered_map<string, string>{
        {"physics_multiplier", "20"},
    });
    comp.stage_macroblock_and_render(AudioSegment("But that reasoning only goes so far."));
    if(FOR_REAL)for (int i = 0; i < 500; i++) {
        float theta  = (rand() / double(RAND_MAX)) * 6.283;
        float radius = (rand() / double(RAND_MAX)) * 0.12;
        float dx = 0.6 + radius * cos(theta);
        float dy = -.4 + radius * sin(theta);
        glm::dvec3 pos(dx, dy, 0);
        int color = sim.predict_fate_of_object(pos, *(scene.state);
        sim.add_mobile_object(pos, color);
    }
    comp.stage_macroblock_and_render(AudioSegment("It's a mess when we look at points far away from the planets."));
    comp.stage_macroblock_and_render(AudioSegment(3));
    comp.stage_macroblock_and_render(AudioSegment("Now, I won't lead you on- this is not a problem with a clean solution. There's no simple equation here to tell you."));
    scene.state.set(unordered_map<string, string>{
        {"physics_multiplier", "10"},
    });






    // Turn on the predictions
    scene.state.transition(unordered_map<string, string>{
        {"predictions_opacity", "1"},
    });
    comp.stage_macroblock_and_render(AudioSegment("I'll just go ahead and spoil the pattern."));
    comp.stage_macroblock_and_render(AudioSegment("This plot shows, if an object is dropped at each point, the planet which it would crash into."));



    // Pan around by setting screen_center_x/y to some moving formula, and slightly increase zoom.
    scene.state.transition(unordered_map<string, string>{
        {"screen_center_x", "{t} 5 / cos"},
        {"screen_center_y", "{t} 5 / sin"},
        {"zoom", "1.5"}
    });
    comp.stage_macroblock_and_render(AudioSegment("There's a lot of emergent complexity going on here."));
    scene.state.transition(unordered_map<string, string>{
        {"zoom", "2"}
    });
    comp.stage_macroblock_and_render(AudioSegment(6));
    comp.stage_macroblock_and_render(AudioSegment("But, it seems like the complexity is finite... more about that later."));



    // Zoom out and restore the position
    // Drop a field of points
    float delta = 0.03;
    int bounds = 1;
    scene.state.set(unordered_map<string, string>{
        {"physics_multiplier", "0"},
    });
    if(FOR_REAL) for(float x = -bounds; x < bounds; x+=delta) for(float y = -bounds; y < bounds; y+=delta){
        glm::dvec3 pos(x,y,0);
        int color = sim.predict_fate_of_object(pos, *(scene.state);
        sim.add_mobile_object(pos, color);
    }
    scene.state.transition(unordered_map<string, string>{
        {"screen_center_x", "0"},
        {"screen_center_y", "0"},
        {"predictions_opacity", "0"},
        {"zoom", "0.5"}
    });
    comp.stage_macroblock_and_render(AudioSegment("Let's watch an object at every point fall."));
    scene.state.set(unordered_map<string, string>{
        {"physics_multiplier", "10"},
    });



    comp.stage_macroblock_and_render(AudioSegment(3)); // Leave some time (3s) to appreciate the simulation
    comp.stage_macroblock_and_render(AudioSegment("Wow."));
    scene.state.set(unordered_map<string, string>{
        {"physics_multiplier", "20"},
    });
    comp.stage_macroblock_and_render(AudioSegment(3)); // Leave some time (3s) to appreciate the simulation
    scene.state.set(unordered_map<string, string>{
        {"physics_multiplier", "10"},
    });



    // Move one of the planets
    scene.state.transition(unordered_map<string, string>{
        {"predictions_opacity", "1"},
        {"planet2.x", "0.3"},
        {"planet2.y", "-.3"},
        {"planet2.z", "0"},
    });
    comp.stage_macroblock_and_render(AudioSegment("We can also move around the planets and see how the plot changes."));



    // Transition the planets to an equilateral triangle
    scene.state.transition(unordered_map<string, string>{
        {"planet1.x", "0"},
        {"planet1.y", "-0.57735 .75 *"},
        {"planet2.x", "0.5 .75 *"},
        {"planet2.y", "0.288675 .75 *"},
        {"planet3.x", "-0.5 .75 *"},
        {"planet3.y", "0.288675 .75 *"}
    });
    comp.stage_macroblock_and_render(AudioSegment("If their shape is symmetric, so is the plot."));



    // Transition the planets to an isosceles triangle, with a vertical axis of symmetry.
    scene.state.transition(unordered_map<string, string>{
        {"planet1.x", "0"},
        {"planet1.y", "-0.3"},
        {"planet2.x", "0.3"},
        {"planet2.y", "0.2"},
        {"planet3.x", "-0.3"},
        {"planet3.y", "0.2"}
    });
    comp.stage_macroblock_and_render(AudioSegment("We can play around with some different shapes and watch what happens to the design."));



    // Transition the planets to a collinear configuration, with one planet at (0,0)
    scene.state.transition(unordered_map<string, string>{
        {"planet1.x", "-0.5"},
        {"planet1.y", "0"},
        {"planet2.x", "0.5"},
        {"planet2.y", "0"},
        {"planet3.x", "0"},
        {"planet3.y", "0"}
    });
    comp.stage_macroblock_and_render(AudioSegment(3));



    // Move one planet far away from the other two (to about (0.8, 0))
    scene.state.transition(unordered_map<string, string>{
        {"planet1.x", "-0.6"},
        {"planet1.y", "0.5"},
        {"planet2.x", "0.3"},
        {"planet2.y", "0.2"},
        {"planet3.x", "0.2"},
        {"planet3.y", "0.3"}
    });
    comp.stage_macroblock_and_render(AudioSegment(3));



    // Move the planets back to where they were
    scene.state.transition(unordered_map<string, string>{
        {"planet1.x", "-.43"},
        {"planet1.y", "-.4"},
        {"planet2.x", "0.4171"},
        {"planet2.y", "0.45"},
        {"planet3.x", "-.5"},
        {"planet3.y", "0.4"}
    });



    comp.stage_macroblock_and_render(AudioSegment("Let's drop a fourth one in there too and see what happens."));
    // Add fourth planet far in the distance, and using the transition mechanics, slide it in from far away.
    sim.add_fixed_object(planet4_color, "planet4");
    scene.state.set(unordered_map<string, string>{
        {"planet4.x", "9.0"},
        {"planet4.y", "5.0"},
        {"planet4.z", "0"},
    });
    scene.state.transition(unordered_map<string, string>{
        {"planet4.x", "0.5"},
        {"planet4.y", "-.5"}
    });
    comp.stage_macroblock_and_render(AudioSegment(3)); // Leave some time (3s) to appreciate the simulation



    comp.stage_macroblock_and_render(AudioSegment("And here's a fifth."));
    // Add fifth planet far in the distance, but using the transition mechanics, slide it in from far away.
    sim.add_fixed_object(planet5_color, "planet5");
    scene.state.set(unordered_map<string, string>{
        {"planet5.x", "-5.0"},
        {"planet5.y", "-8.0"},
        {"planet5.z", "0"},
    });
    scene.state.transition(unordered_map<string, string>{
        {"planet5.x", "0"},
        {"planet5.y", "0"}
    });
    comp.stage_macroblock_and_render(AudioSegment(3)); // Leave some time (3s) to appreciate the simulation



    // Move the planets around a little bit
    scene.state.transition(unordered_map<string, string>{
        {"planet1.x", "-.5"},
        {"planet1.y", "-.5"},
        {"planet2.x", "0.5"},
        {"planet2.y", "-.5"},
        {"planet3.x", "0.5"},
        {"planet3.y", "0.5"},
        {"planet4.x", "-.5"},
        {"planet4.y", "0.5"},
    });
    comp.stage_macroblock_and_render(AudioSegment(5)); // Leave some time (3s) to appreciate the simulation
    comp.stage_macroblock_and_render(AudioSegment(1)); // Leave some time (3s) to appreciate the simulation
    scene.state.transition(unordered_map<string, string>{
        {"planet1.x", "-.3"},
        {"planet1.y", "-.5"},
        {"planet2.x", "0.8"},
        {"planet2.y", "-.5"},
        {"planet3.x", "0.3"},
        {"planet3.y", "0.5"},
        {"planet4.x", "-.8"},
        {"planet4.y", "0.5"},
    });
    comp.stage_macroblock_and_render(AudioSegment(5)); // Leave some time (3s) to appreciate the simulation
    comp.stage_macroblock_and_render(AudioSegment(1)); // Leave some time (3s) to appreciate the simulation



    // TODO: Spend some more time exploring various configurations of 4 objects



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
        scene.state.transition(pentagon_transitions);
    }
    comp.stage_macroblock_and_render(AudioSegment(4)); // Leave some time (3s) to appreciate the simulation
    comp.stage_macroblock_and_render(AudioSegment(2)); // Leave some time (3s) to appreciate the simulation



    // TODO: Drop some points in the 5-object case



    // Get planets 3 4 5 far away before deleting
    scene.state.transition(unordered_map<string, string>{
        {"planet3.x", "9"},
        {"planet3.y", "9"},
        {"planet4.x", "-9"},
        {"planet4.y", "9"},
        {"planet5.x", "0"},
        {"planet5.y", "-13"}
    });



    comp.stage_macroblock_and_render(AudioSegment("You might also be surprised to the amount of complexity observed from merely 2 planets."));
    // Remove all but 2 planets
    sim.remove_fixed_object("planet3");
    sim.remove_fixed_object("planet4");
    sim.remove_fixed_object("planet5");
    scene.state.transition(unordered_map<string, string>{
        {"planet1.x", "0.1"},
        {"planet1.y", "0.1"},
        {"planet2.x", "-.1"},
        {"planet2.y", "-.1"},
    });
    comp.stage_macroblock_and_render(AudioSegment(4)); // Leave some time (3s) to appreciate the simulation
    comp.stage_macroblock_and_render(AudioSegment(2)); // Leave some time (3s) to appreciate the simulation
    scene.state.transition(unordered_map<string, string>{
        {"planet1.x", "0.3"},
        {"planet1.y", "0.3"},
        {"planet2.x", "-.3"},
        {"planet2.y", "-.3"},
    });
    comp.stage_macroblock_and_render(AudioSegment(4)); // Leave some time (3s) to appreciate the simulation
    comp.stage_macroblock_and_render(AudioSegment(2)); // Leave some time (3s) to appreciate the simulation



    // Let's drop a field of points again just for fun.
    scene.state.set(unordered_map<string, string>{
        {"physics_multiplier", "0"},
    });
    scene.state.transition(unordered_map<string, string>{
        {"predictions_opacity", "0"},
    });
    if(FOR_REAL) for(float x = -bounds; x < bounds; x+=delta) for(float y = -bounds; y < bounds; y+=delta){
        glm::dvec3 pos(x,y,0);
        int color = sim.predict_fate_of_object(pos, *(scene.state);
        sim.add_mobile_object(pos, color);
    }
    comp.stage_macroblock_and_render(AudioSegment(2)); // Leave some time (3s) to appreciate the simulation
    scene.state.set(unordered_map<string, string>{
        {"physics_multiplier", "10"},
    });
    comp.stage_macroblock_and_render(AudioSegment(6)); // Leave some time (3s) to appreciate the simulation



    // Add a third planet back
    sim.add_fixed_object(planet3_color, "planet3");
    scene.state.set(unordered_map<string, string>{
        {"planet3.x", "-9.0"},
        {"planet3.y", "6.0"},
        {"planet3.z", "0"}
    });
    scene.state.transition(unordered_map<string, string>{
        {"planet3.x", "-0.4"},
        {"planet3.y", "0.3"},
        {"predictions_opacity", "1"},
        {"planet3.z", "0"}
    });
    // Set the planets to be all black with the exception of planet 1, by accessing their data in the sim.
    comp.stage_macroblock_and_render(AudioSegment("Now, one thing that I found super interesting is the presence of disjoint areas of the same color."));
    comp.stage_macroblock_and_render(AudioSegment("What would cause two distinct areas which don't contact each other to drop to the same point?"));



    scene.state.set(unordered_map<string, string>{
        {"point_path.x", "-.27 {t} cos 4 / +"},
        {"point_path.y", "-.24 {t} sin 5 / +"},
    });
    scene.state.transition(unordered_map<string, string>{
        {"point_path.opacity", "1"},
    });
    comp.stage_macroblock_and_render(AudioSegment("Well, let's look around and see!"));
    comp.stage_macroblock_and_render(AudioSegment(3));
    scene.state.transition(unordered_map<string, string>{
        {"point_path.x", "-.285 {t} cos 40 / +"},
        {"point_path.y", "-.8 {t} sin 10 / +"},
    });
    comp.stage_macroblock_and_render(AudioSegment("The answer seems to be that a different region is associated with a different path of winding around the planets to get to the same spot."));
    comp.stage_macroblock_and_render(AudioSegment(3));
    scene.state.transition(unordered_map<string, string>{
        {"point_path.x", "-.49 {t} cos 40 / +"},
        {"point_path.y", "-.88 {t} cos 10 / +"},
    });
    comp.stage_macroblock_and_render(AudioSegment("But I wasn't really able to formalize that idea- leave a comment if you know of a way."));
    comp.stage_macroblock_and_render(AudioSegment(3));
    //Disable predictions cause we are about to set drag to 0, so there is no convergence
    scene.state.transition(unordered_map<string, string>{
        {"predictions_opacity", "0"},
        {"point_path.opacity", "0"},
    });
    comp.stage_macroblock_and_render(AudioSegment(3));



    // Set drag to zero
    scene.state.set(unordered_map<string, string>{
        {"drag", "1"},
    });
    comp.stage_macroblock_and_render(AudioSegment("Now, those who know a thing or two about physics could probably tell you that, if these gravitational attractors are actually points, then in reality, you shouldn't ever actually crash."));



    // Drop an object, which since drag is now zero, should spin around forever.
    if(FOR_REAL) sim.add_mobile_object(glm::dvec3(0.5, -0.5, 0), OPAQUE_WHITE);
    comp.stage_macroblock_and_render(AudioSegment("You might fall into a stable orbit or spin around chaotically forever, but your kinetic energy wouldn't dissipate enough to cause your orbit to decay into a crash."));



    // Start increasing drag back to what it was
    StateSliderScene drag("drag_slider", latex_text("Drag"), OPAQUE_WHITE, 0, .1, VIDEO_WIDTH*.4, VIDEO_HEIGHT*.1);
    comp.add_scene(&drag, "drag_s", .05, .85, .4, .1, true); 
    comp.stage_macroblock_and_render(AudioSegment("And that's true! I am slightly cheating here- I am applying a constant force, a 'drag', per se, which slows the particle down."));



    scene.state.transition(unordered_map<string, string>{
        {"drag", "0.94"},
    });
    comp.stage_macroblock_and_render(AudioSegment(3));
    scene.state.transition(unordered_map<string, string>{
        {"drag", "0.98"},
    });
    comp.stage_macroblock_and_render(AudioSegment(1));



    // Display the parameters at play
    scene.state.transition(unordered_map<string, string>{
        {"predictions_opacity", "1"},
    });
    comp.stage_macroblock_and_render(AudioSegment("Let's mess around with this parameter and see what happens...!"));



    // Increase drag
    scene.state.transition(unordered_map<string, string>{
        {"drag", "0.94"},
    });
    comp.stage_macroblock_and_render(AudioSegment("Check that out. Perhaps unsurprisingly, the greater the drag, the less complicated the orbital patterns are, because each particle will just slowly march to the nearest attractor."));



    // Increase drag
    scene.state.transition(unordered_map<string, string>{
        {"drag", "0.91"},
    });
    comp.stage_macroblock_and_render(AudioSegment(3));



    // Decrease drag
    scene.state.transition(unordered_map<string, string>{
        {"drag", "0.9925"},
    });
    comp.stage_macroblock_and_render(AudioSegment("As drag decreases, the more time the object spends winding in complicated orbits before its fate is decided, and thus, the more scrambled the colors get."));



    // Restore drag
    scene.state.transition(unordered_map<string, string>{
        {"drag", "0.98"},
    });
    comp.stage_macroblock_and_render(AudioSegment(3));



    // Display the parameters at play
    StateSliderScene tick_duration_scene("tick_duration", "\\Delta_t", OPAQUE_WHITE, 0, 1.5, VIDEO_WIDTH*.4, VIDEO_HEIGHT*.1);
    comp.add_scene(&tick_duration_scene, "tick_s", .55, .85, .4, .1, true); 
    scene.state.transition(unordered_map<string, string>{
        {"planet1.x", "{t} 20 / sin .6 *"},
        {"planet1.y", "{t} 20 / cos .6 *"},
        {"planet2.x", "{t} 16 / sin .4 *"},
        {"planet2.y", "{t} 16 / cos .4 *"},
        {"planet3.x", "{t} 24 / sin .5 *"},
        {"planet3.y", "{t} 24 / cos .5 *"},
    });
    comp.stage_macroblock_and_render(AudioSegment("Let me show you one other parameter which affects the output of the plot."));
    scene.state.transition(unordered_map<string, string>{
        {"tick_duration", "1.3"},
    });
    comp.stage_macroblock_and_render(AudioSegment("This represents the resolution of time between which we recompute the forces and velocities on the falling objects."));
    scene.state.transition(unordered_map<string, string>{
        {"tick_duration", ".1"},
    });
    comp.stage_macroblock_and_render(AudioSegment("As it decreases, the lines between areas become perfectly flat."));
    scene.state.transition(unordered_map<string, string>{
        {"tick_duration", ".6"},
    });
    comp.stage_macroblock_and_render(AudioSegment("As it increases- that is, as our simulation loses precision, these ripple-like artifacts are produced."));
    scene.state.transition(unordered_map<string, string>{
        {"tick_duration", "1.5"},
    });
    comp.stage_macroblock_and_render(AudioSegment("This isn't representative of any physical property, but it was a hurdle for me to overcome while rendering these animations."));
    scene.state.transition(unordered_map<string, string>{
        {"tick_duration", ".05"},
    });
    comp.stage_macroblock_and_render(AudioSegment("At first I wasn't certain whether they were an artifact of imprecise computations, or actually part of the diagram."));
    scene.state.transition(unordered_map<string, string>{
        {"predictions_opacity", "0"},
    });
    comp.stage_macroblock_and_render(AudioSegment("But it seems like they're indeed a mirage!"));



    // Remove all of the fixed objects, add 3 mobile objects, and turn the mobile object interactions on for the simulator
    comp.remove_scene(&tick_duration_scene); 
    comp.remove_scene(&drag); 
    scene.state.set(unordered_map<string, string>{
        {"drag", "1"},
        {"physics_multiplier", "6"},
    });
    sim.fixed_objects.clear();
    if(FOR_REAL) {
        sim.add_mobile_object(glm::dvec3(-0.5, 0, 0), cs.get_color());
        sim.add_mobile_object(glm::dvec3(0.3, 0, 0), cs.get_color());
        sim.add_mobile_object(glm::dvec3(0, 0.5, 0), cs.get_color());
        sim.mobile_interactions = true;
    }
    comp.stage_macroblock_and_render(AudioSegment("This phenomenon is reminiscent of the three-body problem, where predicting the exact trajectory of each body becomes incredibly complex."));
    comp.stage_macroblock_and_render(AudioSegment(3));
}

void render_3d() {
    ColorScheme cs("0079ff00dfa2f6fa70ff0060333333ff8c008a2be27fff00ff69b4");
    OrbitSim sim;
    OrbitScene3D os3d(&sim);
    CompositeScene comp;
    comp.add_scene(&os3d, "orbit3d_s", 0, 0, 1, 1, true);

    unsigned int planet1_color = cs.get_color();
    unsigned int planet2_color = cs.get_color();
    unsigned int planet3_color = cs.get_color();
    unsigned int planet4_color = cs.get_color();
    unsigned int planet5_color = cs.get_color();
    unsigned int planet6_color = cs.get_color();
    unsigned int planet7_color = cs.get_color();
    unsigned int planet8_color = cs.get_color();
    unsigned int planet9_color = cs.get_color();

    comp.state.set(unordered_map<string, string>{
        {"planet1.opacity", "0"},
        {"planet2.opacity", "0"},
        {"planet3.opacity", "0"},
        {"planet4.opacity", "0"},
        {"planet5.opacity", "0"},
        {"planet6.opacity", "0"},
        {"planet7.opacity", "0"},
        {"planet8.opacity", "0"},
        {"planet9.opacity", "0"},
        {"nonconverge.opacity", "0"},
        {"boundingbox.opacity", "0"},
    });

    sim.add_fixed_object(planet1_color, "planet1");
    sim.add_fixed_object(planet2_color, "planet2");
    sim.add_fixed_object(planet3_color, "planet3");
    sim.add_fixed_object(planet4_color, "planet4");

    comp.state.set(unordered_map<string, string>{
        {"planet1.x", "-.3"}, {"planet1.y", "-.3"}, {"planet1.z", "-.3"},
        {"planet2.x", "0.3"}, {"planet2.y", "0.3"}, {"planet2.z", "-.3"},
        {"planet3.x", "-.3"}, {"planet3.y", "0.3"}, {"planet3.z", "0.3"},
        {"planet4.x", "0.3"}, {"planet4.y", "-.3"}, {"planet4.z", "0.3"}
    });

    comp.state.set(unordered_map<string, string>{
        {"fov", ".5"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"surfaces_opacity", "1"},
        {"lines_opacity", "1"},
        {"points_opacity", "1"}
    });

    comp.state.set(unordered_map<string, string>{
        {"tick_duration", ".1"},
        {"collision_threshold", "0.005"},
        {"drag", "0.95"},
        {"drag_slider", "1 <drag> -"},
        {"physics_multiplier", "1"},
        {"wireframe_width" , "100"},
        {"wireframe_height", "100"},
        {"wireframe_depth" , "100"},
        {"zoom", "0.2 <wireframe_width> *"},
    });

    comp.state.set(unordered_map<string, string>{
        {"q1", "{t} 20 / cos"},
        {"qi", ".5"},
        {"qj", "{t} 20 / sin"},
        {"qk", "0"},
        {"d", "6"},
    });

    comp.stage_macroblock_and_render(AudioSegment("We can try the same thing in 3D too! I've picked 4 tetrahedrally arranged points."));
    comp.state.transition(unordered_map<string, string>{
        {"planet1.opacity", "0.5"},
        {"boundingbox.opacity", "1"},
    });

    comp.stage_macroblock_and_render(AudioSegment("This shape in blue is the boundary of the area which converges towards the blue planet."));
    comp.state.transition(unordered_map<string, string>{
        {"planet1.opacity", "0.5"},
        {"planet2.opacity", ".03"},
        {"planet3.opacity", ".03"},
        {"planet4.opacity", ".03"},
        {"planet5.opacity", ".03"},
        {"planet6.opacity", ".03"},
        {"planet7.opacity", ".03"},
        {"planet8.opacity", ".03"},
        {"nonconverge.opacity", ".03"},
    });

    comp.stage_macroblock_and_render(AudioSegment("Let's move the planets around a little and see what happens."));
    comp.state.transition(unordered_map<string, string>{
        {"planet1.x", "-.6"},
        {"planet1.y", "-.3"},
        {"planet1.z", "-.4"},
        {"planet2.x", "0.3"},
        {"planet2.y", "0.2"},
        {"planet2.z", "-.4"},
        {"planet3.x", "-.3"},
        {"planet3.y", "0.4"},
        {"planet3.z", "0.3"},
        {"planet4.x", "0.5"},
        {"planet4.y", "-.6"},
        {"planet4.z", "0.3"}
    });

    comp.stage_macroblock_and_render(AudioSegment(3));
    StateSliderScene drag("drag_slider", latex_text("Drag"), OPAQUE_WHITE, 0, 0.08, VIDEO_WIDTH*.4, VIDEO_HEIGHT*.1);
    comp.add_scene(&drag, "drag_s", .05, .85, .4, .1, true);

    comp.stage_macroblock_and_render(AudioSegment("Let's try playing with drag too!"));
    comp.state.transition(unordered_map<string, string>{
        {"drag", "0.99"},
    });

    comp.stage_macroblock_and_render(AudioSegment(3));
    comp.state.transition(unordered_map<string, string>{
        {"drag", "0.93"},
    });

    comp.stage_macroblock_and_render(AudioSegment(3));
    comp.state.transition(unordered_map<string, string>{
        {"drag", "0.97"},
    });

    comp.stage_macroblock_and_render(AudioSegment(3));
    comp.state.transition(unordered_map<string, string>{
        {"planet3.x", "11"},
        {"planet4.x", "-11"},
    });
    comp.stage_macroblock_and_render(AudioSegment("Now, let's simplify to just two planets."));

    // Demonstrate a 2-planet configuration, and comment on its radial symmetry
    sim.remove_fixed_object("planet3");
    sim.remove_fixed_object("planet4");
    comp.stage_macroblock_and_render(AudioSegment(3));
    comp.state.transition(unordered_map<string, string>{
        {"planet1.x", "0"}, {"planet1.y", "0"}, {"planet1.z", "0.45"},
        {"planet2.x", "0"}, {"planet2.y", "0"}, {"planet2.z", "-.45"},
    });

    comp.state.transition(unordered_map<string, string>{
        {"planet2.opacity", "0.5"},
    });

    comp.stage_macroblock_and_render(AudioSegment("Check out the radial symmetry!"));

    // Subscene 3: Increase drag
    comp.state.transition(unordered_map<string, string>{
        {"drag", "0.96"},
    });

    comp.stage_macroblock_and_render(AudioSegment(3));

    // Demonstrate a 3-planet configuration, and comment on its mirror-plane symmetry
    sim.add_fixed_object(planet3_color, "planet3");
    comp.state.set(unordered_map<string, string>{
        {"planet3.x", "9"},
        {"planet3.y", "0"},
        {"planet3.z", "0"}
    });
    comp.stage_macroblock_and_render(AudioSegment("With 3 planets you get this nice mirror-plane symmetry about the plane all 3 points lie on."));

    comp.state.transition(unordered_map<string, string>{
        {"planet3.opacity", "0.5"},
    });

    comp.stage_macroblock_and_render(AudioSegment(3));

    // Subscene 1: Initial positions
    comp.state.transition(unordered_map<string, string>{
        {"planet3.x", "0.3"},
        {"planet3.y", "0"},
        {"planet3.z", "0"},
    });

    comp.stage_macroblock_and_render(AudioSegment(3));

    // Subscene 3: Increase drag
    comp.state.transition(unordered_map<string, string>{
        {"drag", "0.98"},
    });

    comp.stage_macroblock_and_render(AudioSegment(3));

    // Introduce a configuration with 5 arbitrarily arranged planets
    comp.stage_macroblock_and_render(AudioSegment("Here's 5 planets."));
    sim.add_fixed_object(planet4_color, "planet4");
    sim.add_fixed_object(planet5_color, "planet5");
    comp.state.set(unordered_map<string, string>{
        {"planet4.x", "-0.2"},
        {"planet4.y", "13"},
        {"planet4.z", "-0.2"},
        {"planet5.x", "1"},
        {"planet5.y", "-13"},
        {"planet5.z", "-0.2"}
    });

    comp.state.transition(unordered_map<string, string>{
        {"planet2.opacity", "0.03"},
        {"planet3.opacity", "0.03"},
    });

    comp.stage_macroblock_and_render(AudioSegment(5));
    comp.state.transition(unordered_map<string, string>{
        {"planet1.x", "-.4"}, {"planet1.y", "-.4"}, {"planet1.z", "-.4"},
        {"planet2.x", "0.4"}, {"planet2.y", "0.4"}, {"planet2.z", "0.4"},
        {"planet3.x", "0"  }, {"planet3.y", "0.4"}, {"planet3.z", "0"  },
        {"planet4.x", "-.4"}, {"planet4.y", "0.4"}, {"planet4.z", "-.4"},
        {"planet5.x", "-.4"}, {"planet5.y", "0.4"}, {"planet5.z", "0.4"},
    });

    comp.stage_macroblock_and_render(AudioSegment(3));

    // Subscene 3: Increase drag
    comp.state.transition(unordered_map<string, string>{
        {"drag", "0.96"},
    });

    comp.stage_macroblock_and_render(AudioSegment(3));

    // Demonstrate a configuration with 8 planets arranged in a cube
    sim.add_fixed_object(planet6_color, "planet6");
    sim.add_fixed_object(planet7_color, "planet7");
    sim.add_fixed_object(planet8_color, "planet8");
    comp.state.set(unordered_map<string, string>{
        {"planet6.x", "-0.2"}, {"planet6.y", "13"}, {"planet6.z", "-0.2"},
        {"planet7.x", "-0.2"}, {"planet7.y", "-13"}, {"planet7.z", "-0.2"},
        {"planet8.x", "-22"}, {"planet8.y", "3"}, {"planet8.z", "-0.2"},
    });

    // Subscene 1: Initial positions
    comp.state.transition(unordered_map<string, string>{
        {"planet1.x", "-.4"}, {"planet1.y", "-.4"}, {"planet1.z", "-.4"},
        {"planet2.x", "-.4"}, {"planet2.y", "-.4"}, {"planet2.z", "0.4"},
        {"planet3.x", "-.4"}, {"planet3.y", "0.4"}, {"planet3.z", "-.4"},
        {"planet4.x", "-.4"}, {"planet4.y", "0.4"}, {"planet4.z", "0.4"},
        {"planet5.x", "0.4"}, {"planet5.y", "-.4"}, {"planet5.z", "-.4"},
        {"planet6.x", "0.4"}, {"planet6.y", "-.4"}, {"planet6.z", "0.4"},
        {"planet7.x", "0.4"}, {"planet7.y", "0.4"}, {"planet7.z", "-.4"},
        {"planet8.x", "0.4"}, {"planet8.y", "0.4"}, {"planet8.z", "0.4"},
    });


    comp.stage_macroblock_and_render(AudioSegment("And here's 8."));
    comp.stage_macroblock_and_render(AudioSegment(3));

    // Subscene 2: Rotate positions
    comp.state.transition(unordered_map<string, string>{
        {"planet2.x", "-.4"}, {"planet2.y", "-.4"}, {"planet2.z", "-.4"},
        {"planet4.x", "-.4"}, {"planet4.y", "-.4"}, {"planet4.z", "0.4"},
        {"planet1.x", "-.4"}, {"planet1.y", "0.4"}, {"planet1.z", "-.4"},
        {"planet3.x", "-.4"}, {"planet3.y", "0.4"}, {"planet3.z", "0.4"},
    });

    comp.stage_macroblock_and_render(AudioSegment(3));

    // Subscene 3: Increase drag
    comp.state.transition(unordered_map<string, string>{
        {"drag", "0.94"},
    });

    comp.stage_macroblock_and_render(AudioSegment(3));
    comp.state.transition(unordered_map<string, string>{
        {"lines_opacity", "0"},
        {"points_opacity", "0"},
    });
    comp.stage_macroblock_and_render(AudioSegment(3));
}

void render_credits() {
    TwoswapScene ts(VIDEO_WIDTH * .5, VIDEO_HEIGHT * .5);
    LatexScene ls(latex_text("Music by 6884") + "\\\\\\\\" + latex_text("\\tiny Links in description!"), 1, VIDEO_WIDTH * .5, VIDEO_HEIGHT * .5);
    CompositeScene cs;
    cs.add_scene(&ts, "t_s", -10, -10, .5, .5, true);
    cs.add_scene(&ls, "l_s",  10,  10, .5, .5, true);
    cs.state.set(unordered_map<string, string>{
        {"t_s.x", "0.125 8 0.2 <transition_fraction> - 6 * ^ -"},
        {"t_s.y", "0.125"},
        {"l_s.x", "0.375 8 0.6 <transition_fraction> - 6 * ^ +"},
        {"l_s.y", "0.375"},
    });
    cs.stage_macroblock_and_render(AudioSegment("This has been 2swap, with music from 6884!"));
    cs.state.set(unordered_map<string, string>{
        {"t_s.x", "0.125"},
        {"t_s.y", "0.125"},
        {"l_s.x", "0.375"},
        {"l_s.y", "0.375"},
    });
    cs.stage_macroblock_and_render(AudioSegment(2));
}

int main() {
    Timer timer;
    render_2d();
    render_3d();
    render_credits();
    timer.stop_timer();
    return 0;
}

