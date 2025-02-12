#include "../Scenes/Common/ThreeDimensionScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Media/StateSliderScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/CoordinateSceneWithTrail.cpp"
#include "../Scenes/Common/CoordinateScene.cpp"
#include "../Scenes/Physics/PendulumScene.cpp"
#include "../Scenes/Physics/PendulumGridScene.cpp"
#include "../Scenes/Physics/MovingPendulumGridScene.cpp"
#include "../Scenes/Physics/PendulumPointsScene.cpp"

struct IslandShowcase {
    PendulumState ps;
    double range;
    double fingerprint_zoom;
    string name;
};

vector<IslandShowcase> isv{{{2.49, .25     , .0, .0}, 0.4 , .02, "The Pretzel"},
                           {{2.658, -2.19  , .0, .0}, 0.2 , .02, "The Shoelace"},
                           {{2.453, -2.7727, .0, .0}, 0.05, .02, "The Heart"},
                           {{1.351, 2.979  , .0, .0}, 0.2 , .06, "The Bird"}};

IslandShowcase momentum_island = {{2.14, 0.8  , .4, .6}, 0.2, 0.04, "Island with a starting momentum"};

void showcase_an_island(PendulumGridScene& pgs, const IslandShowcase& is, const string& script) {
    const double range = is.range;
    const double cx = is.ps.theta1;
    const double cy = is.ps.theta2;
    pgs.state_manager.set({
        {"physics_multiplier", "0"},
        {"ticks_opacity", "0"},
        {"rk4_step_size", "0"},
        {"zoom", "2.718281828 <zoomexp> ^"},
        {"zoomexp", "1 6.283 / log"},
        // Leave mode as-is
    });
    pgs.state_manager.microblock_transition({
        {"center_x", to_string(cx)},
        {"center_y", to_string(cy)},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"zoomexp", to_string(log(1/is.range))},
        {"circles_opacity", "0"},
    });

    CompositeScene cs;
    PendulumScene ps(is.ps, 0.5, 1);
    ps.global_publisher_key = true;
    LatexScene ls(latex_text(is.name), 1, 0.5, 0.2);

    // delete trailing zeros
    string str_cx = to_string(cx);
    string str_cy = to_string(cy);
    string str_p1 = to_string(is.ps.p1);
    string str_p2 = to_string(is.ps.p2);
    str_cx = str_cx.erase(str_cx.find_last_not_of('0') + 1);
    str_cy = str_cy.erase(str_cy.find_last_not_of('0') + 1);
    str_p1 = str_p1.erase(str_cx.find_last_not_of('0') + 1);
    str_p2 = str_p2.erase(str_cy.find_last_not_of('0') + 1);
    string latex_str = "\\theta_1 = " + str_cx + ", \\theta_2 = " + str_cy;
    if(is.ps.p1 != 0) latex_str += ", \\l_1 = " + str_p1;
    if(is.ps.p2 != 0) latex_str += ", \\l_1 = " + str_p2;
    LatexScene ls2(latex_str, 1, 0.3, 0.2);

    CoordinateSceneWithTrail ts(0.5, 1);
    cs.add_scene        (&pgs, "pgs", 0.5 , 0.5 );
    cs.add_scene_fade_in(&ps , "ps" , 0.75, 0.5 );
    cs.add_scene_fade_in(&ls , "ls" , 0.5 , 0.15);
    cs.add_scene_fade_in(&ls2, "ls2", 0.5 , 0.25);
    cs.add_scene_fade_in(&ts , "ts" , 0.25, 0.5 );

    if(false) {
        vector<float> audio_left;
        vector<float> audio_right;
        ps.generate_audio(10, audio_left, audio_right);
        cs.inject_audio(GeneratedSegment(audio_left, audio_right), 4);
    }
    else {
        cs.inject_audio(FileSegment(script), 4);
    }

    cs.state_manager.microblock_transition({
        {"pgs.opacity", ".4"},
    });
    ts.state_manager.set({
        {"center_x", to_string(cx)},
        {"center_y", to_string(cy)},
        {"zoom", to_string(is.fingerprint_zoom)},
        {"trail_opacity", "1"},
        {"trail_x", "{pendulum_theta1}"},
        {"trail_y", "{pendulum_theta2}"},
        {"center_x", "{pendulum_theta1}"},
        {"center_y", "{pendulum_theta2}"},
    });
    ps.state_manager.set({
        {"background_opacity", "0"},
        {"top_angle_opacity", "0"},
        {"bottom_angle_opacity", "0"},
        {"pendulum_opacity", "1"},
        {"physics_multiplier", "400"},
        {"path_opacity", "1"},
        {"rk4_step_size", "1 30 / <physics_multiplier> /"},
    });
    cs.render();
    cs.render();
    cs.render();
    cs.state_manager.microblock_transition({
        {"pgs.opacity", "1"},
        {"ps.opacity", "0"},
        {"ls.opacity", "0"},
        {"ls2.opacity", "0"},
        {"ts.opacity", "0"},
    });
    pgs.state_manager.microblock_transition({
        {"zoomexp", "1 6.283 / log"},
        {"circles_opacity", "1"},
    });
    cs.render();
    cs.remove_scene(&pgs);
}

void move_fractal(PendulumGridScene& pgs){
    pgs.state_manager.set({
        {"physics_multiplier", "0"},
        {"rk4_step_size", "0"},
    });
    pgs.state_manager.microblock_transition({
        {"zoom", "1 6.283 /"},
        {"circles_opacity", "0"},
        {"mode", "2"},
        {"center_x", "0"},
        {"center_y", "0"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));

    MovingPendulumGridScene mpgs;
    mpgs.state_manager.set({
        {"iterations", "200"},
        {"mode", "0"},
        {"rk4_step_size", "1 30 / .4 *"},
        {"zoomexp", "1 6.283 / log"},
        {"zoom", "2.718281828 <zoomexp> ^"},
        {"theta_or_momentum", "0"},
        {"contrast", "100"},
        {"p1", "[p1]"},
        {"p2", "[p2]"},
    });
    CompositeScene cs;
    cs.add_scene(&pgs, "pgs", 0.5, 0.5);
    cs.add_scene(&mpgs, "mpgs", 0.5, 0.5);
    cs.state_manager.set({
        {"p1", "0"},
        {"p2", "0"},
        {"pgs.opacity", "1"},
        {"mpgs.opacity", "0"},
    });
    cs.state_manager.microblock_transition({
        {"pgs.opacity", "0"},
        {"mpgs.opacity", "1"},
    });
    cs.inject_audio_and_render(SilenceSegment(2));
    StateSliderScene ssp1("[p1]", "l_1", -1, 1, .4, .1);
    StateSliderScene ssp2("[p2]", "l_2", -1, 1, .4, .1);
    cs.add_scene(&ssp1, "ssp1", 0.25, 0.9); 
    cs.add_scene(&ssp2, "ssp2", 0.75, 0.9); 
    cs.state_manager.set({
        {"ssp1.opacity", "0"},
        {"ssp2.opacity", "0"},
    });
    cs.state_manager.microblock_transition({
        {"ssp1.opacity", ".4"},
        {"ssp2.opacity", ".4"},
    });

    cs.inject_audio_and_render(FileSegment("So far, I've only been dropping the pendulums from a motionless state."));
    cs.state_manager.microblock_transition({
        {"p1", "<t> sin"},
        {"p2", "<t> cos"},
    });
    cs.inject_audio_and_render(FileSegment("What if, instead, we start the pendulum off with some momentum?"));
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.inject_audio(SilenceSegment(2), 2);
    cs.state_manager.macroblock_transition({
        {"p1", to_string(momentum_island.ps.p1)},
        {"p2", to_string(momentum_island.ps.p2)},
    });
    cs.render();
    mpgs.state_manager.microblock_transition({
        {"iterations", "500"},
    });
    cs.render();
    vector<PendulumGrid> grids{PendulumGrid(VIDEO_WIDTH, VIDEO_HEIGHT, -M_PI, M_PI, -M_PI, M_PI, momentum_island.ps.p1, momentum_island.ps.p1, momentum_island.ps.p2, momentum_island.ps.p2)};
    {
        const double ro2 = momentum_island.range/2;
        const double t1  = momentum_island.ps.theta1;
        const double t2  = momentum_island.ps.theta2;
        grids.push_back(PendulumGrid(VIDEO_WIDTH, VIDEO_HEIGHT, t1-ro2, t1+ro2, t2-ro2, t2+ro2, momentum_island.ps.p1, momentum_island.ps.p1, momentum_island.ps.p2, momentum_island.ps.p2));
    }
    if(FOR_REAL) for (PendulumGrid& g : grids) g.iterate_physics(10000, .1/30);
    PendulumGridScene mom(grids);
    mom.state_manager.set({
        {"center_x", "0"},
        {"center_y", "0"},
    });
    mom.state_manager.microblock_transition({
        {"center_x", to_string(momentum_island.ps.theta1)},
        {"center_y", to_string(momentum_island.ps.theta2)},
    });
    cs.add_scene_fade_in(&mom, "mom");
    mom.state_manager.set({
        {"mode", "3"},
    });
    cs.state_manager.microblock_transition({
        {"pgs.opacity", "0"},
        {"mpgs.opacity", "0"},
        {"ssp1.opacity", "0"},
        {"ssp2.opacity", "0"},
    });
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.remove_scene(&mom);
    showcase_an_island(mom, momentum_island, "Sure enough, there are islands of stability for other starting momenta too.");
    cs.add_scene(&mom, "mom");
    cs.state_manager.microblock_transition({
        {"pgs.opacity", ".4"},
        {"mpgs.opacity", "0"},
        {"mom.opacity", "0"},
    });
    PendulumState down = {.1, -.2, .0, .0};
    pgs.state_manager.set({
        {"circle0_x", to_string(down.theta1)},
        {"circle0_y", to_string(down.theta2)},
        {"circle0_r", ".2"},
        {"circles_opacity", "1"},
    });
    pgs.circles_to_render = 1;
    cs.inject_audio(FileSegment("Now, we've already seen how the pendulums which start near the angle zero-zero are very well-behaved."), 2);
    PendulumScene ps(down, .5, 1);
    ps.state_manager.set({
        {"physics_multiplier", "30"},
        {"rk4_step_size", "1 30 / <physics_multiplier> /"},
    });
    cs.add_scene_fade_in(&ps, "ps", 0.75, 0.5);
    cs.render();
    cs.render();
    cs.inject_audio_and_render(FileSegment("Those pendulums have extremely low mechanical energy."));
    cs.inject_audio_and_render(FileSegment("So maybe energy is somehow involved?"));
    PendulumState vert = {0, 3, .0, .0};
    PendulumScene ps2(vert, .5, 1);
    ps2.state_manager.set({
        {"physics_multiplier", "30"},
        {"rk4_step_size", "1 30 / <physics_multiplier> /"},
    });
    pgs.state_manager.microblock_transition({
        {"circle0_x", to_string(vert.theta1)},
        {"circle0_y", to_string(vert.theta2)},
    });
    cs.add_scene(&ps2, "ps2", 0.75, 1.5);
    cs.state_manager.microblock_transition({
        {"ps.opacity", "0"},
        {"ps2.y", ".5"},
    });
    cs.inject_audio_and_render(FileSegment("Taking this borderline-chaotic starting position,"));
    double vert_energy = compute_kinetic_energy(vert) + compute_potential_energy(vert);
    cout << "Vert energy: " << vert_energy << endl;
    pgs.state_manager.microblock_transition({
        {"energy_max", to_string(vert_energy)},
    });
    cs.inject_audio(FileSegment("If I color in red all the pendulums with less mechanical energy than this one"), 2);
    cs.render();
    cs.render();
    cs.inject_audio_and_render(FileSegment("It overlaps nicely with the lissajous-esque pendulums."));
    mpgs.state_manager.set({
        {"iterations", "200"},
    });
    cs.state_manager.microblock_transition({
        {"pgs.opacity", "0"},
        {"ps2.opacity", "0"},
        {"mom.opacity", "0"},
        {"mpgs.opacity", "1"},
        {"ssp1.opacity", "0"},
        {"ssp2.opacity", "0"},
    });
    cs.inject_audio_and_render(FileSegment("So, is that it? Are low-energy pendulums coherent, while high-energy pendulums are chaotic?"));
    mpgs.state_manager.microblock_transition({
        {"theta_or_momentum", "1"},
    });
    cs.inject_audio_and_render(FileSegment("It's time for a change in perspective."));
    cs.inject_audio_and_render(SilenceSegment(2));
    mpgs.state_manager.microblock_transition({
        {"zoomexp", "1 20 / log"},
    });
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.inject_audio_and_render(FileSegment("I've now reoriented the axes of our fractal to be in momentum-space instead of angle-space."));
    cs.inject_audio_and_render(FileSegment("In other words, we are looking at a grid of pendulums all with starting angle at 0, but with different starting speeds."));
    vector<PendulumGrid> grids_momentum{PendulumGrid(VIDEO_WIDTH, VIDEO_HEIGHT, 0,0,0,0, -10, 10, -10, 10)};
    if(FOR_REAL) for (PendulumGrid& g : grids_momentum) g.iterate_physics(10000, .1/30);
    PendulumGridScene perp(grids_momentum);
    cs.add_scene_fade_in(&perp, "perp");
    perp.state_manager.set({
        {"mode", "3"},
    });
    cs.inject_audio_and_render(FileSegment("It's still the case that the pendulums with low energy are stable,"));
    perp.state_manager.microblock_transition({
        {"energy_max", to_string(vert_energy)},
    });
    cs.inject_audio_and_render(FileSegment("and the area of pendulums with slightly higher energy tends to be chaotic,"));
    perp.state_manager.microblock_transition({
        {"energy_min", to_string(vert_energy)},
        {"energy_max", to_string(2*vert_energy)},
    });
    cs.inject_audio_and_render(FileSegment("But these ultra-high-energy pendulums don't easily fit into one box or another."));
    perp.state_manager.microblock_transition({
        {"energy_min", to_string(2*vert_energy)},
        {"energy_max", to_string(50*vert_energy)},
    });
}

void showcase_islands(PendulumGridScene& pgs) {
    showcase_an_island(pgs, isv[0], "This big island of stability is the Pretzel we saw earlier.");
    showcase_an_island(pgs, isv[1], "This one which I call the shoelace traces a more complex pattern.");
    showcase_an_island(pgs, isv[2], "This one draws a picture of a heart. Of the ones I'm showing, it's the smallest.");
    showcase_an_island(pgs, isv[3], "Although the bird is separate from the lissajous pendulums, it doesn't make any flips.");
}

void fine_grid(PendulumGridScene& pgs){
    pgs.state_manager.set({
        {"physics_multiplier", "5"},
        {"mode", "3"},
        {"rk4_step_size", "1 30 / .1 *"},
        {"zoom", "1 8 /"},
        {"trail_start_x", "0.5"},
        {"trail_start_y", "-0.5"},
        {"center_x", "3.1415"},
        {"center_y", "3.1415"},
    });
    pgs.inject_audio_and_render(FileSegment("Here we go!"));
    pgs.state_manager.microblock_transition({
        {"physics_multiplier", "30"},
    });
    pgs.inject_audio_and_render(SilenceSegment(7));
    pgs.state_manager.microblock_transition({
        {"mode", "2"},
        {"zoom", "1 6.283 /"},
        {"center_x", "0"},
        {"center_y", "0"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"trail_opacity", "1"},
    });
    pgs.inject_audio_and_render(FileSegment("Now, I'm gonna pick a certain point, corresponding to a pendulum in the black region meaning its behavior is non-chaotic."));
    pgs.state_manager.microblock_transition({
        {"trail_length", "1200"},
    });
    pgs.inject_audio_and_render(FileSegment("We can plot its path in angle-space just like we did before!"));
    pgs.state_manager.set({
        {"physics_multiplier", "0"},
    });
    pgs.state_manager.microblock_transition({
        {"zoom", "1 4 /"},
        {"trail_start_x", "0.25"},
        {"trail_start_y", "0.5"},
        {"mode", "1.5"},
    });
    pgs.inject_audio_and_render(FileSegment("Moving the point around in the black region, this curve moves smoothly and cleanly."));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", "1"},
        {"trail_start_y", "1"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", "1.5"},
        {"trail_start_y", "1"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", ".5"},
        {"trail_start_y", "0"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", ".5"},
        {"trail_start_y", "-.5"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", ".5"},
        {"trail_start_y", "1.1"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", "1.5"},
        {"trail_start_y", "1.1"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", "1"},
        {"trail_start_y", "1.1"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", "1.2"},
        {"trail_start_y", "1.1"},
    });
    pgs.inject_audio_and_render(FileSegment("This main black region is the home of all the lissajous style pendulums."));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", "1.6"},
        {"trail_start_y", "1.9"},
        {"mode", "2"},
        {"zoom", "0.05"},
    });
    pgs.inject_audio_and_render(FileSegment("But as soon as you leave and step into the chaotic region..."));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", "1.7 <t> 2 * sin 4 / +"},
        {"trail_start_y", "1.9 <t> 2 * cos 4 / +"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.inject_audio_and_render(FileSegment("It goes crazy."));
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"center_x", "3.1415"},
        {"center_y", "0"},
        {"trail_opacity", "0"},
        {"zoom", "1 6.283 /"},
    });
    pgs.inject_audio_and_render(FileSegment("Let's look a little closer at the chaotic region colored in white."));
    StateSet circles;
    for(int i = 0; i < isv.size(); i++){
        IslandShowcase is = isv[i];
        circles.insert(make_pair("circle" + to_string(i) + "_x", to_string(is.ps.theta1)));
        circles.insert(make_pair("circle" + to_string(i) + "_y", to_string(is.ps.theta2)));
        circles.insert(make_pair("circle" + to_string(i) + "_r", to_string(max(is.range/2, .1))));
    }
    pgs.state_manager.set(circles);
    pgs.state_manager.microblock_transition({
        {"circles_opacity", "1"},
    });
    pgs.inject_audio_and_render(FileSegment("You'll notice there are a few spots of black in here."));
    pgs.circles_to_render = isv.size();
    pgs.inject_audio_and_render(FileSegment("These so-called 'islands of stability' correspond to special groups of pendulums which take stable paths without diverging into chaos."));
    pgs.state_manager.microblock_transition({
        {"center_x", "2.49"},
        {"center_y", ".25"},
        {"ticks_opacity", "0"},
    });
    pgs.inject_audio_and_render(FileSegment("Let's take a closer look."));
}


/*
void what_is_chaos(){
    pgs.inject_audio_and_render(FileSegment("What even is chaos though?"));
    pgs.inject_audio_and_render(FileSegment("It's quite hard to define."));
    pgs.inject_audio_and_render(FileSegment("I draw these plots showing a clear white and black region, subtly implying a sort of phase transition, if you will, between a dichotomous chaotic and orderly state."));
    pgs.inject_audio_and_render(FileSegment("But this plot shows that it isn't that simple."));
    pgs.inject_audio_and_render(FileSegment("This band here is somewhere halfway in between, diverging faster than the coherent section, but slower than the chaotic sections."));
    pgs.inject_audio_and_render(FileSegment("It is the case that the differential equations for double pendulums has no closed form solution, regardless of starting position,"));
    pgs.inject_audio_and_render(FileSegment("But still, some of these pendulums are evidently much more well-behaved than the others."));
    pgs.inject_audio_and_render(FileSegment("We can formalize this by plotting the difference between our two pendulums over time."));
}
*/

void grids_and_points(){
    const string phys_mult = "16";
    PendulumGrid grid(VIDEO_WIDTH, VIDEO_HEIGHT, -M_PI, M_PI, -M_PI, M_PI, 0, 0, 0, 0);
    PendulumGridScene pgs(vector<PendulumGrid>{grid});
    pgs.state_manager.set({
        {"physics_multiplier", "0"},
        {"mode", "0"},
        {"rk4_step_size", "1 30 / "+phys_mult+" /"},
        {"zoom", "1 6.283 /"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"contrast", ".01"},
    });
    pgs.inject_audio_and_render(FileSegment("Let's increase the resolution to one pendulum per pixel."));
    pgs.state_manager.set({
        {"physics_multiplier", phys_mult},
    });
    pgs.inject_audio_and_render(SilenceSegment(10));
    CompositeScene cs;
    PendulumState pendulum_state = {5, 8, .0, .0};
    PendulumScene pend(pendulum_state);
    pend.state_manager.set({
        {"manual_mode", "1"},
        {"rk4_step_size", "1"},
        {"physics_multiplier", "0"},
        {"theta1_manual", "5"},
        {"theta2_manual", "8"},
    });
    cs.add_scene(&pgs, "pgs");
    cs.add_scene(&pend, "pend");
    
    cs.inject_audio_and_render(FileSegment("A nice feature of this fractal is that it tiles the plane."));
    cs.inject_audio(FileSegment("Since rotating either angle by 2pi yields the exact same pendulum, the image itself is periodic."), 2);
    pgs.state_manager.microblock_transition({
        {"center_x", "6.283"},
    });
    pend.state_manager.microblock_transition({
        {"theta1_manual", "5 6.283 +"},
    });
    cs.render();
    pgs.state_manager.microblock_transition({
        {"center_y", "6.283"},
    });
    pend.state_manager.microblock_transition({
        {"theta2_manual", "8 6.283 +"},
    });
    cs.render();
    pgs.inject_audio(SilenceSegment(5), 7);
    pgs.state_manager.microblock_transition({
        {"zoom", "1 8 /"},
    });
    pgs.render();
    pgs.render();
    pgs.state_manager.microblock_transition({
        {"center_x", "2"},
        {"center_y", "7"},
        {"zoom", "1 8 /"},
    });
    pgs.render();
    pgs.render();
    pgs.state_manager.microblock_transition({
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 40 /"},
    });
    pgs.render();
    pgs.render();
    pgs.state_manager.microblock_transition({
        {"zoom", "1 6.283 /"},
    });
    pgs.render();
    pgs.state_manager.microblock_transition({
        {"zoom", "1 3.1415 /"},
    });
    pgs.inject_audio_and_render(FileSegment("Pay attention to how there are two distinct modes of behavior here."));
    pgs.inject_audio(FileSegment("There is a region of chaotic noise, as well as a region of coherent behavior."), 4);
    pgs.state_manager.microblock_transition({
        {"center_x", "3.1415"},
        {"center_y", "3.1415"},
    });
    pgs.render();
    pgs.render();
    pgs.state_manager.microblock_transition({
        {"center_x", "0"},
        {"center_y", "0"},
    });
    pgs.render();
    pgs.render();
    pgs.state_manager.microblock_transition({
        {"mode", "2"},
    });
    pgs.inject_audio_and_render(FileSegment("Now, for each pixel, I'll track not only one pendulum but instead two, separated by a microscopic starting difference in angle, and plot their difference as time passes."));
//TODO the grid is insufficiently developed.
    pgs.inject_audio_and_render(FileSegment("In other words, this plot shows how quickly the pendulums in our grid diverge to chaos."));
    pgs.state_manager.microblock_transition({
        {"center_x", "3.1415"},
        {"center_y", "3.1415"},
        {"zoom", "1 8 /"},
    });
    pgs.inject_audio_and_render(SilenceSegment(4));
    pgs.state_manager.microblock_transition({
        {"contrast", ".0005"},
    });
    pgs.inject_audio_and_render(FileSegment("Let's watch that again from the start."));
}

void grid() {
    CompositeScene cs;
    vector<PendulumScene> vps;
    int gridsize = 13;
    double gridstep = 1./gridsize;
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            PendulumState pendulum_state = {(x-gridsize/2)*.2, (y-gridsize/2)*.2, .0, .0};
            PendulumScene ps(pendulum_state, gridstep*2.5, gridstep*2.5);
            StateSet state = {
                {"pendulum_opacity",   "[pendulum_opacity]"  },
                {"background_opacity", "[background_opacity]"},
                {"physics_multiplier", "[physics_multiplier]"},
                {"rk4_step_size",      "[rk4_step_size]"     },
                {"rainbow",            "[rainbow]"           },
            };
            ps.state_manager.set(state);
            vps.push_back(ps);
        }
    }
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            string key = "ps" + to_string(x+y*gridsize);
            cs.add_scene_fade_in(&(vps[x+y*gridsize]), key, gridstep*(x+.5), gridstep*(y+.5));
            cs.state_manager.set({
                {key + ".x", to_string(gridstep*(x+.5)) + " 1 <scrunch> 2 / lerp"},
            });
        }
    }
    StateSet state = {
        {"scrunch", "0"},
        {"pendulum_opacity", "1"},
        {"background_opacity", "0"},
        {"physics_multiplier", "0"},
        {"rk4_step_size", "1 30 / <physics_multiplier> .0001 + /"},
        {"rainbow", "0"},
    };
    cs.state_manager.set(state);
    int selected_pendulum = vps.size()*.2;
    string key_str = "ps" + to_string(selected_pendulum);
    cs.inject_audio_and_render(FileSegment("Instead, we can make a large array of pendulums like this."));
    string return_x = to_string(cs.state_manager.get_state({key_str + ".x"})[key_str + ".x"]);
    string return_y = to_string(cs.state_manager.get_state({key_str + ".y"})[key_str + ".y"]);
    vps[selected_pendulum].state_manager.set({
        {"pendulum_opacity", "1"},
    });
    vps[selected_pendulum].state_manager.microblock_transition({
        {"w", "1"},
        {"h", "1"},
        {"top_angle_opacity", "1"},
    });
    cs.state_manager.microblock_transition({
        {key_str + ".x", ".5"},
        {key_str + ".y", ".5"},
        {"pendulum_opacity", "0.4"},
    });
    cs.inject_audio(FileSegment("The pendulum's x position corresponds to the top angle,"), 3);
    cs.render();
    cs.render();
    vps[selected_pendulum].state_manager.microblock_transition({
        {"top_angle_opacity", "0"},
        {"bottom_angle_opacity", "1"},
    });
    cs.render();
    cs.inject_audio_and_render(FileSegment("and its y position corresponds to the bottom angle."));
    string size_str = to_string(gridstep*2.5);
    cs.state_manager.microblock_transition({
        {"rainbow", "1"},
        {key_str + ".x", return_x},
        {key_str + ".y", return_y},
        {"pendulum_opacity", "1"},
    });
    vps[selected_pendulum].state_manager.microblock_transition({
        {"w", size_str},
        {"h", size_str},
        {"bottom_angle_opacity", "0"},
    });
    cs.inject_audio_and_render(FileSegment("By associating angle positions with a particular color, it's easier to tell what is going on."));
    PendulumGrid pointgrid(100, 100, -M_PI, M_PI, -M_PI, M_PI, 0, 0, 0, 0);
    PendulumPointsScene pps(pointgrid, 0.5, 1);
    pps.state_manager.set({
        {"physics_multiplier", "0"},
        {"rk4_step_size", "1 30 / 5 /"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 6.283 /"},
    });
    cs.add_scene(&pps, "pps", -0.25, 0.5);
    cs.state_manager.macroblock_transition({
        {"scrunch", "1"},
        {"pps.x", ".25"},
    });
    cs.inject_audio_and_render(FileSegment("As a bonus, we can add points in angle space for these pendulums and see how they move."));
    cs.state_manager.set({
        {"physics_multiplier", "5"},
    });
    pps.state_manager.set({
        {"physics_multiplier", "5"},
    });
    cs.inject_audio_and_render(SilenceSegment(10));
    cs.state_manager.macroblock_transition({
        {"scrunch", "0"},
        {"pps.x", "-.25"},
    });
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.inject_audio_and_render(SilenceSegment(2));
}

void intro() {
    const double fov = 12;
    const double start_dist = 15*fov;
    const double after_move = start_dist-3;
    ThreeDimensionScene tds;
    vector<double> notes{pow(2, 3/12.), pow(2, 8/12.), pow(2, 10/12.), pow(2, 15/12.), pow(2, 20/12.), };
    for(int i = 0; i < 5; i++){
        PendulumState pendulum_state = {5+.0000001*i, 8, .0, .0};
        shared_ptr<PendulumScene> ps = make_shared<PendulumScene>(pendulum_state);
        ps->state_manager.set({
            {"path_opacity", "[parent_path_opacity]"},
            {"background_opacity", "0"},
            {"tone", to_string(notes[i])},
            {"volume", "[volume_set1]"},
            {"pendulum_opacity", "1"},
            {"physics_multiplier", "30"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        tds.add_surface(Surface(glm::vec3(0,-fov*.1,(i-2)*fov*.5), glm::vec3(fov,0,0), glm::vec3(0,fov,0), ps));
    }
    vector<shared_ptr<LatexScene>> ls;
    ls.push_back(make_shared<LatexScene>(latex_text("Double"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("Pendulums"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("are"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("NOT"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("Chaotic"), 1));
    tds.state_manager.set({
        {"parent_path_opacity", "0"},
        {"surfaces_opacity", "1"},
        {"lines_opacity", "0"},
        {"points_opacity", "1"},
        {"volume_set1", "1"},
        {"volume_set2", "0"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", to_string(start_dist)},
        {"fov", to_string(fov)},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
    });
    double word_size = .4/fov;
    tds.inject_audio(FileSegment("Double pendulums are NOT chaotic."), 13);
    tds.render();
    tds.render();
    tds.render();
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,-.24/fov,-after_move-.8), glm::vec3(word_size,0,0), glm::vec3(0,word_size,0), ls[0]));
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,-.1/fov,-after_move-.5), glm::vec3(word_size,0,0), glm::vec3(0,word_size,0), ls[1]));
    tds.render();
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,.04/fov,-after_move-.4), glm::vec3(.35*word_size,0,0), glm::vec3(0,.35*word_size,0), ls[2]));
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,.18/fov,-after_move-.7), glm::vec3(word_size,0,0), glm::vec3(0,word_size,0), ls[3]));
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,0.36/fov,-after_move-.7), glm::vec3(word_size,0,0), glm::vec3(0,word_size,0), ls[4]));
    tds.render();
    tds.render();
    tds.render();
    tds.render();
    ls[3]->begin_latex_transition(latex_text("NOT")+"^*");
    tds.inject_audio_and_render(FileSegment("Or, at least, not all of them."));
    tds.state_manager.macroblock_transition({
        {"d", to_string(after_move)},
    });
    tds.inject_audio_and_render(FileSegment("You've probably seen videos like these,"));
    for(int i = 0; i < ls.size(); i++) tds.remove_surface(ls[i]);
    tds.state_manager.macroblock_transition({
        {"qj", ".1"},
    });
    tds.inject_audio_and_render(FileSegment("where a tiny deviation in similar double pendulums amplifies over time,"));
    tds.inject_audio_and_render(FileSegment("until they completely desynchronize."));
    shared_ptr<LatexScene> chaotic = make_shared<LatexScene>(latex_text("Chaotic System"), 1);
    tds.add_surface(Surface(glm::vec3(0, -fov*.2, 0), glm::vec3(fov/2.,0,0), glm::vec3(0,fov/2.,0), chaotic));
    tds.inject_audio_and_render(FileSegment("This sensitivity to initial conditions renders the system unpredictable, and so we call it chaotic."));
    vector<double> notes2{pow(2, 0/12.), pow(2, 4/12.), pow(2, 7/12.), pow(2, 11/12.), pow(2, 12/12.), };
    for(int i = 0; i < 5; i++){
        PendulumState pendulum_state = {2.49+.0001*i, .25, .0, .0};
        shared_ptr<PendulumScene> ps = make_shared<PendulumScene>(pendulum_state);
        ps->state_manager.set({
            {"path_opacity", "[parent_path_opacity]"},
            {"background_opacity", "0"},
            {"volume", "[volume_set2]"},
            {"tone", to_string(notes2[i])},
            {"pendulum_opacity", "0"},
            {"physics_multiplier", "[stable_physics_multiplier]"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        ps->state_manager.macroblock_transition({
            {"pendulum_opacity", "1"},
        });
        tds.add_surface(Surface(glm::vec3(fov*3, -fov*.1, (i-2)*fov*.5), glm::vec3(fov,0,0), glm::vec3(0,fov,0), ps));
    }
    tds.state_manager.macroblock_transition({
        {"x", to_string(fov*3)},
        {"volume_set1", "0"},
        {"volume_set2", "1"},
    });
    tds.state_manager.set({
        {"stable_physics_multiplier", "0"},
    });
    tds.inject_audio_and_render(FileSegment("But you probably haven't seen this:"));
    tds.state_manager.set({
        {"stable_physics_multiplier", "30"},
    });
    tds.inject_audio_and_render(FileSegment("These pendulums also have slightly different starting positions."));
    tds.inject_audio_and_render(FileSegment("However, these will _not_ diverge."));
    tds.state_manager.macroblock_transition({
        {"parent_path_opacity", "1"},
        {"qj", "0"},
        {"d", to_string(start_dist*1.5)},
    });
    tds.inject_audio_and_render(FileSegment("They even trace a repeating pattern,"));
    tds.inject_audio_and_render(FileSegment("for which I call this the Pretzel Pendulum."));
//TODO insert pretzel picture
    tds.remove_surface(chaotic);
    tds.inject_audio_and_render(SilenceSegment(3));
    tds.state_manager.macroblock_transition({
        {"volume_set1", "1"},
        {"volume_set2", "0"},
        {"x", "0"},
    });
    tds.inject_audio_and_render(FileSegment("A stark contrast with the first ones, which are... all over the place."));
    tds.inject_audio_and_render(SilenceSegment(1));
    tds.inject_audio_and_render(FileSegment("So... what's the deal?"));
    tds.state_manager.macroblock_transition({
        {"volume_set1", "0.5"},
        {"volume_set2", "0.5"},
        {"x", to_string(fov*1.5)},
    });
    tds.inject_audio_and_render(FileSegment("These pendulums follow the same laws of physics."));
    tds.state_manager.macroblock_transition({
        {"volume_set1", "0"},
        {"volume_set2", "0"},
        {"z", to_string(start_dist*2)},
    });
    tds.inject_audio_and_render(FileSegment("The only difference is the position from which they started."));
}

void fractal() {
    PendulumGrid pg(VIDEO_WIDTH, VIDEO_HEIGHT, 0, 6.283, 0, 6.283, 0, 0, 0, 0);
    PendulumGridScene pgs(vector<PendulumGrid>{pg});
    pgs.state_manager.set({
        {"physics_multiplier", "16"},
        {"mode", "3"},
        {"rk4_step_size", "1 30 / .05 *"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 10 /"},
        {"ticks_opacity", "0"},
    });
    pgs.state_manager.macroblock_transition({
        {"ticks_opacity", "1"},
    });
    pgs.inject_audio_and_render(FileSegment("Behavior as a function of starting position can be graphed,"));
    pgs.state_manager.macroblock_transition({
        {"mode", "2"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 6.283 /"},
    });
    pgs.inject_audio_and_render(FileSegment("revealing fractals like these,"));
    pgs.inject_audio_and_render(SilenceSegment(1));
    CompositeScene cs;
    cs.add_scene(&pgs, "pgs");
    cs.inject_audio_and_render(FileSegment("where each point shows how chaotic a certain pendulum is."));
    vector<PendulumState> start_states = {
                                          {1.2, 1.5, 0, 0},
                                          {2.49, 0.25, 0, 0},
                                          {0.6   , 0.13, 0, 0},
                                         };
    StateSet state = {
        {"path_opacity", "0"},
        {"background_opacity", "0"},
        {"volume", "0"},
        {"pendulum_opacity", "1"},
        {"physics_multiplier", "10"},
        {"rk4_step_size", "1 30 / <physics_multiplier> /"},
    };
    pgs.state_manager.microblock_transition({
        {"circles_opacity", "1"},
    });
    cs.state_manager.microblock_transition({
        {"pgs.opacity", "1"},
    });
    for(int i = 0; i < start_states.size(); i++) {
        pgs.state_manager.set({
            {"circle"+to_string(i)+"_x", to_string(start_states[i].theta1)},
            {"circle"+to_string(i)+"_y", to_string(start_states[i].theta2)},
            {"circle"+to_string(i)+"_r", to_string(0.1)},
        });
    }
    pgs.circles_to_render = start_states.size();
    cs.inject_audio(FileSegment("But before diving into the fractals, let's get to know a few particular specimen."), 3);
    vector<PendulumScene> specimens;
    for(int i = 0; i < start_states.size(); i++) {
        specimens.push_back(PendulumScene(start_states[i], 1./3, 1./3));
    }
    for(int i = 0; i < start_states.size(); i++) {
        PendulumScene& ps = specimens[i];
        ps.state_manager.set(state);
        ps.state_manager.set({{"tone", to_string(i/4.+1)}});
        string name = "p" + to_string(i);
        cs.add_scene_fade_in(&ps, name, (M_PI+start_states[i].theta1 + 0.2)/6.283, 1-(M_PI+start_states[i].theta2)/6.283);
        cs.render();
    }
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.state_manager.microblock_transition({
        {"pgs.opacity", "0"},
        {"p0.pointer_opacity", "0"},
        {"p1.pointer_opacity", "0"},
        {"p2.pointer_opacity", "0"},
    });
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.remove_scene(&pgs);
    cs.state_manager.microblock_transition({
        {"pgs.opacity", "0"},
        {"p0.x", ".5"},
        {"p0.y", ".5"},
        {"p1.y", "1.5"},
        {"p2.y", "1.5"},
    });
    specimens[0].state_manager.microblock_transition({
        {"w", ".5"},
        {"h", "1"},
        {"volume", "1"},
    });
    cs.inject_audio_and_render(FileSegment("This pendulum is one of the chaotic ones."));
    specimens[0].state_manager.microblock_transition({
        {"top_angle_opacity", "1"},
        {"bottom_angle_opacity", "1"},
    });
    cs.inject_audio_and_render(FileSegment("We are particularly interested in the angles that separate each bar from the vertical."));
    specimens[0].global_publisher_key = true;
    CoordinateSceneWithTrail coord(.5, 1);
    cs.state_manager.microblock_transition({
        {"p0.x", ".75"},
    });
    specimens[1].state_manager.microblock_transition({
        {"w", ".5"},
        {"h", "1"},
    });
    cs.state_manager.microblock_transition({
        {"p1.x", ".75"},
        {"p2.x", ".75"},
        {"p1.y", "1.5"},
        {"p2.y", "1.5"},
    });
    coord.state_manager.set({
        {"center_x", to_string(start_states[0].theta1)},
        {"center_y", to_string(start_states[0].theta2)},
        {"zoom", ".04"},
        {"trail_opacity", "1"},
        {"trail_x", "{pendulum_theta1}"},
        {"trail_y", "{pendulum_theta2}"},
    });
    cs.add_scene_fade_in(&coord, "coord", 0.25, 0.5);
    cs.inject_audio_and_render(FileSegment("Plotting the top pendulum's angle as X and the bottom pendulum's angle as Y, we can make a graph like this."));
    coord.state_manager.microblock_transition({
        {"center_x", "<trail_x>"},
        {"center_y", "<trail_y>"},
    });
    cs.inject_audio_and_render(FileSegment("You can sort of already tell that this pendulum is chaotic from this graph."));
    cs.inject_audio_and_render(SilenceSegment(5));
    cs.inject_audio_and_render(SilenceSegment(10));
    specimens[0].state_manager.microblock_transition({
        {"volume", "0"},
    });
    cs.state_manager.microblock_transition({
        {"p0.y", "-.5"},
        {"p2.y", ".5"},
    });
    coord.state_manager.microblock_transition({
        {"trail_opacity", "0"},
        {"center_x", "0"},
        {"center_y", "0"},
    });
    specimens[2].state_manager.microblock_transition({
        {"volume", "25"},
    });
    specimens[2].state_manager.set({
        {"w", ".5"},
        {"h", "1"},
        {"top_angle_opacity", "1"},
        {"bottom_angle_opacity", "1"},
    });
    cs.inject_audio_and_render(FileSegment("Here's a non-chaotic pendulum."));
    specimens[0].global_publisher_key = false;
    specimens[2].global_publisher_key = true;
    coord.state_manager.microblock_transition({
        {"trail_opacity", "1"},
        {"zoom", ".1"},
    });
    cs.inject_audio_and_render(SilenceSegment(1));
    cs.inject_audio_and_render(SilenceSegment(5));
    coord.state_manager.microblock_transition({
        {"zoom", ".2"},
    });
    cs.inject_audio_and_render(FileSegment("This particular pendulum is drawing a Lissajous curve,"));
    cs.inject_audio_and_render(FileSegment("which is what you get by plotting two different sinusoids against each other."));
    CoordinateSceneWithTrail left(0.5, 1);
    left.state_manager.set({
        {"center_x", "<t> 2 /"},
        {"center_y", "0"},
        {"trail_opacity", "1"},
        {"ticks_opacity", "0"},
        {"zoom", ".2"},
        {"trail_x", "<t> 2 / 1 -"},
        {"trail_y", "{pendulum_theta2}"},
    });
    CoordinateSceneWithTrail right(0.5, 1);
    right.state_manager.set({
        {"center_x", "0"},
        {"center_y", "<t> 2 /"},
        {"trail_opacity", "1"},
        {"ticks_opacity", "0"},
        {"zoom", ".2"},
        {"trail_x", "{pendulum_theta1}"},
        {"trail_y", "<t> 2 / 1.5 -"},
    });
    left.trail_color = 0xffff00ff;
    right.trail_color = 0xff00ffff;
    cs.add_scene_fade_in(&left , "left" , 0.25, 0.5);
    cs.add_scene_fade_in(&right, "right", 0.25, 0.5);
    cs.inject_audio_and_render(FileSegment("Here is the signal from the top and bottom angles, separated out."));
    cs.inject_audio_and_render(SilenceSegment(5));
    specimens[2].state_manager.macroblock_transition({
        {"volume", "0"},
    });
    cs.inject_audio_and_render(FileSegment("We can run this pendulum for several hours, re-interpret these signals as sound waves on the left and right speaker, and 'listen' to the pendulum!"));
    vector<float> audio_left;
    vector<float> audio_right;
    specimens[2].generate_audio(4, audio_left, audio_right);
    cs.inject_audio_and_render(GeneratedSegment(audio_left, audio_right));
    left.state_manager.microblock_transition({
        {"trail_opacity", "0"},
    });
    right.state_manager.microblock_transition({
        {"trail_opacity", "0"},
    });
    coord.state_manager.microblock_transition({
        {"trail_opacity", "0"},
    });
    cs.state_manager.microblock_transition({
        {"p2.y", "1.5"},
        {"p0.y", ".5"},
    });
    cs.inject_audio_and_render(FileSegment("It doesn't precisely sound beautiful, but compare that with the chaotic pendulum!"));
    double pendulum_center_x = specimens[0].pend.state.theta1;
    double pendulum_center_y = specimens[0].pend.state.theta2;
    coord.state_manager.set({
        {"zoom", ".05"},
        {"trail_opacity", "1"},
        {"center_x", to_string(pendulum_center_x)},
        {"center_y", to_string(pendulum_center_y)},
    });
    left.state_manager.set({
        {"zoom", ".05"},
        {"center_y", to_string(pendulum_center_y)},
        {"trail_opacity", "1"},
    });
    right.state_manager.set({
        {"zoom", ".05"},
        {"center_x", to_string(pendulum_center_x)},
        {"trail_opacity", "1"},
    });
    specimens[0].global_publisher_key = true;
    specimens[2].global_publisher_key = false;
    cs.inject_audio_and_render(SilenceSegment(2));
    vector<float> audio_left_c;
    vector<float> audio_right_c;
    specimens[0].generate_audio(4, audio_left_c, audio_right_c);
    cs.inject_audio_and_render(GeneratedSegment(audio_left_c, audio_right_c));
    cs.inject_audio_and_render(FileSegment("And here's the pretzel pendulum."));
    pendulum_center_x = specimens[1].pend.state.theta1;
    pendulum_center_y = specimens[1].pend.state.theta2;
    left.state_manager.microblock_transition({
        {"trail_opacity", "0"},
        {"zoom", ".02"},
        {"trail_x", "<t> 5 * 10 -"},
        {"center_y", to_string(pendulum_center_y)},
    });
    right.state_manager.microblock_transition({
        {"trail_opacity", "0"},
        {"zoom", ".02"},
        {"trail_y", "<t> 5 * 15 -"},
        {"center_x", to_string(pendulum_center_x)},
    });
    coord.state_manager.microblock_transition({
        {"trail_opacity", "0"},
        {"zoom", ".02"},
        {"center_x", to_string(pendulum_center_x)},
        {"center_y", to_string(pendulum_center_y)},
    });
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.state_manager.microblock_transition({
        {"p0.y", "-.5"},
        {"p1.y", ".5"},
    });
    coord.state_manager.set({
        {"trail_opacity", "1"},
    });
    left.state_manager.set({
        {"trail_opacity", "1"},
    });
    right.state_manager.set({
        {"trail_opacity", "1"},
    });
    specimens[1].global_publisher_key = true;
    specimens[0].global_publisher_key = false;
    cs.inject_audio_and_render(SilenceSegment(2));
    vector<float> audio_left_p;
    vector<float> audio_right_p;
    specimens[1].generate_audio(4, audio_left_p, audio_right_p);
    cs.inject_audio_and_render(GeneratedSegment(audio_left_p, audio_right_p));
    coord.state_manager.microblock_transition({
        {"zoom", ".04"},
//TODO these numbers are off
        {"center_x", "8"},
        {"center_y", "14"},
    });
    cs.inject_audio_and_render(FileSegment("It traces a repetitive curve in angle-space, so it sounds the cleanest."));
    cs.fade_out_all_scenes();
    cs.inject_audio_and_render(FileSegment("Listening to a pendulum can tell us if it's chaotic, but unfortunately, we have to do it one-by-one, pendulum-by-pendulum."));
}

void render_video() {
    //PRINT_TO_TERMINAL = false;
    SAVE_FRAME_PNGS = false;
    //FOR_REAL = false;

    intro();
    fractal();
    grid();
    grids_and_points();
    vector<PendulumGrid> grids{PendulumGrid(VIDEO_WIDTH, VIDEO_HEIGHT, -M_PI, M_PI, -M_PI, M_PI, 0, 0, 0, 0)};
    for(const IslandShowcase& is : isv) {
        const double ro2 = is.range/2;
        const double t1 = is.ps.theta1;
        const double t2 = is.ps.theta2;
        grids.push_back(PendulumGrid(VIDEO_WIDTH, VIDEO_HEIGHT, t1-ro2, t1+ro2, t2-ro2, t2+ro2, 0, 0, 0, 0));
    }
    for (PendulumGrid& g : grids) g.iterate_physics(10000, .1/30);
    PendulumGridScene pgs(grids);
    fine_grid(pgs);
    showcase_islands(pgs);
    move_fractal(pgs);
}
