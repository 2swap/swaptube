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
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"

struct IslandShowcase {
    PendulumState ps;
    double range;
    double fingerprint_zoom;
    string name;
    string blurb;
};

vector<IslandShowcase> isv{{{2.49  ,  .25   , .0, .0}, 0.4 , .02, "The Pretzel", "This big island of stability contains the Pretzel we saw earlier."},
                           {{2.658 , -2.19  , .0, .0}, 0.2 , .02, "The Shoelace", "This one which I call the shoelace traces a more complex pattern."},
                           {{2.453 , -2.7727, .0, .0}, 0.05, .02, "The Heart", "This one draws a picture of a heart. Of the islands I'm showing, it's the smallest."}};

vector<IslandShowcase> isv2{{{1.351 ,  2.979 , .0, .0}, 0.2 , .06, "The Bird", "Although on a separate island from the lissajous pendulums,\\\\\\\\neither of its two arms does a flip."},
                           {{0.567 ,  2.995 , .0, .0}, 0.2 , .08, "Big-Ears", "Similar to The Bird,\\\\\\\\this pendulum makes no flips."},
                           {{1.484 ,  2.472 , .0, .0}, 0.2 , .06, "The Fishbowl", "Also makes no flips."},
                           {{1.244 , -0.258 , .0, .0}, 0.2 , .05, "The Pistachio", "This island is in a region dotted with small islands of stability."},
                           {{1.238 , -0.854 , .0, .0}, 0.2 , .06, "The ???", "This pendulum's period is very long,\\\\\\\\hence its sound being so deep."},
                           {{1.311 , -0.804 , .0, .0}, 0.2 , .06, "Jake the Dog", "This pendulum lives right next to the ???"},
                           {{0.028 ,  2.971 , .0, .0}, 0.2 , .06, "Asymmetric A", "The majority of the pendulums I found exhibit left-right symmetry.\\\\\\\\This is the first of 3 exceptions."},
                           {{-.478 ,  2.633 , .0, .0}, 0.2 , .06, "Asymmetric B", "The second asymmetric island."},
                           {{1.348 , -0.299 , .0, .0}, 0.2 , .06, "Asymmetric C", "This island is very close to the Pistachio."},
                           {{0.572 ,  2.539 , .0, .0}, 0.2 , .06, "Seneca Lake", "This island is the largest of a region rich with\\\\\\\\long but thin islands of stability, named after\\\\\\\\the largest of the Fingerlakes of Upstate New York."},
                           {{2.808 ,  0.046 , .0, .0}, 0.2 , .06, "The Jumper", "Nothing too complicated."},
                           {{3.0224,  0.0295, .0, .0}, 0.2 , .06, "The High-Jumper", "This island is particularly close\\\\\\\\to the top angle being pi and the bottom angle being 0."},
                           {{1.782 ,  0.137 , .0, .0}, 0.2 , .02, "The Necktie", "This pendulum traces a path very similar to the Pretzel,\\\\\\\\and its island is close to the Pretzel island of stability."}};

IslandShowcase momentum_island = {{2.14, 0.8  , .4, .6}, 0.2, 0.04, "", "Sure enough, there are islands of stability for other starting momenta too."};

void showcase_an_island(PendulumGridScene& pgs, const IslandShowcase& is, bool spoken) {
    cout << "Showcasing " << is.name << endl;
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
        {"ticks_opacity", "0"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"zoomexp", to_string(log(1/is.range))},
        {"circles_opacity", "0"},
    });

    CompositeScene cs;
    PendulumScene ps(is.ps, 0.5, 1);
    ps.global_publisher_key = true;
    LatexScene ls(latex_text(is.name), 1, 1, 0.2);

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
    if(is.ps.p1 != 0) latex_str += ", p_1 = " + str_p1;
    if(is.ps.p2 != 0) latex_str += ", p_1 = " + str_p2;
    LatexScene ls2(latex_str, 1, 1, 0.12);

    CoordinateSceneWithTrail ts(0.5, 1);
    cs.add_scene        (&pgs, "pgs", 0.5 , 0.5 );
    cs.add_scene_fade_in(&ps , "ps" , 0.75, 0.5 );
    cs.add_scene_fade_in(&ls , "ls" , 0.5 , 0.15);
    cs.add_scene_fade_in(&ls2, "ls2", 0.5 , 0.25);
    cs.add_scene_fade_in(&ts , "ts" , 0.25, 0.5 );
    LatexScene blurb(latex_text(is.blurb), .5, .2, 0.12);
    if(!spoken){
        cs.add_scene_fade_in(&blurb , "blurb" , 0.75, 0.9 );
    }

    if(spoken) cs.inject_audio(FileSegment(is.blurb), 3);
    else {
        vector<float> audio_left;
        vector<float> audio_right;
        ps.generate_audio(5, audio_left, audio_right);
        cs.inject_audio(GeneratedSegment(audio_left, audio_right), 3);
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
    });
    cs.render();
    cs.render();
    cs.render();
    vector<float> audio2_left;
    vector<float> audio2_right;
    ps.generate_audio(3, audio2_left, audio2_right);
    cs.inject_audio_and_render(GeneratedSegment(audio2_left, audio2_right));
    if(!spoken) cs.state_manager.microblock_transition({
        {"blurb.opacity", "0"},
    });
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
    cs.inject_audio_and_render(SilenceSegment(1));
    cs.remove_scene(&pgs);
}

void showcase_momentum_space(PendulumGridScene& perp, double p1, double p2, double volume_mult = 1) {
    perp.state_manager.set({
        {"circle0_x", to_string(p1)},
        {"circle0_y", to_string(p2)},
        {"circle0_r", "1"},
    });
    perp.circles_to_render = 1;
    perp.state_manager.microblock_transition({
        {"circles_opacity", "1"},
    });

    CompositeScene cs;
    PendulumScene ps({0, 0, p1, p2}, 0.5, 1);

    cs.add_scene        (&perp, "perp", 0.5 , 0.5 );
    cs.add_scene_fade_in(&ps  , "ps"  , 0.75, 0.5 );

    vector<float> audio_left;
    vector<float> audio_right;
    ps.generate_audio(5, audio_left, audio_right, volume_mult);
    cs.inject_audio(GeneratedSegment(audio_left, audio_right), 3);

    cs.state_manager.microblock_transition({
        {"perp.opacity", ".4"},
    });
    ps.state_manager.set({
        {"background_opacity", "0"},
        {"top_angle_opacity", "0"},
        {"bottom_angle_opacity", "0"},
        {"pendulum_opacity", "1"},
        {"physics_multiplier", "400"},
        {"path_opacity", "1"},
    });
    cs.render();
    cs.render();
    cs.render();
    cs.state_manager.microblock_transition({
        {"perp.opacity", "1"},
        {"ps.opacity", "0"},
    });
    perp.state_manager.microblock_transition({
        {"circles_opacity", "0"},
    });
    cs.inject_audio_and_render(SilenceSegment(.8));
    cs.remove_scene(&perp);
}

void discuss_energy(PendulumGridScene& pgs){
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
    PendulumState down = {.1, -.2, .0, .0};
    pgs.state_manager.microblock_transition({
        {"center_x", "0"},
        {"center_y", "0"},
    });
    pgs.state_manager.set({
        {"circle0_x", to_string(down.theta1)},
        {"circle0_y", to_string(down.theta2)},
        {"circle0_r", ".2"},
    });
    pgs.state_manager.microblock_transition({
        {"circles_opacity", "1"},
    });
    pgs.circles_to_render = 1;
    CompositeScene cs;
    cs.add_scene(&pgs, "pgs");
    cs.inject_audio(FileSegment("We've seen how the pendulums which start near the angle zero-zero are very well-behaved."), 2);
    PendulumScene ps(down, .5, 1);
    cs.add_scene_fade_in(&ps, "ps", 0.75, 0.5);
    cs.render();
    cs.render();
    LatexScene low_energy(latex_color(0xffff0000, latex_text("Low-Energy Pendulum")), 1, .5, .2);
    cs.add_scene_fade_in(&low_energy, "low_energy", .75, .3);
    cs.inject_audio_and_render(FileSegment("Those pendulums have extremely low mechanical energy."));
    cs.inject_audio_and_render(FileSegment("So maybe energy is somehow involved?"));
    PendulumState vert = {0, 3.1, .0, .0};
    PendulumScene ps2(vert, .5, 1);
    pgs.state_manager.microblock_transition({
        {"circle0_x", to_string(vert.theta1)},
        {"circle0_y", to_string(vert.theta2)},
    });
    cs.add_scene(&ps2, "ps2", 0.25, 1.5);
    cs.state_manager.microblock_transition({
        {"ps.opacity", "0"},
        {"low_energy.opacity", "0"},
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
    cs.state_manager.microblock_transition({
        {"ps2.opacity", "0"},
    });
    cs.inject_audio(FileSegment("So, is that it? Are high-energy pendulums chaotic, while low-energy pendulums are coherent?"), 5);
    cs.render();
    pgs.state_manager.microblock_transition({
        {"energy_max", to_string(vert_energy*5)},
        {"energy_min", to_string(vert_energy)},
    });
    cs.render();
    cs.render();
    pgs.state_manager.microblock_transition({
        {"energy_max", to_string(vert_energy)},
        {"energy_min", "-1"},
    });
    cs.render();
    cs.render();
    cs.remove_scene(&pgs);
}

void move_fractal(PendulumGridScene& pgs){
    PendulumState vert = {0, 3, .0, .0};
    double vert_energy = compute_kinetic_energy(vert) + compute_potential_energy(vert);
    MovingPendulumGridScene mpgs;
    CompositeScene cs;
    mpgs.state_manager.set({
        {"physics_multiplier", "300"},
        {"mode", "0"},
        {"rk4_step_size", ".4 30 /"},
        {"zoomexp", "1 6.283 / log"},
        {"zoom", "2.718281828 <zoomexp> ^"},
        {"theta_or_momentum", "0"},
        {"contrast", "1"},
        {"p1", "[p1]"},
        {"p2", "[p2]"},
    });
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
    cs.inject_audio_and_render(FileSegment("It's time for a change in perspective."));
    StateSliderScene ssp1("[p1]", "p_1", -1, 1, .4, .1);
    StateSliderScene ssp2("[p2]", "p_2", -1, 1, .4, .1);
    cs.add_scene(&ssp1, "ssp1", 0.25, 0.9); 
    cs.add_scene(&ssp2, "ssp2", 0.75, 0.9); 
    cs.state_manager.set({
        {"ssp1.opacity", "0"},
        {"ssp2.opacity", "0"},
    });
    cs.state_manager.microblock_transition({
        {"ssp1.opacity", ".6"},
        {"ssp2.opacity", ".6"},
    });

    cs.inject_audio_and_render(FileSegment("So far, I've only been dropping the pendulums from a motionless state."));
    cs.state_manager.microblock_transition({
        {"p1", "<t> sin"},
        {"p2", "<t> cos"},
    });
    cs.inject_audio_and_render(FileSegment("What if, instead, we start the pendulum off with some momentum?"));
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.state_manager.macroblock_transition({
        {"p1", to_string(momentum_island.ps.p1)},
        {"p2", to_string(momentum_island.ps.p2)},
    });
    cs.inject_audio_and_render(SilenceSegment(2));
    vector<PendulumGrid> grids{PendulumGrid(VIDEO_HEIGHT, VIDEO_HEIGHT, -M_PI, M_PI, -M_PI, M_PI, momentum_island.ps.p1, momentum_island.ps.p1, momentum_island.ps.p2, momentum_island.ps.p2)};
    {
        const double ro2 = momentum_island.range/2;
        const double t1  = momentum_island.ps.theta1;
        const double t2  = momentum_island.ps.theta2;
        grids.push_back(PendulumGrid(VIDEO_HEIGHT, VIDEO_HEIGHT, t1-ro2*VIDEO_WIDTH/VIDEO_HEIGHT, t1+ro2*VIDEO_WIDTH/VIDEO_HEIGHT, t2-ro2, t2+ro2, momentum_island.ps.p1, momentum_island.ps.p1, momentum_island.ps.p2, momentum_island.ps.p2));
    }
    if(FOR_REAL) for(PendulumGrid& g : grids) {
        g.iterate_physics(300*4, .1/30);
    }
    PendulumGridScene mom(grids);
    mom.state_manager.microblock_transition({
        {"physics_multiplier", "400"},
        {"center_x", "0"},
        {"center_y", "0"},
    });
    mom.state_manager.set({
        {"mode", "2"},
    });
    cs.add_scene_fade_in(&mom, "mom");
    cs.state_manager.macroblock_transition({
        {"pgs.opacity", "0"},
        {"mpgs.opacity", "0"},
        {"ssp1.opacity", "0"},
        {"ssp2.opacity", "0"},
    });
    mom.state_manager.microblock_transition({
        {"ticks_opacity", "0"},
    });
    cs.inject_audio(SilenceSegment(2), 2);
    cs.render();
    cs.render();
    cs.remove_scene(&ssp1);
    cs.remove_scene(&ssp2);
    mom.state_manager.set({
        {"physics_multiplier", "0"},
    });
    cs.remove_scene(&mom);
    showcase_an_island(mom, momentum_island, true);
    mom.state_manager.microblock_transition({
        {"center_x", "0"},
        {"center_y", "0"},
    });
    cs.add_scene(&mom, "mom");
    cs.inject_audio_and_render(SilenceSegment(1));
    cs.state_manager.microblock_transition({
        {"mom.opacity", "0"},
        {"mpgs.opacity", "1"},
    });
    cs.inject_audio_and_render(SilenceSegment(1));






    cs.state_manager.microblock_transition({
        {"p1", "0"},
        {"p2", "0"},
    });
    mpgs.state_manager.microblock_transition({
        {"theta_or_momentum", "1"},
        {"zoomexp", "1 40 / log"},
    });
    cs.inject_audio_and_render(FileSegment("I'm now reorienting the axes of our fractal to be in momentum-space instead of angle-space."));
    vector<PendulumGrid> grids_momentum{PendulumGrid(VIDEO_WIDTH, VIDEO_HEIGHT, 0,0,0,0, -20.*VIDEO_WIDTH/VIDEO_HEIGHT, 20.*VIDEO_WIDTH/VIDEO_HEIGHT, -20, 20)};
    if(FOR_REAL) for(PendulumGrid& g : grids_momentum) g.iterate_physics(4000, .05/30);
    PendulumGridScene perp(grids_momentum);
    cs.add_scene_fade_in(&perp, "perp");
    cs.state_manager.microblock_transition({
        {"mpgs.opacity", "0"},
    });
    perp.state_manager.set({
        {"mode", "2"},
        {"center_y", "-0.0001"},
        {"zoom", "1 40 /"},
        {"physics_multiplier", "300"},
        {"rk4_step_size", "1 30 / .05 *"},
    });
    cs.inject_audio_and_render(FileSegment("In other words, we're looking at a grid of pendulums all with starting angle at 0, but with different starting speeds."));
    cs.remove_scene(&pgs);
    cs.remove_scene(&perp);
    perp.state_manager.microblock_transition({
        {"physics_multiplier", "0"},
    });
    perp.state_manager.microblock_transition({
        {"energy_max", to_string(vert_energy)},
    });
    perp.inject_audio_and_render(FileSegment("It's still the case that the pendulums with low energy are stable,"));
    showcase_momentum_space(perp, 1, 1);
    perp.state_manager.microblock_transition({
        {"energy_min", to_string(2*vert_energy)},
        {"energy_max", to_string(5*vert_energy)},
    });
    perp.inject_audio_and_render(FileSegment("and the pendulums with slightly higher energy tend to be chaotic,"));
    showcase_momentum_space(perp, 4, 3);
    perp.state_manager.microblock_transition({
        {"energy_min", to_string(5*vert_energy)},
        {"energy_max", to_string(2000*vert_energy)},
    });
    perp.inject_audio_and_render(FileSegment("But these ultra-high-energy pendulums don't easily fit into one box or another."));
    perp.state_manager.microblock_transition({
        {"energy_min", to_string(40000*vert_energy)},
        {"energy_max", "0"},
    });
    showcase_momentum_space(perp, 10, 0, 0.3);
    showcase_momentum_space(perp, 0, 15);
    showcase_momentum_space(perp, 10, 10, 0.4);
    showcase_momentum_space(perp, -10, 10);
    showcase_momentum_space(perp, 15, 10, 0.5);
    CompositeScene cs_spiro;
    cs_spiro.add_scene(&perp, "perp");
    int gridsize = 10;
    vector<PendulumScene> vps;
    double gridstep = 1./gridsize;
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            PendulumState pendulum_state = {.0, .0, ((x+.5)/gridsize - .5) * 40*VIDEO_WIDTH/VIDEO_HEIGHT, -((y+.5)/gridsize - .5) * 40};
            PendulumScene ps(pendulum_state, gridstep, gridstep);
            ps.state_manager.set({
                {"pendulum_opacity", "0"},
                {"path_opacity", "1"},
            });
            vps.push_back(ps);
        }
    }
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            string key = "ps" + to_string(x+y*gridsize);
            cs_spiro.add_scene(&(vps[x+y*gridsize]), key, gridstep*(x+.5), gridstep*(y+.5));
            cs_spiro.state_manager.set({
                {key+".opacity", "<spiro_opacity>"},
            });
        }
    }
    cs_spiro.state_manager.set({
        {"spiro_opacity", "0"},
    });
    cs_spiro.state_manager.microblock_transition({
        {"perp.opacity", "0.4"},
        {"spiro_opacity", "1"},
    });
    cs_spiro.inject_audio_and_render(FileSegment("Here's the spirographs associated with pendulums in momentum-space."));
    cs_spiro.inject_audio_and_render(SilenceSegment(4));
    cs_spiro.state_manager.microblock_transition({
        {"spiro_opacity", "0"},
    });
    cs_spiro.inject_audio_and_render(SilenceSegment(1));
    cs_spiro.remove_scene(&perp);
    cs.add_scene(&perp, "perp");
    cs.state_manager.set({
        {"perp.opacity", "0.4"},
    });
    cs.state_manager.microblock_transition({
        {"mpgs.opacity", "1"},
        {"perp.opacity", "0"},
    });
    mpgs.state_manager.set({
        {"mode", "3"},
    });
    cs.inject_audio_and_render(SilenceSegment(1));
    cs.remove_scene(&perp);
    mpgs.state_manager.microblock_transition({
        {"zoomexp", "1 6.283 / log"},
        {"theta_or_momentum", "0"},
    });
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.add_scene_fade_in(&pgs, "pgs");
    pgs.circles_to_render = 0;
    pgs.state_manager.set({
        {"energy_max", "0"},
    });
    cs.state_manager.microblock_transition({
        {"mpgs.opacity", "0"},
    });
    cs.remove_scene(&pgs);
}

void showcase_more_islands(PendulumGridScene& pgs) {
    cout << "Doing showcase_more_islands" << endl;
    for(IslandShowcase is : isv2) showcase_an_island(pgs, is, false);
    cout << "Done showcase_more_islands" << endl;
}

void showcase_islands(PendulumGridScene& pgs) {
    cout << "Doing showcase_islands" << endl;
    for(IslandShowcase is : isv) showcase_an_island(pgs, is, true);
}

void fine_grid(PendulumGridScene& pgs){
    cout << "Doing fine_grid" << endl;
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
        {"physics_multiplier", "80"},
        {"trail_opacity", "1"},
    });
    pgs.inject_audio_and_render(FileSegment("Now, I'm gonna pick a certain point, corresponding to a pendulum in the black region, meaning its behavior is non-chaotic."));
    pgs.state_manager.microblock_transition({
        {"trail_length", "1200"},
        {"zoom", "1 4 /"},
    });
    pgs.inject_audio_and_render(FileSegment("We can plot its path in angle-space just like we did before!"));
    pgs.state_manager.set({
        {"physics_multiplier", "0"},
    });
    pgs.state_manager.microblock_transition({
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
        {"mode", "2"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", "1"},
        {"trail_start_y", "1.1"},
    });
    pgs.inject_audio_and_render(FileSegment("This main black region is the home of all the lissajous style pendulums."));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", "1.6"},
        {"trail_start_y", "1.9"},
        {"zoom", "0.05"},
    });
    pgs.inject_audio_and_render(FileSegment("But as soon as you leave and step into the chaotic region..."));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", "1.7 <t> 2 * sin 4 / +"},
        {"trail_start_y", "1.9 <t> 2 * cos 4 / +"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.inject_audio_and_render(FileSegment("All hell breaks loose."));
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"center_x", "3.1415"},
        {"center_y", "0"},
        {"trail_opacity", "0"},
        {"zoom", "1 6.283 /"},
    });
    pgs.inject_audio_and_render(FileSegment("Let's look closer at the chaotic region in white."));
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
    pgs.circles_to_render = isv.size();
    pgs.inject_audio_and_render(FileSegment("There are a few spots of black in here."));
    pgs.state_manager.microblock_transition({
        {"ticks_opacity", "0"},
    });
    pgs.inject_audio_and_render(FileSegment("These 'islands of stability' correspond to special pendulums which trace stable paths."));
}

void outtro(){
    vector<PendulumGrid> grids{PendulumGrid(VIDEO_HEIGHT, VIDEO_HEIGHT, -M_PI, M_PI, -M_PI, M_PI, 0, 0, 0, 0)};
    for(PendulumGrid& p : grids) p.iterate_physics(300, .05/30);
    PendulumGridScene pgs(grids);
    pgs.state_manager.set({
        {"mode", "3"},
    });
    pgs.state_manager.microblock_transition({
        {"mode", "0"},
    });
    CompositeScene cs;
    cs.add_scene(&pgs, "pgs");
    cs.inject_audio_and_render(FileSegment("Chaos is a hard thing to define."));
    cs.state_manager.microblock_transition({{"pgs.opacity", "0.3"}});
    pgs.state_manager.set({
        {"physics_multiplier", "10"},
        {"rk4_step_size", ".05 30 /"},
    });
    cs.inject_audio_and_render(FileSegment("You may have even noticed that I subtly changed definitions throughout the video: aperiodicity, unpredictibility, divergence..."));

    LatexScene ls1("\\Delta = \\theta_1 - \\theta_2", 1, 1, 0.1);
    LatexScene ls2("D = 16 - 9\\cos^2\\Delta", 1, 1, 0.1);
    LatexScene ls3("\\dot{\\theta}_1 = \\frac{6}{m l^2 D} (2 p_1 - 3 \\cos\\Delta \\, p_2)", 1, 1, 0.1);
    LatexScene ls4("\\dot{\\theta}_2 = \\frac{6}{m l^2 D} (8 p_2 - 3 \\cos\\Delta \\, p_1)", 1, 1, 0.1);
    LatexScene ls5("\\dot{p}_1 = -\\frac{1}{2} m l \\left( 3 g \\sin\\theta_1 + l \\dot{\\theta}_1 \\dot{\\theta}_2 \\sin\\Delta \\right)", 1, 1, 0.1);
    LatexScene ls6("\\dot{p}_2 = -\\frac{1}{2} m l \\left( g \\sin\\theta_2 - l \\dot{\\theta}_1 \\dot{\\theta}_2 \\sin\\Delta \\right)", 1, 1, 0.1);
    LatexScene ls7(latex_text("...have no analytic solution."), 1, 1, 0.1);
    cs.inject_audio(FileSegment("It is the case that the differential equations describing double pendulums have no analytic solution for _any_ starting position,"), 11);
    cs.add_scene_fade_in(&ls1, "ls1", .5, .2);
    cs.render();
    cs.add_scene_fade_in(&ls2, "ls2", .5, .3);
    cs.render();
    cs.add_scene_fade_in(&ls3, "ls3", .5, .4);
    cs.render();
    cs.add_scene_fade_in(&ls4, "ls4", .5, .5);
    cs.render();
    cs.add_scene_fade_in(&ls5, "ls5", .5, .6);
    cs.render();
    cs.add_scene_fade_in(&ls6, "ls6", .5, .7);
    cs.render();
    cs.render();
    cs.render();
    cs.render();
    cs.add_scene_fade_in(&ls7, "ls7", .5, .8);
    cs.render();
    cs.render();
    cs.state_manager.microblock_transition({
        {"ls1.opacity", "0"},
        {"ls2.opacity", "0"},
        {"ls3.opacity", "0"},
        {"ls4.opacity", "0"},
        {"ls5.opacity", "0"},
        {"ls6.opacity", "0"},
        {"ls7.opacity", "0"},
    });
    cs.inject_audio_and_render(FileSegment("but even so,"));
    PendulumScene ps1({2.49, .25, .0, .0}, .31, .31);
    PendulumScene ps2({1.5 , 1  , 8, 5}, .31, .31);
    PendulumScene ps3({.0  , .0 ,10.,-15.}, .31, .31);
    StateSet ss = {
        {"rainbow", "0"},
        {"path_opacity", "1"},
    };
    ps1.state_manager.set(ss);
    ps2.state_manager.set(ss);
    ps3.state_manager.set(ss);
    cs.inject_audio(FileSegment("this system is sufficiently fertile to support little gems of order in the rough, as any good chaos should."), 9);
    cs.add_scene_fade_in(&ps1, "ps1", .5, .5);
    cs.render();
    cs.render();
    cs.render();
    cs.add_scene_fade_in(&ps2, "ps2", .75, .5);
    cs.render();
    cs.render();
    cs.state_manager.microblock_transition({
        {"ps1.opacity", "0"},
    });
    cs.render();
    cs.add_scene_fade_in(&ps3, "ps3", .25, .5);
    cs.render();
    cs.render();
    cs.state_manager.microblock_transition({
        {"ps2.opacity", "0"},
    });
    cs.render();
    TwoswapScene ts;
    ts.state_manager.set({{"circle_opacity", "1"}});
    cs.state_manager.microblock_transition({{"pgs.opacity", "0.0"}});
    cs.add_scene_fade_in(&ts, "ts");
    cs.inject_audio_and_render(FileSegment("This has been 2swap."));
    cs.state_manager.microblock_transition({
        {"ts.opacity", "0"},
    });
    cs.inject_audio_and_render(SilenceSegment(2));
}

void intro() {
    cout << "Doing intro()" << endl;
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
        tds.add_surface(Surface(glm::vec3(0,-fov*.1,(i-2)*fov*.5), glm::vec3(fov/2,0,0), glm::vec3(0,fov/2,0), ps));
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
//TODO make tds more like composite
    shared_ptr<LatexScene> chaotic = make_shared<LatexScene>(latex_text("Chaotic System"), 1);
    int num_renders = 5;
    tds.inject_audio(FileSegment("This system is so sensitive to initial conditions that it's practically unpredictable, so we call it chaotic."), num_renders);
    for(int i = 0; i < num_renders; i++){
        if(i == num_renders-1) tds.add_surface(Surface(glm::vec3(0, -fov*.2, 0), glm::vec3(fov/2.,0,0), glm::vec3(0,fov/2.,0), chaotic));
        tds.render();
    }
    vector<double> notes2{pow(2, 0/12.), pow(2, 4/12.), pow(2, 7/12.), pow(2, 11/12.), pow(2, 12/12.), };
    double x_separation = fov*1.4;
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
        tds.add_surface(Surface(glm::vec3(x_separation, -fov*.1, (i-2)*fov*.5), glm::vec3(fov/2,0,0), glm::vec3(0,fov/2,0), ps));
    }
    tds.state_manager.macroblock_transition({
        {"x", to_string(x_separation)},
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
    tds.inject_audio_and_render(FileSegment("These pendulums also have slightly different starting positions, but they will _not_ diverge."));
    tds.state_manager.macroblock_transition({
        {"parent_path_opacity", "1"},
        {"qj", "0"},
    });
    tds.inject_audio_and_render(FileSegment("They even trace a repeating pattern,"));
    tds.inject_audio(FileSegment("for which I call this the Pretzel Pendulum."), 3);
    shared_ptr<PngScene> pretzel = make_shared<PngScene>("pretzel");
    tds.render();
    tds.render();
    tds.add_surface(Surface(glm::vec3(x_separation, -fov*.1, 3*fov*.5), glm::vec3(fov/2,0,0), glm::vec3(0,fov/2,0), pretzel));
    tds.render();
    tds.remove_surface(chaotic);
    tds.inject_audio_and_render(SilenceSegment(1));
    tds.state_manager.macroblock_transition({
        {"volume_set1", "1"},
        {"volume_set2", "0"},
        {"x", "0"},
    });
    tds.inject_audio_and_render(FileSegment("A stark contrast with the first ones, which are... making a complete mess."));
    tds.remove_surface(pretzel);
    CompositeScene tds_cs;
    MovingPendulumGridScene mpgs;
    tds_cs.add_scene(&mpgs, "mpgs");
    tds_cs.add_scene(&tds, "tds");
    tds_cs.state_manager.set({
        {"mpgs.opacity", "0.2"},
    });
    mpgs.state_manager.set({
        {"physics_multiplier", "0"},
        {"mode", "3"},
        {"rk4_step_size", "1 30 / .4 *"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 20 /"},
        {"ticks_opacity", "0"},
    });
    int iterations = 400;
    mpgs.state_manager.microblock_transition({
        {"physics_multiplier", to_string(iterations/2)},
    });
    tds_cs.inject_audio_and_render(FileSegment("So... what's the deal?"));
    tds.state_manager.macroblock_transition({
        {"volume_set1", "0.5"},
        {"volume_set2", "0.5"},
        {"x", to_string(x_separation/2)},
        {"fov", to_string(fov/2)},
    });
    tds_cs.inject_audio_and_render(FileSegment("These pendulums follow the same laws of physics."));
    tds_cs.state_manager.macroblock_transition({
        {"mpgs.opacity", "1"},
    });
    tds.state_manager.macroblock_transition({
        {"volume_set1", "0"},
        {"volume_set2", "0"},
        {"z", to_string(start_dist*x_separation/10)},
    });


    mpgs.state_manager.macroblock_transition({
        {"mode", "2"},
    });
    tds_cs.inject_audio_and_render(FileSegment("The only difference is the position from which they started."));
    tds_cs.remove_scene(&mpgs);
    tds_cs.remove_scene(&tds);
    mpgs.state_manager.macroblock_transition({
        {"physics_multiplier", to_string(iterations)},
        {"ticks_opacity", "1"},
        {"theta_or_momentum", "0.5"},
        {"center_x", "1.14159"},
        {"center_y", "1.14159"},
        {"zoom", "1 6.283 /"},
    });
    mpgs.inject_audio_and_render(FileSegment("And behavior as a function of starting position can be graphed,"));
    mpgs.state_manager.macroblock_transition({
        {"theta_or_momentum", "0"},
    });
    mpgs.inject_audio_and_render(FileSegment("revealing fractals like these,"));
    CompositeScene cs;
    vector<PendulumGrid> grids{PendulumGrid(VIDEO_HEIGHT, VIDEO_HEIGHT, -M_PI, M_PI, -M_PI, M_PI, 0, 0, 0, 0)};
    for(PendulumGrid& p : grids) p.iterate_physics(iterations*8, .05/30);
    PendulumGridScene pgs(grids);
    cs.add_scene(&pgs, "pgs");
    pgs.state_manager.set({
        {"mode", "2"},
        {"center_x", "1.14159"},
        {"center_y", "1.14159"},
        {"zoom", "1 6.283 /"},
    });
    cs.inject_audio_and_render(FileSegment("where each point shows how chaotic a certain pendulum is."));
    vector<PendulumState> start_states = {
                                          {1.3, 1.5, 0, 0},
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
    cs.inject_audio(FileSegment("But before diving into the fractals, let's get to know a few particular specimen."), 3);
    vector<PendulumScene> specimens;
    for(int i = 0; i < start_states.size(); i++) {
        specimens.push_back(PendulumScene(start_states[i], 1./3, 1./3));
    }
    for(int i = 0; i < start_states.size(); i++) {
        pgs.state_manager.set({
            {"circle"+to_string(i)+"_x", to_string(start_states[i].theta1)},
            {"circle"+to_string(i)+"_y", to_string(start_states[i].theta2)},
            {"circle"+to_string(i)+"_r", to_string(0.1)},
        });
        pgs.circles_to_render = i+1;
        PendulumScene& ps = specimens[i];
        ps.state_manager.set(state);
        ps.state_manager.set({{"tone", to_string(i/4.+1)}});
        string name = "p" + to_string(i);
//TODO this is bugged
        cs.add_scene_fade_in(&ps, name, VIDEO_HEIGHT*((start_states[i].theta1 + 2)/6.283-.5)/VIDEO_WIDTH+.5, 1-(start_states[i].theta2 + 2)/6.283);
        cs.render();
    }
    cs.state_manager.microblock_transition({
        {"pgs.opacity", "0"},
        {"p0.pointer_opacity", "0"},
        {"p1.pointer_opacity", "0"},
        {"p2.pointer_opacity", "0"},
    });
    cs.state_manager.microblock_transition({
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
    cs.remove_scene(&pgs);
    specimens[0].state_manager.microblock_transition({
        {"top_angle_opacity", "1"},
        {"bottom_angle_opacity", "1"},
    });
    cs.inject_audio_and_render(FileSegment("We are particularly interested in the angles that separate each bar from the vertical."));
    specimens[0].global_publisher_key = true;
    CoordinateSceneWithTrail coord(1, 1);
    coord.state_manager.set({
        {"center_y", "-5"},
    });
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
    cs.add_scene_fade_in(&coord, "coord", 0.5, 0.5);
    cs.inject_audio_and_render(FileSegment("Plotting the top angle as X and the bottom angle as Y, we can make a graph like this."));
    coord.state_manager.microblock_transition({
        {"center_x", "<trail_x>"},
        {"center_y", "<trail_y>"},
    });
    cs.inject_audio_and_render(FileSegment("You can sort of already tell that this pendulum is chaotic from the graph."));
    cs.inject_audio_and_render(FileSegment("There's no consistent pattern that it follows."));
    cs.inject_audio_and_render(SilenceSegment(3));
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
    coord.state_manager.microblock_transition({
        {"zoom", ".2"},
    });
    cs.inject_audio_and_render(SilenceSegment(5));
    cs.inject_audio_and_render(FileSegment("It's drawing a shape known as a Lissajous curve."));
    CoordinateSceneWithTrail left(1, 1);
    left.state_manager.set({
        {"center_x", "<t> 2 /"},
        {"center_y", "0"},
        {"trail_opacity", "1"},
        {"ticks_opacity", "0"},
        {"zoom", ".2"},
        {"trail_x", "<t> 2 / 1 -"},
        {"trail_y", "{pendulum_theta2}"},
    });
    CoordinateSceneWithTrail right(1, 1);
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
    cs.add_scene_fade_in(&left , "left" , 0.5, 0.5);
    cs.add_scene_fade_in(&right, "right", 0.5, 0.5);
    cs.inject_audio_and_render(FileSegment("Here's the signal from the top and bottom angles, separated out."));
    cs.inject_audio_and_render(SilenceSegment(2));
    specimens[2].state_manager.macroblock_transition({
        {"volume", "0"},
    });
    PngScene ear_left("ear_left", .2, .2);
    PngScene ear_right("ear_right", .2, .2);
    cs.add_scene_fade_in(&ear_left , "ear_left" , 0.25, 0.15);
    cs.add_scene_fade_in(&ear_right, "ear_right", 0.75, 0.15);
    left.state_manager.macroblock_transition({
        {"center_x", "<t> 2 / 5 -"},
        {"w", ".5"},
        {"h", "1"},
    });
    right.state_manager.macroblock_transition({
        {"center_y", "<t> 2 / 5 -"},
        {"w", ".5"},
        {"h", "1"},
    });
    cs.state_manager.macroblock_transition({
        {"left.x", "0.25"},
        {"right.x", "0.75"},
        {"left.y", "0.5"},
        {"right.y", "0.5"},
        {"coord.opacity", ".3"},
        {"p2.opacity", ".3"},
    });
    cs.inject_audio_and_render(FileSegment("We can run this pendulum for several hours, re-interpret these signals as sound waves on the left and right speaker, and 'listen' to the pendulum!"));
    vector<float> audio_left;
    vector<float> audio_right;
    coord.state_manager.microblock_transition({
        {"trail_opacity", "0"},
    });
    specimens[2].generate_audio(4, audio_left, audio_right);
    cs.inject_audio_and_render(GeneratedSegment(audio_left, audio_right));
    specimens[0].global_publisher_key = true;
    specimens[2].global_publisher_key = false;
    cs.state_manager.microblock_transition({
        {"left.opacity", "0"},
        {"right.opacity", "0"},
        {"ear_left.opacity", "0"},
        {"ear_right.opacity", "0"},
        {"p2.y", "1.5"},
        {"p0.y", ".5"},
        {"coord.opacity", "1"},
        {"p0.opacity", "1"},
    });
    coord.state_manager.microblock_transition({
        {"zoom", ".05"},
        {"center_x", "<trail_x>"},
        {"center_y", "<trail_y>"},
    });
    coord.state_manager.set({
        {"trail_opacity", "1"},
    });
    cs.inject_audio_and_render(FileSegment("It doesn't precisely sound beautiful, but compare that with the chaotic pendulum!"));
    vector<float> audio_left_c;
    vector<float> audio_right_c;
    specimens[0].generate_audio(2.5, audio_left_c, audio_right_c);
    cs.inject_audio_and_render(GeneratedSegment(audio_left_c, audio_right_c));
    left.state_manager.microblock_transition({
        {"trail_opacity", "0"},
    });
    right.state_manager.microblock_transition({
        {"trail_opacity", "0"},
    });
    coord.state_manager.microblock_transition({
        {"trail_opacity", "0"},
        {"zoom", ".02"},
    });
    cs.state_manager.microblock_transition({
        {"p0.y", "-.5"},
        {"p1.y", ".5"},
    });
    coord.state_manager.microblock_transition({
        {"center_x", "0"},
        {"center_y", "-5"},
    });
    cs.inject_audio_and_render(FileSegment("And here's the pretzel pendulum."));
    coord.state_manager.set({
        {"trail_opacity", "1"},
    });
    specimens[1].global_publisher_key = true;
    specimens[0].global_publisher_key = false;
    cs.inject_audio_and_render(SilenceSegment(1));
    vector<float> audio_left_p;
    vector<float> audio_right_p;
    specimens[1].generate_audio(2.5, audio_left_p, audio_right_p);
    cs.inject_audio_and_render(GeneratedSegment(audio_left_p, audio_right_p));
    coord.state_manager.microblock_transition({
        {"zoom", ".04"},
        {"center_x", "<trail_x>"},
        {"center_y", "<trail_y>"},
    });
    cs.inject_audio_and_render(FileSegment("It traces a repetitive curve in angle-space, so it sounds the cleanest."));
    cs.inject_audio_and_render(FileSegment("Listening to a pendulum tells us if it's chaotic, but we have to listen one-by-one."));



    cs.fade_out_all_scenes();
    cout << "Doing grid()" << endl;
    int gridsize = 13;
    vector<PendulumScene> vps;
    double gridstep = 1./gridsize;
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            double x_mod = x + ((y%2==0) ? 0.75 : 0.25);
            PendulumState pendulum_state = {-(x_mod-gridsize/2)*.2, (y-gridsize/2)*.2, .0, .0};
            PendulumScene ps(pendulum_state, gridstep*2.5, gridstep*2.5);
            StateSet state = {
                {"pendulum_opacity",   "[pendulum_opacity]"  },
                {"background_opacity", "[background_opacity]"},
                {"physics_multiplier", "[physics_multiplier]"},
                {"rk4_step_size",      "[rk4_step_size]"     },
                {"rainbow",            "[rainbow]"           },
                {"manual_mode",        "[manual_mode]"       },
                {"theta1_manual",      "[manual_transition_1] <start_t1> *"},
                {"theta2_manual",      "[manual_transition_2] <start_t2> *"},
                {"start_t1",           to_string(pendulum_state.theta1)},
                {"start_t2",           to_string(pendulum_state.theta2)},
            };
            ps.state_manager.set(state);
            vps.push_back(ps);
        }
    }
    PendulumGrid pg13(gridsize*2, gridsize*2, -.2*gridsize/2, .2*gridsize/2, -.2*gridsize/2, .2*gridsize/2, 0, 0, 0, 0);
    PendulumGridScene pgs13(vector<PendulumGrid>{pg13});
    PendulumGrid pgfull(VIDEO_HEIGHT, VIDEO_HEIGHT, 0, M_PI*2, 0, M_PI*2, 0, 0, 0, 0);
    pgs = PendulumGridScene(vector<PendulumGrid>{pgfull});
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            double x_mod = x + ((y%2==0) ? 0.75 : 0.25);
            string key = "ps" + to_string(x+y*gridsize);
            cs.add_scene_fade_in(&(vps[x+y*gridsize]), key, 0, gridstep*(y+.5));
            cs.state_manager.set({
                {key + ".x", to_string(gridstep*(x_mod)) + " 1 <scrunch> 2 / lerp"},
            });
        }
    }
    cs.state_manager.set({
        {"scrunch", "0"},
        {"pendulum_opacity", "1"},
        {"background_opacity", "0"},
        {"physics_multiplier", "0"},
        {"rk4_step_size", "1 30 / 5 /"},
        {"rainbow", "1"},
        {"manual_mode", "1"},
        {"manual_transition_1", "0"},
        {"manual_transition_2", "0"},
    });
    cs.remove_scene(&ear_left);
    cs.remove_scene(&ear_right);
    cs.remove_scene(&left);
    cs.remove_scene(&right);
    cs.remove_scene(&specimens[0]);
    cs.remove_scene(&specimens[2]);
    int selected_pendulum_x = gridsize*.1;
    int selected_pendulum_y = gridsize*.2;
    int selected_pendulum = gridsize*selected_pendulum_y + selected_pendulum_x;
    string key_str = "ps" + to_string(selected_pendulum);
    cs.inject_audio_and_render(FileSegment("Instead, we can make a large array of pendulums like this."));
    cs.remove_scene(&coord);
    cs.remove_scene(&specimens[1]);
    vps[selected_pendulum].state_manager.set({
        {"pendulum_opacity", "1"},
    });
    vps[selected_pendulum].state_manager.microblock_transition({
        {"w", "1"},
        {"h", "1"},
    });
    cs.state_manager.microblock_transition({
        {key_str + ".x", ".5"},
        {key_str + ".y", ".5"},
        {"pendulum_opacity", "0.6"},
    });
    cs.inject_audio_and_render(SilenceSegment(1));
    vps[selected_pendulum].state_manager.microblock_transition({
        {"top_angle_opacity", "1"},
    });
    cs.state_manager.microblock_transition({
        {"manual_transition_1", "1"},
    });
    cs.inject_audio(FileSegment("The pendulum's x position corresponds to the top angle,"), 2);
    cs.render();
    cs.render();
    vps[selected_pendulum].state_manager.microblock_transition({
        {"top_angle_opacity", "0"},
        {"bottom_angle_opacity", "1"},
    });
    cs.state_manager.microblock_transition({
        {"manual_transition_2", "1"},
    });
    cs.inject_audio_and_render(FileSegment("and its y position corresponds to the bottom angle."));
    string size_str = to_string(gridstep*2.5);
    double selected_x_mod = selected_pendulum_x + ((selected_pendulum_y%2==0) ? 0.75 : 0.25);
    cs.state_manager.microblock_transition({
        {"rainbow", "1"},
        {key_str + ".x", to_string(gridstep*(selected_x_mod)) + " 1 <scrunch> 2 / lerp"},
        {key_str + ".y", to_string(gridstep*(selected_pendulum_y+.5))},
        {"pendulum_opacity", "1"},
    });
    vps[selected_pendulum].state_manager.microblock_transition({
        {"w", size_str},
        {"h", size_str},
        {"bottom_angle_opacity", "0"},
    });
    cs.inject_audio_and_render(FileSegment("By associating angle positions with colors, it's easier to tell what is going on."));
    PendulumGrid pointgrid(100, 100, -M_PI, M_PI, -M_PI, M_PI, 0, 0, 0, 0);
    PendulumPointsScene pps(pointgrid, 0.5, 1);
    vps[selected_pendulum].state_manager.set({
        {"pendulum_opacity", "[pendulum_opacity]"},
    });
    pps.state_manager.set({
        {"physics_multiplier", "0"},
        {"rk4_step_size", "1 30 / 5 /"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 6.283 /"},
    });
    cs.add_scene(&pps, "pps", -0.25, 0.5);
    pgs.state_manager.set({
        {"physics_multiplier", "0"},
        {"rk4_step_size", "1 30 / 5 /"},
    });
    string zoomval = "5 " + to_string(gridsize) + " /";
    pgs13.state_manager.set({
        {"physics_multiplier", "0"},
        {"zoom", zoomval},
        {"rk4_step_size", "1 30 / 5 /"},
    });
    cs.add_scene(&pgs, "pgs");
    cs.add_scene(&pgs13, "pgs13");
    cs.state_manager.set({
        {"pgs13.opacity", "0"},
        {"pgs.opacity", "0"},
    });
    cs.state_manager.macroblock_transition({
        {"scrunch", "1"},
        {"pps.x", ".25"},
        {"manual_mode", "0"},
    });
    cs.inject_audio_and_render(FileSegment("As a bonus, we can add points in angle space for these pendulums and see how they move."));
    cs.state_manager.set({
        {"physics_multiplier", "5"},
    });
    pps.state_manager.set({
        {"physics_multiplier", "5"},
    });
    pgs.state_manager.set({
        {"physics_multiplier", "5"},
    });
    pgs13.state_manager.set({
        {"physics_multiplier", "5"},
    });
    cs.inject_audio_and_render(SilenceSegment(10));
    cs.state_manager.macroblock_transition({
        {"scrunch", "0"},
        {"pps.x", "-.25"},
    });
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.state_manager.microblock_transition({
        {"pgs13.opacity", "0.5"},
    });
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.inject_audio_and_render(SilenceSegment(2));



    pgs.state_manager.set({
        {"zoom", zoomval},
    });
    cs.state_manager.microblock_transition({
        {"pgs.opacity", "1"},
        {"pgs13.opacity", "0"},
        {"pendulum_opacity", "0"},
    });
    cs.inject_audio_and_render(FileSegment("Let's up the resolution to one pendulum per pixel."));
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            cs.remove_scene(&(vps[x+y*gridsize]));
        }
    }
    cs.remove_scene(&pps);
    cs.remove_scene(&pgs13);




    pgs.state_manager.microblock_transition({
        {"zoom", "1 8 /"},
    });
    cs.inject_audio_and_render(SilenceSegment(4));
    PendulumState pendulum_state = {5, 7, .0, .0};
    PendulumScene pend(pendulum_state);
    pend.state_manager.set({
        {"manual_mode", "1"},
        {"rk4_step_size", "1"},
        {"physics_multiplier", "0"},
        {"theta1_manual", "5"},
        {"theta2_manual", "7"},
    });
    cs.add_scene_fade_in(&pend, "pend");
    cs.state_manager.microblock_transition({
        {"pgs.opacity", "0.4"},
    });
    
    cs.inject_audio_and_render(FileSegment("A nice feature of this fractal is that it tiles the plane."));
    cs.inject_audio(FileSegment("Rotating either angle by 2pi yields the exact same pendulum, so the pattern is periodic."), 3);
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
        {"theta2_manual", "7 6.283 +"},
    });
    cs.render();
    cs.state_manager.microblock_transition({
        {"pgs.opacity", "1"},
        {"pend.opacity", "0"},
    });
    pgs.state_manager.microblock_transition({
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 15 /"},
    });
    pgs.state_manager.set({
        {"physics_multiplier", "5"},
    });
    cs.render();
    cs.remove_scene(&pgs);
    pgs.state_manager.microblock_transition({
        {"zoom", "1 3.1415 /"},
    });
    pgs.inject_audio_and_render(FileSegment("Pay attention to how there are two distinct modes of behavior here."));
    pgs.inject_audio(FileSegment("There's a region of chaotic pendulums sensitive to their initial conditions,"), 2);
    pgs.state_manager.microblock_transition({
        {"center_x", "3.1415"},
        {"center_y", "3.1415"},
    });
    pgs.render();
    pgs.render();
    pgs.inject_audio(FileSegment("as well as a region of coherent ones."), 2);
    pgs.state_manager.microblock_transition({
        {"center_x", "0"},
        {"center_y", "0"},
    });
    pgs.render();
    pgs.render();
    pgs.state_manager.microblock_transition({
        {"mode", "2.5"},
    });
    pgs.inject_audio_and_render(FileSegment("Now, for each pixel, we track two pendulums, separated by a slight starting difference, and plot their difference over time."));
    pgs.state_manager.microblock_transition({
        {"center_x", "3.1415"},
        {"center_y", "3.1415"},
        {"zoom", "1 8 /"},
    });
    pgs.inject_audio_and_render(FileSegment("So, this plot shows how quickly the pendulums in our grid diverge to chaos."));
    pgs.state_manager.microblock_transition({
        {"mode", "2"},
        {"contrast", ".000005"},
    });
    pgs.inject_audio_and_render(FileSegment("Let's reset the pendulums and watch that again from the start."));
}

void identify_vibrations(double t1, double t2) {
    CompositeScene cs;
    vector<PendulumState> start_states = {
                                          {t1, t2, 0, 0},
                                          {t1, t2, 0, 0},
                                         };
    double step_sz = 1./30/300;
    double period = 0;
    PendulumState periodfinder = start_states[1];
    double minsquare = 100;
    for(int i = 0; i < 7/step_sz; i++) {
        double p1 = periodfinder.p1;
        double p2 = periodfinder.p2;
        double squaresum = square(p1) + square(p2);
        if(squaresum < minsquare && i*step_sz > 6) {
            period = i*step_sz;
            minsquare = squaresum;
        }
        periodfinder = rk4Step(periodfinder, step_sz);
    }
    for(int i = 0; i < period/step_sz; i++) {
        start_states[1] = rk4Step(start_states[1], step_sz);
    }
    for(int i = 0; i < period/step_sz/4.; i++) {
        start_states[0] = rk4Step(start_states[0], step_sz);
        start_states[1] = rk4Step(start_states[1], step_sz);
    }
    vector<PendulumScene> specimens;
    for(int i = 0; i < start_states.size(); i++) {
        specimens.push_back(PendulumScene(start_states[i], .5, .5));
    }
    double anim_step = 1800;
    for(int i = 0; i < start_states.size(); i++) {
        PendulumScene& ps = specimens[i];
        ps.state_manager.set({
            {"rk4_step_size", "1 "+to_string(anim_step)+" /"},
            {"physics_multiplier", to_string(anim_step/30)},
            //one frame per cycle{"physics_multiplier", to_string(anim_step*period)},
        });
        string name = "p" + to_string(i);
        cs.add_scene(&ps, name, .75, .25+.5*i);
    }
    specimens[0].global_publisher_key = true;
    specimens[0].global_identifier = "p0.";
    specimens[1].global_publisher_key = true;
    specimens[1].global_identifier = "p1.";
    CoordinateSceneWithTrail coord(.5, 1);
    coord.state_manager.set({
        {"zoom", "0.05"},
        //{"zoom", "6000"},
        {"trail_opacity", "1"},
        {"trail_x", "{p1.pendulum_p1}"},
        {"trail_y", "{p1.pendulum_p2}"},
        //{"trail_x", "{p1.pendulum_theta1} {p0.pendulum_theta1} -"},
        //{"trail_y", "{p1.pendulum_theta2} {p0.pendulum_theta2} -"},
    });
    cs.add_scene(&coord, "coord", 0.25, 0.5);
    string str_cx = to_string(t1);
    string str_cy = to_string(t2);
    str_cx = str_cx.erase(str_cx.find_last_not_of('0') + 1);
    str_cy = str_cy.erase(str_cy.find_last_not_of('0') + 1);
    string latex_str = "\\theta_1 = " + str_cx + ", \\theta_2 = " + str_cy + ", \\pi = " + to_string(period);
    LatexScene ls(latex_str, 1, 1, 0.12);
    cs.add_scene(&ls, "ls", 0.5, 0.1);
    cs.inject_audio_and_render(SilenceSegment(7));
}
void sample_vibrations(){
    for(int x = -2; x <= 2; x++) for(int y = -2; y <= 2; y++) identify_vibrations(2.496147+.0000004*x, .2505+.000008*y);
}

void render_video() {
    SAVE_FRAME_PNGS = false;
    //PRINT_TO_TERMINAL = false;
    //FOR_REAL = false;


    intro();
    vector<PendulumGrid> grids{PendulumGrid(VIDEO_HEIGHT, VIDEO_HEIGHT, -M_PI, M_PI, -M_PI, M_PI, 0, 0, 0, 0)};
    for (const vector<IslandShowcase>& isvh : {isv, isv2}) for(const IslandShowcase& is : isvh) {
        const double ro2 = is.range/2;
        const double t1 = is.ps.theta1;
        const double t2 = is.ps.theta2;
        grids.push_back(PendulumGrid(VIDEO_WIDTH, VIDEO_HEIGHT, t1-ro2*VIDEO_WIDTH/VIDEO_HEIGHT, t1+ro2*VIDEO_WIDTH/VIDEO_HEIGHT, t2-ro2, t2+ro2, 0, 0, 0, 0));
    }
    if(!FOR_REAL) for(PendulumGrid& p : grids) p.iterate_physics(50, .2/30);
    PendulumGridScene pgs(grids);
    fine_grid(pgs);
    showcase_islands(pgs);
    discuss_energy(pgs);
    move_fractal(pgs);
    outtro();
}
