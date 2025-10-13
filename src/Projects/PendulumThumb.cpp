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
#include "../Scenes/Media/Mp4Scene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"

void render_video(){
    perp->stage_macroblock(FileSegment("But these ultra-high-energy pendulums don't nicely fit into one box or another."), 1);
    perp->render_microblock();
    perp->state.microblock_transition({
        {"energy_min", to_string(40000*vert_energy)},
        {"energy_max", "0"},
    });
    showcase_momentum_space(perp, 10, 0, 0.2);
    showcase_momentum_space(perp, 0, 15, 0.6);
    showcase_momentum_space(perp, 10, 10, 0.3);
    showcase_momentum_space(perp, -10, 10, 0.7);
    showcase_momentum_space(perp, 15, 10, 0.4);
    CompositeScene cs_spiro;
    cs_spiro.add_scene(perp, "perp");
    int gridsize = 10;
    vector<shared_ptr<PendulumScene> > vps;
    double gridstep = 1./gridsize;
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            PendulumState pendulum_state = {.0, .0, ((x+.5)/gridsize - .5) * 40*VIDEO_WIDTH/VIDEO_HEIGHT, -((y+.5)/gridsize - .5) * 40};
            shared_ptr<PendulumScene> ps = make_shared<PendulumScene>(pendulum_state, gridstep, gridstep);
            ps->state.set({
                {"pendulum_opacity", "0"},
                {"path_opacity", "1"},
            });
            vps.push_back(ps);
        }
    }
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            string key = "ps" + to_string(x+y*gridsize);
            cs_spiro.add_scene((vps[x+y*gridsize]), key, gridstep*(x+.5), gridstep*(y+.5));
            cs_spiro.state.set({
                {key+".opacity", "<spiro_opacity>"},
            });
        }
    }
    cs_spiro.state.set({
        {"spiro_opacity", "0"},
    });
    cs_spiro.state.microblock_transition({
        {"perp.opacity", "0.3"},
        {"spiro_opacity", "1"},
    });
    perp->state.microblock_transition({
        {"ticks_opacity", "0"},
    });
    cs_spiro.stage_macroblock(FileSegment("Here's some spirographs from all over momentum-space."), 1);
    cs_spiro.render_microblock();
    cs_spiro.stage_macroblock(SilenceSegment(11), 1);
    cs_spiro.render_microblock();
    cs_spiro.state.microblock_transition({
        {"spiro_opacity", "0"},
    });
    cs_spiro.stage_macroblock(SilenceSegment(1), 1);
    cs_spiro.render_microblock();
    cs_spiro.remove_subscene("perp");
    cs.add_scene(perp, "perp");
    cs.state.set({
        {"perp.opacity", "0.4"},
    });
    cs.state.microblock_transition({
        {"mpgs.opacity", "1"},
        {"perp.opacity", "0"},
    });
    mpgs->state.set({
        {"mode", "3"},
    });
    cs.stage_macroblock(SilenceSegment(1), 1);
    cs.render_microblock();
    cs.remove_subscene("perp");
    mpgs->state.microblock_transition({
        {"zoomexp", "1 6.283 / log"},
        {"theta_or_momentum", "0"},
    });
    cs.stage_macroblock(SilenceSegment(2), 1);
    cs.render_microblock();
    cs.add_scene_fade_in(pgs, "pgs");
    pgs->circles_to_render = 0;
    pgs->state.set({
        {"energy_max", "0"},
    });
    cs.state.microblock_transition({
        {"mpgs.opacity", "0"},
    });
    cs.remove_subscene("pgs");
}

    pgs->state.microblock_transition({
        {"physics_multiplier", "30"},
    });
    pgs->stage_macroblock(SilenceSegment(7), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"mode", "2"},
        {"zoom", "1 6.283 /"},
        {"center_x", "0"},
        {"center_y", "0"},
    });
    pgs->stage_macroblock(SilenceSegment(2), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"physics_multiplier", "80"},
        {"trail_opacity", "1"},
    });
    pgs->stage_macroblock(FileSegment("Now, I'm gonna pick a certain point, corresponding to a pendulum in the black region, meaning its behavior is non-chaotic."), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"trail_length", "1200"},
        {"zoom", "1 4 /"},
    });
    pgs->stage_macroblock(FileSegment("We can plot its path in angle-space just like we did before!"), 1);
    pgs->render_microblock();
    pgs->state.set({
        {"physics_multiplier", "0"},
    });
    pgs->state.microblock_transition({
        {"trail_start_x", "0.25"},
        {"trail_start_y", "0.5"},
        {"mode", "1.5"},
    });
    pgs->stage_macroblock(FileSegment("Moving the point around in the black region, this curve moves smoothly and cleanly."), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"trail_start_x", "1"},
        {"trail_start_y", "1"},
    });
    pgs->stage_macroblock(SilenceSegment(2), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"trail_start_x", ".5"},
        {"trail_start_y", "0"},
    });
    pgs->stage_macroblock(SilenceSegment(2), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"trail_start_x", ".5"},
        {"trail_start_y", "-.5"},
    });
    pgs->stage_macroblock(SilenceSegment(2), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"trail_start_x", ".5"},
        {"trail_start_y", "1.1"},
    });
    pgs->stage_macroblock(SilenceSegment(2), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"trail_start_x", "1.5"},
        {"trail_start_y", "1.1"},
        {"mode", "2"},
    });
    pgs->stage_macroblock(SilenceSegment(2), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"trail_start_x", "1"},
        {"trail_start_y", "1.1"},
    });
    pgs->stage_macroblock(FileSegment("This main black region is home to all the lissajous style pendulums."), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"trail_start_x", "1.6"},
        {"trail_start_y", "1.9"},
        {"zoom", "0.05"},
    });
    pgs->stage_macroblock(FileSegment("But as soon as you leave and step into the chaotic region..."), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"trail_start_x", "1.7 {t} 2 * sin 4 / +"},
        {"trail_start_y", "1.9 {t} 2 * cos 4 / +"},
    });
    pgs->stage_macroblock(SilenceSegment(2), 1);
    pgs->render_microblock();
    pgs->stage_macroblock(FileSegment("All hell breaks loose."), 1);
    pgs->render_microblock();
    pgs->stage_macroblock(SilenceSegment(2), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"center_x", "3.1415"},
        {"center_y", "-1.8"},
        {"trail_opacity", "0"},
        {"zoom", "1 6.283 /"},
    });
    pgs->stage_macroblock(FileSegment("Let's look closer at the chaotic region in white."), 1);
    pgs->render_microblock();
    StateSet circles;
    for(int i = 0; i < isv.size(); i++){
        IslandShowcase is = isv[i];
        circles.insert(make_pair("circle" + to_string(i) + "_x", to_string(is.ps.theta1)));
        circles.insert(make_pair("circle" + to_string(i) + "_y", to_string(is.ps.theta2)));
        circles.insert(make_pair("circle" + to_string(i) + "_r", to_string(max(is.range/2, .1))));
    }
    pgs->state.set(circles);
    pgs->state.microblock_transition({
        {"circles_opacity", "1"},
    });
    pgs->circles_to_render = isv.size();
    pgs->state.microblock_transition({
        {"ticks_opacity", "0"},
    });
    pgs->stage_macroblock(FileSegment("There are a few spots of black in here."), 1);
    pgs->render_microblock();
}

void outtro(){
    vector<PendulumGrid> grids{PendulumGrid(VIDEO_HEIGHT, VIDEO_HEIGHT, 0.0001, -M_PI, M_PI, -M_PI, M_PI, 0, 0, 0, 0)};
    for(PendulumGrid& p : grids) p.iterate_physics(300*8, .05/30);
    shared_ptr<PendulumGridScene> pgs = make_shared<PendulumGridScene>(grids);
    pgs->state.set({
        {"rk4_step_size", ".05 30 /"},
        {"mode", "3"},
    });
    pgs->state.microblock_transition({
        {"physics_multiplier", "10"},
        {"mode", "0"},
    });
    CompositeScene cs;
    cs.add_scene(pgs, "pgs");
    cs.stage_macroblock(FileSegment("Chaos is a hard thing to define."), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.state.microblock_transition({{"pgs.opacity", "0.3"}});
    cs.stage_macroblock(FileSegment("You may have even noticed that I subtly changed definitions throughout the video: aperiodicity, unpredictibility, divergence..."), 1);
    cs.render_microblock();

    shared_ptr<LatexScene> ls1 = make_shared<LatexScene>("\\Delta = \\theta_1 - \\theta_2", 1, 1, 0.1);
    shared_ptr<LatexScene> ls2 = make_shared<LatexScene>("D = 16 - 9\\cos^2\\Delta", 1, 1, 0.1);
    shared_ptr<LatexScene> ls3 = make_shared<LatexScene>("\\dot{\\theta}_1 = \\frac{6}{m l^2 D} (2 p_1 - 3 \\cos\\Delta \\, p_2)", 1, 1, 0.1);
    shared_ptr<LatexScene> ls4 = make_shared<LatexScene>("\\dot{\\theta}_2 = \\frac{6}{m l^2 D} (8 p_2 - 3 \\cos\\Delta \\, p_1)", 1, 1, 0.1);
    shared_ptr<LatexScene> ls5 = make_shared<LatexScene>("\\dot{p}_1 = -\\frac{1}{2} m l \\left( 3 g \\sin\\theta_1 + l \\dot{\\theta}_1 \\dot{\\theta}_2 \\sin\\Delta \\right)", 1, 1, 0.1);
    shared_ptr<LatexScene> ls6 = make_shared<LatexScene>("\\dot{p}_2 = -\\frac{1}{2} m l \\left( g \\sin\\theta_2 - l \\dot{\\theta}_1 \\dot{\\theta}_2 \\sin\\Delta \\right)", 1, 1, 0.1);
    shared_ptr<LatexScene> ls7 = make_shared<LatexScene>(latex_text("...have no analytic solution."), 1, 1, 0.1);
    cs.stage_macroblock(FileSegment("These definitions usually match up... but, the differential equations describing double pendulums have no analytic solution for _any_ nontrivial starting position,"), 11);
    cs.add_scene_fade_in(ls1, "ls1", .5, .2);
    cs.render_microblock();
    cs.add_scene_fade_in(ls2, "ls2", .5, .3);
    cs.render_microblock();
    cs.add_scene_fade_in(ls3, "ls3", .5, .4);
    cs.render_microblock();
    cs.add_scene_fade_in(ls4, "ls4", .5, .5);
    cs.render_microblock();
    cs.add_scene_fade_in(ls5, "ls5", .5, .6);
    cs.render_microblock();
    cs.add_scene_fade_in(ls6, "ls6", .5, .7);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.add_scene_fade_in(ls7, "ls7", .5, .8);
    cs.render_microblock();
    cs.render_microblock();
    cs.state.microblock_transition({
        {"ls1.opacity", "0"},
        {"ls2.opacity", "0"},
        {"ls3.opacity", "0"},
        {"ls4.opacity", "0"},
        {"ls5.opacity", "0"},
        {"ls6.opacity", "0"},
        {"ls7.opacity", "0"},
    });
    cs.stage_macroblock(FileSegment("but even so,"), 1);
    cs.render_microblock();
    shared_ptr<PendulumScene> ps1 = make_shared<PendulumScene>(PendulumState{2.49, .25, .0, .0}, .3, .3);
    shared_ptr<PendulumScene> ps2 = make_shared<PendulumScene>(PendulumState{1.5 , 1  , 8, 5}, .3, .3);
    shared_ptr<PendulumScene> ps3 = make_shared<PendulumScene>(PendulumState{.0  , .0 ,10.,-15.}, .3, .3);
    StateSet ss = {
        {"rainbow", "0"},
        {"path_opacity", "1"},
    };
    ps1->state.set(ss);
    ps2->state.set(ss);
    ps3->state.set(ss);
    cs.stage_macroblock(FileSegment("this system is sufficiently fertile to support little gems of order in the rough."), 9);
    cs.add_scene_fade_in(ps1, "ps1", .5, .5);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.add_scene_fade_in(ps2, "ps2", .75, .5);
    cs.render_microblock();
    cs.render_microblock();
    cs.state.microblock_transition({
        {"ps1.opacity", "0"},
    });
    cs.render_microblock();
    cs.add_scene_fade_in(ps3, "ps3", .25, .5);
    cs.render_microblock();
    cs.render_microblock();
    cs.state.microblock_transition({
        {"ps2.opacity", "0"},
    });
    cs.render_microblock();
    cs.stage_macroblock(SilenceSegment(1), 1);
    cs.render_microblock();
    shared_ptr<TwoswapScene> ts = make_shared<TwoswapScene>();
    ts->state.set({{"circle_opacity", "1"}});
    cs.state.microblock_transition({{"pgs.opacity", "0.0"}});
    cs.add_scene_fade_in(ts, "ts");
    cs.stage_macroblock(FileSegment("This has been 2swap."), 1);
    cs.render_microblock();
    cs.state.set({
        {"ps3.opacity", "0"},
    });
    shared_ptr<PngScene> note = make_shared<PngScene>("note", 0.18, 0.18);
    shared_ptr<LatexScene> seef = make_shared<LatexScene>(latex_text("6884"), 1, .6, .27);
    cs.add_scene_fade_in(seef, "seef", 0.6, 0.73);
    cs.add_scene_fade_in(note, "note", 0.44, 0.73);
    cs.stage_macroblock(FileSegment("with music by 6884"), 1);
    cs.render_microblock();
    cs.fade_all_subscenes(0);
}

void intro() {
    cout << "Doing intro()" << endl;
    const double fov = 12;
    const double start_dist = 15*fov;
    const double after_move = start_dist-3;
    //Mp4Scene mp4s("pendulum");
    CompositeScene cs_mp4;
    shared_ptr<ThreeDimensionScene> tds = make_shared<ThreeDimensionScene>();
    //cs_mp4.add_scene(mp4s, "mp4s");
    cs_mp4.add_scene(tds, "tds");
    vector<double> notes{pow(2, 3/12.), pow(2, 8/12.), pow(2, 10/12.), pow(2, 15/12.), pow(2, 20/12.), };
    for(int i = 0; i < 5; i++){
        PendulumState pendulum_state = {5+.0000001*i, 8, .0, .0};
        shared_ptr<PendulumScene> ps = make_shared<PendulumScene>(pendulum_state);
        ps->state.set({
            {"path_opacity", "[parent_path_opacity]"},
            {"background_opacity", "0"},
            {"tone", to_string(notes[i])},
            {"volume", "[volume_set1]"},
            {"pendulum_opacity", "1"},
            {"physics_multiplier", "30"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        tds->add_surface(Surface(glm::dvec3(0,-fov*.1,(i-2)*fov*.5), glm::dvec3(fov/2,0,0), glm::dvec3(0,fov/2,0), "3dpend" + to_string(i)), ps);
    }
    vector<shared_ptr<LatexScene>> ls;
    ls.push_back(make_shared<LatexScene>(latex_text("Double"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("Pendulums"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("are"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("NOT"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("Chaotic"), 1));
    tds->state.set({
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
    cs_mp4.stage_macroblock(FileSegment("Double pendulums are NOT chaotic."), 13);
    cs_mp4.render_microblock();
    cs_mp4.render_microblock();
    cs_mp4.render_microblock();
    cs_mp4.render_microblock();
    tds->add_surface(Surface(glm::dvec3(0,-.24/fov,-after_move-.8), glm::dvec3(word_size,0,0), glm::dvec3(0,word_size,0), "Double"), ls[0]);
    cs_mp4.render_microblock();
    tds->add_surface(Surface(glm::dvec3(0,-.1/fov,-after_move-.5), glm::dvec3(word_size,0,0), glm::dvec3(0,word_size,0), "Pendulums"), ls[1]);
    cs_mp4.render_microblock();
    cs_mp4.render_microblock();
    tds->add_surface(Surface(glm::dvec3(0,.04/fov,-after_move-.4), glm::dvec3(.35*word_size,0,0), glm::dvec3(0,.35*word_size,0), "are"), ls[2]);
    cs_mp4.render_microblock();
    tds->add_surface(Surface(glm::dvec3(0,.18/fov,-after_move-.7), glm::dvec3(word_size,0,0), glm::dvec3(0,word_size,0), "NOT"), ls[3]);
    cs_mp4.render_microblock();
    tds->add_surface(Surface(glm::dvec3(0,0.36/fov,-after_move-.7), glm::dvec3(word_size,0,0), glm::dvec3(0,word_size,0), "Chaotic"), ls[4]);
    cs_mp4.render_microblock();
    cs_mp4.render_microblock();
    cs_mp4.render_microblock();
    cs_mp4.render_microblock();
    ls[3]->begin_latex_transition(latex_text("NOT")+"^*");
    cs_mp4.state.microblock_transition({
        //{"mp4s.opacity", "0"},
    });
    cs_mp4.stage_macroblock(FileSegment("Or, at least, not all of them."), 1);
    cs_mp4.render_microblock();
    tds->state.macroblock_transition({
        {"d", to_string(after_move)},
    });
    cs_mp4.remove_subscene("tds");
    tds->stage_macroblock(FileSegment("You've probably seen videos like these,"), 1);
    tds->render_microblock();
    tds->remove_surface("Double");
    tds->remove_surface("Pendulums");
    tds->remove_surface("are");
    tds->remove_surface("NOT");
    tds->remove_surface("Chaotic");
    tds->state.macroblock_transition({
        {"qj", ".1"},
    });
    tds->stage_macroblock(FileSegment("where a tiny deviation in similar double pendulums amplifies over time,"), 1);
    tds->render_microblock();
    tds->stage_macroblock(FileSegment("until they completely desynchronize."), 1);
    tds->render_microblock();
    shared_ptr<LatexScene> chaotic = make_shared<LatexScene>(latex_text("Chaotic System"), 1);
    int num_renders = 5;
    tds->stage_macroblock(FileSegment("This system is so sensitive to initial conditions that it's practically unpredictable, so we call it chaotic."), num_renders);
    for(int i = 0; i < num_renders; i++){
        if(i == num_renders-1) tds->add_surface(Surface(glm::dvec3(0, -fov*.2, 0), glm::dvec3(fov/2.,0,0), glm::dvec3(0,fov/2.,0), "chaotic" + to_string(i)), chaotic);
        tds->render_microblock();
    }
    vector<double> notes2{pow(2, 0/12.), pow(2, 4/12.), pow(2, 7/12.), pow(2, 11/12.), pow(2, 12/12.), };
    double x_separation = fov*1.4;
    for(int i = 0; i < 5; i++){
        PendulumState pendulum_state = {2.49+.0001*i, .25, .0, .0};
        shared_ptr<PendulumScene> ps = make_shared<PendulumScene>(pendulum_state);
        ps->state.set({
            {"path_opacity", "[parent_path_opacity]"},
            {"background_opacity", "0"},
            {"volume", "[volume_set2]"},
            {"tone", to_string(notes2[i])},
            {"pendulum_opacity", "0"},
            {"physics_multiplier", "[stable_physics_multiplier]"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        ps->state.macroblock_transition({
            {"pendulum_opacity", "1"},
        });
        tds->add_surface(Surface(glm::dvec3(x_separation, -fov*.1, (i-2)*fov*.5), glm::dvec3(fov/2,0,0), glm::dvec3(0,fov/2,0), "ps" + to_string(i)), ps);
    }
    tds->state.macroblock_transition({
        {"x", to_string(x_separation)},
        {"volume_set1", "0"},
        {"volume_set2", "1"},
    });
    tds->state.set({
        {"stable_physics_multiplier", "0"},
    });
    tds->stage_macroblock(FileSegment("But you probably haven't seen this:"), 1);
    tds->render_microblock();
    tds->state.set({
        {"stable_physics_multiplier", "30"},
    });
    tds->stage_macroblock(FileSegment("These pendulums also have slightly different starting positions, but they will _not_ diverge."), 1);
    tds->render_microblock();
    tds->state.macroblock_transition({
        {"parent_path_opacity", "1"},
        {"qj", "0"},
    });
    tds->stage_macroblock(FileSegment("They even trace a repeating pattern,"), 1);
    tds->render_microblock();
    tds->stage_macroblock(FileSegment("for which I call this the Pretzel Pendulum."), 3);
    shared_ptr<PngScene> pretzel = make_shared<PngScene>("pretzel");
    tds->render_microblock();
    tds->render_microblock();
    tds->add_surface(Surface(glm::dvec3(x_separation, -fov*.1, 3*fov*.5), glm::dvec3(fov/2,0,0), glm::dvec3(0,fov/2,0), "pretzel"), pretzel);
    tds->render_microblock();
    for(int i = 0; i < 5; i++) tds->remove_surface("chaotic" + to_string(i));
    tds->stage_macroblock(SilenceSegment(1), 1);
    tds->render_microblock();
    tds->state.macroblock_transition({
        {"volume_set1", "1"},
        {"volume_set2", "0"},
        {"x", "0"},
    });
    tds->stage_macroblock(FileSegment("A stark contrast with the first ones, which are... making a complete mess."), 1);
    tds->render_microblock();
    tds->remove_surface("pretzel");
    CompositeScene tds_cs;
    shared_ptr<MovingPendulumGridScene> mpgs = make_shared<MovingPendulumGridScene>();
    tds_cs.add_scene(mpgs, "mpgs");
    tds_cs.add_scene(tds, "tds");
    tds_cs.state.set({
        {"mpgs.opacity", "0.2"},
    });
    mpgs->state.set({
        {"physics_multiplier", "0"},
        {"mode", "3"},
        {"rk4_step_size", "1 30 / .4 *"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 20 /"},
        {"ticks_opacity", "0"},
    });
    int iterations = 400;
    mpgs->state.microblock_transition({
        {"physics_multiplier", to_string(iterations)},
    });
    tds_cs.stage_macroblock(FileSegment("So... what's the deal?"), 1);
    tds_cs.render_microblock();
    tds_cs.stage_macroblock(SilenceSegment(0.5), 1);
    tds_cs.render_microblock();
    tds->state.macroblock_transition({
        {"volume_set1", "0.5"},
        {"volume_set2", "0.5"},
        {"x", to_string(x_separation/2)},
        {"fov", to_string(fov/2)},
    });
    tds_cs.stage_macroblock(FileSegment("These pendulums follow the same laws of physics."), 1);
    tds_cs.render_microblock();
    tds_cs.state.macroblock_transition({
        {"mpgs.opacity", "1"},
    });
    tds->state.macroblock_transition({
        {"volume_set1", "0"},
        {"volume_set2", "0"},
        {"z", to_string(start_dist*x_separation/10)},
    });


    tds_cs.stage_macroblock(FileSegment("The only difference is the position from which they started."), 1);
    tds_cs.render_microblock();
    tds_cs.remove_subscene("mpgs");
    tds_cs.remove_subscene("tds");
    mpgs->stage_macroblock(SilenceSegment(0.5), 1);
    mpgs->render_microblock();
    mpgs->state.macroblock_transition({
        {"ticks_opacity", "1"},
        {"theta_or_momentum", "1"},
        {"center_x", "1.14159"},
        {"center_y", "1.14159"},
        {"zoom", "1 15 /"},
    });
    mpgs->stage_macroblock(FileSegment("And behavior as a function of starting position can be graphed,"), 1);
    mpgs->render_microblock();
    mpgs->state.macroblock_transition({
        {"theta_or_momentum", "0.5"},
    });
    mpgs->stage_macroblock(FileSegment("revealing fractals like these,"), 1);
    mpgs->render_microblock();
    mpgs->state.macroblock_transition({
        {"theta_or_momentum", "0"},
        {"zoom", "1 6.283 /"},
    });
    mpgs->stage_macroblock(FileSegment("where each point shows how chaotic a certain pendulum is."), 1);
    mpgs->render_microblock();
    CompositeScene cs;
    vector<PendulumGrid> grids{PendulumGrid(VIDEO_HEIGHT, VIDEO_HEIGHT, 0.0001, -M_PI, M_PI, -M_PI, M_PI, 0, 0, 0, 0)};
    for(PendulumGrid& p : grids) p.iterate_physics(iterations*8, .05/30);
    shared_ptr<PendulumGridScene> pgs = make_shared<PendulumGridScene>(grids);
    cs.add_scene(pgs, "pgs");
    pgs->state.set({
        {"mode", "3"},
        {"center_x", "1.14159"},
        {"center_y", "1.14159"},
        {"zoom", "1 6.283 /"},
    });
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
    pgs->state.microblock_transition({
        {"circles_opacity", "1"},
    });
    cs.state.microblock_transition({
        {"pgs.opacity", "1"},
    });
    cs.stage_macroblock(FileSegment("But before diving into the fractals, let's get to know a few individual specimen."), 3);
    vector<shared_ptr<PendulumScene>> specimens;
    for(int i = 0; i < start_states.size(); i++) {
        specimens.push_back(make_shared<PendulumScene>(start_states[i], 1./3, 1./3));
    }
    for(int i = 0; i < start_states.size(); i++) {
        pgs->state.set({
            {"circle"+to_string(i)+"_x", to_string(start_states[i].theta1)},
            {"circle"+to_string(i)+"_y", to_string(start_states[i].theta2)},
            {"circle"+to_string(i)+"_r", to_string(0.1)},
        });
        pgs->circles_to_render = i+1;
        shared_ptr<PendulumScene> ps = specimens[i];
        ps->state.set(state);
        ps->state.set({{"tone", to_string(i/4.+1)}});
        string name = "p" + to_string(i);
        cs.add_scene_fade_in(ps, name, VIDEO_HEIGHT*((start_states[i].theta1 + 2)/6.283-.5)/VIDEO_WIDTH+.5, 1-(start_states[i].theta2 + 2)/6.283);
        cs.render_microblock();
    }
    cs.stage_macroblock(SilenceSegment(0.5), 1);
    cs.render_microblock();
    cs.state.microblock_transition({
        {"pgs.opacity", "0"},
        {"p0.pointer_opacity", "0"},
        {"p1.pointer_opacity", "0"},
        {"p2.pointer_opacity", "0"},
    });
    cs.state.microblock_transition({
        {"p0.x", ".5"},
        {"p0.y", ".5"},
        {"p1.y", "1.5"},
        {"p2.y", "1.5"},
    });
    specimens[0]->state.microblock_transition({
        {"w", ".5"},
        {"h", "1"},
        {"volume", "1"},
    });
    cs.stage_macroblock(FileSegment("This pendulum is one of the chaotic ones."), 1);
    cs.render_microblock();
    cs.remove_subscene("pgs");
    specimens[0]->state.microblock_transition({
        {"top_angle_opacity", "1"},
        {"bottom_angle_opacity", "1"},
    });
    specimens[0]->global_publisher_key = true;
    cs.stage_macroblock(FileSegment("We are particularly interested in the angles that separate each bar from the vertical."), 1);
    cs.render_microblock();
    shared_ptr<CoordinateSceneWithTrail> coord = make_shared<CoordinateSceneWithTrail>(1, 1);
    coord->state.set({
        {"center_y", "-5"},
    });
    cs.state.microblock_transition({
        {"p0.x", ".75"},
    });
    specimens[1]->state.microblock_transition({
        {"w", ".5"},
        {"h", "1"},
    });
    cs.state.microblock_transition({
        {"p1.x", ".75"},
        {"p2.x", ".75"},
        {"p1.y", "1.5"},
        {"p2.y", "1.5"},
    });
    coord->state.set({
        {"center_x", to_string(start_states[0].theta1)},
        {"center_y", to_string(start_states[0].theta2)},
        {"zoom", ".04"},
        {"trail_opacity", "1"},
        {"trail_x", "{pendulum_theta1}"},
        {"trail_y", "{pendulum_theta2}"},
    });
    cs.add_scene_fade_in(coord, "coord", 0.5, 0.5);
    cs.stage_macroblock(FileSegment("Plotting the top angle as X and the bottom angle as Y, we can make a graph like this."), 1);
    cs.render_microblock();
    coord->state.microblock_transition({
        {"center_x", "<trail_x>"},
        {"center_y", "<trail_y>"},
    });
    cs.stage_macroblock(FileSegment("This pendulum is evidently chaotic from the graph."), 1);
    cs.render_microblock();
    cs.stage_macroblock(FileSegment("It follows no consistent pattern."), 1);
    cs.render_microblock();
    cs.stage_macroblock(SilenceSegment(3), 1);
    cs.render_microblock();
    specimens[0]->state.microblock_transition({
        {"volume", "0"},
    });
    cs.state.microblock_transition({
        {"p0.y", "-.5"},
        {"p2.y", ".5"},
    });
    coord->state.microblock_transition({
        {"trail_opacity", "0"},
        {"center_x", "0"},
        {"center_y", "0"},
    });
    specimens[2]->state.microblock_transition({
        {"volume", "25"},
    });
    specimens[2]->state.set({
        {"w", ".5"},
        {"h", "1"},
        {"top_angle_opacity", "1"},
        {"bottom_angle_opacity", "1"},
    });
    cs.stage_macroblock(FileSegment("Here's a non-chaotic pendulum."), 1);
    cs.render_microblock();
    specimens[0]->global_publisher_key = false;
    specimens[2]->global_publisher_key = true;
    coord->state.microblock_transition({
        {"trail_opacity", "1"},
        {"zoom", ".1"},
    });
    cs.stage_macroblock(SilenceSegment(1), 1);
    cs.render_microblock();
    coord->state.microblock_transition({
        {"zoom", ".2"},
    });
    cs.stage_macroblock(SilenceSegment(5), 1);
    cs.render_microblock();
    shared_ptr<LatexScene> lissa = make_shared<LatexScene>(latex_text("Lissajous Curve"), 1, 0.3, 0.1);
    cs.add_scene_fade_in(lissa, "lissa", 0.5, 0.75);
    cs.stage_macroblock(FileSegment("It's drawing a shape known as a Lissajous curve."), 1);
    cs.render_microblock();
    shared_ptr<CoordinateSceneWithTrail> left = make_shared<CoordinateSceneWithTrail>(1, 1);
    left->state.set({
        {"center_x", "{t} 2 /"},
        {"center_y", "0"},
        {"trail_opacity", "1"},
        {"ticks_opacity", "0"},
        {"zoom", ".2"},
        {"trail_x", "{t} 2 / 1 -"},
        {"trail_y", "{pendulum_theta2}"},
    });
    shared_ptr<CoordinateSceneWithTrail> right = make_shared<CoordinateSceneWithTrail>(1, 1);
    right->state.set({
        {"center_x", "0"},
        {"center_y", "{t} 2 /"},
        {"trail_opacity", "1"},
        {"ticks_opacity", "0"},
        {"zoom", ".2"},
        {"trail_x", "{pendulum_theta1}"},
        {"trail_y", "{t} 2 / 1.5 -"},
    });
    left->trail_color = 0xffff00ff;
    right->trail_color = 0xff00ffff;
    cs.add_scene_fade_in(left , "left" , 0.5, 0.5);
    cs.add_scene_fade_in(right, "right", 0.5, 0.5);
    cs.state.microblock_transition({
        {"lissa.opacity", "0"},
    });
    cs.stage_macroblock(FileSegment("Here's the signal from the top and bottom angles, separated out."), 1);
    cs.render_microblock();
    cs.remove_subscene("lissa");
    cs.stage_macroblock(SilenceSegment(2), 1);
    cs.render_microblock();
    specimens[2]->state.macroblock_transition({
        {"volume", "0"},
    });
    shared_ptr<PngScene> ear_left  = make_shared<PngScene>("ear_left", .2, .2);
    shared_ptr<PngScene> ear_right = make_shared<PngScene>("ear_right", .2, .2);
    cs.add_scene_fade_in(ear_left , "ear_left" , 0.25, 0.15);
    cs.add_scene_fade_in(ear_right, "ear_right", 0.75, 0.15);
    left->state.macroblock_transition({
        {"center_x", "{t} 2 / 5 -"},
        {"w", ".5"},
        {"h", "1"},
    });
    right->state.macroblock_transition({
        {"center_y", "{t} 2 / 5 -"},
        {"w", ".5"},
        {"h", "1"},
    });
    cs.state.macroblock_transition({
        {"left.x", "0.25"},
        {"right.x", "0.75"},
        {"left.y", "0.5"},
        {"right.y", "0.5"},
        {"coord.opacity", ".1"},
        {"p2.opacity", ".1"},
    });
    cs.stage_macroblock(FileSegment("We can run this pendulum for a long time, reinterpret these signals as sound waves on the left and right speaker, and 'listen' to the pendulum!"), 1);
    cs.render_microblock();
    vector<float> audio_left;
    vector<float> audio_right;
    coord->state.microblock_transition({
        {"trail_opacity", "0"},
    });
    specimens[2]->generate_audio(4, audio_left, audio_right);
    cs.stage_macroblock(GeneratedSegment(audio_left, audio_right), 1);
    cs.render_microblock();
    specimens[0]->global_publisher_key = true;
    specimens[2]->global_publisher_key = false;
    cs.state.microblock_transition({
        {"left.opacity", "0"},
        {"right.opacity", "0"},
        {"ear_left.opacity", "0"},
        {"ear_right.opacity", "0"},
        {"p2.y", "1.5"},
        {"p0.y", ".5"},
        {"coord.opacity", "1"},
        {"p0.opacity", "1"},
    });
    coord->state.microblock_transition({
        {"zoom", ".05"},
        {"center_x", "<trail_x>"},
        {"center_y", "<trail_y>"},
    });
    coord->state.set({
        {"trail_opacity", "1"},
    });
    cs.stage_macroblock(FileSegment("It doesn't precisely sound beautiful, but compare that with the chaotic pendulum!"), 1);
    cs.render_microblock();
    vector<float> audio_left_c;
    vector<float> audio_right_c;
    specimens[0]->generate_audio(2.5, audio_left_c, audio_right_c);
    cs.stage_macroblock(GeneratedSegment(audio_left_c, audio_right_c), 1);
    cs.render_microblock();
    left->state.microblock_transition({
        {"trail_opacity", "0"},
    });
    right->state.microblock_transition({
        {"trail_opacity", "0"},
    });
    coord->state.microblock_transition({
        {"trail_opacity", "0"},
        {"zoom", ".02"},
    });
    cs.state.microblock_transition({
        {"p0.y", "-.5"},
        {"p1.y", ".5"},
    });
    coord->state.microblock_transition({
        {"center_x", "0"},
        {"center_y", "-5"},
    });
    cs.stage_macroblock(FileSegment("And here's the pretzel pendulum."), 1);
    cs.render_microblock();
    coord->state.set({
        {"trail_opacity", "1"},
    });
    specimens[1]->global_publisher_key = true;
    specimens[0]->global_publisher_key = false;
    cs.stage_macroblock(SilenceSegment(.1), 1);
    cs.render_microblock();
    vector<float> audio_left_p;
    vector<float> audio_right_p;
    specimens[1]->generate_audio(2.5, audio_left_p, audio_right_p);
    cs.stage_macroblock(GeneratedSegment(audio_left_p, audio_right_p), 1);
    cs.render_microblock();
    coord->state.microblock_transition({
        {"zoom", ".04"},
        {"center_x", "<trail_x>"},
        {"center_y", "<trail_y>"},
    });
    cs.stage_macroblock(FileSegment("This one traces a repetitive curve in angle-space, so it sounds the cleanest."), 1);
    cs.render_microblock();
    cs.stage_macroblock(FileSegment("Listening to a pendulum tells us if it's chaotic, but we have to listen one-by-one."), 1);
    cs.render_microblock();



    cs.fade_all_subscenes(0);
    int gridsize = 13;
    vector<shared_ptr<PendulumScene>> vps;
    double gridstep = 1./gridsize;
    double wh_ratio = static_cast<double>(VIDEO_WIDTH)/VIDEO_HEIGHT;
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            double x_mod = x + ((y%2==0) ? 0.75 : 0.25);
            PendulumState pendulum_state = {(y-gridsize/2)*.1*wh_ratio, -(x_mod-gridsize/2)*.1, .0, .0};
            shared_ptr<PendulumScene> ps = make_shared<PendulumScene>(pendulum_state, gridstep*2.5, gridstep*2.5);
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
            ps->state.set(state);
            vps.push_back(ps);
        }
    }
    PendulumGrid pg13(gridsize*2.*wh_ratio, gridsize*2, -.1*gridsize*wh_ratio/2, 0.0001, .1*gridsize*wh_ratio/2, -.1*gridsize/2, .1*gridsize/2, 0, 0, 0, 0);
    shared_ptr<PendulumGridScene> pgs13 = make_shared<PendulumGridScene>(vector<PendulumGrid>{pg13});
    PendulumGrid pgfull(VIDEO_HEIGHT, VIDEO_HEIGHT, 0.0001, 0, M_PI*2, 0, M_PI*2, 0, 0, 0, 0);
    pgs = make_shared<PendulumGridScene>(vector<PendulumGrid>{pgfull});
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            double x_mod = x + ((y%2==0) ? 0.75 : 0.25);
            string key = "ps" + to_string(x+y*gridsize);
            cs.add_scene_fade_in(vps[x+y*gridsize], key, 0, gridstep*(y+.5));
            cs.state.set({
                {key + ".x", to_string(gridstep*(x_mod)) + " 1 <scrunch> 2 / lerp"},
            });
        }
    }
    cs.state.set({
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
    cs.remove_subscene("ear_left");
    cs.remove_subscene("ear_right");
    cs.remove_subscene("left");
    cs.remove_subscene("right");
    cs.remove_subscene("p0");
    cs.remove_subscene("p2");
    int selected_pendulum_x = gridsize*.1;
    int selected_pendulum_y = gridsize*.2;
    int selected_pendulum = gridsize*selected_pendulum_y + selected_pendulum_x;
    string key_str = "ps" + to_string(selected_pendulum);
    cs.stage_macroblock(FileSegment("Instead, we can make a large array of pendulums like this."), 1);
    cs.render_microblock();
    cs.remove_subscene("coord");
    cs.remove_subscene("p1");
    vps[selected_pendulum]->state.set({
        {"pendulum_opacity", "1"},
    });
    vps[selected_pendulum]->state.microblock_transition({
        {"w", "1"},
        {"h", "1"},
    });
    cs.state.microblock_transition({
        {key_str + ".x", ".5"},
        {key_str + ".y", ".5"},
        {"pendulum_opacity", "0.6"},
    });
    cs.stage_macroblock(SilenceSegment(1), 1);
    cs.render_microblock();
    vps[selected_pendulum]->state.microblock_transition({
        {"top_angle_opacity", "1"},
    });
    cs.state.microblock_transition({
        {"manual_transition_1", "1"},
    });
    cs.stage_macroblock(FileSegment("The pendulum's x position in the grid corresponds to the top angle,"), 2);
    cs.render_microblock();
    cs.render_microblock();
    vps[selected_pendulum]->state.microblock_transition({
        {"top_angle_opacity", "0"},
        {"bottom_angle_opacity", "1"},
    });
    cs.state.microblock_transition({
        {"manual_transition_2", "1"},
    });
    cs.stage_macroblock(FileSegment("and its y position corresponds to the bottom angle."), 1);
    cs.render_microblock();
    vps[selected_pendulum]->state.microblock_transition({
        {"bottom_angle_opacity", "0"},
    });
    cs.stage_macroblock(SilenceSegment(0.5), 1);
    cs.render_microblock();
    string size_str = to_string(gridstep*2.5);
    double selected_x_mod = selected_pendulum_x + ((selected_pendulum_y%2==0) ? 0.75 : 0.25);
    cs.state.microblock_transition({
        {"rainbow", "1"},
        {key_str + ".x", to_string(gridstep*(selected_x_mod)) + " 1 <scrunch> 2 / lerp"},
        {key_str + ".y", to_string(gridstep*(selected_pendulum_y+.5))},
        {"pendulum_opacity", "1"},
    });
    vps[selected_pendulum]->state.microblock_transition({
        {"w", size_str},
        {"h", size_str},
    });
    cs.stage_macroblock(FileSegment("By associating angle positions with colors, it's easier to tell what's going on."), 1);
    cs.render_microblock();
    PendulumGrid pointgrid(80, 80, 0.0001, -M_PI*.6, M_PI*.6, -M_PI*.6, M_PI*.6, 0, 0, 0, 0);
    shared_ptr<PendulumPointsScene> pps = make_shared<PendulumPointsScene>(pointgrid, 0.5, 1);
    vps[selected_pendulum]->state.set({
        {"pendulum_opacity", "[pendulum_opacity]"},
    });
    pps->state.set({
        {"physics_multiplier", "0"},
        {"rk4_step_size", "1 30 / 5 /"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 6.283 /"},
    });
    cs.add_scene(pps, "pps", -0.25, 0.5);
    pgs->state.set({
        {"physics_multiplier", "0"},
        {"rk4_step_size", "1 30 / 5 /"},
    });
    string zoomval = "10 " + to_string(gridsize) + " /";
    pgs13->state.set({
        {"physics_multiplier", "0"},
        {"zoom", zoomval},
        {"rk4_step_size", "1 30 / 5 /"},
    });
    cs.add_scene(pgs, "pgs");
    cs.add_scene(pgs13, "pgs13");
    cs.state.set({
        {"pgs13.opacity", "0"},
        {"pgs.opacity", "0"},
    });
    cs.state.macroblock_transition({
        {"scrunch", "1"},
        {"pps.x", ".25"},
        {"manual_mode", "0"},
    });
    cs.stage_macroblock(FileSegment("As a bonus, we can add points in angle space for these pendulums and see how they move."), 1);
    cs.render_microblock();
    cs.state.set({
        {"physics_multiplier", "5"},
    });
    pps->state.set({
        {"physics_multiplier", "5"},
    });
    pgs->state.set({
        {"physics_multiplier", "5"},
    });
    pgs13->state.set({
        {"physics_multiplier", "5"},
    });
    cs.stage_macroblock(SilenceSegment(10), 1);
    cs.render_microblock();
    cs.state.macroblock_transition({
        {"scrunch", "0"},
        {"pps.x", "-.25"},
    });
    cs.stage_macroblock(SilenceSegment(2), 1);
    cs.render_microblock();
    cs.stage_macroblock(SilenceSegment(2), 1);
    cs.render_microblock();
    cs.state.microblock_transition({
        {"pgs13.opacity", "0.5"},
    });
    cs.stage_macroblock(SilenceSegment(2), 1);
    cs.render_microblock();
    cs.stage_macroblock(SilenceSegment(2), 1);
    cs.render_microblock();



    pgs->state.set({
        {"zoom", zoomval},
    });
    cs.state.microblock_transition({
        {"pgs.opacity", "1"},
        {"pgs13.opacity", "0"},
        {"pendulum_opacity", "0"},
    });
    cs.stage_macroblock(FileSegment("Let's increase the resolution to one pendulum per pixel."), 1);
    cs.render_microblock();
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            cs.remove_subscene("p" + to_string(x+y*gridsize));
        }
    }
    cs.remove_subscene("pps");
    cs.remove_subscene("pgs13");




    pgs->state.microblock_transition({
        {"zoom", "1 8 /"},
    });
    cs.stage_macroblock(SilenceSegment(4), 1);
    cs.render_microblock();
    PendulumState pendulum_state = {5, 7, .0, .0};
    shared_ptr<PendulumScene> pend = make_shared<PendulumScene>(pendulum_state);
    pend->state.set({
        {"manual_mode", "1"},
        {"rk4_step_size", "1"},
        {"physics_multiplier", "0"},
        {"theta1_manual", "5"},
        {"theta2_manual", "7"},
    });
    cs.add_scene_fade_in(pend, "pend");
    cs.state.microblock_transition({
        {"pgs.opacity", "0.4"},
    });
    
    cs.stage_macroblock(FileSegment("A nice feature of this fractal is that it tiles the plane."), 1);
    cs.render_microblock();
    cs.stage_macroblock(FileSegment("Rotating either pendulum arm by 2pi yields the exact same position, so the fractal is periodic."), 2);
    pgs->state.microblock_transition({
        {"center_x", "6.283"},
    });
    pend->state.microblock_transition({
        {"theta1_manual", "5 6.283 +"},
    });
    cs.render_microblock();
    pgs->state.microblock_transition({
        {"center_y", "6.283"},
    });
    pend->state.microblock_transition({
        {"theta2_manual", "7 6.283 +"},
    });
    cs.render_microblock();
    cs.state.microblock_transition({
        {"pgs.opacity", "1"},
        {"pend.opacity", "0"},
    });
    pgs->state.microblock_transition({
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 15 /"},
    });
    pgs->state.set({
        {"physics_multiplier", "5"},
    });
    cs.stage_macroblock(SilenceSegment(1), 1);
    cs.render_microblock();
    cs.remove_subscene("pgs");
    pgs->state.microblock_transition({
        {"zoom", "1 3.1415 /"},
    });
    pgs->stage_macroblock(FileSegment("Pay attention to how there are two distinct modes of behavior here."), 1);
    pgs->render_microblock();
    pgs->stage_macroblock(FileSegment("There's a region of chaotic pendulums sensitive to their initial conditions,"), 2);
    pgs->state.microblock_transition({
        {"center_x", "3.1415"},
        {"center_y", "3.1415"},
    });
    pgs->render_microblock();
    pgs->render_microblock();
    pgs->stage_macroblock(FileSegment("as well as a region of coherent ones which are not."), 2);
    pgs->state.microblock_transition({
        {"center_x", "0"},
        {"center_y", "0"},
    });
    pgs->render_microblock();
    pgs->render_microblock();
    pgs->stage_macroblock(SilenceSegment(0.5), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"mode", "2.5"},
    });
    pgs->stage_macroblock(FileSegment("Now, for each pixel, we track two pendulums, separated by a slight starting difference, and plot their difference over time."), 1);
    pgs->render_microblock();
    pgs->stage_macroblock(SilenceSegment(0.5), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"center_x", "3.1415"},
        {"center_y", "3.1415"},
        {"zoom", "1 8 /"},
    });
    pgs->stage_macroblock(FileSegment("So, this plot shows how quickly the pendulums in our grid diverge."), 1);
    pgs->render_microblock();
    pgs->state.microblock_transition({
        {"mode", "2"},
        {"contrast", ".000005"},
    });
    pgs->stage_macroblock(FileSegment("Let's reset the pendulums and watch that again from the start."), 1);
    pgs->render_microblock();
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
    vector<shared_ptr<PendulumScene>> specimens;
    for(int i = 0; i < start_states.size(); i++) {
        specimens.push_back(make_shared<PendulumScene>(start_states[i], .5, .5));
    }
    double anim_step = 1800;
    for(int i = 0; i < start_states.size(); i++) {
        shared_ptr<PendulumScene> ps = specimens[i];
        ps->state.set({
            {"rk4_step_size", "1 "+to_string(anim_step)+" /"},
            {"physics_multiplier", to_string(anim_step/30)},
            //one frame per cycle{"physics_multiplier", to_string(anim_step*period)},
        });
        string name = "p" + to_string(i);
        cs.add_scene(ps, name, .75, .25+.5*i);
    }
    specimens[0]->global_publisher_key = true;
    specimens[0]->global_identifier = "p0.";
    specimens[1]->global_publisher_key = true;
    specimens[1]->global_identifier = "p1.";
    shared_ptr<CoordinateSceneWithTrail> coord = make_shared<CoordinateSceneWithTrail>(.5, 1);
    coord->state.set({
        {"zoom", "0.05"},
        //{"zoom", "6000"},
        {"trail_opacity", "1"},
        {"trail_x", "{p1.pendulum_p1}"},
        {"trail_y", "{p1.pendulum_p2}"},
        //{"trail_x", "{p1.pendulum_theta1} {p0.pendulum_theta1} -"},
        //{"trail_y", "{p1.pendulum_theta2} {p0.pendulum_theta2} -"},
    });
    cs.add_scene(coord, "coord", 0.25, 0.5);
    string str_cx = to_string(t1);
    string str_cy = to_string(t2);
    str_cx = str_cx.erase(str_cx.find_last_not_of('0') + 1);
    str_cy = str_cy.erase(str_cy.find_last_not_of('0') + 1);
    string latex_str = "\\theta_1 = " + str_cx + ", \\theta_2 = " + str_cy + ", \\pi = " + to_string(period);
    shared_ptr<LatexScene> ls = make_shared<LatexScene>(latex_str, 1, 1, 0.12);
    cs.add_scene(ls, "ls", 0.5, 0.1);
    cs.stage_macroblock(SilenceSegment(7), 1);
    cs.render_microblock();
}
void sample_vibrations(){
    for(int x = -2; x <= 2; x++) for(int y = -2; y <= 2; y++) identify_vibrations(2.496147+.0000004*x, .2505+.000008*y);
}

void render_video() {
    SAVE_FRAME_PNGS = false;
    //PRINT_TO_TERMINAL = false;
    //FOR_REAL = false;


    intro();
    vector<PendulumGrid> grids{PendulumGrid(VIDEO_HEIGHT, VIDEO_HEIGHT, 0.0001, -M_PI, M_PI, -M_PI, M_PI, 0.f, 0.f, 0.f, 0.f)};
    for (const vector<IslandShowcase>& isvh : {isv}) for(const IslandShowcase& is : isvh) {
        const double ro2 = is.range/2;
        const double t1 = is.ps.theta1;
        const double t2 = is.ps.theta2;
        grids.push_back(PendulumGrid(VIDEO_WIDTH, VIDEO_HEIGHT, 0.0001, t1-ro2*VIDEO_WIDTH/VIDEO_HEIGHT, t1+ro2*VIDEO_WIDTH/VIDEO_HEIGHT, t2-ro2, t2+ro2, 0.f, 0.f, 0.f, 0.f));
    }
    if(!FOR_REAL) for(PendulumGrid& p : grids) p.iterate_physics(50, .2/30);
    shared_ptr<PendulumGridScene> pgs = make_shared<PendulumGridScene>(grids);
    fine_grid(pgs);
    showcase_islands(pgs);
    discuss_energy(pgs);
    move_fractal(pgs);
    outtro();
}
