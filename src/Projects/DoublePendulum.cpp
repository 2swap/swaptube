#include "../Scenes/Common/ThreeDimensionScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/CoordinateSceneWithTrail.cpp"
#include "../Scenes/Common/CoordinateScene.cpp"
#include "../Scenes/Physics/PendulumScene.cpp"
#include "../Scenes/Physics/PendulumGridScene.cpp"
#include "../Scenes/Physics/PendulumPointsScene.cpp"

void island(const double cx, const double cy, const double range){
    const double zoom = 1/range; 
    const double addsub = range/2;
    PendulumGrid pg2(VIDEO_WIDTH, VIDEO_HEIGHT, cx-addsub, cx+addsub, cy-addsub, cy+addsub, 0, 0, 0, 0);
    PendulumGridScene pgs2(cx-addsub, cx+addsub, cy-addsub, cy+addsub, pg2);
    pgs2.state_manager.set({
        {"physics_multiplier", "40"},
        {"mode", "0"},
        {"rk4_step_size", "1 30 / .05 *"},
        {"zoom", to_string(zoom)},
        {"center_x", to_string(cx)},
        {"center_y", to_string(cy)},
    });
    pgs2.inject_audio_and_render(SilenceSegment(6));
    pgs2.state_manager.microblock_transition({
        {"mode", "2"},
        {"physics_multiplier", "0"},
    });
    pgs2.inject_audio_and_render(SilenceSegment(2));
}

void island1(){
    island(2.49, .25, .4);
}

void island2(){
    island(2.658, -2.19, .2);
}

void island3(){
    island(2.453, -2.7727, .05);
}

void island4(){
    // False island?
    island(2.631, -2.95, .02);
}

void island5(){
    island(1.351, 2.979, .2);
}

void fine_grid_2(){
    PendulumGrid pg(VIDEO_WIDTH, VIDEO_HEIGHT, -M_PI, M_PI, -M_PI, M_PI, 0, 0, 0, 0);
    PendulumGridScene pgs(-M_PI, M_PI, -M_PI, M_PI, pg);
    pgs.state_manager.set({
        {"physics_multiplier", "20"},
        {"mode", "3"},
        {"rk4_step_size", "1 30 / .1 *"},
        {"zoom", "1 8 /"},
        {"trail_start_x", "0.5"},
        {"trail_start_y", "-0.5"},
        {"center_x", "3.1415"},
        {"center_y", "3.1415"},
    });
    pgs.inject_audio_and_render(FileSegment("Here we go!"));
    pgs.inject_audio_and_render(SilenceSegment(7));
    pgs.state_manager.set({
        {"physics_multiplier", "0"},
    });
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
        {"trail_length", "1000"},
    });
    pgs.inject_audio_and_render(FileSegment("We can plot its path in angle-space just like we did before!"));
    pgs.state_manager.microblock_transition({
        {"zoom", "1 4 /"},
        {"trail_start_x", "0.25"},
        {"trail_start_y", "0.5"},
        {"mode", "1"},
    });
    pgs.inject_audio_and_render(FileSegment("Moving the point around in the black region, this curve moves smoothly and traces a recognizable pattern."));
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
        {"trail_start_x", "1.4"},
        {"trail_start_y", "1.7"},
        {"mode", "2"},
        {"zoom", "0.05"},
    });
    pgs.inject_audio_and_render(FileSegment("But as soon as you leave and step into the chaotic region..."));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", "1.5"},
        {"trail_start_y", "1.7"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"trail_start_x", "1.5 <t> sin 5 / +"},
        {"trail_start_y", "1.8 <t> cos 5 / +"},
    });
    pgs.inject_audio_and_render(FileSegment("It goes crazy."));
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.state_manager.microblock_transition({
        {"center_x", "2.49"},
        {"center_y", ".25"},
        {"trail_start_x", "2.49"},
        {"trail_start_y", ".25"},
        {"zoom", "1"},
    });
    pgs.inject_audio_and_render(FileSegment("Also, check out this spot:"));
    pgs.inject_audio_and_render(SilenceSegment(2));
    pgs.inject_audio_and_render(FileSegment("This small black island of stability is the home of the pretzel pendulum!"));
    pgs.state_manager.microblock_transition({
        {"zoom", "2.5"},
    });
    pgs.inject_audio_and_render(FileSegment("Let's increase the resolution and try that again from the start..."));

    pgs.state_manager.microblock_transition({
        {"zoom", "0.5"},
    });
    pgs.inject_audio_and_render(FileSegment("Zooming back out a bit,"));
    pgs.inject_audio_and_render(FileSegment("You'll notice that there are a whole bunch of little islands of stability here."));
    pgs.state_manager.microblock_transition({
        {"center_x", "2.66"},
        {"center_y", "-2.19"},
        {"trail_start_x", "2.66"},
        {"trail_start_y", "-2.19"},
    });
    pgs.inject_audio_and_render(FileSegment("Here's another one."));
    pgs.state_manager.microblock_transition({
        {"zoom", "2.5"},
    });
    pgs.inject_audio_and_render(SilenceSegment(2));
    island2();
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

void fine_grid(){
    PendulumGrid pg(VIDEO_WIDTH, VIDEO_HEIGHT, -M_PI, M_PI, -M_PI, M_PI, 0, 0, 0, 0);
    PendulumGridScene pgs(-M_PI, M_PI, -M_PI, M_PI, pg);
    pgs.state_manager.set({
        {"physics_multiplier", "0"},
        {"mode", "0"},
        {"rk4_step_size", "1 30 / .05 *"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 6.283 /"},
    });
    pgs.inject_audio_and_render(FileSegment("Let's up the resolution, with one pendulum per pixel."));
    pgs.state_manager.set({
        {"physics_multiplier", "16"},
    });
    pgs.inject_audio_and_render(SilenceSegment(9));
    pgs.inject_audio(FileSegment("A nice feature of this fractal is that it tiles the plane- in other words, since rotating either angle by 2pi yields the exact same position, the image itself is periodic."), 7);
    pgs.state_manager.microblock_transition({
        {"center_x", "3.1415"},
        {"center_y", "3.1415"},
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
    pgs.inject_audio_and_render(FileSegment("Now I want you to pay attention to how there are seemingly two distinct modes of behavior here."));
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
    pgs.inject_audio_and_render(FileSegment("Now what we can do is, for each pixel, track not only one pendulum but instead two, separated by a microscopic difference in angle, and plot their difference as time passes."));
    pgs.inject_audio_and_render(FileSegment("This is what you get."));
    pgs.state_manager.microblock_transition({
        {"center_x", "3.1415"},
        {"center_y", "3.1415"},
        {"zoom", "1 10 /"},
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
            cs.add_scene(&(vps[x+y*gridsize]), key, gridstep*(x+.5), gridstep*(y+.5));
        }
    }
    StateSet state = {
        {"pendulum_opacity", "1"},
        {"background_opacity", "0"},
        {"physics_multiplier", "[physics_multiplier]"},
        {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        {"rainbow", "0"},
    };
    cs.state_manager.set(state);
    CompositeScene meta_cs;
    meta_cs.add_scene(&cs, "cs", 0.5, 0.5);
    meta_cs.state_manager.set({
        {"physics_multiplier", "0.5"},
    });
    meta_cs.inject_audio_and_render(FileSegment("Well, we can try creating a large array of pendulums like this."));
    cs.state_manager.macroblock_transition({
        {"w", ".5"},
    });
    meta_cs.state_manager.macroblock_transition({
        {"cs.x", ".75"},
    });
    meta_cs.inject_audio_and_render(SilenceSegment(2));

    PendulumGrid grid(60, 60, -M_PI, M_PI, -M_PI, M_PI, 0, 0, 0, 0);
    PendulumPointsScene pps(grid, 0.5, 1);
    pps.state_manager.set({
        {"physics_multiplier", "[physics_multiplier]"},
        {"rk4_step_size", "1 30 / <physics_multiplier> / 4 /"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 6.283 /"},
    });
    meta_cs.add_scene_fade_in(&pps, "pps", 0.25, 0.5);
    meta_cs.inject_audio_and_render(FileSegment("If we make a point for each of these pendulums and track them in angle-space all at once, this is what it looks like:"));
    meta_cs.state_manager.set({
        {"physics_multiplier", "16"},
    });
    meta_cs.inject_audio_and_render(SilenceSegment(10));
    cs.state_manager.macroblock_transition({
        {"w", "1"},
    });
    meta_cs.state_manager.macroblock_transition({
        {"cs.x", ".5"},
        {"pps.opacity", "0"},
    });
    meta_cs.inject_audio_and_render(SilenceSegment(2));

    cs.state_manager.set({
        {"physics_multiplier", "16"},
    });
    cs.state_manager.microblock_transition({
        {"rainbow", "1"},
    });
    cs.inject_audio_and_render(FileSegment("Another way to make it easier to follow what's going on is by associating all possible pendulum positions with a color."));
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.state_manager.microblock_transition({
        {"background_opacity", "1"},
        {"pendulum_opacity", "0"},
    });
    cs.inject_audio_and_render(FileSegment("Coloring each pendulum with the color associated with its position, this is what we get:"));
    cs.inject_audio_and_render(SilenceSegment(2));
}

void simple() {
    vector<PendulumState> vps{{2.49, .25     , .0, .0},
                              {2.658, -2.19   , .0, .0},
                              {2.453, -2.7727, .0, .0},
                              {2.631, -2.95 , .0, .0},//False but there are morsels of true
                              {1.351, 2.979  , .0, .0}};
    //for(PendulumState pendulum_state : vps){
    PendulumState ps = vps[3];
/*
*/
for(int x = -2; x < 3; x++)
for(int y = -2; y < 3; y++){
        PendulumState pendulum_state{ps.theta1+x*.002, ps.theta2+y*.002, ps.p1, ps.p2};
        PendulumScene ps(pendulum_state);
        ps.state_manager.set({
            {"background_opacity", "0"},
            {"angles_opacity", "0"},
            {"pendulum_opacity", "1"},
            {"physics_multiplier", "400"},
            {"path_opacity", "1"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        vector<float> audio_left;
        vector<float> audio_right;
        ps.generate_audio(1, audio_left, audio_right);
        ps.inject_audio_and_render(GeneratedSegment(audio_left, audio_right));
    }
}

void intro() {
    ThreeDimensionScene tds;
    //vector<double> notes{pow(2, 0/12.), pow(2, 5/12.), pow(2, 7/12.), pow(2, 12/12.), pow(2, 17/12.), };
    vector<double> notes{pow(2, 0/12.), pow(2, 4/12.), pow(2, 7/12.), pow(2, 11/12.), pow(2, 12/12.), };
    for(int i = 0; i < 5; i++){
        PendulumState pendulum_state = {5+.0000001*i, 8, .0, .0};
        shared_ptr<PendulumScene> ps = make_shared<PendulumScene>(pendulum_state);
        ps->state_manager.set({
            {"path_opacity", "[parent_path_opacity]"},
            {"background_opacity", "0"},
            {"tone", to_string(notes[i])},
            {"angles_opacity", "0"},
            {"volume", "[volume_set1]"},
            {"pendulum_opacity", "1"},
            {"physics_multiplier", "30"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        tds.add_surface(Surface(glm::vec3(0,0,i-2), glm::vec3(8,0,0), glm::vec3(0,8,0), ps));
    }
    vector<shared_ptr<LatexScene>> ls;
    ls.push_back(make_shared<LatexScene>(latex_text("Double"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("Pendulums"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("are"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("NOT"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("Chaotic"), 1));
    int start_dist = 60;
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
        {"fov", "4"},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
    });
    tds.inject_audio(FileSegment("Double pendulums are NOT chaotic."), 13);
    tds.render();
    tds.render();
    tds.render();
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,-.06,-start_dist+3-.8), glm::vec3(.1,0,0), glm::vec3(0,.1,0), ls[0]));
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,-.025,-start_dist+3-.5), glm::vec3(.1,0,0), glm::vec3(0,.1,0), ls[1]));
    tds.render();
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,.01,-start_dist+3-.4), glm::vec3(.035,0,0), glm::vec3(0,.035,0), ls[2]));
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,.045,-start_dist+3-.7), glm::vec3(.1,0,0), glm::vec3(0,.1,0), ls[3]));
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,0.09,-start_dist+3-.7), glm::vec3(.1,0,0), glm::vec3(0,.1,0), ls[4]));
    tds.render();
    tds.render();
    tds.render();
    tds.render();
    ls[3]->begin_latex_transition(latex_text("NOT")+"^*");
    tds.inject_audio_and_render(FileSegment("Or, at least, not all of them."));
    tds.state_manager.macroblock_transition({
        {"d", to_string(start_dist - 3)},
    });
    tds.inject_audio_and_render(FileSegment("You've probably seen videos like these,"));
    for(int i = 0; i < ls.size(); i++) tds.remove_surface(ls[i]);
    tds.state_manager.macroblock_transition({
        {"qj", ".2"},
    });
    tds.inject_audio_and_render(FileSegment("where a tiny deviation in similar double pendulums amplifies over time,"));
    tds.inject_audio_and_render(FileSegment("until they eventually completely desynchronize."));
    tds.inject_audio_and_render(FileSegment("This is known as a chaotic system, because small changes in starting conditions yield vastly different behavior in the long run."));
    vector<double> notes2{pow(2, 3/12.), pow(2, 8/12.), pow(2, 10/12.), pow(2, 15/12.), pow(2, 20/12.), };
    for(int i = 0; i < 5; i++){
        PendulumState pendulum_state = {2.49+.0001*i, .25, .0, .0};
        shared_ptr<PendulumScene> ps = make_shared<PendulumScene>(pendulum_state);
        ps->state_manager.set({
            {"path_opacity", "[parent_path_opacity]"},
            {"background_opacity", "0"},
            {"angles_opacity", "0"},
            {"volume", "[volume_set2]"},
            {"tone", to_string(notes2[i])},
            {"pendulum_opacity", "0"},
            {"physics_multiplier", "[stable_physics_multiplier]"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        ps->state_manager.macroblock_transition({
            {"pendulum_opacity", "1"},
        });
        tds.add_surface(Surface(glm::vec3(18, 0, i-2), glm::vec3(8,0,0), glm::vec3(0,8,0), ps));
    }
    tds.state_manager.macroblock_transition({
        {"x", "18"},
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
    tds.inject_audio_and_render(FileSegment("Here are a few more pendulums with slightly different starting positions."));
    tds.inject_audio_and_render(FileSegment("Unlike the others, these will _not_ diverge."));
    tds.state_manager.macroblock_transition({
        {"parent_path_opacity", "1"},
        {"qj", "0"},
        {"fov", "8"},
        {"d", to_string(start_dist*2)},
    });
    tds.inject_audio_and_render(FileSegment("Not only that, but they trace a regular, repeating pattern."));
    tds.inject_audio_and_render(SilenceSegment(3));
    tds.state_manager.macroblock_transition({
        {"volume_set1", "1"},
        {"volume_set2", "0"},
        {"x", "0"},
    });
    tds.inject_audio_and_render(FileSegment("A stark contrast with the first ones, which are... all over the place."));
    tds.inject_audio_and_render(SilenceSegment(1));
    tds.inject_audio_and_render(FileSegment("So... what's going on here?"));
    tds.state_manager.macroblock_transition({
        {"volume_set1", "0.5"},
        {"volume_set2", "0.5"},
        {"x", "9"},
        {"fov", "4"},
    });
    tds.inject_audio_and_render(FileSegment("These pendulums follow the same laws of physics."));
    tds.state_manager.macroblock_transition({
        {"volume_set1", "0"},
        {"volume_set2", "0"},
        {"z", "100"},
    });
    tds.inject_audio_and_render(FileSegment("The only difference is what their starting position was."));
}

void fractal() {
    PendulumGrid pg(VIDEO_WIDTH, VIDEO_HEIGHT, 0, 6.283, 0, 6.283, 0, 0, 0, 0);
    PendulumGridScene pgs(-M_PI, M_PI, -M_PI, M_PI, pg);
    pgs.state_manager.set({
        {"physics_multiplier", "16"},
        {"mode", "2"},
        {"rk4_step_size", "1 30 / .05 *"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 40 /"},
    });
    pgs.inject_audio_and_render(FileSegment("And behavior as a function of starting position can be graphed,"));
    pgs.state_manager.macroblock_transition({
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 6.283 /"},
    });
    pgs.inject_audio_and_render(FileSegment("revealing fractals like these,"));
    pgs.inject_audio_and_render(SilenceSegment(1));
    CompositeScene cs;
    cs.add_scene(&pgs, "pgs");
    cs.inject_audio_and_render(FileSegment("where each point shows how chaotic a certain pendulum's behavior is."));
    vector<PendulumState> start_states = {
                                          {1, 1.5, 0, 0},
                                          {6.283-2.49, 6.283-0.25, 0, 0},
                                          {0.6   , 0.13, 0, 0},
                                         };
    StateSet state = {
        {"path_opacity", "0"},
        {"background_opacity", "0"},
        {"angles_opacity", "0"},
        {"volume", "0"},
        {"pendulum_opacity", "1"},
        {"physics_multiplier", "10"},
        {"rk4_step_size", "1 30 / <physics_multiplier> /"},
    };
    cs.inject_audio(FileSegment("But before diving into the fractals, let's get to know a few particular specimen."), 3);
    vector<PendulumScene> specimens;
    for(int i = 0; i < start_states.size(); i++) {
        specimens.push_back(PendulumScene(start_states[i], 1./3, 1./3));
    }
    for(int i = 0; i < start_states.size(); i++) {
        PendulumScene& ps = specimens[i];
        ps.state_manager.set(state);
        ps.state_manager.set({{"tone", to_string(i/2.+1)}});
        string name = "p" + to_string(i);
        cs.add_scene_fade_in(&ps, name, 0.25*(1+i), 0.8);
        cs.state_manager.set({
            {name + ".pointer_x", to_string(  start_states[i].theta1/6.283)},
            {name + ".pointer_y", to_string(1-start_states[i].theta2/6.283)},
        });
        cs.state_manager.microblock_transition({
            {name + ".pointer_opacity", "1"},
        });
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
    cs.inject_audio_and_render(FileSegment("This first pendulum is one of the chaotic ones."));
    specimens[0].state_manager.microblock_transition({
        {"angles_opacity", "1"},
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
        {"zoom", ".05"},
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
        {"volume", "10"},
    });
    specimens[2].state_manager.set({
        {"w", ".5"},
        {"h", "1"},
        {"angles_opacity", "1"},
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
    cs.inject_audio_and_render(FileSegment("This particular pendulum is drawing what's known as a Lissajous curve,"));
    cs.inject_audio_and_render(FileSegment("which is basically what you get when you plot two different sinusoid frequencies against each other."));
    cs.inject_audio_and_render(FileSegment("Now... it's actually rotated at an angle, so both of the frequencies have components present in both of the pendulums."));
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
    specimens[2].state_manager.macroblock_transition({
        {"volume", "0"},
    });
    cs.inject_audio_and_render(FileSegment("Here is the sinusoid for the top and bottom angles, separated out."));
    cs.inject_audio_and_render(SilenceSegment(5));
    cs.inject_audio_and_render(FileSegment("We can then re-interpret these signals as sound waves on the left and right speaker, and 'listen' to the pendulum!"));
    vector<float> audio_left;
    vector<float> audio_right;
    specimens[2].generate_audio(6, audio_left, audio_right);
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
    coord.state_manager.set({
        {"zoom", ".05"},
        {"trail_opacity", "1"},
    });
    left.state_manager.set({
        {"zoom", ".05"},
        {"trail_opacity", "1"},
    });
    right.state_manager.set({
        {"zoom", ".05"},
        {"trail_opacity", "1"},
    });
    specimens[0].global_publisher_key = true;
    specimens[2].global_publisher_key = false;
    cs.inject_audio_and_render(SilenceSegment(2));
    vector<float> audio_left_c;
    vector<float> audio_right_c;
    specimens[0].generate_audio(6, audio_left_c, audio_right_c);
    cs.inject_audio_and_render(GeneratedSegment(audio_left_c, audio_right_c));
    cs.inject_audio_and_render(FileSegment("Let's also check out what that special non-chaotic pendulum from the beginning sounds like."));
    left.state_manager.microblock_transition({
        {"trail_opacity", "0"},
        {"zoom", ".02"},
        {"center_x", "5"},
        {"center_y", "5"},
    });
    right.state_manager.microblock_transition({
        {"trail_opacity", "0"},
        {"zoom", ".02"},
        {"center_x", "5"},
        {"center_y", "5"},
    });
    coord.state_manager.microblock_transition({
        {"trail_opacity", "0"},
        {"zoom", ".02"},
        {"center_x", "5"},
        {"center_y", "5"},
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
    specimens[1].generate_audio(6, audio_left_p, audio_right_p);
    cs.inject_audio_and_render(GeneratedSegment(audio_left_p, audio_right_p));
    cs.inject_audio_and_render(FileSegment("It traces a completely repetitive curve in angle-space, so it unsurprisingly sounds the cleanest."));
    cs.fade_out_all_scenes();
    cs.inject_audio_and_render(FileSegment("Listening to a pendulum can tell us if it is chaotic, but unfortunately, we have to do it one-by-one, pendulum-by-pendulum."));
    cs.inject_audio_and_render(FileSegment("How can we find new pendulums with unique properties, other than picking one at random, and checking after the fact?"));
}

void render_video() {
    //PRINT_TO_TERMINAL = false;
    SAVE_FRAME_PNGS = false;
    //FOR_REAL = false;

/*
    intro();
    fractal();
    grid();
    fine_grid();
    fine_grid_2();
*/
    //island2();
    //island3();
    //island4();
    //island5();
    simple();
}
