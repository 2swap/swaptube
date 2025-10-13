#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/CoordinateSceneWithTrail.cpp"
#include "../Scenes/Common/CoordinateScene.cpp"
#include "../Scenes/Physics/PendulumScene.cpp"
#include "../Scenes/Physics/PendulumGridScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"

struct IslandShowcase {
    PendulumState ps;
    double range;
    double fingerprint_zoom;
    string name;
    string blurb;
};

vector<IslandShowcase> isv{
                           // Class 1: S-shapes
                           {{2.49  ,  .25   , .0, .0}, 0.6 , -2, "The Pretzel", "This big island of stability contains the Pretzel we saw earlier."},
                           {{2.658 , -2.19  , .0, .0}, 0.2 , -2.5, "The Shoelace", "This one, which I call the shoelace, traces a more complex pattern."},
                           {{1.782 ,  0.137 , .0, .0}, 0.2 , -2, "The Necktie", "This pendulum traces a path very similar to the Pretzel,\\\\\\\\and its island is close to the Pretzel island of stability."},

                           // Class 2: 
                           {{1.351 ,  2.979 , .0, .0}, 0.2 , -1, "The Bird", "Although on a separate island from the lissajous pendulums,\\\\\\\\neither of its two arms does a flip."},
                           {{0.567 ,  2.995 , .0, .0}, 0.2 , -1, "Big-Ears", "Similar to The Bird,\\\\\\\\this pendulum makes no flips."},
                           {{1.484 ,  2.472 , .0, .0}, 0.2 , -1, "The Fishbowl", "Also makes no flips."},

                           // Class 3: 
                           {{1.244 , -0.258 , .0, .0}, 0.2 , -2, "The Pistachio", "This island is in a region dotted with small islands of stability."},
                           {{1.311 , -0.804 , .0, .0}, 0.2 , -2, "Jake the Dog", "This pendulum lives right next to the ???"},

                           // Class 4: Asymmetrics
                           {{0.028 ,  2.971 , .0, .0}, 0.2 , -1, "Asymmetric A", "The majority of the pendulums I found exhibit left-right symmetry.\\\\\\\\This is the first of 3 exceptions."},
                           {{-.478 ,  2.633 , .0, .0}, 0.2 , -1, "Asymmetric B", "The second asymmetric island."},
                           {{1.348 , -0.299 , .0, .0}, 0.2 , -1, "Asymmetric C", "This island is very close to the Pistachio."},

                           // Class 5: Jumpers
                           {{2.808 ,  0.046 , .0, .0}, 0.2 , -1.5, "The Jumper", "Nothing too complicated."},
                           {{3.0224,  0.0295, .0, .0}, 0.2 , -1.5, "The High-Jumper", "This island is particularly close\\\\\\\\to the top angle being pi and the bottom angle being 0."},

                           // Others
                           {{2.453 , -2.7727, .0, .0}, 0.05, -2.5, "The Heart", "This one draws a picture of a heart. This island is particularly small."},
                           {{2.2486 , 0.4565, .0, .0}, 0.05, -2.5, "The ???", "aaaa"},
                           {{2.07302 , 0.32, .0, .0}, 0.02, -2.5, "The ???", "aaaa"},


                           {{1.238 , -0.854 , .0, .0}, 0.2 , -3, "The ???", "This pendulum's period is very long,\\\\\\\\hence its sound being so deep."},
                           {{0.572 ,  2.539 , .0, .0}, 0.2 , -3, "Seneca Lake", "This island is the largest of a region rich with\\\\\\\\long but thin islands of stability, named after\\\\\\\\the largest of the Fingerlakes of Upstate New York."},
};

void showcase_an_island(shared_ptr<PendulumGridScene>& pgs, const IslandShowcase& is) {
    cout << "Showcasing " << is.name << endl;
    const double range = is.range;
    const double cx = is.ps.theta1;
    const double cy = is.ps.theta2;

    /*
    pgs.state.set({
        {"physics_multiplier", "0"},
        {"ticks_opacity", "0"},
        {"rk4_step_size", "0"},
        {"zoom", "2.718281828 <zoomexp> ^"},
        {"zoomexp", "1 6.283 / log"},
        // Leave mode as-is
    });
    pgs.state.transition(MICRO, {
        {"center_x", to_string(cx)},
        {"center_y", to_string(cy)},
        {"ticks_opacity", "0"},
    });
    pgs.stage_macroblock(SilenceBlock(2), 1);
    pgs.render_microblock();
    pgs.state.transition(MICRO, {
        {"zoomexp", to_string(log(1/is.range))},
        //{"circles_opacity", "0"},
    });
    */

    CompositeScene cs;
    shared_ptr<PendulumScene> ps = make_shared<PendulumScene>(is.ps, 0.5, 1);
    ps->alpha_subtract = 0;
    ps->global_publisher_key = true;
    shared_ptr<LatexScene> ls = make_shared<LatexScene>("\\text{" + is.name + "}", 1, 1, 0.2);

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
    bool moveup = false;
    if(is.ps.p1 != 0) { latex_str += ", p_1 = " + str_p1; moveup = true; }
    if(is.ps.p2 != 0)   latex_str += ", p_2 = " + str_p2;
    shared_ptr<LatexScene> ls2 = make_shared<LatexScene>(latex_str, 1, 1, 0.12);

    shared_ptr<CoordinateSceneWithTrail> ts = make_shared<CoordinateSceneWithTrail>(0.5, 1);
    //cs.add_scene        (make_shared<PendulumGridScene>(pgs), "pgs", 0.5 , 0.5 );
    cs.add_scene_fade_in(MICRO, ps , "ps" , 0.75, 0.5 );
    cs.add_scene_fade_in(MICRO, ls , "ls" , 0.5 , 0.15);
    cs.add_scene_fade_in(MICRO, ls2, "ls2", 0.5 , moveup?0.15:0.25);
    cs.add_scene_fade_in(MICRO, ts , "ts" , 0.25, 0.5 );

    //cs.stage_macroblock(FileBlock(is.blurb), 3);
    vector<sample_t> audio_left;
    vector<sample_t> audio_right;
    ps->generate_audio(10, audio_left, audio_right);
    cs.stage_macroblock(GeneratedBlock(audio_left, audio_right), 3);

    cs.state.transition(MICRO, {
        //{"pgs.opacity", ".4"},
    });
    ts->state.set({
        {"zoom", to_string(is.fingerprint_zoom)},
        {"trail_opacity", "1"},
        {"trail_x", "{pendulum_theta1}"},
        {"trail_y", "{pendulum_theta2}"},
        {"center_x", to_string(is.ps.theta1)},
        {"center_y", to_string(is.ps.theta2)},
    });
    ps->state.set({
        {"background_opacity", "0"},
        {"top_angle_opacity", "0"},
        {"bottom_angle_opacity", "0"},
        {"pendulum_opacity", "1"},
        {"physics_multiplier", "400"},
        {"path_opacity", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.state.transition(MICRO, {
        //{"pgs.opacity", "1"},
        {"ps.opacity", "0"},
        {"ls.opacity", "0"},
        {"ls2.opacity", "0"},
        {"ts.opacity", "0"},
    });
    pgs->state.transition(MICRO, {
        //{"zoomexp", "1 6.283 / log"},
        //{"circles_opacity", "1"},
    });
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();
    //cs.remove_subscene("pgs");
}

void identify_vibrations(float t1, float t2) {
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
        ps.state.set({
            {"rk4_step_size", "1 "+to_string(anim_step)+" /"},
            {"physics_multiplier", to_string(anim_step/30)},
            //one frame per cycle{"physics_multiplier", to_string(anim_step*period)},
        });
        string name = "p" + to_string(i);
        cs.add_scene(make_shared<PendulumScene>(ps), name, .75, .25+.5*i);
    }
    specimens[0].global_publisher_key = true;
    specimens[0].global_identifier = "p0.";
    specimens[1].global_publisher_key = true;
    specimens[1].global_identifier = "p1.";
    CoordinateSceneWithTrail coord(.5, 1);
    coord.state.set({
        {"zoom", "0.05"},
        {"trail_opacity", "1"},
        {"trail_x", "{p1.pendulum_p1}"},
        {"trail_y", "{p1.pendulum_p2}"},
    });
    cs.add_scene(make_shared<CoordinateSceneWithTrail>(coord), "coord", 0.25, 0.5);
    string str_cx = to_string(t1);
    string str_cy = to_string(t2);
    str_cx = str_cx.erase(str_cx.find_last_not_of('0') + 1);
    str_cy = str_cy.erase(str_cy.find_last_not_of('0') + 1);
    string latex_str = "\\theta_1 = " + str_cx + ", \\theta_2 = " + str_cy + ", \\pi = " + to_string(period);
    LatexScene ls(latex_str, 1, 1, 0.12);
    cs.add_scene(make_shared<LatexScene>(ls), "ls", 0.5, 0.1);
    cs.stage_macroblock(SilenceBlock(14), 1);
    cs.render_microblock();
}

void sample_vibrations(){
    for(int x = -2; x <= 2; x++) for(int y = -2; y <= 2; y++) identify_vibrations(2.496147+.0000004*x, .2505+.000008*y);
}

void stack_diagrams(){
    CompositeScene cs;

    int bb = isv.size();
    for(int i = 0; i < bb; i++){
        PendulumScene ps(PendulumScene(isv[i].ps, .5, 1));
        string key = to_string(i);
        ps.global_publisher_key = true;
        ps.global_identifier = "p"+key+".";
        cs.add_scene(make_shared<PendulumScene>(ps), "ps"+key, .75, .5);
        cs.state.set({{"ps"+key+".opacity", "0"}});
    }
    for(int i = 0; i < bb; i++){
        string key = to_string(i);
        CoordinateSceneWithTrail cswt(1, 1);
        cswt.state.set({
            {"zoom", "0.02"},
            {"ticks_opacity", "0"},
            {"trail_opacity", "1"},
            {"trail_x", "{p"+key+".pendulum_theta1}"},
            {"trail_y", "{p"+key+".pendulum_theta2}"},
        });
        cs.add_scene(make_shared<CoordinateSceneWithTrail>(cswt), "coord"+key, (i+.5)/bb, 0.5);
    }
    cs.stage_macroblock(SilenceBlock(10), 1);
    cs.render_microblock();
}

void render_video() {
    SAVE_FRAME_PNGS = false;
    //PRINT_TO_TERMINAL = false;
    //FOR_REAL = false;

    const int frequency = 10;
    const int physmult = frequency/FRAMERATE;
    const double stepsize = 1./frequency;
    //stack_diagrams();
    //return;

    double delta = 0.001;
    vector<PendulumGrid> grids{PendulumGrid(VIDEO_HEIGHT*2, VIDEO_HEIGHT*2, delta, -M_PI, M_PI, -M_PI, M_PI, 0, 0, 0, 0)};
    for (const vector<IslandShowcase>& isvh : {isv}) for(const IslandShowcase& is : isvh) {
        const double ro2 = is.range/2;
        const double t1 = is.ps.theta1;
        const double t2 = is.ps.theta2;
        grids.push_back(PendulumGrid(VIDEO_WIDTH, VIDEO_HEIGHT, delta, t1-ro2*VIDEO_WIDTH/VIDEO_HEIGHT, t1+ro2*VIDEO_WIDTH/VIDEO_HEIGHT, t2-ro2, t2+ro2, 0, 0, 0, 0));
    }
    shared_ptr<PendulumGridScene> pgs = make_shared<PendulumGridScene>(grids);
    /*
    stack_diagrams(); return;
    pgs.state.set({
        {"rk4_step_size", to_string(stepsize)},
        {"physics_multiplier", to_string(physmult)},
    });
    pgs.stage_macroblock(SilenceBlock(4), 1);
    pgs.render_microblock();
    pgs.state.transition(MICRO, {
        {"mode", "2"},
    });
    pgs.stage_macroblock(SilenceBlock(2), 1);
    pgs.render_microblock();
    pgs.state.set("physics_multiplier", "0");
    */
    for(IslandShowcase is : isv) {
        showcase_an_island(pgs, is);
    }
}
