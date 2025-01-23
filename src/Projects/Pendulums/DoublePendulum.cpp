#include "../Scenes/Common/ThreeDimensionScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Physics/PendulumScene.cpp"
#include "../Scenes/Physics/PendulumGridScene.cpp"

void butterfly(const double sx, const double sy) {
    CompositeScene cs;
    vector<PendulumScene> vps;
    for(int i = 0; i < 9; i++){
        PendulumState pendulum_state = {sx+(i/3-1)*.01, sy + (i%3-1) * .01, .0, .0};
        PendulumScene ps(pendulum_state);
        ps.state_manager.set({
            {"background_opacity", "0"},
            {"pendulum_opacity", "1"},
            {"physics_multiplier", "16"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        vps.push_back(ps);
    }
    for(int i = 0; i < vps.size(); i++){
        string key = "ps" + to_string(i);
        cs.add_scene(&(vps[i]), key);
    }
    cs.inject_audio_and_render(SilenceSegment(10));
}

void grid() {
    CompositeScene cs;
    vector<PendulumScene> vps;
    int gridsize = 41;
    double gridstep = 1./gridsize;
    for(int x = 0; x < gridsize; x++){
        for(int y = 0; y < gridsize; y++){
            PendulumState pendulum_state = {(x-gridsize/2)*.1, (y-gridsize/2)*.1, .0, .0};
            PendulumScene ps(pendulum_state, gridstep*1.02, gridstep*1.02);
            StateSet state = {
                {"pendulum_opacity",   "[pendulum_opacity]"  },
                {"background_opacity", "[background_opacity]"},
                {"physics_multiplier", "[physics_multiplier]"},
                {"rk4_step_size",      "[rk4_step_size]"     },
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
        {"pendulum_opacity", "0"},
        {"background_opacity", "1"},
        {"physics_multiplier", "1"},
        {"rk4_step_size", "1 30 / <physics_multiplier> /"},
    };
    cs.state_manager.set(state);
    cs.inject_audio_and_render(SilenceSegment(2));
}

void fractal() {
    PendulumGridScene pgs(0, 0);
    pgs.state_manager.set({
        {"physics_multiplier", "32"},
        {"mode", "1"},
        {"rk4_step_size", "1 30 / .05 *"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1 40 /"},
    });
    pgs.inject_audio_and_render(SilenceSegment(5));
}

void lissajous() {
    vector<PendulumState> ps;
    ps.push_back({0.3, .3, .0, .0});
    //ps.push_back({3.6, 3, .0, .0});
    //ps.push_back({2.49, .25, .0, .0});
    //ps.push_back({2, .3, .0, .0});
    //ps.push_back({0.1, .3, .0, .0});
    vector<double> zooms{0.5, .01, .02, .01, 0.5};
    for(int i = 0; i < ps.size(); i++){
        vector<float> left;
        vector<float> right;
        CompositeScene cs;
        PendulumScene this_ps(ps[i], 0.5, 0.5);
        this_ps.global_publisher_key = true;
        this_ps.state_manager.set({
            {"physics_multiplier", "16"},
            {"pendulum_opacity", "1"},
            {"background_opacity", "0.1"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        CoordinateScene coord;
        coord.state_manager.set({
            {"center_x", "0"},
            {"center_y", "0"},
            {"zoom", to_string(zooms[i])},
            {"trail_x", "{pendulum_theta1}"},
            {"trail_y", "{pendulum_theta2}"},
        });
        cs.add_scene(&this_ps, "this_ps", 0.25, 0.25);
        cs.add_scene(&coord, "coord", 0.5, 0.5);
        this_ps.generate_audio(6, left, right);
        //cs.inject_audio_and_render(GeneratedSegment(left, right));
        cs.inject_audio_and_render(SilenceSegment(6));
    }
}

void intro() {
FOR_REAL = false;
    ThreeDimensionScene tds;
    for(int i = 0; i < 4; i++){
        PendulumState pendulum_state = {5+.0001*i, 8, .0, .0};
        shared_ptr<PendulumScene> ps = make_shared<PendulumScene>(pendulum_state);
        ps->state_manager.set({
            {"background_opacity", "0"},
            {"pendulum_opacity", "1"},
            {"physics_multiplier", "16"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        tds.add_surface(Surface(glm::vec3(0,0,6+i*.01), glm::vec3(5,0,0), glm::vec3(0,5,0), ps));
    }
    vector<shared_ptr<LatexScene>> ls;
    ls.push_back(make_shared<LatexScene>(latex_text("Double"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("Pendulums"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("are"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("NOT"), 1));
    ls.push_back(make_shared<LatexScene>(latex_text("Chaotic"), 1));
    tds.state_manager.set({
        {"surfaces_opacity", "1"},
        {"lines_opacity", "0"},
        {"points_opacity", "1"},
        {"x", "0"},
        {"y", "0"},
        {"z", "3"},
        {"d", "6"},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
    });
    tds.inject_audio(FileSegment("Double pendulums are NOT chaotic."), 6);
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,-.6,-.8), glm::vec3(1,0,0), glm::vec3(0,1,0), ls[0]));
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,-.25,-.5), glm::vec3(1,0,0), glm::vec3(0,1,0), ls[1]));
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,.1,-.4), glm::vec3(.35,0,0), glm::vec3(0,.35,0), ls[2]));
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,.45,-.7), glm::vec3(1,0,0), glm::vec3(0,1,0), ls[3]));
    tds.render();
    tds.add_surface(Surface(glm::vec3(0,0.9,-.7), glm::vec3(1,0,0), glm::vec3(0,1,0), ls[4]));
    tds.render();
    ls[3]->begin_latex_transition(latex_text("NOT")+"^*");
    tds.inject_audio_and_render(FileSegment("Or, at least, not all of them."));
    tds.state_manager.macroblock_transition({
        {"d", "3"},
    });
    tds.inject_audio_and_render(FileSegment("You've probably seen videos like these,"));
    for(int i = 0; i < ls.size(); i++) tds.remove_surface(ls[i]);
    tds.inject_audio_and_render(FileSegment("where a tiny deviation in similar double pendulums amplifies over time,"));
    tds.inject_audio_and_render(FileSegment("until they eventually completely desynchronize."));
    tds.inject_audio_and_render(FileSegment("This is known as a chaotic system, because small changes in starting conditions yield vastly different behavior in the long run."));
    for(int i = 0; i < 4; i++){
        PendulumState pendulum_state = {2.49+.0001*i, 25, .0, .0};
        shared_ptr<PendulumScene> ps = make_shared<PendulumScene>(pendulum_state);
        ps->state_manager.set({
            {"background_opacity", "0"},
            {"pendulum_opacity", "0"},
            {"physics_multiplier", "[stable_physics_multiplier]"},
            {"rk4_step_size", "1 30 / <physics_multiplier> /"},
        });
        ps->state_manager.macroblock_transition({
            {"pendulum_opacity", "1"},
        });
        tds.add_surface(Surface(glm::vec3(0,4+i*0.2,6+i*.01), glm::vec3(5,0,0), glm::vec3(0,5,0), ps));
    }
FOR_REAL = true;
    tds.state_manager.set({
        {"stable_physics_multiplier", "0"},
    });
    tds.inject_audio_and_render(FileSegment("But you probably haven't seen this:"));
    tds.state_manager.set({
        {"stable_physics_multiplier", "16"},
    });
    tds.inject_audio_and_render(FileSegment("Here are a few more pendulums with slightly different starting positions."));
}

void render_video() {
    PRINT_TO_TERMINAL = false;
    SAVE_FRAME_PNGS = false;

    intro();
    /*butterfly(1, 1.25);
    butterfly(2.49, .25);
    lissajous();
    grid();
    fractal();*/
}
