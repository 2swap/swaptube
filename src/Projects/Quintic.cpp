#include "../Scenes/Math/ComplexPlotScene.cpp"
#include "../Scenes/Math/RealFunctionScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/ThreeDimensionScene.cpp"

/*
    cps->stage_swap_roots_when_in_root_mode(MICRO, "0","2");
    cps->render_microblock();
    cps->stage_swap_roots_when_in_root_mode(MICRO, "1","0");
    cps->render_microblock();
    cps->stage_swap_roots_when_in_root_mode(MICRO, "2","1");
    cps->render_microblock();
    cps->stage_swap_roots_when_in_root_mode(MICRO, "0","2");
    cps->render_microblock();
    */

void render_video(){
    cout << "Rendering Quintic video..." << endl;
    shared_ptr<ComplexPlotScene> cps = make_shared<ComplexPlotScene>(3);
    cps->stage_macroblock(FileBlock("This is the relationship between a polynomial's coefficients and its roots."), 10);
    cps->state_manager.set("ticks_opacity", "0");

    for(int i = 0; i < cps->degree; i++) {
        cps->state_manager.transition(MICRO, {
            {"root"+to_string(i)+"_r", "<t> 1.2 * 6.28 .3333 " + to_string(i) + " * * + sin"},
            {"root"+to_string(i)+"_i", "<t> .8 * 6.28 .3333 " + to_string(i) + " * * + cos .2 +"},
        });
    }
    cps->render_microblock();
    cps->render_microblock();
    cps->render_microblock();
    cps->render_microblock();
    cps->render_microblock();
    cps->state_manager.transition(MICRO, {
        {"coefficient0_ring", "1"},
        {"coefficient1_ring", "1"},
        {"coefficient2_ring", "1"},
    });
    cps->render_microblock();
    cps->state_manager.transition(MICRO, {
        {"coefficient0_ring", "0"},
        {"coefficient1_ring", "0"},
        {"coefficient2_ring", "0"},
    });
    cps->render_microblock();
    cps->state_manager.transition(MICRO, {
        {"root0_opacity", "1"},
        {"root1_opacity", "1"},
        {"root2_opacity", "1"},
    });
    cps->render_microblock();
    cps->state_manager.transition(MICRO, {
        {"root0_opacity", "0"},
        {"root1_opacity", "0"},
        {"root2_opacity", "0"},
    });
    cps->render_microblock();
    cps->state_manager.transition(MICRO, {
        {"root0_r", "1"},
        {"root0_i", ".2"},
        {"root1_r", ".5"},
        {"root1_i", "1.2"},
        {"root2_r", "-1"},
        {"root2_i", "-.3"},
    });
    cps->render_microblock();

    cps->stage_macroblock(FileBlock("Notice how moving a single root has a hard-to-predict effect on the coefficients,"), 4);
    cps->state_manager.set({
        {"root0_opacity", "0"},
    });
    cps->state_manager.transition(MICRO, {
        {"root0_opacity", "1"},
    });
    cps->render_microblock();
    cps->state_manager.set({
        {"root0_r", "1 <spin_coefficient_r> +"},
        {"root0_i", ".2 <spin_coefficient_i> +"},
        {"spin_coefficient_r", "<t> 3 * sin <spin_multiplier> *"},
        {"spin_coefficient_i", "<t> 5 * cos <spin_multiplier> *"},
        {"spin_multiplier", "0"},
    });
    cps->state_manager.transition(MICRO, {
        {"spin_multiplier", ".5"},
    });
    cps->render_microblock();
    cps->render_microblock();
    cps->state_manager.transition(MICRO, {
        {"spin_multiplier", "0"},
        {"root0_opacity", "0"},
    });
    cps->render_microblock();

    cps->state_manager_roots_to_coefficients();
    cps->stage_macroblock(FileBlock("and moving a single coefficient has a hard-to-predict effect on the roots."), 4);
    cps->state_manager.transition(MICRO, {
        {"coefficient0_ring", "1"},
    });
    cps->render_microblock();
    cps->state_manager.set({
        {"coefficient0_r", cps->state_manager.get_equation("coefficient0_r") + " <spin_coefficient_r> +"},
        {"coefficient0_i", cps->state_manager.get_equation("coefficient0_i") + " <spin_coefficient_i> +"},
    });
    cps->state_manager.transition(MICRO, {
        {"spin_multiplier", ".5"},
    });
    cps->render_microblock();
    cps->render_microblock();
    cps->state_manager.transition(MICRO, {
        {"coefficient0_ring", "0"},
        {"spin_multiplier", "0"},
    });
    cps->render_microblock();

    cps->stage_macroblock(FileBlock("It's a shame you've never seen this plot,"), 1);
    cps->state_manager_coefficients_to_roots();
    for(int i = 0; i < cps->degree; i++) {
        cps->state_manager.transition(MICRO, {
            {"root"+to_string(i)+"_r", "<t> 1.2 * 6.28 .3333 " + to_string(i) + " * * + sin"},
            {"root"+to_string(i)+"_i", "<t> .8 * 6.28 .3333 " + to_string(i) + " * * + cos"},
        });
    }
    cps->render_microblock();

    cps->stage_macroblock(CompositeBlock(FileBlock("because their relationship _is the core, the essence of algebra_."), SilenceBlock(1)), 1);
    cps->state_manager.transition(MICRO, {
        {"root0_r", ".4"},
        {"root0_i", "0"},
        {"root1_r", ".5"},
        {"root1_i", "1.2"},
        {"root2_r", "-1"},
        {"root2_i", "-.3"},
        {"center_y", ".6"},
    });
    cps->render_microblock();

    CompositeScene cs;
    cs.add_scene(cps, "cps");

    shared_ptr<LatexScene> ls = make_shared<LatexScene>("ax^0+bx^1+cx^2+x^3", 1, 1, .5);
    cs.add_scene_fade_in(MICRO, ls, "ls", .5, .2);
    cs.stage_macroblock(FileBlock("Polynomials have a standard form, where each term has an associated coefficient."), 4);
    cs.render_microblock();
    string colory = latex_color(0xffff0000, "a")+"x^0+"+latex_color(0xff00ff00, "b")+"x^1+"+latex_color(0xff0000ff, "c")+"x^2+"+/*latex_color(0xffcccc00, "d")+*/"x^3";
    ls->begin_latex_transition(MICRO, colory);
    cs.render_microblock();
    ls->begin_latex_transition(MICRO, "ax^0+bx^1+cx^2+x^3");
    cs.render_microblock();
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("They're drawn on the graph here."), 5);
    cs.render_microblock();
    for(int i = 0; i < 2; i++) {
        cps->state_manager.transition(MICRO, {{"coefficient0_ring","1"}, {"coefficient1_ring","1"}, {"coefficient2_ring","1"}});
        ls->begin_latex_transition(MICRO, colory);
        cs.render_microblock();

        cps->state_manager.transition(MICRO, {{"coefficient0_ring","0"}, {"coefficient1_ring","0"}, {"coefficient2_ring","0"}});
        ls->begin_latex_transition(MICRO, "ax^0+bx^1+cx^2+x^3");
        cs.render_microblock();
    }
    cps->state_manager.transition(MICRO, {{"coefficient0_opacity","0"}, {"coefficient1_opacity","0"}, {"coefficient2_opacity","0"}});
    cps->state_manager.transition(MICRO, {{"center_y", "0"}});
    cs.fade_subscene(MICRO, "ls", 0);

    shared_ptr<LatexScene> ls2 = make_shared<LatexScene>("(x-r_1)(x-r_2)(x-r_3)", .7, 1, .5);
    cs.add_scene_fade_in(MICRO, ls2, "ls2", .5, .8);
    cs.stage_macroblock(FileBlock("There's also a factored form, with one term for each root,"), 4);
    cs.render_microblock();
    ls2->begin_latex_transition(MICRO, "(x-" + latex_color(0xffff0000, "r_1")+")(x-"+latex_color(0xff00ff00, "r_2")+")(x-"+latex_color(0xff0000ff, "r_3")+")");
    cs.render_microblock();
    ls2->begin_latex_transition(MICRO, "(x-r_1)(x-r_2)(x-r_3)");
    cs.render_microblock();
    cs.render_microblock();
    cs.remove_subscene("ls");

    cs.stage_macroblock(FileBlock("which are clearly visible in the polynomial's graph."), 4);
    for(int i = 0; i < 2; i++) {
        cps->state_manager.transition(MICRO, {{"root0_opacity", ".5"}, {"root1_opacity", ".5"}, {"root2_opacity", ".5"}});
        ls2->begin_latex_transition(MICRO, "(x-" + latex_color(0xffff0000, "r_1")+")(x-"+latex_color(0xff00ff00, "r_2")+")(x-"+latex_color(0xff0000ff, "r_3")+")");
        cs.render_microblock();

        cps->state_manager.transition(MICRO, {{"root0_opacity", "0"}, {"root1_opacity", "0"}, {"root2_opacity", "0"}});
        ls2->begin_latex_transition(MICRO, "(x-r_1)(x-r_2)(x-r_3)");
        cs.render_microblock();
    }

    cs.fade_subscene(MICRO, "ls2", 0);
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cps->state_manager.transition(MICRO, {
        {"ticks_opacity", "1"},
    });
    cps->state_manager.transition(MACRO, {
        {"zoom", "0.1"},
    });
    cs.stage_macroblock(FileBlock("The space we're looking at is the complex plane, home to numbers like i and 2-i."), 6);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 1), "i"));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(2, -1), "2-i"));
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MICRO, {{"ticks_opacity", "0"}, {"zoom", ".2"}});
    cs.render_microblock();

    cs.fade_subscene(MICRO, "ls2", 1);
    cps->state_manager.transition(MICRO, "geometry_opacity", "0");
    ls2->begin_latex_transition(MICRO, "(" + latex_color(0xffff0000, "x")+"-r_1)(" + latex_color(0xffff0000, "x")+"-r_2)(" + latex_color(0xffff0000, "x")+"-r_3)");
    cs.stage_macroblock(FileBlock("For each pixel on the screen, I passed that complex number as an input to the polynomial,"), 4);
    // TODO animate the input and output
    cs.render_microblock();
    cps->construction.clear();
    cps->state_manager.set("geometry_opacity", "1");
    cs.render_microblock();
    ls2->begin_latex_transition(MICRO, "(" + latex_color(0xffff0000, "2-i")+"-r_1)(" + latex_color(0xffff0000, "2-i")+"-r_2)(" + latex_color(0xffff0000, "2-i")+"-r_3)");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and colored it according to the output."), 1);
    cs.render_microblock();

    cs.fade_subscene(MICRO, "ls2", 0);
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "0"},
    });
    cs.stage_macroblock(FileBlock("The brightness shows magnitude of the output, or distance to zero,"), 5);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.remove_subscene("ls2");
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "20"},
    });
    cs.stage_macroblock(FileBlock("and the color shows the angle of the output."), 3);
    cs.render_microblock();
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", ".8"},
    });
    cps->state_manager_coefficients_to_roots();
    cps->state_manager.transition(MICRO, {
        {"root0_r", "-1.2"},
        {"root0_i", "0"},
        {"root1_r", "-.3"},
        {"root1_i", "0"},
        {"root2_r", "1.2"},
        {"root2_i", "0"},
    });
    cs.render_microblock();
    cs.remove_all_subscenes();

    ThreeDimensionScene tds;
    tds.add_surface(Surface(glm::vec3(0, 0, 0), glm::vec3(.5, 0, 0), glm::vec3(0, .5, 0), "cps"), cps);
    tds.stage_macroblock(FileBlock("Doing this for every point, we can graph our complex-valued function."), 2);
    tds.state_manager.transition(MICRO, {
        {"d", "1.5"},
    });
    tds.render_microblock();

    shared_ptr<RealFunctionScene> rfs = make_shared<RealFunctionScene>();
    rfs->add_function("? 1.2 - ? .3 + ? 1.2 + * *", 0xffff0000);
    tds.add_surface(Surface(glm::vec3(0, 0, 0), glm::vec3(.5, 0, 0), glm::vec3(0, 0, .5), "rfs"), rfs);
    tds.state_manager.transition(MICRO, {
        {"qi", "-.3 <t> sin .07 * +"},
        {"qj", "<t> cos .025 *"},
    });
    tds.render_microblock();

    shared_ptr<LatexScene> f_complex = make_shared<LatexScene>("\\text{Complex Function}", 1, .5, .5);
    shared_ptr<LatexScene> f_real    = make_shared<LatexScene>("\\text{Real Function}", 1, .5, .5);
    tds.add_surface(Surface(glm::vec3(0, .25,-.05), glm::vec3(.25, 0, 0), glm::vec3(0, .25, 0), "f_complex"), f_complex);
    tds.add_surface(Surface(glm::vec3(0, .05,-.25), glm::vec3(.25, 0, 0), glm::vec3(0, 0, .25), "f_real"), f_real);
    tds.stage_macroblock(FileBlock("You might ask, what's all this complex number business? Why leave the familiar land of the reals?"), 1);
    tds.render_microblock();

    tds.state_manager.transition(MICRO, {
        {"d", "1"},
        {"qi", "0"},
        {"qj", "0"},
    });
    tds.stage_macroblock(FileBlock("After all, complex numbers were invented by big math to sell more calculators, right?"), 1);
    tds.fade_subscene(MICRO, "f_complex", 0);
    tds.fade_subscene(MICRO, "f_real", 0);
    tds.fade_subscene(MICRO, "rfs", 0);
    tds.render_microblock();
    tds.remove_all_subscenes();

    cs = CompositeScene();
    cs.add_scene(cps, "cps");
    cps->state_manager_coefficients_to_roots();
    for(int i = 0; i < cps->degree; i++) {
        cps->state_manager.transition(MICRO, {
            {"root"+to_string(i)+"_r", "<t> 1.2 * 6.28 .3333 " + to_string(i) + " * * + sin"},
            {"root"+to_string(i)+"_i", "<t> .8 * 6.28 .3333 " + to_string(i) + " * * + cos 5 -"},
        });
    }
    cs.stage_macroblock(FileBlock("But in turn, I would ask _you_, who needs decimals or negatives either?"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Imagine there's nothing but natural numbers- 0, 1, 2, 3, sitting happily on the right side of the number line."), 5);
    for(int x = 0; x <= 4; x++) {
        cps->construction.add(GeometricPoint(glm::vec2(x, 0), to_string(x)));
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("In such a world, there's a big problem..."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("There are equations with no solution, like x+2=0."), 1);
    shared_ptr<LatexScene> impossible = make_shared<LatexScene>("x+2=0", .5, 1, .4);
    cs.add_scene_fade_in(MICRO, impossible, "impossible", .5, .7);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("There's nothing we can write here for x that would make the left equal zero."), 1);
    impossible->begin_latex_transition(MACRO, latex_color(0xffff0000, "x")+"+2=0");
    cps->state_manager_roots_to_coefficients();
    cps->state_manager.transition(MICRO, {
        {"coefficient3_r", "0.000001"},
        {"coefficient3_i", "0"},
        {"coefficient3_opacity", "0"},
        {"coefficient2_r", "0.000001"},
        {"coefficient2_i", "0"},
        {"coefficient2_opacity", "0"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"coefficient1_opacity", "1"},
        {"coefficient0_r", "2"},
        {"coefficient0_i", "0"},
        {"coefficient0_opacity", "1"},
        {"geometry_opacity", ".15"},
    });
    cs.render_microblock();
    cps->decrement_degree();
    cps->decrement_degree();

    cs.stage_macroblock(FileBlock("If I plot x+2, the coefficients land on numbers which exist in our number system,"), 4);
    cps->state_manager.transition(MICRO, {{"coefficient0_ring", "1"}, {"coefficient1_ring", "1"}});
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {{"coefficient0_ring", "0"}, {"coefficient1_ring", "0"}});
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {{"coefficient0_ring", "1"}, {"coefficient1_ring", "1"}});
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {{"coefficient0_ring", "0"}, {"coefficient1_ring", "0"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but the solution to the equation is -2... it lands outside of our number system."), 4);
    cps->state_manager.transition(MICRO, "root0_opacity", "1");
    cs.render_microblock();
    impossible->begin_latex_transition(MICRO, latex_color(0xff00ff00, "-2")+"+2=0");
    cps->state_manager.transition(MICRO, "root0_opacity", "0");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_opacity", "1");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_opacity", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("So, to be able to solve this, we _need_ to invent -2..."), 2);
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(-2, 0), "-2"));
    cs.fade_subscene(MICRO, "impossible", 0);
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("as well as the other negatives."), SilenceBlock(2)), 6);
    for(int x = -1; x >= -4; x--) {
        if(x == -2) continue; // already added
        cps->state_manager.transition(MICRO, {
            {"coefficient0_r", to_string(-x)},
            {"coefficient0_i", "0"},
        });
        cs.render_microblock();
        cps->construction.add(GeometricPoint(glm::vec2(x, 0), to_string(x)));
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("Sadly, our number system is still lacking."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MICRO, {
        {"coefficient1_r", "2"},
        {"coefficient0_r", "1"},
    });
    impossible->jump_latex("2x+1=0");
    cs.fade_subscene(MICRO, "impossible", 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Once again, with our coefficients on existing numbers, the solution lands on -1/2, which is not yet in our number system."), 3);
    cps->state_manager.transition(MICRO, {
        {"coefficient1_ring", "1"},
        {"coefficient0_ring", "1"},
    });
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(-.5, 0), "-.5", .45));
    cps->state_manager.transition(MICRO, {
        {"coefficient1_ring", "0"},
        {"coefficient0_ring", "0"},
        {"coefficient1_opacity", "0"},
        {"coefficient0_opacity", "0"},
        {"root0_opacity", "1"},
    });
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"root0_opacity", "0"},
        {"root1_opacity", "0"},
    });
    cs.render_microblock();

    int microblock_count = 0;
    impossible->begin_latex_transition(MACRO, "2\\cdot"+latex_color(0xff00ff00, "-\\frac{1}{2}")+"+1=0");
    for(int counting = 0; counting < 2; counting++) {
        for(int power_of_two = -1; power_of_two > -3; power_of_two--) {
            float dx = 1.0 / (1 << -power_of_two);
            for(float x = -5 + dx; x <= 5 - dx; x+=2*dx) {
                if(counting == 1) {
                    if(x != -0.5) cps->construction.add(GeometricPoint(glm::vec2(x, 0), float_to_pretty_string(x), dx*.9));
                    cs.render_microblock();
                }
                else microblock_count++;
            }
        }
        if(counting == 0) cs.stage_macroblock(FileBlock("It looks like we need fractions too..."), microblock_count);
    }

    cs.stage_macroblock(CompositeBlock(FileBlock("Heck, let's just add all decimals. We get a nice continuum of numbers."), FileBlock("Surely our number system is finally complete!")), 2);
    cs.fade_subscene(MICRO, "impossible", 0);
    cps->construction.add(GeometricLine(glm::vec2(-5, 0), glm::vec2(5, 0)));
    cs.render_microblock();
    cs.render_microblock();


    cs.stage_macroblock(FileBlock("Well then, what about this equation?"), 1);
    impossible->jump_latex("x^2+1=0");
    cs.fade_subscene(MACRO, "impossible", 1);
    cs.render_microblock();
    cps->increment_degree();

    cs.remove_all_subscenes();

    cs.stage_macroblock(SilenceBlock(1), 1);
    shared_ptr<ThreeDimensionScene> tds2 = make_shared<ThreeDimensionScene>();
    cs.add_scene(tds2, "tds2");
    cs.add_scene(impossible, "impossible", .5, .7);
    tds2->add_surface(Surface(glm::vec3(0, 0, 0), glm::vec3(.5, 0, 0), glm::vec3(0, .5, 0), "cps"), cps);

    rfs = make_shared<RealFunctionScene>();
    rfs->add_function("? 2 * 1 +", 0xffff0000);
    tds2->add_surface(Surface(glm::vec3(0, 0, 0), glm::vec3(.5, 0, 0), glm::vec3(0, 0, .5), "rfs"), rfs);
    tds2->state_manager.transition(MICRO, {
        {"qi", "-.1 <t> 2 / sin .02 * +"},
        {"qj", "<t> 2 / cos .01 *"},
        {"d", "1.5"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    impossible->begin_latex_transition(MICRO, latex_color(0xff00ff00, "x^2")+"+1=0");
    cps->state_manager.transition(MICRO, {
        {"coefficient2_r", "1"},
        {"coefficient0_r", "0"},
        {"coefficient0_i", "0"},
    });
    rfs->begin_transition(MICRO, 0, "? ? * ? 2 * +");
    cs.render_microblock();

    cps->state_manager.transition(MICRO, {
        {"coefficient0_r", "1"},
        {"coefficient0_i", "0"},
        {"coefficient1_r", "0"},
        {"coefficient1_i", "0"},
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"construction_opacity", ".7"},
    });
    rfs->begin_transition(MICRO, 0, "? ? * 1 +");
    cs.stage_macroblock(CompositeBlock(FileBlock("No real number squared gives us negative 1..."), FileBlock("Just like before, our number line must be missing something...")), 2);
    cs.render_microblock();
    cs.render_microblock();

    tds2->state_manager.transition(MICRO, {
        {"d", "1"},
        {"qi", "0"},
        {"qj", "0"},
    });
    tds2->fade_subscene(MICRO, "rfs", 0);
    cs.stage_macroblock(FileBlock("Let's call the solutions to this equation 'i' and '-i'."), 4);
    cs.fade_subscene(MICRO, "impossible", 0);
    //impossible->begin_latex_transition(MICRO, "i^2+1=0");
    cs.render_microblock();
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 1), "i"));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, -1), "-i"));
    cs.render_microblock();
    tds2->remove_subscene("cps");
    cs.remove_subscene("tds2");
    cs.add_scene(cps, "cps");

    cs.render_microblock();

    cs.stage_macroblock(FileBlock("along with 2i, negative 2i, and so on,"), 3);
    cs.fade_subscene(MACRO, "impossible", 0);
    for(int y : {2, -2}) {
        string label = to_string(y) + "i";
        if(y == -1) label = "-i";
        cps->construction.add(GeometricPoint(glm::vec2(0, y), label));
        cs.render_microblock();
    }
    cps->construction.add(GeometricLine(glm::vec2(0, -3), glm::vec2(0, 3)));
    cs.render_microblock();
    cs.remove_subscene("impossible");

    // Add gaussian integer grid
    cs.stage_macroblock(FileBlock("which, combined with the real numbers, forms the complex plane."), 2*4*8);
    for(int x = -4; x <= 4; x++) {
        for(int y = -2; y <= 2; y++) {
            if(x == 0 || y == 0) continue;
            string label = to_string(x);
            if(y != 0){
                if(y < 0) label += "-";
                else label += "+";
                if(abs(y) != 1) label += to_string(abs(y));
                label += "i";
            }
            cps->construction.add(GeometricPoint(glm::vec2(x, y), label, .5));
            cs.render_microblock();
        }
    }
    while(cs.microblocks_remaining()) cs.render_microblock();

    cps->state_manager_coefficients_to_roots();
    for(int i = 0; i < cps->degree; i++) {
        cps->state_manager.transition(MICRO, {
            {"root"+to_string(i)+"_r", "<t> 1.2 * 6.28 .3333 " + to_string(i) + " * * + sin"},
            {"root"+to_string(i)+"_i", "<t> .8 * 6.28 .3333 " + to_string(i) + " * * + cos"},
        });
    }

    cs.stage_macroblock(FileBlock("Up until now, we've been playing whack-a-mole..."), 1);
    cps->state_manager.transition(MICRO, "geometry_opacity", "0");
    cs.render_microblock();
    cps->construction.clear();
    cps->state_manager.set("geometry_opacity", "1");

    cs.stage_macroblock(FileBlock("_You_ invent numbers, and _I_ make an equation that can't be solved without _even more numbers_."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But that game ends here!"), 1);
    cps->state_manager.set({{"coefficient0_ring", "0"}, {"coefficient1_ring", "0"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Wherever I put the coefficients of the polynomial inside the complex plane,"), 2);
    cps->state_manager.transition(MICRO, {
        {"coefficient0_opacity", "1"},
        {"coefficient1_opacity", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("the solutions may move around, but they don't jump out of the existing number system like they did before."), 2);
    cps->state_manager.transition(MICRO, {
        {"coefficient0_opacity", "0"},
        {"coefficient1_opacity", "0"},
        {"root0_opacity", "1"},
        {"root1_opacity", "1"},
    });
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"root0_opacity", "0"},
        {"root1_opacity", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("What happens on the complex plane stays on the complex plane."), 1);
    cs.render_microblock();

    shared_ptr<LatexScene> fta = make_shared<LatexScene>("\\mathbb{C} \\text{ is algebraically closed.}", 1, .6, .5);
    cs.add_scene_fade_in(MICRO, fta, "fta", .5, .6);
    cs.stage_macroblock(FileBlock("The fancy way to say this is that the complex field is algebraically closed."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This is so important, that it's called:"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("THE FUNDAMENTAL THEOREM OF ALGEBRA"), 15);
    cs.render_microblock();
    cs.render_microblock();
    shared_ptr<LatexScene> fta_title_1 = make_shared<LatexScene>("\\text{The}", 1, .12, .12);
    cs.add_scene(fta_title_1, "fta_title_1", .07, .1);
    cs.render_microblock();
    shared_ptr<LatexScene> fta_title_2 = make_shared<LatexScene>("\\text{Fundamental}", 1, .6, .3);
    cs.add_scene(fta_title_2, "fta_title_2", .37, .1);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    shared_ptr<LatexScene> fta_title_3 = make_shared<LatexScene>("\\text{Theorem}", 1, .45, .25);
    cs.add_scene(fta_title_3, "fta_title_3", .82, .1);
    cs.render_microblock();
    shared_ptr<LatexScene> fta_title_4 = make_shared<LatexScene>("\\text{of}", 1, .12, .12);
    cs.add_scene(fta_title_4, "fta_title_4", .35, .3);
    cs.render_microblock();
    shared_ptr<LatexScene> fta_title_5 = make_shared<LatexScene>("\\text{Algebra}", 1, .7, .35);
    cs.add_scene(fta_title_5, "fta_title_5", .55, .3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 2);
    cs.render_microblock();
    cs.fade_all_subscenes_except(MICRO, "cps", 0);
    cs.render_microblock();
    cs.remove_all_subscenes_except("cps");

    cs.stage_macroblock(FileBlock("Our graph shows us something else too:"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The number of roots never changes either."), 2);
    cps->state_manager_coefficients_to_roots();
    cps->state_manager.transition(MICRO, {
        {"root0_opacity", "1"},
        {"root1_opacity", "1"},
        {"root0_r", "-1"},
        {"root0_i", "0"},
        {"root1_r", "2"},
        {"root1_i", "0"},
    });
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"root0_opacity", "0"},
        {"root1_opacity", "0"},
    });
    cs.render_microblock();

    shared_ptr<LatexScene> quadratic = make_shared<LatexScene>("x^2-x-2", .7, 1, .5);
    cs.add_scene_fade_in(MICRO, quadratic, "quadratic", .5, .2);
    cs.stage_macroblock(FileBlock("We're looking at a quadratic polynomial."), 1);
    cs.render_microblock();

    quadratic->begin_latex_transition(MICRO, "x^"+latex_color(0xffff0000, "2")+"-x-2");
    cs.stage_macroblock(FileBlock("The highest exponent is 2, so there's 2 roots."), 3);
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"root0_opacity", "1"},
        {"root1_opacity", "1"},
    });
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"root0_opacity", "0"},
        {"root1_opacity", "0"},
    });
    cs.render_microblock();

    quadratic->begin_latex_transition(MICRO, "x^3+x^2-x-2");
    cps->increment_degree();
    cps->state_manager_roots_to_coefficients();
    cps->state_manager.transition(MICRO, {
        {"coefficient3_r", "1"},
        {"coefficient3_i", "0"},
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "-1"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "-2"},
        {"coefficient0_i", "0"},
    });
    cs.stage_macroblock(FileBlock("With a cubic polynomial, there are 3 roots."), 2);
    cs.render_microblock();
    cs.fade_subscene(MICRO, "quadratic", 0);
    cs.render_microblock();
    cs.remove_all_subscenes_except("cps");

    cs.stage_macroblock(FileBlock("Well, unless we intentionally scrunch them on top of each other... but that's cheating."), 1);
    cps->state_manager_coefficients_to_roots();
    cps->state_manager.transition(MICRO, {
        {"root0_r", "0"},
        {"root0_i", "0"},
        {"root1_r", "0"},
        {"root1_i", "0"},
        {"root2_r", "-2"},
        {"root2_i", "2"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We call this a root of multiplicity 2. It counts as 2 normal roots."), 2);
    cps->state_manager.transition(MICRO, {
        {"root0_opacity", "1"},
    });
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"root0_opacity", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("You can actually identify the roots' multiplicities from the graph. Can you see it?"), 1);
    cps->state_manager.transition(MICRO, {
        {"ticks_opacity", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Tracing around this normal root, we see the colors blue-green-red."), 1);
    cps->state_manager.transition(MACRO, {
        {"center_x", "-2"},
        {"center_y", "2"},
        {"zoom", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Tracing around the multiplicity 2 root, we see the colors blue-green-red-blue-green-red."), 1);
    cps->state_manager.transition(MACRO, {
        {"center_x", "0"},
        {"center_y", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The color wheel is duplicated!"), 1);
    cps->state_manager.transition(MICRO, {
        {"zoom", "0.2"},
        {"ticks_opacity", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Plotting the function but evaluated at this point,"), 1);
    cs.render_microblock();

    return;
    cs.stage_macroblock(FileBlock("If Algebra was a play,"), 1);
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("the Complex Numbers are the stage it was written to be performed on."), 1);
    cs.render_microblock();
}
