#include "../Scenes/Math/ComplexPlotScene.cpp"
#include "../Scenes/Math/ComplexArbitraryFunctionScene.cpp"
#include "../Scenes/Math/RealFunctionScene.cpp"
#include "../Scenes/Math/RootFractalScene.cpp"
#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/ThreeDimensionScene.cpp"

void render_video(){
    shared_ptr<ComplexPlotScene> cps = make_shared<ComplexPlotScene>(3);
    cps->stage_macroblock(FileBlock("This is the relationship between a polynomial's coefficients and its solutions."), 10);
    cps->state_manager.set("ticks_opacity", "0");

    for(int i = 0; i < cps->get_degree(); i++) {
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
    cps->transition_coefficient_rings(MICRO, 1);
    cps->render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cps->render_microblock();
    cps->transition_root_rings(MICRO, 1);
    cps->render_microblock();
    cps->transition_root_rings(MICRO, 0);
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

    cps->stage_macroblock(CompositeBlock(FileBlock("Notice how moving a single solution has a hard-to-predict effect on the coefficients,"), SilenceBlock(1)), 4);
    cps->state_manager.set({
        {"root0_ring", "0"},
    });
    cps->state_manager.transition(MICRO, {
        {"root0_ring", "1"},
    });
    cps->render_microblock();
    cps->state_manager.set({
        {"root0_r", "1 <spin_coefficient_r> +"},
        {"root0_i", ".2 <spin_coefficient_i> +"},
        {"spin_coefficient_r", "<t> 2 * sin <spin_multiplier> *"},
        {"spin_coefficient_i", "<t> 3 * cos <spin_multiplier> *"},
        {"spin_multiplier", "0"},
    });
    cps->state_manager.transition(MICRO, {
        {"spin_multiplier", "1"},
    });
    cps->render_microblock();
    cps->render_microblock();
    cps->state_manager.transition(MICRO, {
        {"spin_multiplier", "0"},
        {"root0_ring", "0"},
    });
    cps->render_microblock();

    cps->roots_to_coefficients();
    cps->stage_macroblock(CompositeBlock(FileBlock("and moving a single coefficient has a hard-to-predict effect on the solutions."), SilenceBlock(1.5)), 4);
    cps->state_manager.transition(MICRO, {
        {"coefficient0_ring", "1"},
    });
    cps->render_microblock();
    cps->state_manager.set({
        {"coefficient0_r", cps->state_manager.get_equation("coefficient0_r") + " <spin_coefficient_r> +"},
        {"coefficient0_i", cps->state_manager.get_equation("coefficient0_i") + " <spin_coefficient_i> +"},
    });
    cps->state_manager.transition(MICRO, {
        {"spin_multiplier", "1"},
    });
    cps->render_microblock();
    cps->render_microblock();
    cps->state_manager.transition(MICRO, {
        {"coefficient0_ring", "0"},
        {"spin_multiplier", "0"},
    });
    cps->render_microblock();

    cps->stage_macroblock(FileBlock("It's a shame you've never seen this plot,"), 1);
    cps->coefficients_to_roots();
    for(int i = 0; i < cps->get_degree(); i++) {
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

    shared_ptr<LatexScene> ls = make_shared<LatexScene>(latex_color(0xff333333, "ax^3+bx^2+cx+d"), 1, 1, .5);
    cs.add_scene_fade_in(MICRO, ls, "ls", .5, .2);
    cs.stage_macroblock(FileBlock("Polynomials have a standard form, where each term has an associated coefficient."), 5);
    cs.render_microblock();
    cs.render_microblock();
    string colory = latex_color(0xff333333, latex_color(0xffff8080, "a")+"x^3+"+latex_color(0xff80ff80, "b")+"x^2+"+latex_color(0xff8080ff, "c")+"x+"+latex_color(0xffcccc00, "d"));
    ls->begin_latex_transition(MICRO, colory);
    cs.render_microblock();
    ls->begin_latex_transition(MICRO, latex_color(0xff333333, "ax^3+bx^2+cx+d"));
    cs.render_microblock();
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("They're drawn on the graph here."), 5);
    cs.render_microblock();
    for(int i = 0; i < 2; i++) {
        cps->transition_coefficient_rings(MICRO, 1);
        ls->begin_latex_transition(MICRO, colory);
        cs.render_microblock();
        cps->transition_coefficient_rings(MICRO, 0);
        ls->begin_latex_transition(MICRO, latex_color(0xff333333, "ax^3+bx^2+cx+d"));
        cs.render_microblock();
    }

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->transition_coefficient_opacities(MICRO, 0);
    cps->state_manager.transition(MICRO, {{"center_y", "0"}});
    cs.fade_subscene(MICRO, "ls", 0);
    cs.render_microblock();

    shared_ptr<LatexScene> factored = make_shared<LatexScene>(latex_color(0xff333333, "(x-r_1)(x-r_2)(x-r_3)"), .7, 1, .5);
    cs.add_scene_fade_in(MICRO, factored, "factored", .5, .8);
    cs.stage_macroblock(FileBlock("There's also a factored form, with one term for each solution,"), 4);
    cs.render_microblock();
    string colory_factored = latex_color(0xff333333, "(x-" + latex_color(0xff80ff80, "r_1")+")(x-"+latex_color(0xff8080ff, "r_2")+")(x-"+latex_color(0xffff8080, "r_3")+")");
    factored->begin_latex_transition(MICRO, colory_factored);
    cs.render_microblock();
    factored->begin_latex_transition(MICRO, latex_color(0xff333333, "(x-r_1)(x-r_2)(x-r_3)"));
    cs.render_microblock();
    cs.render_microblock();
    cs.remove_subscene("ls");

    cs.stage_macroblock(FileBlock("which are clearly visible in the polynomial's graph."), 4);
    for(int i = 0; i < 2; i++) {
        cps->transition_root_rings(MICRO, 1);
        factored->begin_latex_transition(MICRO, colory_factored);
        cs.render_microblock();
        cps->transition_root_rings(MICRO, 0);
        factored->begin_latex_transition(MICRO, latex_color(0xff333333, "(x-r_1)(x-r_2)(x-r_3)"));
        cs.render_microblock();
    }

    cs.fade_subscene(MICRO, "factored", 0);
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cps->state_manager.transition(MICRO, {
        {"ticks_opacity", "1"},
    });
    cps->state_manager.transition(MACRO, {
        {"zoom", "2"},
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
    cps->state_manager.transition(MICRO, {{"ticks_opacity", "0"}});
    cs.render_microblock();

    cs.fade_subscene(MICRO, "factored", 1);
    cps->state_manager.transition(MICRO, "geometry_opacity", "0");
    factored->begin_latex_transition(MICRO, "(" + latex_color(0xffff8080, "x")+"-r_1)(" + latex_color(0xffff8080, "x")+"-r_2)(" + latex_color(0xffff8080, "x")+"-r_3)");

    cps->coefficients_to_roots(); // Just to be safe
    cs.stage_macroblock(FileBlock("For each pixel on the screen, we can pass it into the polynomial, and see what comes out..."), 4);
    cps->state_manager.set({
        {"point_in_x", "0"},
        {"point_in_y", "1"},
    });
    cs.render_microblock();
    cps->construction.clear();
    cps->state_manager.set("geometry_opacity", "1");
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "in", .7, true));
    cs.render_microblock();
    factored->begin_latex_transition(MICRO, "(" + latex_color(0xffff8080, "-i")+"-r_1)(" + latex_color(0xffff8080, "-i")+"-r_2)(" + latex_color(0xffff8080, "-i")+"-r_3)");
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "out", .7, true));
    cps->state_manager.set({
        {"d0r", "<root0_r> <point_in_x> -"},
        {"d1r", "<root1_r> <point_in_x> -"},
        {"d2r", "<root2_r> <point_in_x> -"},
        {"d0i", "<root0_i> <point_in_y> -"},
        {"d1i", "<root1_i> <point_in_y> -"},
        {"d2i", "<root2_i> <point_in_y> -"},
        {"m01r", "<d0r> <d1r> * <d0i> <d1i> * -"},
        {"m01i", "<d0r> <d1i> * <d0i> <d1r> * +"},
        {"point_out_x", "<m01r> <d2r> * <m01i> <d2i> * -"},
        {"point_out_y", "<m01r> <d2i> * <m01i> <d2r> * +"},
    });
    cs.fade_subscene(MICRO, "factored", 0);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(5), 1);
    cps->state_manager.transition(MICRO, {
        {"point_in_x", "<t> sin .9 * 1.2 *"},
        {"point_in_y", "<t> cos .8 * 1.2 *"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We color the input point depending on where the output lands."), 1);
    cps->state_manager.transition(MICRO, {
        {"zoom", "1"},
    });
    cs.render_microblock();

    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "0"},
        {"geometry_opacity", "0"},
    });
    cs.stage_macroblock(FileBlock("The brightness shows magnitude of the output, or distance to zero,"), 5);
    cs.render_microblock();
    cps->construction.clear();
    cps->state_manager.transition(MICRO, {
        {"geometry_opacity", "1"},
    });
    cps->state_manager.remove({
        {"d0r"},
        {"d1r"},
        {"d2r"},
        {"d0i"},
        {"d1i"},
        {"d2i"},
        {"m01r"},
        {"m01i"},
        {"point_in_x"},
        {"point_in_y"},
        {"point_out_x"},
        {"point_out_y"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.remove_subscene("factored");
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "20"},
    });
    cs.stage_macroblock(FileBlock("and the color shows the angle of the output."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Doing this for every point, we can graph our complex-valued function."), 1);
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", ".8"},
    });
    cps->coefficients_to_roots();
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
    tds.stage_macroblock(SilenceBlock(2), 2);
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
    tds.add_surface_fade_in(MICRO, Surface(glm::vec3(0,.25,-.05), glm::vec3(.25, 0, 0), glm::vec3(0, .25, 0), "f_complex"), f_complex);
    tds.add_surface_fade_in(MICRO, Surface(glm::vec3(0,.05,-.25), glm::vec3(.25, 0, 0), glm::vec3(0, 0, .25), "f_real"), f_real);
    tds.stage_macroblock(FileBlock("You might ask, what's all this complex number business? Why leave the familiar land of real numbers?"), 1);
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
    cps->coefficients_to_roots();
    for(int i = 0; i < cps->get_degree(); i++) {
        cps->state_manager.transition(MICRO, {
            {"root"+to_string(i)+"_r", "<t> 1.2 * 6.28 .3333 " + to_string(i) + " * * + sin"},
            {"root"+to_string(i)+"_i", "<t> .8 * 6.28 .3333 " + to_string(i) + " * * + cos 5 -"},
        });
    }
    cs.stage_macroblock(FileBlock("In turn, I would ask _you_, who needs decimals or negatives either?"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Imagine there's nothing but natural numbers- 0, 1, 2, 3, sitting happily on the right side of the number line."), 9);
    cs.render_microblock();
    cs.render_microblock();
    for(int x = 0; x <= 4; x++) {
        cps->construction.add(GeometricPoint(glm::vec2(x, 0), to_string(x)));
        cs.render_microblock();
    }
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("In such a world, there's a big problem..."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("There are equations with no solution, like 2x+4=0."), 1);
    shared_ptr<LatexScene> impossible = make_shared<LatexScene>("2x+4=0", .5, 1, .4);
    cs.add_scene_fade_in(MICRO, impossible, "impossible", .5, .7);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We can write this equation, since 2, 4, and 0 are in our number system."), 10);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    impossible->begin_latex_transition(MICRO, latex_color(0xff333333, latex_color(0xffff8080, "2")+"x+"+latex_color(0xffcccc00, "4")+"="+latex_color(0xff333333, "0")));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(2, 0), ""));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(4, 0), ""));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), ""));
    cs.render_microblock();
    impossible->begin_latex_transition(MICRO, "2x+4=0");
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("However, there's nothing we can write here for x that would make this come true."), 1);
    impossible->begin_latex_transition(MACRO, latex_color(0xff333333, "2"+latex_color(0xffff8080, "x")+"+4=0"));
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's plot this polynomial."), 2);
    cps->state_manager.transition(MACRO, {
        {"geometry_opacity", ".15"},
    });
    cs.render_microblock();
    cps->roots_to_coefficients();
    cps->state_manager.set({
        {"linear_transitioner", "0"},
        {"exponential_transitioner", "<linear_transitioner> -2 * exp"},
        {"coefficient3_r", cps->state_manager.get_equation("coefficient3_r") + " <exponential_transitioner> * " + new_coefficient_val + " .5 * +"},
        {"coefficient3_i", cps->state_manager.get_equation("coefficient3_i") + " <exponential_transitioner> *"},
        {"coefficient2_r", cps->state_manager.get_equation("coefficient2_r") + " <exponential_transitioner> * " + new_coefficient_val + " .5 * +"},
        {"coefficient2_i", cps->state_manager.get_equation("coefficient2_i") + " <exponential_transitioner> *"},
    });
    cps->state_manager.transition(MICRO, {
        {"linear_transitioner", "5"},
        {"coefficient3_opacity", "0"},
        {"coefficient2_opacity", "0"},
        {"coefficient1_r", "2"},
        {"coefficient1_i", "0"},
        {"coefficient1_opacity", "1"},
        {"coefficient0_r", "4"},
        {"coefficient0_i", "0"},
        {"coefficient0_opacity", "1"},
    });
    cs.render_microblock();
    cps->set_degree(1);

    int flashes = 4;
    cs.stage_macroblock(FileBlock("The coefficients land on numbers which exist in our number system,"), flashes*2);
    for(int i = 0; i < flashes; i++) {
        cps->transition_coefficient_rings(MICRO, 1);
        cs.render_microblock();
        cps->transition_coefficient_rings(MICRO, 0);
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("but the solution to the equation is -2... which isn't one of the numbers we invented!"), flashes*2+1);
    cps->state_manager.transition(MACRO, "geometry_opacity", ".4");
    impossible->begin_latex_transition(MICRO, latex_color(0xff00ff00, "2\\small{(-2)}")+"+4=0");
    cs.render_microblock();
    for(int i = 0; i < flashes; i++) {
        cps->transition_root_rings(MICRO, 1);
        cs.render_microblock();
        cps->transition_root_rings(MICRO, 0);
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("To be able to solve this, we _need_ to invent -2..."), 2);
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(-2, 0), "-2"));
    cs.fade_subscene(MICRO, "impossible", 0);
    cs.render_microblock();
    cs.remove_subscene("impossible");

    cps->state_manager.transition(MICRO, {
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
    });
    cs.stage_macroblock(FileBlock("And, I can place the coefficients on other values, "), 1);
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("which force us to invent all the other negatives."), SilenceBlock(2)), 6);
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

    cs.stage_macroblock(CompositeBlock(CompositeBlock(SilenceBlock(1), FileBlock("Sadly, our number system is still lacking.")), SilenceBlock(1)), 1);
    cps->state_manager.transition(MICRO, {
        {"coefficient1_r", "2"},
        {"coefficient0_r", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    impossible = make_shared<LatexScene>("2x+1=0", .5, 1, .4);
    cs.add_scene_fade_in(MICRO, impossible, "impossible", .5, .7);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This equation, 2x+1=0, has the same problem."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Our coefficients lie on existing numbers, but the solution lands outside of our number system."), 3);
    /*cps->state_manager.transition(MICRO, {
        {"coefficient1_ring", "1"},
        {"coefficient0_ring", "1"},
    });*/
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(-.5, 0), "-.5", .45));
    //cps->transition_coefficient_rings(MICRO, 0);
    cps->transition_coefficient_opacities(MICRO, 0);
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    int microblock_count = 0;
    impossible->begin_latex_transition(MACRO, "2\\cdot"+latex_color(0xff00ff00, "\\small{-\\frac{1}{2}}")+"+1=0");
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
        if(counting == 0) cs.stage_macroblock(FileBlock("So, it looks like we need fractions too..."), microblock_count);
    }

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Heck, let's just add all decimals. We get a nice continuum of numbers."), 1);
    cs.fade_subscene(MICRO, "impossible", 0);
    cps->construction.add(GeometricLine(glm::vec2(-5, 0), glm::vec2(5, 0)));
    cs.render_microblock();
    cs.remove_all_subscenes();

    shared_ptr<ThreeDimensionScene> tds2 = make_shared<ThreeDimensionScene>();
    cs.stage_macroblock(CompositeBlock(FileBlock("Surely our number system is finally complete!"), SilenceBlock(2)), 1);
    cs.add_scene(tds2, "tds2");
    rfs = make_shared<RealFunctionScene>();
    rfs->add_function("? 2 * 1 +", 0xffff0000);
    rfs->state_manager.set("ticks_opacity", "0");
    tds2->add_surface(Surface(glm::vec3(0, 0, 0), glm::vec3(.5, 0, 0), glm::vec3(0, .5, 0), "cps"), cps);
    shared_ptr<RealFunctionScene> rfs2 = make_shared<RealFunctionScene>();
    rfs2->state_manager.set("ticks_opacity", "0");
    rfs2->add_function("1 ? ? * -", 0xffcc00ff);
    tds2->add_surface(Surface(glm::vec3(0, 0, 0), glm::vec3(0, .5, 0), glm::vec3(0, 0, .5), "rfs2"), rfs2);
    tds2->add_surface(Surface(glm::vec3(0, 0, 0), glm::vec3(.5, 0, 0), glm::vec3(0, 0, .5), "rfs"), rfs);
    tds2->state_manager.transition(MICRO, {
        {"qi", "-.1 <t> 2 / sin .02 * +"},
        {"d", "1.5"},
    });
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("Well then, what about this equation?"), SilenceBlock(1)), 2);
    cps->set_degree(2);
    cps->state_manager.transition(MACRO, {
        {"coefficient0_r", "1"},
        {"coefficient0_i", "0"},
        {"coefficient1_r", "0"},
        {"coefficient1_i", "0"},
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"geometry_opacity", ".7"},
    });
    rfs->begin_transition(MACRO, 0, "? ? * 1 +");
    shared_ptr<LatexScene> impossible_sq = make_shared<LatexScene>("x^2+1=0", .5, 1, .4);
    cs.add_scene_fade_in(MICRO, impossible_sq, "impossible_sq", .5, .7);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(1), CompositeBlock(FileBlock("No real number squared gives us negative 1..."), SilenceBlock(1))), 3);
    impossible_sq->begin_latex_transition(MICRO, latex_color(0xff333333, latex_color(0xff00ff00, "x^2")+"+1=0"));
    cs.render_microblock();
    cs.render_microblock();
    impossible_sq->begin_latex_transition(MICRO, "x^2+1=0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and you can see that here- none of the solutions are on our number line."), 4);
    for(int flash = 0; flash < 2; flash++) {
        cps->transition_root_rings(MICRO, 1);
        cs.render_microblock();
        cps->transition_root_rings(MICRO, 0);
        cs.render_microblock();
    }

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Just like before, our number line must be missing something..."), 1);
    tds2->state_manager.transition(MICRO, {
        {"qi", "-.1 <t> 2 / sin .02 * +"},
        {"qj", "-.2 <t> 2 / cos .01 * +"},
        {"d", "1.5"},
    });
    cs.render_microblock();

    tds2->state_manager.transition(MICRO, {
        {"d", "1"},
        {"qi", "0"},
        {"qj", "0"},
    });
    tds2->fade_subscene(MICRO, "rfs", 0);
    tds2->fade_subscene(MICRO, "rfs2", 0);
    cs.fade_subscene(MICRO, "impossible_sq", 0);

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's call these solutions 'i' and '-i'."), 4);
    cs.render_microblock();
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 1), "i"));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, -1), "-i"));
    cs.render_microblock();
    tds2->remove_all_subscenes();
    cs.remove_all_subscenes();
    cs.add_scene(cps, "cps");

    cs.stage_macroblock(FileBlock("and add 2i, negative 2i, and so on,"), 3);
    cs.fade_subscene(MACRO, "impossible_sq", 0);
    for(int y : {2, -2}) {
        string label = to_string(y) + "i";
        if(y == -1) label = "-i";
        cps->construction.add(GeometricPoint(glm::vec2(0, y), label));
        cs.render_microblock();
    }
    cps->construction.add(GeometricLine(glm::vec2(0, -2.7), glm::vec2(0, 2.7)));
    cs.render_microblock();
    cs.remove_subscene("impossible_sq");

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

    cps->coefficients_to_roots();
    for(int i = 0; i < cps->get_degree(); i++) {
        cps->state_manager.transition(MICRO, {
            {"root"+to_string(i)+"_r", "<t> 1.2 * 6.28 .3333 " + to_string(i) + " * * + sin"},
            {"root"+to_string(i)+"_i", "<t> .8 * 6.28 .3333 " + to_string(i) + " * * + cos"},
        });
    }

    cs.stage_macroblock(SilenceBlock(2), 1);
    cps->state_manager.transition(MICRO, "geometry_opacity", "0");
    cs.render_microblock();
    cps->construction.clear();
    cps->state_manager.set("geometry_opacity", "1");

    cs.stage_macroblock(FileBlock("Up until now, we've been playing whack-a-mole..."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("_You_ invent numbers, and _I_ make an equation that can't be solved without _more numbers_."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But that game ends here!"), 1);
    cps->state_manager.set({{"coefficient0_ring", "0"}, {"coefficient1_ring", "0"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Wherever we put the coefficients inside the complex plane,"), 2);
    cps->transition_coefficient_rings(MICRO, 1);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("the solutions may move around, but they don't jump out of the number system like they did before."), 2);
    cps->transition_coefficient_opacities(MICRO, 0);
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("What happens on the complex plane stays on the complex plane."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    shared_ptr<LatexScene> fta = make_shared<LatexScene>("\\mathbb{C} \\text{ is algebraically closed.}", 1, .6, .5);
    cs.add_scene_fade_in(MICRO, fta, "fta", .5, .5);
    cs.stage_macroblock(FileBlock("The fancy way to say this is that the complex field is algebraically closed."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This is so important, that it's called the Fundamental Theorem of Algebra."), 28);
    for(int i = 0; i < 14; i++) cs.render_microblock();
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

    cs.stage_macroblock(FileBlock("The number of solutions never changes either."), 2);
    cps->coefficients_to_roots();
    cps->transition_root_rings(MICRO, 1);
    cps->state_manager.transition(MICRO, {
        {"root0_r", "2"},
        {"root0_i", "0"},
        {"root1_r", "-1"},
        {"root1_i", "0"},
    });
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    shared_ptr<LatexScene> quadratic = make_shared<LatexScene>("x^2-x-2", .7, 1, .5);
    cs.add_scene_fade_in(MICRO, quadratic, "quadratic", .5, .2);
    cs.stage_macroblock(FileBlock("We're looking at a quadratic polynomial."), 1);
    cs.render_microblock();

    quadratic->begin_latex_transition(MICRO, latex_color(0xff333333, "x^"+latex_color(0xffff8080, "2")+"-x-2"));
    cs.stage_macroblock(FileBlock("The highest exponent is 2, so there's 2 solutions."), 3);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    quadratic->begin_latex_transition(MICRO, "x^3+x^2-x-2");
    cps->set_degree(3);
    cps->roots_to_coefficients();
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
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("With a cubic polynomial, there are 3 solutions."), 4);
    cs.render_microblock();
    cs.fade_subscene(MICRO, "quadratic", 0);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();
    cs.remove_all_subscenes_except("cps");

    cs.stage_macroblock(FileBlock("Well, unless we intentionally scrunch them on top of each other... but that's cheating."), 1);
    cps->coefficients_to_roots();
    cps->state_manager.transition(MICRO, {
        {"root0_r", "1"},
        {"root0_i", "0"},
        {"root1_r", "-1"},
        {"root1_i", "0"},
        {"root2_r", "-1"},
        {"root2_i", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We call that a solution of multiplicity 2. It counts as 2 normal solutions."), 2);
    cps->state_manager.transition(MICRO, {
        {"root2_ring", "1"},
        {"center_x", "-1"},
        {"zoom", "1.7"},
    });
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"root2_ring", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1.5), 1);
    cps->state_manager.transition(MICRO, {
        {"center_x", "0"},
        {"zoom", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("You can identify the solutions' multiplicities from the graph."), 1);
    cps->state_manager.transition(MICRO, {
        {"ticks_opacity", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MACRO, {
        {"center_x", "<root0_r>"},
        {"center_y", "<root0_i>"},
        {"zoom", "2.5"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Tracing around this normal solution,"), 1);
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "2"},
    });
    cps->state_manager.begin_timer("theta");
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "", .7, true));
    cps->state_manager.set({
        {"point__x", "<root0_r>"},
        {"point__y", "<root0_i>"},
    });
    cps->state_manager.transition(MACRO, {
        {"point__x", "<root0_r> <theta> 2 * cos .1 * +"},
        {"point__y", "<root0_i> <theta> 2 * sin .1 * +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("we see red, then green, then blue."), 8);
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"theta", "3.4"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"theta", "4.4"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"theta", "5.4"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We make a spin around the color wheel."), 1);
    cps->state_manager.transition(MICRO, {
        {"theta", "8.54"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.begin_timer("theta2");
    cps->state_manager.transition(MACRO, {
        {"point__x", "<root2_r> <theta2> 2 * cos .2 * +"},
        {"point__y", "<root2_i> <theta2> 2 * sin .2 * +"},
        {"center_x", "<root2_r>"},
        {"center_y", "<root2_i>"},
    });
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", ".8"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Tracing around the multiplicity 2 solution,"), 1);
    cps->state_manager.transition(MACRO, {
        {"ab_dilation", "1.5"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("we see red, green, blue, then red, green, blue again."), 17);
    cs.render_microblock();
    cs.render_microblock();
    for(int i = 0; i < 6; i++) {
        double angle = 3.14 * (i/6. + 1.275);
        if(i == 0) angle += .1;
        cps->state_manager.transition(MICRO, "theta2", to_string(angle));
        cs.render_microblock();
        cs.render_microblock();
        if(i == 2) {
            cs.render_microblock();
        }
    }
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The color wheel is duplicated!"), 1);
    cps->state_manager.transition(MICRO, {
        {"zoom", "1"},
        {"center_x", "0"},
        {"center_y", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MICRO, {
        {"ticks_opacity", "0"},
        {"ab_dilation", ".8"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's plot the output like before."), 2);
    cps->state_manager.transition(MICRO, {
        {"point__x", "<root0_r> <t> 3 * sin .1 * +"},
        {"point__y", "<root0_i> <t> 3 * cos .1 * +"},
    });
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "out", .7, true));
    cps->state_manager.set({
        {"d0r", "<root0_r> <point__x> -"},
        {"d1r", "<root1_r> <point__x> -"},
        {"d2r", "<root2_r> <point__x> -"},
        {"d0i", "<root0_i> <point__y> -"},
        {"d1i", "<root1_i> <point__y> -"},
        {"d2i", "<root2_i> <point__y> -"},
        {"m01r", "<d0r> <d1r> * <d0i> <d1i> * -"},
        {"m01i", "<d0r> <d1i> * <d0i> <d1r> * +"},
        {"point_out_x", "<m01r> <d2r> * <m01i> <d2i> * -"},
        {"point_out_y", "<m01r> <d2i> * <m01i> <d2r> * +"},
    });
    cps->state_manager.transition(MICRO, {
        {"zoom", "1.7"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("As we trace around the multiplicity 1 solution, the output point follows the angle of the input point."), 1);
    cs.render_microblock();

    cps->state_manager.transition(MICRO, {
        {"zoom", "1"},
    });
    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MICRO, {
        {"point__x", "<root2_r> <t> 3 * sin .3 * +"},
        {"point__y", "<root2_i> <t> 3 * cos .3 * +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But tracing around the multiplicity 2 solution, the output point goes around twice as fast."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MICRO, "geometry_opacity", "0");
    cs.render_microblock();
    cps->state_manager.remove({
        {"d0r"},
        {"d1r"},
        {"d2r"},
        {"d0i"},
        {"d1i"},
        {"d2i"},
        {"m01r"},
        {"m01i"},
        {"point__x"},
        {"point__y"},
        {"point_out_x"},
        {"point_out_y"},
    });
    cps->construction.clear();
    cps->state_manager.set("geometry_opacity", "1");

    /*
    cs.stage_macroblock(FileBlock("This is thanks to how complex multiplication works."), 1);
    cps->state_manager.transition(MICRO, {
        {"root0_i", "-4.8"},
        {"root1_i", "-3.2"},
        {"root2_i", "-3.2"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Taking two complex numbers A and B,"), 2);
    complex<float> a(8, 6);
    a /= abs(a);
    cps->construction.add(GeometricPoint(glm::vec2(a.real(), a.imag()), "a"));
    cs.render_microblock();
    complex<float> b(8, 3);
    b /= abs(b);
    cps->construction.add(GeometricPoint(glm::vec2(b.real(), b.imag()), "b"));
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("A times B's angle is the sum of A's angle plus B's angle."), 1);
    const complex<float> ab = a * b;
    cps->construction.add(GeometricPoint(glm::vec2(ab.real(), ab.imag()), "a\\cdot b"));
    cs.render_microblock();
    */

    cs.stage_macroblock(FileBlock("So, the degree of the polynomial,"), 2);
    quadratic = make_shared<LatexScene>(latex_color(0xff333333, "1x^" + latex_color(0xffff8080, "3") + "+1x^2-1x-1"), .7, 1, .5);
    cs.add_scene_fade_in(MICRO, quadratic, "quadratic", .5, .2);
    cps->roots_to_coefficients();
    cps->state_manager.transition(MICRO, {
        {"coefficient3_r", "1"},
        {"coefficient3_i", "0"},
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "-1"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "-1"},
        {"coefficient0_i", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("always matches the number of solutions..."), 3);
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("...as long as we account for their multiplicities."), 1);
    cps->state_manager.transition(MICRO, {
        {"coefficient3_r", "1"},
        {"coefficient3_i", "0"},
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "-1"},
        {"coefficient1_i", "1"},
        {"coefficient0_r", "-1"},
        {"coefficient0_i", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    quadratic->begin_latex_transition(MICRO, latex_color(0xff333333, "x^3+"+latex_color(OPAQUE_WHITE, "1")+"x^2-"+latex_color(OPAQUE_WHITE, "1")+"x-"+latex_color(OPAQUE_WHITE, "1")));
    cs.stage_macroblock(FileBlock("Another way of saying this is that there is always exactly one more coefficient than the number of solutions."), 2);
    cps->transition_coefficient_opacities(MICRO, 1);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_subscene(MACRO, "quadratic", 0);
    cs.render_microblock();
    cs.remove_subscene("quadratic");

    cs.stage_macroblock(FileBlock("But, how do they relate to each other?"), 1);
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    //move the roots around
    cps->roots_to_coefficients();
    for(int i = 0; i < cps->get_degree()+1; i++) {
        cps->state_manager.transition(MICRO, {
            {"coefficient"+to_string(i)+"_r", "<t> 1.2 * 6.28 .3333 " + to_string(i) + " * * + sin"},
            {"coefficient"+to_string(i)+"_i", "<t> .8 * 6.28 .3333 " + to_string(i) + " * * + cos .2 +"},
        });
    }
    cs.stage_macroblock(FileBlock("Given the coefficients,"), 2);
    cps->transition_coefficient_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cs.render_microblock();

    cps->coefficients_to_roots();
    cps->state_manager.transition(MICRO, {
        {"root0_r", "-1"},
        {"root0_i", ".2"},
        {"root1_r", "2"},
        {"root1_i", ".3"},
        {"root2_r", "1"},
        {"root2_i", "-.1"},
    });
    cs.stage_macroblock(FileBlock("how do we know where the solutions should be?"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("It's obvious since I'm plotting the graph for you..."), 1);
    //move roots offscreen
    cps->coefficients_to_roots();
    cps->transition_coefficient_opacities(MICRO, 0);
    cps->state_manager.transition(MICRO, {
        {"root0_r", "-6"},
        {"root0_i", "0"},
        {"root1_r", "4"},
        {"root1_i", "-6"},
        {"root2_r", "4"},
        {"root2_i", "6"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but if I give you a, b, and c numerically, how do you find solutions 1, 2, and 3?"), 9);
    shared_ptr<LatexScene> abc = make_shared<LatexScene>("\\begin{tabular}{cc} a=2+i & \\end{tabular}", .4);
    cs.add_scene_fade_in(MICRO, abc, "abc");
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=2+i & \\\\\\\\ b=-i & \\end{tabular}");
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=2+i & \\\\\\\\ b=-i & \\\\\\\\ c=1.5 & \\end{tabular}");
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=2+i & r_1=? \\\\\\\\ b=-i & \\\\\\\\ c=1.5 & \\end{tabular}");
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=2+i & r_1=? \\\\\\\\ b=-i & r_2=? \\\\\\\\ c=1.5 & \\end{tabular}");
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=2+i & r_1=? \\\\\\\\ b=-i & r_2=? \\\\\\\\ c=1.5 & r_3=? \\end{tabular}");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Before I spoil this problem, let me illustrate its complexities."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's take the simplest case imaginable- a polynomial with coefficients that are only zero or one."), 2);
    cps->roots_to_coefficients();
    cps->state_manager.transition(MACRO, {
        {"coefficient3_r", new_coefficient_val},
        {"coefficient3_i", "0"},
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "0"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "1"},
        {"coefficient0_i", "0"},
    });
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=1 & r_1=? \\\\\\\\ b=0 & r_2=? \\\\\\\\ c=1 & r_3=? \\end{tabular}");
    cs.render_microblock();
    cps->roots_to_coefficients();
    cs.render_microblock();
    cps->set_degree(2);

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("There are a few different options of which are ones and which are zeros."), 1);
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=1 & r_1=? \\\\\\\\ b=1 & r_2=? \\\\\\\\ c=0 & r_3=? \\end{tabular}");
    cps->state_manager.transition(MICRO, {
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "0"},
        {"coefficient0_i", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_subscene(MICRO, "abc", 0);
    cs.render_microblock();
    cs.remove_all_subscenes_except("cps");

    flashes = 3;
    cs.stage_macroblock(CompositeBlock(FileBlock("With a quadratic polynomial, there are exactly 4 ways."), FileBlock("Plotting all of their solutions at the same time...")), flashes*4);
    cs.fade_subscene(MACRO, "cps", 0);
    shared_ptr<RootFractalScene> fracs = make_shared<RootFractalScene>();
    cs.add_scene(fracs, "fracs", .5, .5, true);
    fracs->state_manager.set({
        {"terms", "3"},
        {"coefficients_opacity", "0"},
        {"ticks_opacity", "0"},
        {"visibility_multiplier", "9"},
        {"rainbow", "0"},
    });
    cps->transition_coefficient_opacities(MACRO, 1);
    for(int bit = 0; bit < flashes*4; bit++) {
        string b = (bit&2) ? "1" : "0";
        string c = (bit&1) ? "1" : "0";
        abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=1 & r_1=? \\\\\\\\ b="+b+" & r_2=? \\\\\\\\ c="+c+" & r_3=? \\end{tabular}");
        cps->state_manager.transition(MICRO, {
            {"coefficient2_r", "1"},
            {"coefficient2_i", "0"},
            {"coefficient1_r", b},
            {"coefficient1_i", "0"},
            {"coefficient0_r", c},
            {"coefficient0_i", "0"},
        });
        cs.render_microblock();
    }
    cs.remove_subscene("cps");

    fracs->stage_macroblock(FileBlock("we get this shape."), 1);
    fracs->render_microblock();

    fracs->stage_macroblock(FileBlock("Cranking it up to degree 3,"), 1);
    fracs->state_manager.transition(MICRO, {
        {"visibility_multiplier", "7"},
        {"terms", "4"}, // degree 3 means 4 terms
    });
    fracs->render_microblock();

    fracs->stage_macroblock(FileBlock("degree 4,"), 1);
    fracs->state_manager.transition(MICRO, {
        {"visibility_multiplier", "5"},
        {"terms", "5"},
    });
    fracs->render_microblock();

    fracs->stage_macroblock(CompositeBlock(FileBlock("and even higher,"), SilenceBlock(8)), 11);
    fracs->state_manager.transition(MACRO, {
        {"zoom", "1.6"},
        {"visibility_multiplier", "1"},
    });
    int i = 6;
    while(fracs->microblocks_remaining()) {
        fracs->state_manager.transition(MICRO, {
            {"terms", to_string(i)},
        });
        i++;
        fracs->render_microblock();
    }

    fracs->stage_macroblock(FileBlock("We were letting the coefficients be either 0 or 1,"), 1);
    fracs->state_manager.transition(MICRO, {
        {"coefficients_opacity", "1"},
    });
    fracs->render_microblock();

    fracs->stage_macroblock(FileBlock("but what happens if we change those options?"), 1);
    fracs->state_manager.begin_timer("fractal_timer");
    fracs->state_manager.transition(MICRO, {
        {"coefficient0_r", ".1 <fractal_timer> 4 / cos *"},
        {"coefficient0_i", ".1 <fractal_timer> 4 / sin *"},
        {"zoom", "1.3"},
    });
    fracs->render_microblock();

    fracs->stage_macroblock(SilenceBlock(2), 1);
    fracs->state_manager.transition(MICRO, {
        {"center_x", "-.5"},
        {"center_y", ".5"},
    });
    fracs->render_microblock();

    fracs->stage_macroblock(SilenceBlock(2), 1);
    fracs->state_manager.transition(MICRO, {
        {"rainbow", "1"},
        {"zoom", "5"},
    });
    fracs->render_microblock();

    fracs->stage_macroblock(SilenceBlock(40), 13);
    fracs->state_manager.transition(MICRO, {
        {"coefficient0_r", "<fractal_timer> 8 / sin"},
        {"coefficient0_i", "<fractal_timer> 9 / sin"},
    });
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"center_x", "0"},
        {"center_y", "1"},
    });
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"rainbow", "1"},
        {"zoom", "7"},
    });
    fracs->render_microblock();
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"zoom", "1.3"},
        {"rainbow", "0"},
    });
    fracs->render_microblock();
    fracs->state_manager.begin_timer("spin");
    fracs->state_manager.transition(MICRO, {
        {"center_x", "0"},
        {"center_y", "0"},
    });
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"coefficient0_r", "<spin> 2 / cos"},
        {"coefficient0_i", "<spin> 2 / sin"},
    });
    fracs->render_microblock();
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"coefficient0_r", "<spin> 2 / cos 5 *"},
        {"coefficient0_i", "<spin> 2 / sin 5 *"},
    });
    fracs->render_microblock();
    fracs->render_microblock();
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"zoom", "2.5"},
        {"center_x", ".4"},
        {"center_y", "-.4"},
    });
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"rainbow", "1"},
        {"zoom", "4"},
        {"spin", to_string(fracs->state_manager.get_value("spin")) + " 1 +"},
    });
    fracs->render_microblock();

    fracs->stage_macroblock(FileBlock("As we zoom into these point clouds,"), 1);
    fracs->state_manager.transition(MICRO, "zoom", "7");
    fracs->render_microblock();

    cs.state_manager.transition(MICRO, "fracs.x", ".25");
    fracs->state_manager.transition(MICRO, "w", ".5");
    shared_ptr<LatexScene> dragon = make_shared<LatexScene>("\\text{Dragon Curve TODO}", 1, .5, 1);
    cs.add_scene(dragon, "dragon", 1.25, .5);
    cs.slide_subscene(MICRO, "dragon", -.5, 0);
    cs.stage_macroblock(FileBlock("there are shapes resembling the Dragon Curve,"), 2);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(5), 2);
    fracs->state_manager.transition(MACRO, { {"zoom", "4"}, });
    cs.render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"coefficient0_r", "-1"},
        {"coefficient0_i", "0"},
        {"center_x", "-.59522"},
        {"center_y", "-.42693"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("That's the same shape that you get by folding up a strip of paper and letting it spring out."), 1);
    cs.render_microblock();

    cs.state_manager.transition(MICRO, "fracs.x", ".5");
    fracs->state_manager.transition(MICRO, "w", "1");
    cs.slide_subscene(MICRO, "dragon", .5, 0);
    cs.stage_macroblock(FileBlock("The point is, the function mapping coefficients to solutions isn't simple- it must have a bit of magic to it."), 1);
    cs.render_microblock();
    cs.remove_all_subscenes_except("fracs");

    fracs->stage_macroblock(FileBlock("Even with just -1 and 1 as coefficients, we find ourselves in a zoo of emergent complexity."), 1);
    fracs->state_manager.begin_timer("littlewood_timer");
    fracs->state_manager.transition(MICRO, {
        {"rainbow", "0"},
        {"center_x", "<littlewood_timer> 10 / cos -1.2 *"},
        {"center_y", "<littlewood_timer> 10 / sin 1.2 *"},
        {"zoom", "2"},
    });
    fracs->render_microblock();

    fracs->stage_macroblock(SilenceBlock(8), 3);
    fracs->render_microblock();
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"zoom", "1"},
        {"center_x", "0"},
        {"center_y", "0"},
    });
    fracs->render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(1), FileBlock("What _can_ we say about the process which maps coefficients to solutions?")), 1);
    cps->state_manager.set({
        {"coefficient0_opacity", "0"},
        {"coefficient1_opacity", "0"},
        {"coefficient2_opacity", "0"},
    });
    cs.add_scene_fade_in(MICRO, cps, "cps", .5, .5, true);
    cs.fade_subscene(MICRO, "fracs", 0);
    cs.render_microblock();
    cs.remove_subscene("fracs");

    // TODO justify the switch to monic
    /*
    cs.stage_macroblock(FileBlock("Now, just as a note- I have been fixing the leading coefficient as 1 for the whole video."), 1);
    quadratic->begin_latex_transition(MICRO, latex_color(0xffff8080, "1")+latex_color(0xff333333, "x+1=0"));
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("You can try to work out why this doesn't really change the underlying problem,"), 1);
    quadratic->begin_latex_transition(MICRO, "x+1=0");
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("but that's a simplification I'm making, and will continue to make for the rest of the video."), 1);
    cs.render_microblock();
    */

    cs.stage_macroblock(FileBlock("Let's start off easy, with polynomials whose highest exponent is one."), 2);
    quadratic = make_shared<LatexScene>(latex_color(0xff333333, "ax^"+latex_color(0xffff8080, "1")+"+b = 0"), .6, 1, .5);
    cs.add_scene_fade_in(MICRO, quadratic, "quadratic", .5, .2);
    //shared_ptr<LatexScene> current_degree = make_shared<LatexScene>("1 \\\\\\\\ \\tiny{\\text{Linear}}", 1, .2, .2);
    //cs.add_scene_fade_in(MICRO, current_degree, "current_degree", .05, .9);
    cps->roots_to_coefficients();
    cps->state_manager.transition(MICRO, {
        {"coefficient2_r", new_coefficient_val},
        {"coefficient2_i", "0"},
        {"coefficient2_opacity", "0"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", ".8"},
        {"coefficient0_i", "1"},
    });
    cs.render_microblock();
    quadratic->begin_latex_transition(MICRO, "ax+b=0");
    cs.render_microblock();
    cps->set_degree(1);

    cs.stage_macroblock(FileBlock("With a linear polynomial, there's only one solution."), 2);
    cps->transition_coefficient_opacities(MICRO, 1);
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This case is super easy."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("To solve ax+b=0, we can just use algebra."), 2);
    quadratic->begin_latex_transition(MICRO, "ax=-b");
    cs.render_microblock();
    quadratic->begin_latex_transition(MICRO, "x=\\frac{-b}{a}");
    cs.render_microblock();
    cs.export_frame("Sample");

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MICRO, {
        {"coefficient0_r", "<t> sin"},
        {"coefficient0_i", "<t> cos"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Since we are holding A at 1, the root x and the coefficient b are opposites, so they do a mirror dance about the origin."), 3);
    quadratic->begin_latex_transition(MICRO, "x=-b");
    cps->roots_to_coefficients();
    cps->state_manager.transition(MICRO, {
        {"coefficient1_opacity", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "quadratic", 0);
    cs.render_microblock();
    cs.remove_subscene("quadratic");

    cs.stage_macroblock(SilenceBlock(1.5), 1);
    cps->state_manager.transition(MICRO, {
        {"coefficient0_r", "0"},
        {"coefficient0_i", "1"},
        {"coefficient0_opacity", "0"},
    });
    cs.render_microblock();

    cps->set_degree(2);

    StateSet preflip = {
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "1"},
        {"coefficient0_r", "2"},
        {"coefficient0_i", "0"},
    };
    cps->state_manager.transition(MICRO, preflip);

    cs.stage_macroblock(FileBlock("Jumping up to a polynomial with highest exponent 2,"), 1);
    //current_degree->begin_latex_transition(MICRO, "2 \\\\\\\\ \\tiny{\\text{Quadratic}}");
    cps->state_manager.transition(MICRO, {
        {"coefficient2_opacity", "0"},
        {"coefficient1_opacity", "1"},
        {"coefficient0_opacity", "1"},
    });
    cs.render_microblock();

    shared_ptr<LatexScene> quadratic_formula = make_shared<LatexScene>("\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}", 1, .3, .3);
    cs.stage_macroblock(FileBlock("There's a certain formula you might have memorized in school..."), 1);
    cs.add_scene(quadratic_formula, "quadratic_formula", -.2, .5);
    cs.slide_subscene(MACRO, "quadratic_formula", .32, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but pretend you didn't for a moment."), 2);
    cs.slide_subscene(MICRO, "quadratic_formula", -1, 0);
    cs.render_microblock();
    cs.remove_subscene("quadratic_formula");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("I want to point out something that starts to get weird in the quadratic case."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("As I've hopefully made clear, 2 coefficients means 2 solutions and vice versa."), 9);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    // Highlight coefficients then roots
    cps->state_manager.transition(MICRO, {
        {"coefficient1_ring", "1"},
        {"coefficient0_ring", "1"},
    });
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"coefficient1_ring", "0"},
        {"coefficient0_ring", "0"},
    });
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("However, coefficients and roots are structurally very different."), 8);
    cs.render_microblock();
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"coefficient1_ring", "1"},
        {"coefficient0_ring", "1"},
    });
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"coefficient1_ring", "0"},
        {"coefficient0_ring", "0"},
    });
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Looking at the polynomial,"), 1);
    quadratic = make_shared<LatexScene>(latex_color(0xff333333, "x^2 + "+latex_color(0xffff8080, "b")+"x + "+latex_color(0xffff8080, "c")), .7, 1, .5);
    cs.add_scene_fade_in(MICRO, quadratic, "quadratic", .5, .15);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("If we switch b and c here, we don't have the same function anymore."), 1);
    quadratic->begin_latex_transition(MICRO, latex_color(0xff333333, "x^2 + "+latex_color(0xffff8080, "c")+"x + "+latex_color(0xffff8080, "b")));
    cps->state_manager.transition(MICRO, {
        {"coefficient0_ring", "1"},
        {"coefficient1_ring", "1"},
    });
    cps->stage_swap(MICRO, "0", "1", true);
    cs.render_microblock();

    int flips = 15;
    cs.stage_macroblock(FileBlock("The coefficients are an ordered list- they're not interchangeable."), flips);
    cs.fade_subscene(MACRO, "quadratic", 0);
    for(int i = 0; i < flips; i++) {
        if(i == 1 || i == 4 || i == 8 || i == 9 || i == 12) {
            cps->stage_swap(MICRO, "0", "1", true);
        }
        cs.render_microblock();
    }
    cs.remove_subscene("quadratic");

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MICRO, {
        {"coefficient1_ring", "0"},
        {"coefficient0_ring", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The solutions, on the other hand, don't have a particular order to them."), 2);
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("In the factored form of the polynomial, we can rearrange as we please,"), 2);
    factored = make_shared<LatexScene>("(x - "+latex_color(0xffff8080, "r_1")+") (x - "+latex_color(0xffff8080, "r_2")+") (x - "+latex_color(0xffff8080, "r_3")+")", .7, 1, .5);
    cs.add_scene_fade_in(MACRO, factored, "factored", .5, .2);
    cps->coefficients_to_roots();
    cs.render_microblock();
    factored->begin_latex_transition(MICRO, "(x - "+latex_color(0xffff8080, "r_2")+") (x - "+latex_color(0xffff8080, "r_1")+") (x - "+latex_color(0xffff8080, "r_3")+")");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("since it doesn't matter what order we multiply the terms."), 2);
    cs.render_microblock();
    cs.fade_subscene(MICRO, "factored", 0);
    cs.render_microblock();
    cs.remove_subscene("factored");

    cs.stage_macroblock(FileBlock("Swapping the solutions with each other, the coefficients go right back to where they started."), 4);
    cps->transition_root_rings(MICRO, 1);
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The coefficients are distinct from each other, but the solutions aren't."), 4);
    cps->state_manager.transition(MICRO, {
        {"coefficient0_ring", "1"},
        {"coefficient1_ring", "1"},
    });
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"coefficient0_ring", "0"},
        {"coefficient1_ring", "0"},
    });
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This has some bizarre implications."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("First and foremost, there's no function which takes in the coefficients and gives out _some_ solution, and is continuous."), 3);
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.render_microblock();
    shared_ptr<LatexScene> function = make_shared<LatexScene>("f(a, b, c) \\rightarrow r_1", 1, .75, .4);
    shared_ptr<LatexScene> details = make_shared<LatexScene>("f \\text{ returns a single root}", 1, .5, .4);
    cs.add_scene_fade_in(MICRO, function, "function", .5, .4);
    cs.add_scene_fade_in(MICRO, details, "details", .5, .6);
    cs.render_microblock();
    details->begin_latex_transition(MICRO, "f \\text{ returns a single root} \\\\\\\\ f \\text{ continuous}");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This may seem nonsensical, especially in the light of the quadratic formula I hinted at earlier."), 2);
    cs.slide_subscene(MICRO, "function", 0, -.25);
    cs.slide_subscene(MICRO, "details", 0, -.25);
    cs.render_microblock();
    quadratic_formula = make_shared<LatexScene>("\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}", 1, .6, .4);
    cs.add_scene(quadratic_formula, "quadratic_formula", -.3, .75);
    cs.slide_subscene(MICRO, "quadratic_formula", .8, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But we asked for a function which provides us with _one_ solution, not both of them."), 3);
    function->begin_latex_transition(MICRO, latex_color(0xff333333, "f(a, b, c) \\rightarrow " + latex_color(0xffff8080, "r_1")));
    cs.render_microblock();
    cs.render_microblock();
    function->begin_latex_transition(MICRO, "f(a, b, c) \\rightarrow r_1");
    cs.render_microblock();

    shared_ptr<LatexScene> negative_quadratic_formula = make_shared<LatexScene>("\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}", 1, .6, .4);
    cs.add_scene(negative_quadratic_formula, "negative_quadratic_formula", .5, .75);
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_subscene(MICRO, "function", 0);
    cs.fade_subscene(MICRO, "details", 0);
    cs.slide_subscene(MICRO, "quadratic_formula", 0, -.25);
    cs.slide_subscene(MICRO, "negative_quadratic_formula", 0, -.25);
    cs.slide_subscene(MICRO, "function", 0, -.45);
    cs.slide_subscene(MICRO, "details", 0, -.45);
    cs.render_microblock();
    cs.remove_subscene("function");
    cs.remove_subscene("details");


    cs.stage_macroblock(FileBlock("Thanks to the plus-or-minus sign, the quadratic formula is really two functions."), 1);
    negative_quadratic_formula->begin_latex_transition(MICRO, "\\frac{-b - \\sqrt{b^2 - 4ac}}{2a}");
    quadratic_formula->begin_latex_transition(MICRO, "\\frac{-b + \\sqrt{b^2 - 4ac}}{2a}");
    cs.slide_subscene(MICRO, "negative_quadratic_formula", 0, -.3);
    cs.slide_subscene(MICRO, "quadratic_formula", 0, .3);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Naively discarding one of them seems like it should do the trick?"), 1);
    cs.slide_subscene(MICRO, "negative_quadratic_formula", 0, -1);
    cs.slide_subscene(MICRO, "quadratic_formula", 0, -.3);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("I'm gonna pass our coefficients into this modified formula, and highlight what it says."), 3);
    cs.fade_subscene(MICRO, "quadratic_formula", .5);
    cps->roots_to_coefficients();
    cps->transition_coefficient_rings(MICRO, 1);
    cps->transition_coefficient_opacities(MICRO, 1);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"positive_quadratic_formula_opacity", "1"},
    });
    cs.fade_subscene(MICRO, "quadratic_formula", 0);
    cs.render_microblock();
    cs.remove_subscene("quadratic_formula");

    cs.stage_macroblock(CompositeBlock(SilenceBlock(3), FileBlock("Looks like our function is tracking... huh?")), 1);
    cps->roots_to_coefficients();
    cps->state_manager.transition(MICRO, {
        {"coefficient2_r", "-2"},
        {"coefficient2_i", "-.5"},
        {"coefficient1_r", "2"},
        {"coefficient1_i", ".5"},
        {"coefficient0_r", "1.5"},
        {"coefficient0_i", "-1.3"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    // TODO wiggle across the discontinuity
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("It really isn't continuous!"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("As I move the coefficients around, the solution suddenly jumps."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Remember, the quadratic formula has a square root in it..."), 2);
    cs.add_scene_fade_in(MICRO, quadratic_formula, "quadratic_formula");
    quadratic_formula->begin_latex_transition(MICRO, latex_color(0xff333333, "\\frac{-b + " + latex_color(0xffff8080, "\\sqrt{b^2 - 4ac}") + "}{2a}"));
    cs.render_microblock();
    cs.fade_subscene(MICRO, "cps", 0);
    cs.render_microblock();

    // TODO move between plots in 3-space tds
    shared_ptr<ComplexArbitraryFunctionScene> cafs = make_shared<ComplexArbitraryFunctionScene>();
    cs.add_scene_fade_in(MICRO, cafs, "cafs", .5, .5, 1, true);
    cs.stage_macroblock(FileBlock("and on the complex plane, the square root function has a yucky discontinuity!"), 2);
    cs.fade_subscene(MICRO, "quadratic_formula", 0);
    cs.render_microblock();
    cs.remove_subscene("quadratic_formula");
    cafs->construction.add(GeometricLine(glm::vec2(-5, 0), glm::vec2(0, 0)));
    cafs->state_manager.transition(MICRO, "geometry_opacity", "0");
    cs.render_microblock();

    cps->state_manager.set("positive_quadratic_formula_opacity", "0");
    cs.fade_subscene(MICRO, "cafs", 0);
    cs.fade_subscene(MICRO, "cps", 1);
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();
    cs.remove_subscene("cafs");

    cs.stage_macroblock(FileBlock("So, _that_ doesn't work..."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and sure enough, in general, there's no way to make a continuous function give us just one root. Here's why."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Assume we do have a continuous function, and it takes in a, b, and c,"), 4);
    cs.render_microblock();
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and gives us this root."), 3);
    cps->state_manager.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("As we saw before, we can swap the roots and the coefficients will return to their starting position."), 1);
    cps->coefficients_to_roots();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But our 'continuous function' gives us a different root now!"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Given this same a, b, and c,"), 3);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("First it gave us this root, but then it gave us that root."), 8);
    cps->state_manager.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root1_ring", "1");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root1_ring", "0");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root1_ring", "1");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root1_ring", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("A function has to be deterministic- this is against the rules."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("By nature of being continuous, it can't distinguish between the two roots."), 4);
    while(cs.microblocks_remaining()) {
        cps->state_manager.transition(MICRO, "root0_ring", "1");
        cs.render_microblock();
        cps->state_manager.transition(MICRO, "root0_ring", "0");
        cs.render_microblock();
        cps->state_manager.transition(MICRO, "root1_ring", "1");
        cs.render_microblock();
        cps->state_manager.transition(MICRO, "root1_ring", "0");
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("So, creating a continuous function to track a particular root is an impossible task to begin with."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("A so-called 'quadratic formula' has to incorporate discontinuous functions."), 1);
    cs.render_microblock();

    shared_ptr<LatexScene> continuous_operations = make_shared<LatexScene>("+, -, \\times, \\div", .5, 1, .3);
    cs.add_scene_fade_in(MICRO, continuous_operations, "continuous_operations", .5, .15);
    cs.stage_macroblock(FileBlock("There's no way to do it with just basic arithmetic."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Even if we incorporate other functions,"), 1);
    cs.fade_subscene(MICRO, "cps", 0);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    continuous_operations->begin_latex_transition(MICRO, "+, -, \\times, \\div, \\sin");
    cafs->state_manager.set({{"sqrt_coef", "0"}, {"sin_coef", "1"}});
    cs.add_scene_fade_in(MACRO, cafs, "cafs", 0.5, 0.5, 1, true);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("such as sines, (long pause), cosines, (long pause), and exponentials,"), 6);
    cs.render_microblock();
    cs.render_microblock();
    cafs->state_manager.transition(MICRO, {{"sin_coef", "0"}, {"cos_coef", "1"}});
    continuous_operations->begin_latex_transition(MICRO, "+, -, \\times, \\div, \\sin, \\cos");
    cs.render_microblock();
    cs.render_microblock();
    cafs->state_manager.transition(MICRO, {{"cos_coef", "0"}, {"exp_coef", "1"}});
    continuous_operations->begin_latex_transition(MICRO, "+, -, \\times, \\div, \\sin, \\cos, \\exp");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("they feature no discontinuities, so they have no way of selecting a unique solution of the two."), 2);
    cs.render_microblock();
    cs.fade_all_subscenes(MICRO, 0);
    cs.fade_subscene(MICRO, "cps", 1);
    cs.render_microblock();
    cs.remove_subscene("cafs");
    cs.remove_subscene("continuous_operations");

    cs.stage_macroblock(FileBlock("Some kind of discontinuous operation is necessary- and a square root does the trick."), 4);
    cs.render_microblock();
    cs.render_microblock();
    cs.add_scene_fade_in(MICRO, quadratic_formula, "quadratic_formula", .5, .5);
    quadratic_formula->begin_latex_transition(MICRO, latex_color(0xff333333, "\\frac{-b + " + latex_color(0xffff8080, "\\sqrt{b^2 - 4ac}") + "}{2a}"));
    cs.render_microblock();
    cs.fade_subscene(MICRO, "quadratic_formula", 0);
    cs.render_microblock();
    cs.remove_subscene("quadratic_formula");

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    shared_ptr<LatexScene> recap = make_shared<LatexScene>("\\begin{tabular}{ccc} \\text{Degree} & \\qquad \\qquad \\text{Form} \\qquad \\qquad & \\qquad \\qquad \\qquad \\text{Solutions} \\qquad \\qquad \\qquad", 1, 1, 1);
    cs.add_scene_fade_in(MACRO, recap, "recap");
    cs.stage_macroblock(FileBlock("Just to recap..."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Linear polynomials are solved trivially."), 1);
    recap->begin_latex_transition(MICRO, "\\begin{tabular}{ccc} \\text{Degree} & \\qquad \\qquad \\text{Form} \\qquad \\qquad & \\qquad \\qquad \\qquad \\text{Solutions} \\qquad \\qquad \\qquad \\\\\\\\ 1 & ax + b & -\\frac{b}{a}");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Quadratics have this yucky, discontinuous formula."), 1);
    recap->begin_latex_transition(MICRO, "\\begin{tabular}{ccc} \\text{Degree} & \\qquad \\qquad \\text{Form} \\qquad \\qquad & \\qquad \\qquad \\qquad \\text{Solutions} \\qquad \\qquad \\qquad \\\\\\\\ 1 & ax + b & -\\frac{b}{a} \\\\\\\\ 2 & ax^2 + bx + c & \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}");
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("What do you think happens next?"), SilenceBlock(2)), 3);
    recap->begin_latex_transition(MICRO, "\\begin{tabular}{ccc} \\text{Degree} & \\qquad \\qquad \\text{Form} \\qquad \\qquad & \\qquad \\qquad \\qquad \\text{Solutions} \\qquad \\qquad \\qquad \\\\\\\\ 1 & ax + b & -\\frac{b}{a} \\\\\\\\ 2 & ax^2 + bx + c & \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\\\\\\\ 3 & ax^3 + bx^2 + cx + d & ?");
    cs.render_microblock();
    recap->begin_latex_transition(MICRO, "\\begin{tabular}{ccc} \\text{Degree} & \\qquad \\qquad \\text{Form} \\qquad \\qquad & \\qquad \\qquad \\qquad \\text{Solutions} \\qquad \\qquad \\qquad \\\\\\\\ 1 & ax + b & -\\frac{b}{a} \\\\\\\\ 2 & ax^2 + bx + c & \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\\\\\\\ 3 & ax^3 + bx^2 + cx + d & ? \\\\\\\\ 4 & ax^4 + bx^3 + cx^2 + ... & ?");
    cs.render_microblock();
    recap->begin_latex_transition(MICRO, "\\begin{tabular}{ccc} \\text{Degree} & \\qquad \\qquad \\text{Form} \\qquad \\qquad & \\qquad \\qquad \\qquad \\text{Solutions} \\qquad \\qquad \\qquad \\\\\\\\ 1 & ax + b & -\\frac{b}{a} \\\\\\\\ 2 & ax^2 + bx + c & \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\\\\\\\ 3 & ax^3 + bx^2 + cx + d & ? \\\\\\\\ 4 & ax^4 + bx^3 + cx^2 + ... & ? \\\\\\\\ 5 & ax^5 + bx^4 + cx^3 + ... & ?");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We can learn about the form that solutions for cubics would have to take by looking at their symmetries, just like we did with quadratics."), 1);
    //current_degree->begin_latex_transition(MICRO, "3 \\\\\\\\ \\tiny{\\text{Cubic}}");
    //cs.fade_subscene(MICRO, "current_degree", 1);
    recap->begin_latex_transition(MACRO, "ax^3 + bx^2 + cx + d");
    cps->set_degree(3);
    cps->roots_to_coefficients();
    cps->transition_coefficient_opacities(MICRO, 1);
    cps->state_manager.transition(MACRO, {
        {"coefficient3_r", "1"},
        {"coefficient3_i", "0"},
        {"coefficient2_r", "-1"},
        {"coefficient2_i", "1"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "1"},
        {"coefficient0_i", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("If there's a cubic formula, it'll need a discontinuous operation to pick out one of the three roots."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The same argument still applies. If I swap two roots, the coefficients go back to where they started."), 2);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("For our functions to be deterministic, they have to be discontinuous."), 1);
    cs.render_microblock();

}
