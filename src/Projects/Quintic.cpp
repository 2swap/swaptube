#include "../Scenes/Math/ComplexPlotScene.cpp"
#include "../Scenes/Math/ComplexArbitraryFunctionScene.cpp"
#include "../Scenes/Math/RealFunctionScene.cpp"
#include "../Scenes/Math/RootFractalScene.cpp"
#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/ThreeDimensionScene.cpp"
#include "../Scenes/Math/ManifoldScene.cpp"

void render_video(){
    shared_ptr<ComplexPlotScene> cps = make_shared<ComplexPlotScene>(3);
    cps->stage_macroblock(FileBlock("This is the relationship between a polynomial's coefficients and its solutions."), 10);
    cps->state_manager.set("ticks_opacity", "0");

    for(int i = 0; i < cps->get_degree(); i++) {
        cps->state_manager.transition(MICRO, {
            {"root"+to_string(i)+"_r", "{t} 1.2 * 6.28 .3333 " + to_string(i) + " * * + sin"},
            {"root"+to_string(i)+"_i", "{t} .8 * 6.28 .3333 " + to_string(i) + " * * + cos .2 +"},
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
        {"spin_coefficient_r", "{t} 2 * sin <spin_multiplier> *"},
        {"spin_coefficient_i", "{t} 3 * cos <spin_multiplier> *"},
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

    CompositeScene cs_intro;
    cs_intro.add_scene(cps, "cps");
    shared_ptr<RootFractalScene> rfs_intro = make_shared<RootFractalScene>();
    rfs_intro->state_manager.set({{"terms", "13"}, {"coefficients_opacity", "0"}, {"ticks_opacity", "0"}});
    rfs_intro->state_manager.begin_timer("circle");
    cs_intro.stage_macroblock(CompositeBlock(FileBlock("Their relationship yields a lot of beautiful math, like these polynomial solution fractals."), SilenceBlock(3)), 4);
    rfs_intro->state_manager.set({
        {"coefficient0_r", "<circle> 1 - 2 / cos"},
        {"coefficient0_i", "<circle> 1 - 2 / sin"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
    });
    cs_intro.add_scene_fade_in(MICRO, rfs_intro, "rfs_intro");
    cs_intro.fade_subscene(MICRO, "cps", 0);
    cs_intro.render_microblock();
    cs_intro.render_microblock();
    cs_intro.render_microblock();
    cs_intro.render_microblock();

    cs_intro.stage_macroblock(FileBlock("We're gonna figure out what that relationship is, lying at the heart of algebra."), 1);
    cps->coefficients_to_roots();
    for(int i = 0; i < cps->get_degree(); i++) {
        cps->state_manager.set({
            {"root"+to_string(i)+"_r", "{t} 1.2 * 6.28 .3333 " + to_string(i) + " * * + sin"},
            {"root"+to_string(i)+"_i", "{t} .8 * 6.28 .3333 " + to_string(i) + " * * + cos"},
        });
    }
    cs_intro.fade_subscene(MICRO, "rfs_intro", 0);
    cs_intro.fade_subscene(MICRO, "cps", 1);
    cps->state_manager.transition(MICRO, {
        {"root0_r", ".4"},
        {"root0_i", "0"},
        {"root1_r", ".5"},
        {"root1_i", "1.2"},
        {"root2_r", "-1"},
        {"root2_i", "-.3"},
        {"center_y", ".6"},
    });
    cs_intro.render_microblock();
    cs_intro.remove_subscene("rfs_intro");

    cs_intro.stage_macroblock(FileBlock("We'll discover how symmetries between solutions tell us about a polynomial,"), 4);
    cps->coefficients_to_roots();
    cps->stage_swap(MICRO, "0", "1", false);
    shared_ptr<LatexScene> commutator = make_shared<LatexScene>("x \\phantom{y x^-1 y^-1}", 1, .5, .35);
    cs_intro.add_scene_fade_in(MICRO, commutator, "commutator", .5, .2);
    cs_intro.render_microblock();
    commutator->begin_latex_transition(MICRO, "x y \\phantom{x^{-1} y^{-1}}");
    cps->stage_swap(MICRO, "1", "2", false);
    cs_intro.render_microblock();
    commutator->begin_latex_transition(MICRO, "x y x^{-1} \\phantom{y^{-1}}");
    cps->stage_swap(MICRO, "2", "0", false);
    cs_intro.render_microblock();
    commutator->begin_latex_transition(MICRO, "x y x^{-1} y^{-1}");
    cps->stage_swap(MICRO, "0", "1", false);
    cs_intro.render_microblock();

    cs_intro.stage_macroblock(FileBlock("and solve the mystery of the missing quintic formula."), 1);
    cs_intro.fade_subscene(MICRO, "commutator", 0);
    cps->state_manager.transition(MICRO, {
        {"center_y", ".6"},
    });
    cs_intro.render_microblock();
    cs_intro.remove_all_subscenes();

    CompositeScene cs;
    cs.add_scene(cps, "cps");

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    shared_ptr<LatexScene> ls = make_shared<LatexScene>(latex_color(0xff333333, "ax^3+bx^2+cx+d"), 1, 1, .5);
    cs.add_scene_fade_in(MICRO, ls, "ls", .5, .2);
    cs.stage_macroblock(FileBlock("Polynomials have a standard form, where each term has an associated coefficient."), 5);
    cs.render_microblock();
    cs.render_microblock();
    string colory = latex_color(0xff333333, latex_color(OPAQUE_WHITE, "a")+"x^3+"+latex_color(OPAQUE_WHITE, "b")+"x^2+"+latex_color(OPAQUE_WHITE, "c")+"x+"+latex_color(OPAQUE_WHITE, "d"));
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

    shared_ptr<LatexScene> factored = make_shared<LatexScene>(latex_color(0xff333333, "(x-x_1)(x-x_2)(x-x_3)"), .7, 1, .5);
    cs.add_scene_fade_in(MICRO, factored, "factored", .5, .7);
    cs.stage_macroblock(FileBlock("There's also a factored form, with one term for each solution,"), 4);
    cs.render_microblock();
    string colory_factored = latex_color(0xff333333, "(x-" + latex_color(OPAQUE_WHITE, "x_1")+")(x-"+latex_color(OPAQUE_WHITE, "x_2")+")(x-"+latex_color(OPAQUE_WHITE, "x_3")+")");
    factored->begin_latex_transition(MICRO, colory_factored);
    cs.render_microblock();
    factored->begin_latex_transition(MICRO, latex_color(0xff333333, "(x-x_1)(x-x_2)(x-x_3)"));
    cs.render_microblock();
    cs.render_microblock();
    cs.remove_subscene("ls");

    cs.stage_macroblock(FileBlock("which are clearly visible in the polynomial's graph."), 4);
    for(int i = 0; i < 2; i++) {
        cps->transition_root_rings(MICRO, 1);
        factored->begin_latex_transition(MICRO, colory_factored);
        cs.render_microblock();
        cps->transition_root_rings(MICRO, 0);
        factored->begin_latex_transition(MICRO, latex_color(0xff333333, "(x-x_1)(x-x_2)(x-x_3)"));
        cs.render_microblock();
    }

    cs.fade_subscene(MICRO, "factored", 0);
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cps->state_manager.transition(MICRO, {
        {"ticks_opacity", "1"},
    });
    cps->state_manager.transition(MACRO, {
        {"zoom", "-.5"},
    });
    cs.stage_macroblock(FileBlock("This space is the complex plane, home to numbers like i and 2-i."), 6);
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
    cps->state_manager.transition(MICRO, "construction_opacity", "0");
    factored->begin_latex_transition(MICRO, latex_color(0xff333333, "(x-x_1)(x-x_2)(x-x_3)"));

    cps->coefficients_to_roots(); // Just to be safe
    cs.stage_macroblock(FileBlock("For each pixel on the screen, we can pass that complex number into the polynomial, and see what comes out..."), 3);
    cps->state_manager.set({
        {"point_in_x", "0"},
        {"point_in_y", "1"},
    });
    cs.render_microblock();
    factored->begin_latex_transition(MICRO, latex_color(0xff333333, "(" + latex_color(OPAQUE_WHITE, "x")+"-x_1)(" + latex_color(OPAQUE_WHITE, "x")+"-x_2)(" + latex_color(OPAQUE_WHITE, "x")+"-x_3)"));
    cps->construction.clear();
    cps->state_manager.set("construction_opacity", "1");
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "in", .7, true));
    cs.render_microblock();
    //factored->begin_latex_transition(MICRO, "(" + latex_color(0xffff8080, "-i")+"-x_1)(" + latex_color(0xffff8080, "-i")+"-x_2)(" + latex_color(0xffff8080, "-i")+"-x_3)");
    //cs.render_microblock();
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
    cs.remove_subscene("factored");

    cs.stage_macroblock(SilenceBlock(5), 1);
    cps->state_manager.transition(MICRO, {
        {"point_in_x", "{t} sin .9 * 1.2 *"},
        {"point_in_y", "{t} cos .8 * 1.2 *"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We color the input point depending on where the output lands."), 1);
    cps->state_manager.transition(MICRO, {
        {"zoom", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MICRO, "zero_crosshair_opacity", "1");
    cps->construction.add(GeometricLine(glm::vec2(0, 0), glm::vec2(0, 0), "vec", true));
    cps->state_manager.set({
        {"line_vec_start_x", "0"},
        {"line_vec_start_y", "0"},
        {"line_vec_end_x", "<point_out_x>"},
        {"line_vec_end_y", "<point_out_y>"},
    });
    cs.render_microblock();

    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "0"},
    });
    cs.stage_macroblock(FileBlock("The brightness shows how far it is from the origin,"), 5);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cps->coefficients_to_roots();
    cs.stage_macroblock(FileBlock("meaning that the white areas are solutions of the polynomial."), 6);
    for (int i = 0; i < 3; i++) {
        cps->state_manager.transition(MICRO, {
            {"point_in_x", "<root" + to_string(i) + "_r>"},
            {"point_in_y", "<root" + to_string(i) + "_i>"},
        });
        cs.render_microblock();
        cs.render_microblock();
    }

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The color shows the angle of the output,"), 1);
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "24"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("so, for example, pink means above the origin and green means below it."), 5);
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"point_in_x", "1"},
        {"point_in_y", "-.3"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"point_in_x", "-1"},
        {"point_in_y", ".6"},
    });
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MICRO, "zero_crosshair_opacity", "0");
    cps->state_manager.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();
    cps->construction.clear();
    cps->state_manager.set("construction_opacity", "1");
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
        {"line_vec_start_x"},
        {"line_vec_start_y"},
        {"line_vec_end_x"},
        {"line_vec_end_y"},
    });
    cs.state_manager.set("construction_opacity", "1");

    cs.stage_macroblock(FileBlock("Doing this for every complex number on the plane, we can graph our polynomial."), 2);
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "1"},
    });
    cps->coefficients_to_roots();
    cps->state_manager.transition(MACRO, {
        {"root0_r", "-1.2"},
        {"root0_i", "0"},
        {"root1_r", "-.3"},
        {"root1_i", "0"},
        {"root2_r", "1.2"},
        {"root2_i", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.remove_all_subscenes();

    ThreeDimensionScene tds;
    tds.add_surface(Surface("cps"), cps);
    tds.stage_macroblock(SilenceBlock(2), /*2*/1);
    tds.state_manager.transition(MICRO, {
        {"d", "1.5"},
    });
    //tds.render_microblock();

    shared_ptr<RealFunctionScene> rfs = make_shared<RealFunctionScene>();
    rfs->add_function("? 1.2 - ? .3 + ? 1.2 + * *", 0xffff0000);
    rfs->state_manager.set("ticks_opacity", "0");
    tds.add_surface(Surface(glm::vec3(0,0,0), glm::vec3(.5,0,0), glm::vec3(0,0,.5), "rfs"), rfs);
    tds.state_manager.transition(MICRO, {
        {"qi", "-.3 {t} sin .07 * +"},
        {"qj", "{t} cos .025 *"},
    });
    tds.render_microblock();

    shared_ptr<LatexScene> f_complex = make_shared<LatexScene>("\\text{Complex Function}", 1, .5, .5);
    shared_ptr<LatexScene> f_real    = make_shared<LatexScene>("\\text{Real Function}", 1, .5, .5);
    tds.add_surface_fade_in(MICRO, Surface(glm::vec3(0,.25,-.05), glm::vec3(.25, 0, 0), glm::vec3(0, .25, 0), "f_complex"), f_complex);
    tds.add_surface_fade_in(MICRO, Surface(glm::vec3(0,.05,-.25), glm::vec3(.25, 0, 0), glm::vec3(0, 0, .25), "f_real"), f_real);
    tds.stage_macroblock(FileBlock("You might ask, what's all this complex number business? Why leave the familiar land of real numbers?"), 1);
    tds.render_microblock();

    tds.stage_macroblock(FileBlock("In turn, I would ask _you_, who needs decimals or negatives either?"), 2);
    tds.state_manager.transition(MICRO, {
        {"d", "1"},
        {"qi", "0"},
        {"qj", "0"},
    });
    tds.render_microblock();
    cps->coefficients_to_roots();
    for(int i = 0; i < cps->get_degree(); i++) {
        cps->state_manager.transition(MICRO, {
            {"root"+to_string(i)+"_r", "{t} 1.2 * 6.28 .3333 " + to_string(i) + " * * + sin"},
            {"root"+to_string(i)+"_i", "{t} .8 * 6.28 .3333 " + to_string(i) + " * * + cos 5 -"},
        });
    }
    tds.fade_subscene(MICRO, "f_complex", 0);
    tds.fade_subscene(MICRO, "f_real", 0);
    tds.fade_subscene(MICRO, "rfs", 0);
    tds.render_microblock();
    tds.remove_all_subscenes();

    cs = CompositeScene();
    cs.add_scene(cps, "cps");
    cs.stage_macroblock(FileBlock("Imagine there's nothing but natural numbers- 0, 1, 2, 3, sitting happily on the right side of the number line."), 11);
    cs.render_microblock();
    cs.render_microblock();
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
    impossible->begin_latex_transition(MICRO, latex_color(0xff222222, latex_color(OPAQUE_WHITE, "2")+"x+"+latex_color(OPAQUE_WHITE, "4")+"="+latex_color(OPAQUE_WHITE, "0")));
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
    impossible->begin_latex_transition(MACRO, latex_color(0xff222222, "2"+latex_color(OPAQUE_WHITE, "x")+"+4=0"));
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    impossible->begin_latex_transition(MICRO, "2x+4=0");
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("Let's plot this polynomial."), SilenceBlock(2)), 1);
    cps->state_manager.transition(MACRO, {
        {"construction_opacity", ".15"},
    });
    cps->roots_to_coefficients();
    cps->state_manager.set({
        {"linear_transitioner", "0"},
        {"exponential_transitioner", "<linear_transitioner> .25 ^ -1 * exp"},
        {"coefficient3_r", cps->state_manager.get_equation("coefficient3_r") + " <exponential_transitioner> * " + new_coefficient_val + " .5 * +"},
        {"coefficient3_i", cps->state_manager.get_equation("coefficient3_i") + " <exponential_transitioner> *"},
        {"coefficient2_r", cps->state_manager.get_equation("coefficient2_r") + " <exponential_transitioner> * " + new_coefficient_val + " .5 * +"},
        {"coefficient2_i", cps->state_manager.get_equation("coefficient2_i") + " <exponential_transitioner> *"},
    });
    cps->state_manager.transition(MICRO, {
        {"linear_transitioner", "10000"},
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
    cs.stage_macroblock(FileBlock("The coefficients lands on numbers in our number system,"), flashes*2);
    for(int i = 0; i < flashes; i++) {
        cps->transition_coefficient_rings(MICRO, 1);
        cs.render_microblock();
        cps->transition_coefficient_rings(MICRO, 0);
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("but the solution to the equation is -2... which isn't one of them!"), flashes*2+2);
    cps->state_manager.transition(MACRO, "construction_opacity", ".4");
    impossible->begin_latex_transition(MICRO, "2 \\cdot" + latex_color(0xff00ff00, "\\small{-2}")+"+4=0");
    cs.render_microblock();
    for(int i = 0; i < flashes; i++) {
        cps->transition_root_rings(MICRO, 1);
        cs.render_microblock();
        cps->transition_root_rings(MICRO, 0);
        cs.render_microblock();
    }
    cs.fade_subscene(MICRO, "impossible", 0);
    cs.render_microblock();
    cs.remove_subscene("impossible");

    cs.stage_macroblock(FileBlock("To be able to solve this, we _need_ to invent -2..."), 2);
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(-2, 0), "-2"));
    cs.render_microblock();

    cps->state_manager.transition(MICRO, {
        {"coefficient0_r", "1"},
        {"coefficient0_i", "0"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
    });
    cs.stage_macroblock(FileBlock("And furthermore, I can place the coefficients on other values, "), 1);
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

    cs.stage_macroblock(FileBlock("Our coefficients lie on existing numbers, but the solution lands outside of our number system."), 7);
    cps->transition_coefficient_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(-.5, 0), "-.5", .45));
    impossible->begin_latex_transition(MICRO, "2\\cdot"+latex_color(0xff00ff00, "-.5")+"+1=0");
    cs.render_microblock();
    cs.render_microblock();
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.render_microblock();

    int microblock_count = 0;
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
    cs.fade_subscene(MICRO, "impossible", 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Heck, let's just add all decimals. We get a nice continuum of numbers."), 1);
    cps->construction.add(GeometricLine(glm::vec2(-5, 0), glm::vec2(5, 0)));
    cs.render_microblock();
    cs.remove_all_subscenes();

    shared_ptr<ThreeDimensionScene> tds2 = make_shared<ThreeDimensionScene>();
    cs.stage_macroblock(CompositeBlock(FileBlock("Surely our number system is finally complete!"), SilenceBlock(1.5)), 1);
    cs.add_scene(tds2, "tds2");
    rfs = make_shared<RealFunctionScene>();
    rfs->add_function("? 2 * 1 +", 0xffff0000);
    rfs->state_manager.set("ticks_opacity", "0");
    tds2->add_surface(Surface("cps"), cps);
    shared_ptr<RealFunctionScene> rfs2 = make_shared<RealFunctionScene>();
    rfs2->state_manager.set("ticks_opacity", "0");
    rfs2->add_function("1 ? ? * -", 0xffcc00ff);
    tds2->add_surface(Surface(glm::vec3(0, 0, 0), glm::vec3(0, .5, 0), glm::vec3(0, 0, .5), "rfs2"), rfs2);
    tds2->add_surface(Surface(glm::vec3(0, 0, 0), glm::vec3(.5, 0, 0), glm::vec3(0, 0, .5), "rfs"), rfs);
    tds2->state_manager.transition(MICRO, {
        {"qi", "-.1 {t} 2 / sin .02 * +"},
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
        {"construction_opacity", ".7"},
    });
    rfs->begin_transition(MACRO, 0, "? ? * 1 +");
    shared_ptr<LatexScene> impossible_sq = make_shared<LatexScene>("x^2+1=0", .5, 1, .4);
    cs.add_scene_fade_in(MICRO, impossible_sq, "impossible_sq", .5, .77);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(1), CompositeBlock(FileBlock("No real number squared gives us negative 1..."), SilenceBlock(1))), 3);
    impossible_sq->begin_latex_transition(MICRO, latex_color(0xff333333, latex_color(OPAQUE_WHITE, "x^2")+"+1=0"));
    cs.render_microblock();
    impossible_sq->begin_latex_transition(MICRO, latex_color(0xff333333, latex_color(OPAQUE_WHITE, "x^2")+"=-1"));
    cs.render_microblock();
    cs.fade_subscene(MICRO, "impossible_sq", 0);
    cs.render_microblock();
    cs.remove_subscene("impossible_sq");

    cs.stage_macroblock(FileBlock("and you can see it too- none of the solutions are on the number line."), 4);
    for(int flash = 0; flash < 2; flash++) {
        cps->transition_root_rings(MICRO, 1);
        cs.render_microblock();
        cps->transition_root_rings(MICRO, 0);
        cs.render_microblock();
    }

    cs.stage_macroblock(SilenceBlock(.5), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Just like before, our number line must be missing something..."), 1);
    tds2->state_manager.transition(MICRO, {
        {"qi", "-.1 {t} 2 / sin .02 * +"},
        {"qj", "-.2 {t} 2 / cos .01 * +"},
        {"d", "1.5"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    tds2->state_manager.transition(MICRO, {
        {"qi", "0"},
        {"qj", "0"},
        {"d", "1"},
    });
    tds2->fade_subscene(MICRO, "rfs", 0);
    tds2->fade_subscene(MICRO, "rfs2", 0);
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
    cs.stage_macroblock(FileBlock("which, combined with the real numbers, forms the complex plane."), 13*2);
    for(int slice = -6; slice < 6; slice++) {
        for(int x = -4; x <= 4; x++) {
            for(int y = -2; y <= 2; y++) {
                if(x == 0 || y == 0) continue;
                if(x+y != slice) continue;
                string label = to_string(x);
                if(y != 0){
                    if(y < 0) label += "-";
                    else label += "+";
                    if(abs(y) != 1) label += to_string(abs(y));
                    label += "i";
                }
                cps->construction.add(GeometricPoint(glm::vec2(x, y), label, .5));
            }
        }
        cs.render_microblock();
    }
    while(cs.microblocks_remaining()) cs.render_microblock();

    cps->coefficients_to_roots();
    for(int i = 0; i < cps->get_degree(); i++) {
        cps->state_manager.transition(MICRO, {
            {"root"+to_string(i)+"_r", "{t} 1.2 * 6.28 .3333 " + to_string(i) + " * * + sin"},
            {"root"+to_string(i)+"_i", "{t} .8 * 6.28 .3333 " + to_string(i) + " * * + cos"},
        });
    }

    cs.stage_macroblock(SilenceBlock(2), 1);
    cps->state_manager.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();
    cps->construction.clear();
    cps->state_manager.set("construction_opacity", "1");

    cs.stage_macroblock(FileBlock("Up until now, we've been playing whack-a-mole..."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("_You_ invent numbers, and _I_ make an equation that can't be solved without _more numbers_."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But that game ends here!"), 1);
    cps->state_manager.set({{"coefficient0_ring", "0"}, {"coefficient1_ring", "0"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Wherever we put the coefficients inside the complex plane,"), 2);
    cps->transition_coefficient_opacities(MICRO, 1);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("the solutions may move around, but they don't jump out of the number system like they did before."), 2);
    cps->transition_coefficient_opacities(MICRO, 0);
    cps->transition_coefficient_rings(MICRO, 0);
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("What happens on the complex plane stays on the complex plane."), 1);
    shared_ptr<LatexScene> fta = make_shared<LatexScene>("\\mathbb{C} \\text{ is algebraically closed.}", 1, .6, .5);
    cs.add_scene_fade_in(MICRO, fta, "fta", .5, .5);
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

    quadratic->begin_latex_transition(MICRO, latex_color(0xff333333, "x^"+latex_color(OPAQUE_WHITE, "2")+"-x-2"));
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

    cs.stage_macroblock(CompositeBlock(FileBlock("With a cubic polynomial, there's 3 solutions."), SilenceBlock(1)), 4);
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
        {"zoom", ".7"},
    });
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"root2_ring", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1.5), 1);
    cps->state_manager.transition(MICRO, {
        {"center_x", "0"},
        {"zoom", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("You can actually identify the solutions' multiplicities from the graph."), 1);
    cps->state_manager.transition(MICRO, {
        {"ticks_opacity", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MACRO, {
        {"center_x", "<root0_r>"},
        {"center_y", "<root0_i>"},
        {"zoom", "1.5"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Tracing around this normal solution,"), 1);
    cps->state_manager.transition(MICRO, {
        {"ticks_opacity", "0"},
    });
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "2.4"},
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

    cs.stage_macroblock(SilenceBlock(2), 1);
    cps->state_manager.begin_timer("theta2");
    cps->state_manager.set({
        {"theta3", "<theta2> 1 -"},
    });
    cps->state_manager.transition(MACRO, {
        {"point__x", "<root2_r> <theta3> 2 * cos .2 * +"},
        {"point__y", "<root2_i> <theta3> 2 * sin .2 * +"},
        {"center_x", "<root2_r>"},
        {"center_y", "<root2_i>"},
    });
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "1.3"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Tracing around the multiplicity 2 solution,"), 1);
    cps->state_manager.transition(MACRO, {
        {"ab_dilation", "1.8"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("we see red, green, blue, then red, green, blue again."), 19);
    cs.render_microblock();
    cs.render_microblock();
    for(int i = 0; i < 6; i++) {
        double angle = 3.14 * (i/6. + .275 + 1);
        if(i == 0) angle += .1;
        cps->state_manager.transition(MICRO, "theta3", to_string(angle));
        cs.render_microblock();
        cs.render_microblock();
        if(i == 2) {
            cs.render_microblock();
        }
    }
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("The color wheel is duplicated!"), SilenceBlock(1)), 2);
    cps->state_manager.transition(MACRO, {
        {"zoom", "0"},
        {"center_x", "0"},
        {"center_y", "0"},
    });
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"ticks_opacity", "0"},
        {"ab_dilation", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's plot the output like before."), 2);
    cps->state_manager.transition(MICRO, {
        {"point__x", "<root0_r> {t} 3 * sin .1 * +"},
        {"point__y", "<root0_i> {t} 3 * cos .1 * +"},
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
        {"zoom", ".7"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("As we trace around the multiplicity 1 solution, the output point follows the angle of the input point."), 1);
    cs.render_microblock();

    cps->state_manager.transition(MICRO, {
        {"zoom", "0"},
    });
    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MICRO, {
        {"point__x", "<root2_r> {t} 3 * sin .3 * +"},
        {"point__y", "<root2_i> {t} 3 * cos .3 * +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But tracing around the multiplicity 2 solution, the output point goes around twice as fast."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MICRO, "construction_opacity", "0");
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
    cps->state_manager.set("construction_opacity", "1");

    cs.stage_macroblock(FileBlock("The degree of the polynomial,"), 2);
    quadratic = make_shared<LatexScene>(latex_color(0xff333333, "1x^" + latex_color(OPAQUE_WHITE, "3") + "+1x^2-1x-1"), .7, 1, .5);
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
    quadratic->begin_latex_transition(MICRO, latex_color(0xff333333, "1x^3+1x^2-1x-1"));
    cs.render_microblock();

    quadratic->begin_latex_transition(MICRO, latex_color(0xff333333, latex_color(OPAQUE_WHITE, "1")+"x^3+"+latex_color(OPAQUE_WHITE, "1")+"x^2-"+latex_color(OPAQUE_WHITE, "1")+"x-"+latex_color(OPAQUE_WHITE, "1")));
    cs.stage_macroblock(FileBlock("Another way of saying this is that there's always exactly one more coefficient than the number of solutions."), 2);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_subscene(MACRO, "quadratic", 0);
    cs.render_microblock();
    cs.remove_subscene("quadratic");

    cs.stage_macroblock(FileBlock("But, how do they relate to each other?"), 1);
    cps->transition_coefficient_opacities(MICRO, 1);
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    //move the roots around
    cps->roots_to_coefficients();
    for(int i = 0; i < cps->get_degree()+1; i++) {
        cps->state_manager.transition(MICRO, {
            {"coefficient"+to_string(i)+"_r", "{t} 1.2 * 6.28 .47 " + to_string(i) + " * * + sin"},
            {"coefficient"+to_string(i)+"_i", "{t} .8 * 6.28 .36 " + to_string(i) + " * * + cos .2 +"},
        });
    }
    cs.stage_macroblock(FileBlock("Given the coefficients,"), 2);
    cps->transition_coefficient_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cs.render_microblock();

    cps->state_manager.transition(MICRO, {
        {"coefficient3_r", "1"},
        {"coefficient3_i", "0"},
    });
    cps->transition_coefficient_opacities(MACRO, 0);
    cs.stage_macroblock(FileBlock("how do we know where the solutions should be?"), 2);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("It's obvious since I'm plotting the graph for you..."), 1);
    //move roots offscreen
    cps->coefficients_to_roots();
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
    shared_ptr<LatexScene> abc = make_shared<LatexScene>("\\begin{tabular}{cc} a=2+i \\phantom{x_1=?} & \\end{tabular}", .4);
    cs.add_scene_fade_in(MICRO, abc, "abc");
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=2+i & \\phantom{x_1=?} \\\\\\\\ b=-i & \\phantom{x_2=?} \\end{tabular}");
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=2+i & \\phantom{x_1=?} \\\\\\\\ b=-i & \\phantom{x_2=?} \\\\\\\\ c=1.5 & \\phantom{x_3=?} \\end{tabular}");
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=2+i & x_1=? \\\\\\\\ b=-i & \\phantom{x_2=?} \\\\\\\\ c=1.5 & \\phantom{x_3=?} \\end{tabular}");
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=2+i & x_1=? \\\\\\\\ b=-i & x_2=? \\\\\\\\ c=1.5 & \\phantom{x_3=?} \\end{tabular}");
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=2+i & x_1=? \\\\\\\\ b=-i & x_2=? \\\\\\\\ c=1.5 & x_3=? \\end{tabular}");
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Before I spoil this problem, let me illustrate its complexities."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's take the simplest case imaginable- a polynomial with coefficients that are all 1 or -1."), 2);
    cps->roots_to_coefficients();
    cps->state_manager.transition(MACRO, {
        {"coefficient3_r", new_coefficient_val},
        {"coefficient3_i", "0"},
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "-1"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "1"},
        {"coefficient0_i", "0"},
    });
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=1 & x_1=? \\\\\\\\ b=-1 & x_2=? \\\\\\\\ c=1 & x_3=? \\end{tabular}");
    cs.render_microblock();
    cps->roots_to_coefficients();
    cs.render_microblock();
    cps->set_degree(2);

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("There's a few different options of which are 1s and which are -1s."), 3);
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a=1 & x_1=? \\\\\\\\ b=1 & x_2=? \\\\\\\\ c=-1 & x_3=? \\end{tabular}");
    cps->state_manager.transition(MICRO, {
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "-1"},
        {"coefficient0_i", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_subscene(MICRO, "abc", 0);
    cs.render_microblock();
    cs.remove_all_subscenes_except("cps");

    flashes = 2;
    cs.stage_macroblock(CompositeBlock(FileBlock("With a quadratic polynomial, there's exactly 8 options."), FileBlock("Plotting all of their solutions at the same time...")), flashes*8);
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
    for(int bit = 0; bit < flashes*8; bit++) {
        string a = (bit&4) ? "1" : "-1";
        string b = (bit&2) ? "1" : "-1";
        string c = (bit&1) ? "1" : "-1";
        abc->begin_latex_transition(MICRO, "\\begin{tabular}{cc} a="+a+" & x_1=? \\\\\\\\ b="+b+" & x_2=? \\\\\\\\ c="+c+" & x_3=? \\end{tabular}");
        cps->state_manager.transition(MICRO, {
            {"coefficient2_r", a},
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

    fracs->stage_macroblock(CompositeBlock(FileBlock("and even higher,"), SilenceBlock(8)), 13);
    fracs->state_manager.transition(MACRO, {
        {"visibility_multiplier", "1"},
        {"zoom", ".7"},
    });
    int i = 6;
    while(fracs->microblocks_remaining()) {
        fracs->state_manager.transition(MICRO, {
            {"terms", to_string(i)},
        });
        i++;
        fracs->render_microblock();
    }

    fracs->stage_macroblock(FileBlock("We were letting the coefficients be either 1 or -1,"), 1);
    fracs->state_manager.transition(MICRO, {
        {"coefficients_opacity", "1"},
    });
    fracs->render_microblock();

    fracs->stage_macroblock(FileBlock("but what happens if we change those options?"), 1);
    fracs->state_manager.begin_timer("fractal_timer");
    fracs->state_manager.transition(MICRO, {
        {"coefficient0_r", "-.01 <fractal_timer> 120 + 4 / cos *"},
        {"coefficient0_i", ".01 <fractal_timer> 120 + 4 / sin *"},
        {"zoom", "0"},
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
        {"zoom", "2.5"},
    });
    fracs->render_microblock();

    fracs->stage_macroblock(SilenceBlock(40), 10);
    fracs->state_manager.transition(MICRO, {
        {"coefficient0_r", "<fractal_timer> 4 - 8 / sin"},
        {"coefficient0_i", "<fractal_timer> 4 - 9 / sin"},
    });
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"center_x", "0"},
        {"center_y", "1"},
    });
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"rainbow", "1"},
        {"zoom", "3"},
    });
    fracs->render_microblock();
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"zoom", ".3"},
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
        {"coefficient0_r", "<spin> 1.9 / cos"},
        {"coefficient0_i", "<spin> 1.9 / sin"},
    });
    fracs->render_microblock();
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"coefficient0_r", "<spin> 1.9 / cos 5 *"},
        {"coefficient0_i", "<spin> 1.9 / sin 5 *"},
    });
    fracs->render_microblock();
    fracs->render_microblock();

    fracs->stage_macroblock(CompositeBlock(SilenceBlock(3), FileBlock("As we zoom into these point clouds,")), 2);
    fracs->state_manager.transition(MACRO, {
        {"center_x", "-.48165"},
        {"center_y", "-.45802"},
        {"coefficient0_r", "0.1"},
        {"coefficient0_i", "0"},
    });
    fracs->render_microblock();
    fracs->state_manager.transition(MICRO, {
        {"rainbow", "1"},
        {"zoom", "2.5"},
    });
    fracs->render_microblock();

    cs.stage_macroblock(FileBlock("there are shapes resembling the Dragon Curve,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(5), 2);
    fracs->state_manager.transition(MACRO, { {"zoom", "3"}, });
    cs.render_microblock();
    cs.state_manager.transition(MICRO, "fracs.x", ".25");
    fracs->state_manager.transition(MICRO, "w", ".5");
    shared_ptr<LatexScene> dragon = make_shared<LatexScene>("\\text{Dragon Curve TODO}", 1, .5, 1);
    cs.add_scene(dragon, "dragon", 1.25, .5);
    cs.slide_subscene(MICRO, "dragon", -.5, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("That's the same shape that you get by folding up a strip of paper and letting it spring out."), 1);
    cs.render_microblock();

    cs.state_manager.transition(MICRO, "fracs.x", ".5");
    fracs->state_manager.transition(MICRO, "w", "1");
    cs.slide_subscene(MICRO, "dragon", .5, 0);
    cs.stage_macroblock(FileBlock("The point is, the function mapping coefficients to solutions isn't simple- it must have a bit of magic to it."), 1);
    cs.render_microblock();
    cs.remove_all_subscenes_except("fracs");

    fracs->state_manager.transition(MICRO, {
        {"coefficient0_r", "-1"},
        {"coefficient0_i", "0"},
    });
    fracs->stage_macroblock(FileBlock("Even with just 1 and -1 as coefficients, we find ourselves in a zoo of emergent complexity."), 1);
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
        {"zoom", "0"},
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

    cs.stage_macroblock(FileBlock("Let's start off easy, with polynomials whose highest exponent is one."), 2);
    quadratic = make_shared<LatexScene>(latex_color(0xff333333, "ax^"+latex_color(OPAQUE_WHITE, "1")+"+b = 0"), .6, 1, .5);
    cs.add_scene_fade_in(MICRO, quadratic, "quadratic", .5, .2);
    cps->roots_to_coefficients();
    cps->state_manager.transition(MICRO, {
        {"coefficient2_r", new_coefficient_val},
        {"coefficient2_i", "0"},
        {"coefficient2_opacity", "0"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", ".8"},
        {"coefficient0_i", "-.5"},
    });
    cs.render_microblock();
    cps->set_degree(1);
    cps->transition_coefficient_opacities(MICRO, 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    quadratic->begin_latex_transition(MICRO, "ax+b=0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("With such a linear polynomial, there's only one solution."), 4);
    cs.render_microblock();
    cs.render_microblock();
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
        {"coefficient0_r", "{t} sin"},
        {"coefficient0_i", "{t} cos"},
    });
    cs.fade_subscene(MICRO, "quadratic", 0);
    cs.render_microblock();
    cs.remove_subscene("quadratic");

    cs.stage_macroblock(SilenceBlock(1.5), 1);
    cps->roots_to_coefficients();
    cps->state_manager.transition(MICRO, {
        {"coefficient0_r", "0"},
        {"coefficient0_i", "1"},
    });
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.render_microblock();

    cps->set_degree(2);

    StateSet preflip = {
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "1.7"},
        {"coefficient1_i", "1"},
        {"coefficient0_r", ".5"},
        {"coefficient0_i", "1"},
    };
    cps->state_manager.transition(MICRO, preflip);

    cs.stage_macroblock(FileBlock("Jumping up to a polynomial with highest exponent 2,"), 1);
    cps->transition_coefficient_opacities(MICRO, 1);
    cs.render_microblock();

    shared_ptr<LatexScene> quadratic_formula = make_shared<LatexScene>("\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}", 1, .3, .3);
    cs.stage_macroblock(FileBlock("There's a certain formula you might have memorized in school..."), 1);
    cs.add_scene(quadratic_formula, "quadratic_formula", -.2, .5);
    cs.state_manager.begin_timer("quadratic_timer");
    cs.state_manager.set("quadratic_formula.x", "<quadratic_timer> .1 * .2 -");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but pretend you didn't for a moment,"), 2);
    cs.state_manager.transition(MICRO, "quadratic_formula.x", "-.5");
    cs.render_microblock();
    cs.remove_subscene("quadratic_formula");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("cause something starts getting weird in the quadratic case."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("You see, coefficients and solutions are structurally very different."), 10);
    cs.render_microblock();
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.5), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Looking at the polynomial,"), 1);
    quadratic = make_shared<LatexScene>(latex_color(0xff333333, "x^2 + "+latex_color(OPAQUE_WHITE, "b")+"x + "+latex_color(OPAQUE_WHITE, "c")), .7, 1, .5);
    cs.add_scene_fade_in(MICRO, quadratic, "quadratic", .5, .15);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("If we switch b and c here, we don't have the same function anymore."), 4);
    quadratic->begin_latex_transition(MICRO, latex_color(0xff333333, "x^2 + "+latex_color(OPAQUE_WHITE, "c")+"x + "+latex_color(OPAQUE_WHITE, "b")));
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"coefficient0_ring", "1"},
        {"coefficient1_ring", "1"},
    });
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", true);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Notice how the solutions aren't in the same place after I swap the coefficients."), 3);
    cs.fade_subscene(MICRO, "quadratic", 0);
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cs.remove_subscene("quadratic");
    cps->stage_swap(MICRO, "0", "1", true);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    int flips = 15;
    cs.stage_macroblock(FileBlock("The coefficients are an ordered list- they're not interchangeable."), flips);
    for(int i = 0; i < flips; i++) {
        if(i == 1 || i == 4 || i == 8 || i == 9 || i == 12 || i == 14) {
            cps->stage_swap(MICRO, "0", "1", true);
        }
        cs.render_microblock();
    }

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MICRO, {
        {"coefficient1_ring", "0"},
        {"coefficient0_ring", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The solutions, on the other hand, don't have a particular order to them."), 2);
    cps->state_manager.transition(MACRO, {
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
    });
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("In the factored form of the polynomial, we can rearrange as we please,"), 2);
    factored = make_shared<LatexScene>("(x - "+latex_color(OPAQUE_WHITE, "\\small{x_1}")+") (x - "+latex_color(OPAQUE_WHITE, "x_2")+") (x - "+latex_color(OPAQUE_WHITE, "x_3")+")", .7, 1, .5);
    cs.add_scene_fade_in(MACRO, factored, "factored", .5, .2);
    cps->coefficients_to_roots();
    cs.render_microblock();
    factored->begin_latex_transition(MICRO, "(x - "+latex_color(OPAQUE_WHITE, "x_2")+") (x - "+latex_color(OPAQUE_WHITE, "\\small{x_1}")+") (x - "+latex_color(OPAQUE_WHITE, "x_3")+")");
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
    cps->transition_coefficient_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This has some bizarre implications."), 1);
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("First and foremost, there's no function which takes in the coefficients and gives out _some individual solution_, and is continuous."), FileBlock("Here's why.")), 8);
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.render_microblock();
    shared_ptr<LatexScene> function = make_shared<LatexScene>("f(a, b, c) \\rightarrow x_1", 1, .75, .4);
    shared_ptr<LatexScene> details = make_shared<LatexScene>("f \\text{ continuous}", 1, .5, .4);
    cs.add_scene_fade_in(MICRO, function, "function", .5, .4);
    cs.render_microblock();
    function->begin_latex_transition(MICRO, "f("+latex_color(0xff8080ff, "a, b, c")+") \\rightarrow x_1");
    cs.render_microblock();
    function->begin_latex_transition(MICRO, "f(a, b, c) \\rightarrow "+latex_color(0xff80ff80, "x_1"));
    cs.render_microblock();
    function->begin_latex_transition(MICRO, "f(a, b, c) \\rightarrow x_1");
    cs.render_microblock();
    cs.add_scene_fade_in(MICRO, details, "details", .5, .65);
    cs.render_microblock();
    details->begin_latex_transition(MICRO, latex_color(0xffff0000, "f \\text{ continuous}"));
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Assume we do have a continuous function that takes in a, b, and c,"), 4);
    cs.fade_subscene(MICRO, "function", 0);
    cs.fade_subscene(MICRO, "details", 0);
    cps->transition_coefficient_opacities(MICRO, 1);
    cs.render_microblock();
    cs.remove_subscene("function");
    cs.remove_subscene("details");
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and gives us this solution."), 5);
    cps->state_manager.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("As we saw before, we can swap the solutions and the coefficients will return to their starting position."), 3);
    cps->coefficients_to_roots();
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But our 'continuous function' gives us a different solution now!"), 4);
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Given this same a, b, and c,"), 3);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("First it gave us this solution, but then it gave us that solution."), 8);
    cps->state_manager.transition(MICRO, "root1_ring", "1");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root1_ring", "0");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root1_ring", "1");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root1_ring", "0");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("That's a contradiction! The function must not be continuous."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We could handle this by accepting a function which is discontinuous, like this."), 4);
    cps->state_manager.transition(MICRO, "positive_quadratic_formula_opacity", "1");
    cs.render_microblock();
    cps->coefficients_to_roots();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("A 'quadratic formula' couldn't be made without such a discontinuity."), 1);
    cps->state_manager.transition(MICRO, "positive_quadratic_formula_opacity", "0");
    cs.render_microblock();
    cs.remove_all_subscenes_except("cps");

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    shared_ptr<LatexScene> continuous_operations = make_shared<LatexScene>("+, -, \\times, \\div", .5, 1, .3);
    cs.add_scene_fade_in(MICRO, continuous_operations, "continuous_operations", .5, .15);
    cs.stage_macroblock(FileBlock("And that means it can't be written with just basic arithmetic!"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Even if we incorporate other functions,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("such as sines, (long pause), cosines, (long pause), and exponentials,"), 6);
    continuous_operations->begin_latex_transition(MICRO, "+, -, \\times, \\div, \\sin");
    shared_ptr<ComplexArbitraryFunctionScene> cafs = make_shared<ComplexArbitraryFunctionScene>();
    cafs->state_manager.set({{"ticks_opacity", "0"}, {"sqrt_coef", "0"}, {"sin_coef", "1"}});
    cs.fade_subscene(MICRO, "cps", 0);
    cs.add_scene_fade_in(MICRO, cafs, "cafs", .5, .5, 1, true);
    cs.render_microblock();
    cs.render_microblock();
    cafs->state_manager.transition(MICRO, {{"sin_coef", "0"}, {"cos_coef", "1"}});
    continuous_operations->begin_latex_transition(MICRO, "+, -, \\times, \\div, \\sin, \\cos");
    cs.render_microblock();
    cs.render_microblock();
    cafs->state_manager.transition(MICRO, {{"cos_coef", "0"}, {"exp_coef", "1"}});
    continuous_operations->begin_latex_transition(MICRO, "+, -, \\times, \\div, \\sin, \\cos, e^x");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("they feature no discontinuities, so they have no way of selecting a unique solution of the two."), 2);
    cs.render_microblock();
    cs.fade_all_subscenes(MICRO, 0);
    cs.fade_subscene(MICRO, "cps", 1);
    cs.render_microblock();
    cs.remove_subscene("cafs");
    cs.remove_subscene("continuous_operations");

    cs.stage_macroblock(FileBlock("What about the quadratic formula!?"), 2);
    cs.slide_subscene(MICRO, "function", 0, -.25);
    cs.slide_subscene(MICRO, "details", 0, -.25);
    cs.render_microblock();
    quadratic_formula = make_shared<LatexScene>("\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}", 1, .6, .4);
    cs.add_scene(quadratic_formula, "quadratic_formula", -.3, .5);
    cs.slide_subscene(MICRO, "quadratic_formula", .8, 0);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_subscene(MICRO, "function", 0);
    cs.fade_subscene(MICRO, "details", 0);
    cs.slide_subscene(MICRO, "function", 0, -.45);
    cs.slide_subscene(MICRO, "details", 0, -.45);
    cs.render_microblock();
    cs.remove_subscene("function");
    cs.remove_subscene("details");

    cs.stage_macroblock(FileBlock("it has this plus-or-minus square rooty stuff going on."), 1);
    quadratic_formula->begin_latex_transition(MICRO, latex_color(0xff222222, "\\frac{-b\\:" + latex_color(0xffffffff, " \\pm \\sqrt{b^2 - 4ac}") + "}{2a}"));
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_subscene(MICRO, "quadratic_formula", 0);
    cs.render_microblock();
    cs.remove_subscene("quadratic_formula");

    cs.stage_macroblock(FileBlock("What makes a square root different from those other functions?"), 1);
    shared_ptr<LatexScene> sqrt4 = make_shared<LatexScene>("\\sqrt{\\phantom{4}}", 1, .5, .3);
    cs.add_scene_fade_in(MICRO, sqrt4, "sqrt4");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We usually say things like 'the square root of 4 is 2,'"), 2);
    sqrt4->begin_latex_transition(MICRO, "\\sqrt{4}");
    cs.render_microblock();
    sqrt4->begin_latex_transition(MICRO, "2 = \\sqrt{4}");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but what we're asking to begin with is,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("'what number, when multiplied by itself, gives 4?'"), 1);
    sqrt4->begin_latex_transition(MICRO, "x * x = 4");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("what are the solutions of x^2=4?"), 1);
    sqrt4->begin_latex_transition(MICRO, "x^2 = 4");
    cps->set_degree(2);
    cps->roots_to_coefficients();
    cps->state_manager.transition(MICRO, {
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "0"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "-4"},
        {"coefficient0_i", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Plotting that equation, we of course see the result 2,"), 2);
    cs.fade_subscene(MICRO, "sqrt4", 0);
    cps->construction.add(GeometricPoint(glm::vec2(2, 0), "2"));
    cs.render_microblock();
    cs.render_microblock();
    cs.remove_subscene("sqrt4");

    cs.stage_macroblock(FileBlock("as well as the negative square root, -2."), 2);
    cps->construction.add(GeometricPoint(glm::vec2(-2, 0), "-2"));
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("By convention, we define the 'square root' to be positive."), 4);
    cps->state_manager.transition(MICRO, "construction_opacity", "0");
    cps->coefficients_to_roots();
    cps->state_manager.set({
        {"root0_r", "2"},
        {"root0_i", "0"},
        {"root1_r", "-2"},
        {"root1_i", "0"},
    });
    for(int i = 0; i < 4; i++) {
        cps->state_manager.transition(MICRO, "root0_ring", to_string(1 - (i % 2)));
        cs.render_microblock();
    }
    cps->construction.clear();
    cps->state_manager.set("construction_opacity", "1");

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But what about the square root of -1?"), 1);
    cps->roots_to_coefficients();
    cps->state_manager.transition(MICRO, {
        {"coefficient0_r", "1"},
        {"coefficient0_i", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We _define_ i as being precisely that,"), 1);
    cps->construction.add(GeometricPoint(glm::vec2(0, 1), "i"));
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but just like before, -i works too."), 1);
    cps->construction.add(GeometricPoint(glm::vec2(0, -1), "-i"));
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This is more than just a curiosity."), 1);
    cps->state_manager.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();
    cps->construction.clear();
    cps->state_manager.set("construction_opacity", ".3");

    cs.stage_macroblock(FileBlock("Watch as I move the coefficients of the polynomial along the real number line."), 2);
    cps->transition_coefficient_opacities(MICRO, 1);
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", ".25"},
        {"dot_radius", "3"},
    });
    cs.render_microblock();
    cps->roots_to_coefficients();
    cps->state_manager.begin_timer("realline");
    cps->state_manager.transition(MICRO, {
        {"coefficient0_r", "<realline> 1.0 * 0 + 2 / sin 3 *"},
        {"coefficient0_i", "0"},
        {"coefficient1_r", "<realline> 1.1 * 1 + 2 / sin 3 *"},
        {"coefficient1_i", "0"},
        {"coefficient2_r", "<realline> 1.2 * 2 + 2 / sin 3 *"},
        {"coefficient2_i", "0"},
    });
    cps->construction.add(GeometricLine(glm::vec2(-5, 0), glm::vec2(5, 0)));
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.render_microblock();

    int real_axis_degree = 6;
    cs.stage_macroblock(FileBlock("I'll jump up to degree " + to_string(real_axis_degree) + "..."), 1);
    cps->set_degree(real_axis_degree);
    for(int i = 0; i <= real_axis_degree; i++) {
        cps->state_manager.transition(MICRO, "coefficient" + to_string(i) + "_r", "<realline> 1." + to_string(i) + " * " + to_string(i) + " + 2 / sin 3 *");
        cps->state_manager.transition(MICRO, "coefficient" + to_string(i) + "_i", "0");
    }
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("With the coefficients on the real line, the solutions are always vertically symmetrical."), 4);
    cps->transition_coefficient_opacities(MICRO, 1);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(3), 1);
    cps->state_manager.transition(MICRO, {
        {"coefficient0_r", "1"},
        {"coefficient0_i", "0"},
        {"coefficient1_r", "0"},
        {"coefficient1_i", "0"},
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
    });
    for(int i = 3; i <= real_axis_degree; i++) {
        cps->state_manager.transition(MICRO, "coefficient" + to_string(i) + "_r", "0.0001");
        cps->state_manager.transition(MICRO, "coefficient" + to_string(i) + "_i", "0");
    }
    cps->state_manager.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("So any definition for i must simultaneously include -i!"), 2);
    cps->construction.clear();
    cps->state_manager.set("construction_opacity", "1");
    cps->construction.add(GeometricPoint(glm::vec2(0, 1), "i"));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, -1), "-i"));
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "1"},
        {"dot_radius", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("They're algebraically indistinguishable, so which gets to be the square root of -1 is a matter of arbitrary choice."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("If we _do_ define the square root function with such an arbitrary choice, its graph is discontinuous like this..."), 2);
    cafs->state_manager.set({
        {"ticks_opacity", "0"},
        {"exp_coef", "0"},
        {"cos_coef", "0"},
        {"sin_coef", "0"},
        {"sqrt_coef", "1"},
        {"sqrt_branch_cut", to_string(M_PI)},
    });

    cs.add_scene_fade_in(MICRO, cafs, "cafs", .5, .5, 1, true);
    cs.fade_all_subscenes_except(MICRO, "cafs", 0);
    cs.render_microblock();
    cs.remove_all_subscenes_except("cafs");
    cafs->construction.add(GeometricLine(glm::vec2(-5, 0), glm::vec2(0, 0)));
    cafs->state_manager.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();

    cps->state_manager.set("positive_quadratic_formula_opacity", "0");
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We can change our arbitrary choice, and that yucky discontinuity starts to move around."), 1);
    cafs->state_manager.transition(MICRO, "sqrt_branch_cut", "{t} 5 * sin 1 * {t} cos 4 * +");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("In other words, there's not just one square root function, but a whole class of them depending on that choice."), 1);
    cafs->state_manager.transition(MICRO, "sqrt_branch_cut", to_string(M_PI));
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But discontinuities suck. With a bit of a change in perspective, we might be able to eliminate it!"), 2);
    shared_ptr<ManifoldScene> ms = make_shared<ManifoldScene>();
    ms->state_manager.set({
        {"d", "1"},
        {"q1", ".92387953251"},
        {"qi", ".38268343236"},
        {"qj", "0"},
        {"qk", "0"},
    });

    // u is radius, v is angle
    ms->add_manifold("0",
        "(u) (v) cos *", "0", "(u) (v) sin *",
        "(v) 2 / cos (u) .5 ^ * -1 * 10 *", "(v) 2 / sin (u) .5 ^ * -1 * 10 *",
        "0", "1.5", "3000",
        "-3.14159", "3.14159", "9500"
    );
    cs.add_scene_fade_in(MICRO, ms, "ms");
    cs.render_microblock();
    cs.remove_all_subscenes_except("ms");
    ms->state_manager.transition(MICRO, {
        {"d", "5"},
        {"q1", "1"},
        {"qi", "{t} .1 * sin .05 * .08 +"},
        {"qj", "{t} .1 * cos .15 *"},
        {"qk", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We can imagine visualizing this function as jutting out of the complex plane to highlight the discontinuity."), 1);
    ms->state_manager.transition(MICRO, {
        {"manifold0_y", "(v) 2 / sin (u) .5 ^ * -0.5 *"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This is what it looks like if I change that arbitrary choice like before."), 1);
    ms->state_manager.transition(MICRO, {
        {"manifold0_v_min", "6 -3.14159 +"},
        {"manifold0_v_max", "6 3.14159 +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(5), 1);
    ms->state_manager.transition(MICRO, {
        {"qj", "{t} .1 * cos .1 * .05 +"},
        {"manifold0_v_min", "-30 -3.14159 +"},
        {"manifold0_v_max", "-30 3.14159 +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Notice how we can extend this surface continuously until it meets itself."), 1);
    ms->state_manager.transition(MICRO, {
        {"manifold0_v_min", "-30 -6.28318 +"},
        {"manifold0_v_max", "-30 6.28318 +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This is called a Riemann surface."), 1);
    ms->state_manager.transition(MICRO, {
        {"manifold0_v_min", "-6.28318"},
        {"manifold0_v_max", "6.28318"},
        {"qj", "{t} .1 * cos .1 * .5 +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    ms->state_manager.transition(MICRO, {
        {"qj", "{t} .1 * cos .1 * .05 +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Now that our surface is continuous, we can make our square root continuous too."), 1);
    ms->add_manifold("1",
        "0", "0", "0",
        "0.00001", "0.00001",
        "-1.5", "1.5", "1500",
        "-1.5", "1.5", "1500"
    );
    ms->state_manager.transition(MICRO, {
        {"manifold1_x", "(u)"},
        {"manifold1_z", "(v)"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Picking some point on the complex plane, we can evaluate the square root function by seeing where it intersects with the two sheets,"), 3);
    string point_x_start = "<zradius> <ztheta> cos * (u) cos (v) sin * .02 * +";
    string point_y_start = "(v) cos .02 *";
    string point_z_start = "<zradius> <ztheta> sin * (u) sin (v) sin * .02 * +";
    ms->state_manager.set({
        {"zradius", "1.2"},
        {"ztheta", "3.8"},
    });
    ms->state_manager.transition(MICRO, {
        {"manifold1_x", point_x_start},
        {"manifold1_y", point_y_start},
        {"manifold1_z", point_z_start},
        {"manifold1_u_min", "0"},
        {"manifold1_u_max", "3.14159"},
        {"manifold1_v_min", "-3.14159"},
        {"manifold1_v_max", "3.14159"},
        {"manifold1_u_steps", "200"},
        {"manifold1_v_steps", "800"},
    });
    cs.render_microblock();
    ms->state_manager.transition(MICRO, {
        {"manifold1_x", point_x_start},
        {"manifold1_y", point_y_start + " 40 *"},
        {"manifold1_z", point_z_start},
        {"axes_length", ".5"},
    });
    cs.render_microblock();
    // Add another identical manifold
    ms->add_manifold("2",
        ms->state_manager.get_equation("manifold1_x"), ms->state_manager.get_equation("manifold1_y"), ms->state_manager.get_equation("manifold1_z"),
        "0.00001", "0.00001",
        "0", "3.14159", "200",
        "-3.14159", "3.14159", "800"
    );
    ms->state_manager.transition(MICRO, {
        {"manifold1_y", point_y_start + " <ztheta> 2 / sin <zradius> .5 ^ * -0.5 * +"},
        {"manifold2_y", point_y_start + " <ztheta> 2 / sin <zradius> .5 ^ * -0.5 * -"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and we can even track the square root continuously as we move around."), 1);
    ms->state_manager.transition(MICRO, {
        {"ztheta", "-3"},
        {"zradius", "1.1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This should hopefully make the predicament clear-"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We're either stuck with a discontinuous square root function which returns a single value,"), 2);
    ms->state_manager.transition(MICRO, {
        {"manifold0_v_min", "-3.14159"},
        {"manifold0_v_max", "3.14159"},
        {"manifold0_y", "0"},
        {"manifold1_y", point_y_start},
        {"manifold2_y", point_y_start},
    });
    cs.render_microblock();
    ms->state_manager.transition(MICRO, {
        {"q1", ".92387953251"},
        {"qi", ".38268343236"},
        {"qj", "0"},
        {"qk", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("or a continuous square root function which returns two that are always opposite each other!"), 1);
    ms->state_manager.transition(MICRO, {
        {"q1", "{t} .1 * cos"},
        {"qi", ".05"},
        {"qj", "{t} .05 * sin"},
        {"qk", "0"},
        {"manifold0_y", "(v) 2 / sin (u) .5 ^ * -0.5 *"},
        {"manifold0_v_min", "-6.28318"},
        {"manifold0_v_max", "6.28318"},
        {"manifold1_y", point_y_start + " <ztheta> 2 / sin <zradius> .5 ^ * -0.5 * +"},
        {"manifold2_y", point_y_start + " <ztheta> 2 / sin <zradius> .5 ^ * -0.5 * -"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 2);
    cs.render_microblock();
    cs.fade_subscene(MICRO, "ms", 0);
    cs.add_scene_fade_in(MICRO, cps, "cps", .5, .5, 1, true);
    cps->construction.clear();
    cps->roots_to_coefficients();
    cps->state_manager.set({
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "0"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "{t} 4 * cos .3 *"},
        {"coefficient0_i", "{t} 4 * sin .3 *"},
        {"positive_quadratic_formula_opacity", "1"},
    });
    cs.render_microblock();
    cs.remove_subscene("ms");
    ms->remove_manifold("1");
    ms->remove_manifold("2");

    cs.stage_macroblock(FileBlock("And this makes sense! As we saw before, either our formula cannot transform continuously,"), 4);
    cps->set_degree(2);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("or otherwise we have to yield both solutions without distinguishing between them to achieve continuity."), 4);
    cps->state_manager.transition(MICRO, {
        {"positive_quadratic_formula_opacity", "0"},
    });
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We've shown that other functions alone can't solve quadratics, but why can square roots?"), 1);
    cps->coefficients_to_roots();
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", ".5"},
        {"dot_radius", ".5"},
        {"root0_r", "0"},
        {"root0_i", "-.8"},
        {"root1_r", "-1"},
        {"root1_i", "-.4"},
        {"center_x", "2"},
    });
    ms->state_manager.set("w", ".5");
    cs.add_scene_fade_in(MICRO, ms, "ms", .75, .5);
    ms->state_manager.set({
        {"ztheta", "3.14159"},
        {"ab_dilation", ".1"},
        {"dot_radius", ".1"},
        {"manifold0_r", "<zradius> <ztheta> cos * (u) (v) cos * -"},
        {"manifold0_i", "<zradius> <ztheta> sin * (u) (v) sin * -"},
        {"q1", "{t} .1 * cos cos"},
        {"qi", ".05"},
        {"qj", "{t} .05 * sin sin 2 /"},
        {"qk", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Remember: I can grab coefficients and make a loop which switches the solutions."), 3);
    cs.fade_subscene(MICRO, "ms", 0.2);
    cps->transition_coefficient_opacities(MICRO, 1);
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "coefficient0_ring", "1");
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(6), 4);
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->state_manager.transition(MICRO, "coefficient0_ring", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Now watch, as I make a loop in the input value of the square root function..."), 1);
    cs.fade_subscene(MICRO, "ms", 1);
    ms->add_manifold("1",
        point_x_start, point_y_start, point_z_start,
        "0.00001", "0.00001",
        "0", "3.14159", "200",
        "-3.14159", "3.14159", "200"
    );
    ms->state_manager.transition(MICRO, {
        {"ztheta", "-3.14159"},
        {"zradius", ".75"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("the solutions exchange places!"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let me artificially stretch out that surface so it's clear what's going on..."), 1);
    ms->state_manager.transition(MICRO, {
        {"q1", ".92387953251"},
        {"qi", ".38268343236"},
        {"qj", "0"},
        {"qk", "0"},
        {"manifold0_x", "(u) .5 * 1 + (v) 2 / cos *"},
        {"manifold0_y", "0"},
        {"manifold0_z", "(u) .5 * 1 + (v) 2 / sin *"},
        {"axes_length", ".2"},
        {"ztheta", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    ms->state_manager.transition(MICRO, {
        {"ztheta", to_string(3.14159 * 2)},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The behavior is just like our polynomial solutions!"), 1);
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    // TODO comparison with a non-square-rooty function.

    cs.stage_macroblock(FileBlock("A loop of the input yields a swap of the outputs."), 1);
    ms->state_manager.transition(MICRO, {
        {"ztheta", to_string(3.14159 * 4)},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "1"},
        {"dot_radius", "1"},
        {"center_x", "0"},
    });
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.fade_all_subscenes_except(MICRO, "cps", 0);
    cs.render_microblock();
    cs.remove_all_subscenes_except("cps");

    cs.stage_macroblock(FileBlock("So a square root is just the right ingredient to make a quadratic formula."), 2);
    cps->transition_root_rings(MICRO, 0);
    cs.add_scene_fade_in(MACRO, quadratic_formula, "quadratic_formula", .5, .5);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_subscene(MACRO, "quadratic_formula", 0);
    // Stop coefficient from spinning
    cps->state_manager.transition(MICRO, {
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "0"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "1"},
        {"coefficient0_i", "0"},
    });
    cs.render_microblock();
    cs.remove_subscene("quadratic_formula");

    shared_ptr<LatexScene> recap = make_shared<LatexScene>("\\begin{tabular}{ccc} \\text{\\huge{Degree}} & \\qquad \\qquad \\text{\\huge{Form}} \\qquad \\qquad & \\text{\\huge{Solutions}}", 1, 1, 1);
    cs.add_scene_fade_in(MACRO, recap, "recap");
    cs.stage_macroblock(FileBlock("Just to recap..."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Linear polynomials are solved trivially."), 1);
    recap->begin_latex_transition(MICRO, "\\begin{tabular}{ccc} \\text{\\huge{Degree}} & \\qquad \\qquad \\text{\\huge{Form}} \\qquad \\qquad & \\text{\\huge{Solutions}} \\\\\\\\ \\\\\\\\ 1 & ax + b & -\\frac{b}{a}");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Quadratics need a square root."), 1);
    recap->begin_latex_transition(MICRO, "\\begin{tabular}{ccc} \\text{\\huge{Degree}} & \\qquad \\qquad \\text{\\huge{Form}} \\qquad \\qquad & \\text{\\huge{Solutions}} \\\\\\\\ \\\\\\\\ 1 & ax + b & -\\frac{b}{a} \\\\\\\\ \\\\\\\\ 2 & ax^2 + bx + c & \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}");
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("What do you think happens next?"), SilenceBlock(1.5)), 2);
    recap->begin_latex_transition(MICRO, "\\begin{tabular}{ccc} \\text{\\huge{Degree}} & \\qquad \\qquad \\text{\\huge{Form}} \\qquad \\qquad & \\text{\\huge{Solutions}} \\\\\\\\ \\\\\\\\ 1 & ax + b & -\\frac{b}{a} \\\\\\\\ \\\\\\\\ 2 & ax^2 + bx + c & \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\\\\\\\ \\\\\\\\ 3 & ax^3 + bx^2 + cx + d & ?");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(2), FileBlock("If there's a cubic formula, it also can't be written with simple operators, avoiding square roots.")), 2);
    cps->set_degree(3);
    cps->roots_to_coefficients();
    cps->state_manager.transition(MACRO, {
        {"coefficient3_r", "1"},
        {"coefficient3_i", "0"},
        {"coefficient2_r", "-.7"},
        {"coefficient2_i", "1"},
        {"coefficient1_r", ".8"},
        {"coefficient1_i", ".2"},
        {"coefficient0_r", "1.2"},
        {"coefficient0_i", "1"},
    });
    cs.fade_subscene(MICRO, "recap", 0);
    cs.render_microblock();
    cs.remove_subscene("recap");
    cps->transition_coefficient_opacities(MICRO, 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The same argument still applies. If I swap two solutions, the coefficients go back to where they started."), 3);
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"root0_ring", "1"},
        {"root1_ring", "1"},
    });
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.5), 1);
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("It turns out, though, that you need more powerful tools than just square roots to solve cubics."), 1);
    cs.add_scene_fade_in(MICRO, ms, "ms", .75, .5);
    cps->state_manager.transition(MACRO, {
        {"ab_dilation", ".5"},
        {"dot_radius", ".5"},
        {"center_x", "2"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("First of all, a square root yields 2 values, but we need a function that gives 3."), 2);
    ms->state_manager.transition(MICRO, {
        {"q1", "1"},
        {"qi", ".05"},
        {"qj", "{t} .05 * sin"},
        {"qk", "0"},
        {"manifold0_x", "(u) (v) cos *"},
        {"manifold0_y", "(v) 2 / sin (u) .5 ^ * -0.5 *"},
        {"manifold0_z", "(u) (v) sin *"},
    });
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("In that case, how about a cube root?"), 2);
    cs.render_microblock();
    ms->state_manager.transition(MICRO, {
        {"manifold0_x", "(u) (v) cos *"},
        {"manifold0_y", "(u) .333333 ^ (v) 3 / sin *"},
        {"manifold0_z", "(u) (v) sin *"},
        {"manifold0_v_min", to_string(-3 * M_PI)},
        {"manifold0_v_max", to_string(3 * M_PI)},
        {"manifold0_v_steps", "13500"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The associated Riemann surface has 3 sheets, and the cube root function is continuous on it."), 2);
    ms->state_manager.transition(MICRO, "ztheta", to_string(3.14159 * 2));
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Stretching this manifold out so it's easy to see,"), 1);
    ms->state_manager.transition(MICRO, {
        {"q1", ".92387953251"},
        {"qi", ".38268343236"},
        {"qj", "0"},
        {"qk", "0"},
        {"manifold0_x", "(u) .5 * 1 + (v) 3 / cos *"},
        {"manifold0_y", "0"},
        {"manifold0_z", "(u) .5 * 1 + (v) 3 / sin *"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("we can cycle the 3 roots like this..."), 1);
    ms->state_manager.transition(MICRO, {
        {"ztheta", to_string(3.14159 * 4)},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and we can move some coefficients to yield the same behavior on the solutions..."), 2);
    cps->state_manager.transition(MICRO, {
        {"coefficient0_ring", "1"},
        {"coefficient1_ring", "1"},
    });
    cs.render_microblock();
    cps->coefficients_to_roots();
    cps->state_manager.transition(MICRO, {
        {"root0_r", cps->state_manager.get_equation("root1_r")},
        {"root0_i", cps->state_manager.get_equation("root1_i")},
        {"root1_r", cps->state_manager.get_equation("root2_r")},
        {"root1_i", cps->state_manager.get_equation("root2_i")},
        {"root2_r", cps->state_manager.get_equation("root0_r")},
        {"root2_i", cps->state_manager.get_equation("root0_i")},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("That's a good sign, right?"), 1);
    cps->state_manager.transition(MICRO, {
        {"coefficient0_ring", "0"},
        {"coefficient1_ring", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Well, not so fast."), 1);
    cs.slide_subscene(MICRO, "ms", .5, 0);
    StateSet reset = {
        {"root0_r", "-2"},
        {"root0_i", "-2"},
        {"root1_r", "0"},
        {"root1_i", "-2"},
        {"root2_r", "2"},
        {"root2_i", "-2"},
    };
    cps->state_manager.transition(MICRO, reset);
    cps->state_manager.transition(MICRO, "center_x", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("I'm gonna pick up those same coefficients again,"), 1);
    cps->state_manager.transition(MICRO, {
        {"coefficient0_ring", "1"},
        {"coefficient1_ring", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and this time make a very special sequence of 4 loops."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("First, we move them in a way that swaps the left two solutions in a clockwise rotation,"), 1);
    cps->stage_swap(MICRO, "0", "1", false, true);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("then we swap the right two clockwise,"), 1);
    cps->stage_swap(MICRO, "2", "0", false, true);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("now the left two counterclockwise,"), 1);
    cps->stage_swap(MICRO, "2", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and finally the right two counterclockwise."), 1);
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cps->state_manager.set(reset);
    cs.stage_macroblock(FileBlock("We did two things,"), 2);
    cps->stage_swap(MICRO, "0", "1", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "0", false, true);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("then undid them both."), 2);
    cps->stage_swap(MICRO, "2", "1", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cps->state_manager.set(reset);
    cps->state_manager.set({
        {"point_1_x", "<root0_r>"},
        {"point_1_y", "<root0_i>"},
        {"point_2_x", "<root1_r>"},
        {"point_2_y", "<root1_i>"},
        {"point_3_x", "<root2_r>"},
        {"point_3_y", "<root2_i>"},
    });
    cs.stage_macroblock(FileBlock("But watch what happens if I label the solutions this time!"), 3);
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "1", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "2", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "3", .7, true));
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(4), 4);
    cps->stage_swap(MICRO, "0", "1", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "0", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "1", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The solutions changed places, even though in some sense we undid each of our loops!"), 5);
    for(int i = 0; i < 5; i++) {
        cps->state_manager.transition(MICRO, "construction_opacity", to_string(i % 2));
        cs.render_microblock();
    }

    cps->state_manager.set(reset);
    commutator = make_shared<LatexScene>("x \\phantom{y x^-1 y^-1}", 1, .5, .35);

    cs.stage_macroblock(FileBlock("This special way of cycling objects is called a commutator."), 8);
    cps->stage_swap(MICRO, "0", "1", false, true);
    cs.add_scene_fade_in(MICRO, commutator, "commutator", .5, .2);
    cs.render_microblock();
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "0", false, true);
    commutator->begin_latex_transition(MICRO, "x y \\phantom{x^{-1} y^{-1}}");
    cs.render_microblock();
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "1", false);
    commutator->begin_latex_transition(MICRO, "x y x^{-1} \\phantom{y^{-1}}");
    cs.render_microblock();
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    commutator->begin_latex_transition(MICRO, "x y x^{-1} y^{-1}");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But watch what happens if we make commutator loops on the input of the cube root function..."), 1);
    cs.fade_subscene(MICRO, "commutator", 0);
    cs.slide_subscene(MICRO, "ms", -.5, 0);
    cps->state_manager.transition(MICRO, "center_x", "2");
    cs.render_microblock();
    cs.remove_subscene("commutator");

    cs.stage_macroblock(FileBlock("Loop, loop, back, back..."), 4);
    ms->state_manager.set({
        {"ztheta", "<loop1> 6.283 * <loop2> 6.283 * sin .1 * +"},
        {"zradius", "<loop2> 6.283 * sin .25 * .75 +"},
        {"loop1", "0"},
        {"loop2", "0"},
    });
    ms->state_manager.transition(MICRO, "loop1", "1");
    cs.render_microblock();
    ms->state_manager.transition(MICRO, "loop2", "1");
    cs.render_microblock();
    ms->state_manager.transition(MICRO, "loop1", "0");
    cs.render_microblock();
    ms->state_manager.transition(MICRO, "loop2", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("the solutions returned to where they started..."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's try a different commutator on the input..."), 4);
    ms->state_manager.set({
        {"ztheta", "<loop1> 6.283 * sin .1 * <loop2> 6.283 +"},
        {"zradius", "<loop2> 6.283 * sin <loop1> 6.283 * cos + .1 * .75 +"},
    });
    ms->state_manager.transition(MICRO, "loop1", "1");
    cs.render_microblock();
    ms->state_manager.transition(MICRO, "loop2", "1");
    cs.render_microblock();
    ms->state_manager.transition(MICRO, "loop1", "0");
    cs.render_microblock();
    ms->state_manager.transition(MICRO, "loop2", "0");
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("No matter what commutator the input follows, the outputs always finish where they started."), 4);
    ms->state_manager.set({
        {"ztheta", "<loop1> 6.283 * <loop2> 6.283 * +"},
        {"zradius", "<loop2> 6.283 * sin .25 * .75 +"},
    });
    ms->state_manager.transition(MICRO, "loop1", "1");
    cs.render_microblock();
    ms->state_manager.transition(MICRO, "loop2", "1");
    cs.render_microblock();
    ms->state_manager.transition(MICRO, "loop1", "0");
    cs.render_microblock();
    ms->state_manager.transition(MICRO, "loop2", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("In general, the three points do a cycle whenever our loop circles the origin,"), 1);
    ms->state_manager.set({
        {"ztheta", "<loop1> 6.283 * <loop2> 6.283 * sin .1 * +"},
        {"zradius", "<loop2> 6.283 * sin .25 * .75 +"},
    });
    ms->state_manager.transition(MICRO, "loop1", "1");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and don't do anything at all when the loop doesn't wrap the origin."), 1);
    ms->state_manager.transition(MICRO, "loop2", "1");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but the commutator unwinds any accumulated phase around the origin in the second half."), 0);
    ms->state_manager.transition(MICRO, "loop1", "0");
    cs.render_microblock();
    ms->state_manager.transition(MICRO, "loop2", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("If there is some function which maps the coefficients of the cubic to the input of this cube root,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("the former can express behaviors that the cube root function is unable to emulate."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("What kind of operator could be more expressive?"), 1);
    cs.render_microblock();
}
