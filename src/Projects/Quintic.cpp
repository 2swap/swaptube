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
    shared_ptr<ComplexPlotScene> cps = make_shared<ComplexPlotScene>(3);
    cps->stage_macroblock(FileBlock("This is the relationship between a polynomial's coefficients and its roots."), 4);
    cps->state_manager.set("dot_radius", ".1");
    cps->state_manager.set("ticks_opacity", "0");

    for(int i = 0; i < cps->degree; i++) {
        cps->state_manager.transition(MICRO, {
            {"root"+to_string(i)+"_r", "<t> 1.2 * 6.28 .3333 " + to_string(i) + " * * + sin"},
            {"root"+to_string(i)+"_i", "<t> .8 * 6.28 .3333 " + to_string(i) + " * * + cos"},
        });
    }
    cps->render_microblock();
    cps->render_microblock();
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
        {"root1_opacity", "0"},
        {"root2_opacity", "0"},
        {"roots_opacity", "1"},
    });
    cps->state_manager.transition(MICRO, {
        {"root0_opacity", "1"},
    });
    cps->render_microblock();
    cps->state_manager.set({
        {"root0_r", "1 <spin_coefficient_r> +"},
        {"root0_i", ".2 <spin_coefficient_i> +"},
        {"spin_coefficient_r", "<t> 3 * sin <spin_multiplier> *"},
        {"spin_coefficient_i", "<t> 3 * cos <spin_multiplier> *"},
        {"spin_multiplier", "0"},
    });
    cps->state_manager.transition(MICRO, {
        {"spin_multiplier", ".5"},
    });
    cps->render_microblock();
    cps->render_microblock();
    cps->state_manager.transition(MICRO, {
        {"spin_multiplier", "0"},
        {"roots_opacity", "0"},
    });
    cps->render_microblock();

    cps->state_manager.set({
        {"root0_opacity", "1"},
        {"root1_opacity", "1"},
        {"root2_opacity", "1"},
    });
    cps->state_manager_roots_to_coefficients();
    cps->stage_macroblock(FileBlock("and moving a single coefficient has a hard-to-predict effect on the roots."), 4);
    cps->state_manager.transition(MICRO, {
        {"coefficient1_opacity", ".1"},
        {"coefficient2_opacity", ".1"},
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
        {"coefficient1_opacity", "1"},
        {"coefficient2_opacity", "1"},
        {"spin_multiplier", "0"},
    });
    cps->render_microblock();

    // TODO zoom in and out for pretty colors
    cps->stage_macroblock(FileBlock("It's a shame you've never seen this plot,"), 1);
    cps->render_microblock();

    cps->stage_macroblock(FileBlock("because their relationship _is the core, the essence of algebra_."), 1);
    cps->render_microblock();

    CompositeScene cs;
    cs.add_scene(cps, "cps");

    shared_ptr<LatexScene> ls = make_shared<LatexScene>("x^3+ax^2+bx^1+cx^0", 1, 1, .5);
    cs.add_scene_fade_in(MICRO, ls, "ls", .5, .15);
    cs.stage_macroblock(FileBlock("Polynomials have a standard form, where each term has an associated coefficient."), 4);
    cs.render_microblock();
    ls->begin_latex_transition(MICRO, "x^3+"+latex_color(0xffff0000, "a")+"x^2+"+latex_color(0xff00ff00, "b")+"x^1+"+latex_color(0xff0000ff, "c")+"x^0");
    cs.render_microblock();
    ls->begin_latex_transition(MICRO, "x^3+ax^2+bx^1+cx^0");
    cs.render_microblock();
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("They're drawn on the graph here."), 4);
    for(int i = 0; i < 2; i++) {
        cps->state_manager.transition(MICRO, {{"coefficients_opacity","0"}});
        ls->begin_latex_transition(MICRO, "x^3+"+latex_color(0xffff0000, "a")+"x^2+"+latex_color(0xff00ff00, "b")+"x^1+"+latex_color(0xff0000ff, "c")+"x^0");
        cs.render_microblock();

        cps->state_manager.transition(MICRO, {{"coefficients_opacity","1"}});
        ls->begin_latex_transition(MICRO, "x^3+ax^2+bx^1+cx^0");
        cs.render_microblock();
    }
    cps->state_manager.transition(MICRO, "coefficients_opacity", "0");
    cs.fade_subscene(MICRO, "ls", 0);

    shared_ptr<LatexScene> ls2 = make_shared<LatexScene>("(x-r_1)(x-r_2)(x-r_3)", 1, 1, .5);
    cs.add_scene_fade_in(MICRO, ls2, "ls2", .5, .15);
    cs.stage_macroblock(FileBlock("There's also a factored form, with one term for each root,"), 4);
    cs.render_microblock();
    ls2->begin_latex_transition(MICRO, "(x-" + latex_color(0xffff0000, "r_1")+")(x-"+latex_color(0xff00ff00, "r_2")+")(x-"+latex_color(0xff0000ff, "r_3")+")");
    cs.render_microblock();
    ls2->begin_latex_transition(MICRO, "(x-r_1)(x-r_2)(x-r_3)");
    cs.render_microblock();
    cs.render_microblock();
    cs.stage_macroblock(FileBlock("which are clearly visible in the polynomial's graph."), 4);
    for(int i = 0; i < 2; i++) {
        cps->state_manager.transition(MICRO, {{"roots_opacity",".5"}});
        ls2->begin_latex_transition(MICRO, "(x-" + latex_color(0xffff0000, "r_1")+")(x-"+latex_color(0xff00ff00, "r_2")+")(x-"+latex_color(0xff0000ff, "r_3")+")");
        cs.render_microblock();

        cps->state_manager.transition(MICRO, {{"roots_opacity","0"}});
        ls2->begin_latex_transition(MICRO, "(x-r_1)(x-r_2)(x-r_3)");
        cs.render_microblock();
    }

    // add points to show the numbers
    cs.stage_macroblock(FileBlock("The space we're looking at is the complex plane, home to numbers like i and 3-2i."), 3);
    cps->state_manager.transition(MICRO, {{"ticks_opacity", "1"}});
    cs.render_microblock();
    cps->construction.add_point(GeometricPoint(glm::vec2(0, 1)));
    cs.render_microblock();
    cps->construction.add_point(GeometricPoint(glm::vec2(3, -2)));
    cs.render_microblock();
    cps->construction.clear();

    // TODO animate that point going into the polynomial, reduce the polynomial in latex
    cs.stage_macroblock(FileBlock("For each pixel on the screen, I passed that complex number as an input to the polynomial,"), 1);
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

    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "20"},
    });
    cs.stage_macroblock(FileBlock("and the color shows the angle of the output."), 3);
    cs.render_microblock();
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", ".3"},
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

    shared_ptr<LatexScene> f_complex = make_shared<LatexScene>(latex_text("Complex Function"), 1, .5, .5);
    shared_ptr<LatexScene> f_real    = make_shared<LatexScene>(latex_text("Real Function"), 1, .5, .5);
    tds.add_surface(Surface(glm::vec3(0, .25,-.05), glm::vec3(.25, 0, 0), glm::vec3(0, .25, 0), "f_complex"), f_complex);
    tds.add_surface(Surface(glm::vec3(0, .05,-.25), glm::vec3(.25, 0, 0), glm::vec3(0, 0, .25), "f_real"), f_real);
    tds.stage_macroblock(FileBlock("You might ask, what's all this complex number business? Why leave the familiar land of the reals?"), 1);
    tds.render_microblock();

    tds.state_manager.transition(MICRO, {
        {"d", "1"},
        {"qi", "0"},
        {"qj", "0"},
    });
    cps->state_manager.transition(MICRO, "ticks_opacity", "0");
    tds.stage_macroblock(FileBlock("Surely, complex numbers were invented by big math to sell more calculators!"), 1);
    tds.fade_subscene(MICRO, "f_complex", 0);
    tds.fade_subscene(MICRO, "f_real", 0);
    tds.fade_subscene(MICRO, "rfs", 0);
    tds.render_microblock();
    tds.remove_all_subscenes();

    cs = CompositeScene();
    cs.add_scene(cps, "cps");
    cs.stage_macroblock(FileBlock("But in turn, I would ask, why do you need decimals or negatives either?"), 1);
    cs.render_microblock();

    shared_ptr<LatexScene> all_numbers = make_shared<LatexScene>("\\{1, 2, 3, ...\\}", .4, 1, .4);
    cs.add_scene_fade_in(MICRO, all_numbers, "all_numbers", .5, .2);
    cs.stage_macroblock(FileBlock("Imagine there's nothing but natural numbers- 1, 2, 3, and so on."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("In such a world, there's a big problem..."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We can write equations with no solution, such as 1 + x = 1."), 1);
    shared_ptr<LatexScene> impossible1 = make_shared<LatexScene>("1+x=1", 1, .5, .5);
    cs.add_scene_fade_in(MICRO, impossible1, "impossible1", .5, .7);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Without zero, it's as though our number system is incomplete."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("There's nothing we can write here for x that would make the thing on the left equal the thing on the right."), 1);
    impossible1->begin_latex_transition(MICRO, "1+!?=1");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("So, to be able to solve this, we _need_ 0..."), 2);
    cs.render_microblock();
    all_numbers->begin_latex_transition(MICRO, "\\{0, 1, 2, 3, ...\\}");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("What about 2+x=1?"), 1);
    cs.state_manager.transition(MICRO, "impossible1.y", "1.5");
    shared_ptr<LatexScene> impossible2 = make_shared<LatexScene>("2+x=1", 1, .5, .5);
    cs.add_scene_fade_in(MICRO, impossible2, "impossible2", .5, .7);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("x _wants_ to be -1!"), 1);
    impossible2->begin_latex_transition(MICRO, "2+-1=1");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We need to add negatives..."), 1);
    all_numbers->begin_latex_transition(MICRO, "\\{-1, 0, 1, 2, 3, ...\\}");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and so on..."), 1);
    all_numbers->begin_latex_transition(MICRO, "\\{..., -3, -2, -1, 0, 1, 2, 3, ...\\}");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Sadly, our number system is still incomplete."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Two times something gives one..."), 1);
    cs.state_manager.transition(MICRO, "impossible2.y", "1.5");
    shared_ptr<LatexScene> impossible3 = make_shared<LatexScene>("2*x=1", 1, .5, .5);
    cs.add_scene_fade_in(MICRO, impossible3, "impossible3", .5, .7);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("It looks like we need fractions too..."), 1);
    all_numbers->begin_latex_transition(MICRO, "\\{\\frac{a}{b} | a, b \\in \\mathbb{Z}, b \\neq 0\\}");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Heck, let's just add all decimals, surely our number system will finally be complete?"), 6);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We get a nice continuum of numbers."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Well, then, what about this equation?"), 1);
    cs.state_manager.transition(MICRO, "impossible3.y", "1.5");
    shared_ptr<LatexScene> impossible4 = make_shared<LatexScene>("x*x=-1", 1, .5, .5);
    cs.add_scene_fade_in(MICRO, impossible4, "impossible4", .5, .7);
    cs.render_microblock();

    return;
    cs.stage_macroblock(FileBlock("Uh oh, no real number squared gives us negative 1..."), 1);
    cs.stage_macroblock(FileBlock("so we need to add more numbers to make our set whole."), 1);
    cs.stage_macroblock(FileBlock("Let's call the solution to this equation 'i'"), 1);
    cs.stage_macroblock(FileBlock("All the multiples of 'i' then yield another line of numbers, and adding them to real numbers gives us a whole two dimensional plane of numbers."), 1);
    // TODO add some script introducing a real valued plot in a 3d axis and show that it doesn't necessarily have zeros.
    cs.stage_macroblock(FileBlock("Now, believe it or not, our number system is finally whole."), 1);
    cs.stage_macroblock(FileBlock("In other words, wherever I put the coefficients of the polynomial we are drawing,"), 1);
    cs.stage_macroblock(FileBlock("the roots- the little white holes where the output equals zero- you can never get rid of them."), 1);
    cs.stage_macroblock(FileBlock("There is no equation with the complex numbers, pluses, and timeses, which can't be solved with complex number solutions."), 1);
    cs.stage_macroblock(FileBlock("The fancy way to say this is that the complex field is algebraically closed"), 1);
    cs.stage_macroblock(FileBlock("And this is so important that it is called:"), 1);
    cs.stage_macroblock(FileBlock("THE FUNDAMENTAL THEOREM OF ALGEBRA"), 1);
    cs.stage_macroblock(FileBlock("Beyond just knowing that the equation has a root, our plot shows us that the number of roots stays fixed."), 1);
    cs.stage_macroblock(FileBlock("Well, unless you intentionally put them on top of each other... but that's cheating."), 1);
    cs.stage_macroblock(FileBlock("Now, notice- it's always the same number as the equation's highest exponent."), 1);
    cs.stage_macroblock(FileBlock("This"), 1);
}
