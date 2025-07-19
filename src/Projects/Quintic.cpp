#include "../Scenes/Math/ComplexPlotScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"

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
    shared_ptr<ComplexPlotScene> cps = make_shared<ComplexPlotScene>(5);
    cps->stage_macroblock(FileBlock("This is the relationship between a polynomial's coefficients and its roots."), 2);
    cps->state_manager.set("zoom", ".1");

    for(int i = 0; i < 2; i++) {
        cps->state_manager_roots_to_coefficients();
        cps->state_manager.transition(MICRO, {
            {"coefficient"+to_string(i)+"_r", "1.5"},
            {"coefficient"+to_string(i)+"_i", ".5"},
        });
        cps->render_microblock();
        if(i == 1) return;

        cps->state_manager_coefficients_to_roots();
        cps->state_manager.transition(MICRO, {
            {"root"+to_string(i)+"_r", "-2"},
            {"root"+to_string(i)+"_i", ".2"},
        });
        cps->render_microblock();
    }

    CompositeScene cs;
    cs.add_scene(cps, "cps");

    cs.stage_macroblock(FileBlock("You might remember from Algebra class that a polynomial can be written in two forms-"), 1);
    cs.render_microblock();

    shared_ptr<LatexScene> ls = make_shared<LatexScene>("x^3+ax^2+bx^1+cx^0", 1);
    cs.add_scene_fade_in(MICRO, ls, "ls");
    cs.stage_macroblock(FileBlock("There's a standard form, where each exponent of x has an associated coefficient,"), 4);
    cs.render_microblock();
    ls->begin_latex_transition(MICRO, "x^3+"+latex_color(0xffff0000, "a")+"x^2+"+latex_color(0xff00ff00, "b")+"x^1+"+latex_color(0xff0000ff, "c")+"x^0");
    cs.render_microblock();
    ls->begin_latex_transition(MICRO, "x^3+ax^2+bx^1+cx^0");
    cs.render_microblock();
    cs.state_manager.transition(MICRO, {{"ls.x", ".25"}, {"ls.y", ".1"}});
    ls->state_manager.transition(MICRO, {{"w", ".4"}, {"h", ".2"}});
    cs.render_microblock();

    shared_ptr<LatexScene> ls2 = make_shared<LatexScene>("(x-r_1)(x-r_2)(x-r_3)", 1);
    cs.add_scene_fade_in(MICRO, ls2, "ls2");
    cs.stage_macroblock(FileBlock("and also a factored form, with one term for each root."), 4);
    cs.render_microblock();
    ls2->begin_latex_transition(MICRO, "(x-" + latex_color(0xffff0000, "r_1")+")(x-"+latex_color(0xff00ff00, "r_2")+")(x-"+latex_color(0xff0000ff, "r_3")+")");
    cs.render_microblock();
    ls2->begin_latex_transition(MICRO, "(x-r_1)(x-r_2)(x-r_3)");
    cs.render_microblock();
    cs.state_manager.transition(MICRO, {{"ls2.x", ".75"}, {"ls2.y", ".1"}});
    ls2->state_manager.transition(MICRO, {{"w", ".4"}, {"h", ".2"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("These points are the coefficients of each power of the input,"), 4);
    cps->state_manager.transition(MICRO, {{"coefficients_opacity","0"}});
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {{"coefficients_opacity","1"}});
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {{"coefficients_opacity","0"}});
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {{"coefficients_opacity","1"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and these points are the roots of the polynomial, plotted in the complex plane,"), 4);
    cps->state_manager.transition(MICRO, {{"roots_opacity","1"}});
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {{"roots_opacity","0"}});
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {{"roots_opacity","1"}});
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {{"roots_opacity","0"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Notice how moving a single root has a hard-to-predict effect on the coefficients,"), 1);
    cs.stage_macroblock(FileBlock("and moving a single coefficient has a hard-to-predict effect on the roots."), 1);

    cs.stage_macroblock(FileBlock("It's really a shame that, most likely, that algebra teacher never showed you this plot of their relationship..."), 1);
    cs.render_microblock();

    //cs.stage_macroblock(FileBlock("because without much work, it lets us peer right into the soul of the polynomial, and show tons of important results in algebra, culminating in the Abel-Ruffini theorem, stating that there is no Quintic formula."), 1);
    cs.stage_macroblock(FileBlock("because that relationship _is the core, the essence of algebra_."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This space we're looking at is the complex plane, home to numbers like i and 3-2i."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("For each pixel on the screen, I passed that complex number as an input to the polynomial,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and colored it according to the output."), 1);
    cs.render_microblock();

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
    cs.render_microblock();
    return;
    cs.stage_macroblock(FileBlock("Doing this for every point, we can graph our complex-valued function."), 1);
    cs.stage_macroblock(FileBlock("Before we get analytical, I hear you asking..."), 1);
    cs.stage_macroblock(FileBlock("What's all this complex number business? Why leave the familiar land of the reals?"), 1);
    // TODO add some script introducing a real valued plot in a 3d axis and show that it doesn't necessarily have zeros.
    cs.stage_macroblock(FileBlock("But in turn, I would ask, why do you need decimals or negatives either?"), 1);
    cs.stage_macroblock(FileBlock("Imagine there's nothing but natural numbers- 1, 2, 3, and so on, along with the ideas of plus and times."), 1);
    cs.stage_macroblock(FileBlock("In such a world, we quickly run into problems..."), 1);
    cs.stage_macroblock(FileBlock("We can write equations that have no solution, such as 1 + x = 1."), 1);
    cs.stage_macroblock(FileBlock("Without zero, our number system is somehow incomplete."), 1);
    cs.stage_macroblock(FileBlock("There's nothing we can write here for x that would solve make this equation true."), 1);
    cs.stage_macroblock(FileBlock("So, to be able to solve this equation, we need 0 in our set..."), 1);
    cs.stage_macroblock(FileBlock("What about 2+x=1?"), 1);
    cs.stage_macroblock(FileBlock("Well, we then need -1 too,"), 1);
    cs.stage_macroblock(FileBlock("along with -2,"), 1);
    cs.stage_macroblock(FileBlock("and so on..."), 1);
    cs.stage_macroblock(FileBlock("and what if we write an equation like this one?"), 1);
    cs.stage_macroblock(FileBlock("It looks like we're gonna need fractions too..."), 1);
    cs.stage_macroblock(FileBlock("Surely, if we simply add all decimal numbers, our number system will finally be complete?"), 1);
    cs.stage_macroblock(FileBlock("We get a nice continuum of numbers."), 1);
    cs.stage_macroblock(FileBlock("Well, then, what about this equation?"), 1);
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
