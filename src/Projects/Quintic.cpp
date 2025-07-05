#include "../Scenes/Math/ComplexPlotScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"

void render_video(){
    shared_ptr<ComplexPlotScene> cps = make_shared<ComplexPlotScene>(5);
    cps->stage_macroblock(FileBlock("This plot elucidates the relationship between a polynomial's roots and its coefficients."), 5);

    cps->state_manager.set({
        {"roots_or_coefficients_control", "1"},
    });
    cps->state_manager.transition(MICRO, {
        {"coefficient0_r", "4"},
    });
    cps->render_microblock();
    cps->state_manager_coefficients_to_roots();
    cps->state_manager.set({
        {"roots_or_coefficients_control", "0"},
    });

    cps->stage_swap_roots_when_in_root_mode(MICRO, "0","2");
    cps->render_microblock();
    cps->stage_swap_roots_when_in_root_mode(MICRO, "1","0");
    cps->render_microblock();
    cps->stage_swap_roots_when_in_root_mode(MICRO, "2","1");
    cps->render_microblock();
    cps->stage_swap_roots_when_in_root_mode(MICRO, "0","2");
    cps->render_microblock();

    CompositeScene cs;
    cs.add_scene(cps, "cps");

    cs.stage_macroblock(FileBlock("You might remember from Algebra class that a polynomial can be written in two forms-"), 1);
    cs.render_microblock();

    shared_ptr<LatexScene> ls = make_shared<LatexScene>("x^3+ax^2+bx^1+cx^0", 1);
    cs.add_scene_fade_in(MICRO, ls, "ls");
    cs.state_manager.transition(MICRO, {{"ls.x", ".25"}, {"ls.y", ".1"}});
    ls->state_manager.transition(MICRO, {{"w", ".4"}, {"h", ".2"}});
    cs.stage_macroblock(FileBlock("There's a standard form, where each exponent of x has an associated coefficient,"), 1);
    cs.render_microblock();

    shared_ptr<LatexScene> ls2 = make_shared<LatexScene>("(x+r_1)(x+r_2)(x+r_3)", 1);
    cs.add_scene_fade_in(MICRO, ls2, "ls2");
    cs.state_manager.transition(MICRO, {{"ls2.x", ".75"}, {"ls2.y", ".1"}});
    ls2->state_manager.transition(MICRO, {{"w", ".4"}, {"h", ".2"}});
    cs.stage_macroblock(FileBlock("and also a factored form, with one term corresponding to each root."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("These points are the roots of the polynomial, plotted in the complex plane,"), 2);
    cps->state_manager.transition(MICRO, {{"roots_opacity","1"}});
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {{"roots_opacity","0"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and these points are the coefficients of each power of the input."), 4);
    cps->state_manager.transition(MICRO, {{"coefficients_opacity","0"}});
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {{"coefficients_opacity","1"}});
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {{"coefficients_opacity","0"}});
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {{"coefficients_opacity","1"}});
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("It's really a shame that, most likely, that algebra teacher never showed you this plot of the relationship between them..."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("because with relatively little work, it'll let us peer right into the soul of the polynomial, and derive all kinds of important results in algebra, culminating in the Abel-Ruffini theorem, stating that there is no Quintic formula."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The space we're looking at here is the complex plane, home to numbers like i and 3-2i."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("For each point in this plot, I took that complex number, and put it as an input to the polynomial,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and colored the associated pixel accordingly."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("For example, say I take the point 1+i. plugging that into the polynomial gives (?)"), 1);
    cs.render_microblock();

    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "0"},
    });
    cs.stage_macroblock(FileBlock("The brightness shows distance to the origin, where pure white is an output value of zero,"), 5);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cps->state_manager.transition(MICRO, {
        {"ab_dilation", "10"},
    });
    cs.stage_macroblock(FileBlock("and the color is the angle of the output complex number."), 5);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cps->state_manager.transition(MICRO, {
        {"ab_dilation", ".3"},
    });
    cs.render_microblock();
    return;
    cs.stage_macroblock(FileBlock("That means our output value gets assigned this (?) color."), 1);
    cs.stage_macroblock(FileBlock("Doing this for every point, we can view what our complex polynomial looks like."), 1);
    cs.stage_macroblock(FileBlock("But before we get all analytical, I hear you asking..."), 1);
    cs.stage_macroblock(FileBlock("What's all this business with complex numbers? Why leave the familiar land of the reals?"), 1);
    cs.stage_macroblock(FileBlock("But in turn, I would ask, why do you need decimals or negatives either?"), 1);
    cs.stage_macroblock(FileBlock("If we imagine there are no numbers besides the natural numbers- 1, 2, 3, and so on, along with the ideas of plus and times."), 1);
    cs.stage_macroblock(FileBlock("In such a world, there are equations we can write that have no solution, such as 1 + x = 1."), 1);
    cs.stage_macroblock(FileBlock("Without the number zero, our number system is somehow incomplete."), 1);
    cs.stage_macroblock(FileBlock("There's nothing we can write here for x that would solve make this equation true."), 1);
    cs.stage_macroblock(FileBlock("So, in order to solve this equation, we need to accept the number 0 into our set..."), 1);
    cs.stage_macroblock(FileBlock("What about 1+x=0?"), 1);
    cs.stage_macroblock(FileBlock("Well, we then need -1 too,"), 1);
    cs.stage_macroblock(FileBlock("along with -2,"), 1);
    cs.stage_macroblock(FileBlock("and so on..."), 1);
    cs.stage_macroblock(FileBlock("and what if we write an equation like this one?"), 1);
    cs.stage_macroblock(FileBlock("It looks like we're gonna need fractions too..."), 1);
    cs.stage_macroblock(FileBlock("Surely, if we simply add all decimal numbers, our number system will finally be complete?"), 1);
    cs.stage_macroblock(FileBlock("We get a nice continuum of numbers."), 1);
    cs.stage_macroblock(FileBlock("Well, then, what about this equation?"), 1);
    cs.stage_macroblock(FileBlock("Uh oh, no real number squared gives us negative 1..."), 1);
    cs.stage_macroblock(FileBlock("so we need to continue creating more numbers to make our set whole."), 1);
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
    cs.stage_macroblock(FileBlock("Beyond just knowing that the equation has a root, it is also clear from looking at our plot that the number of roots does not change."), 1);
    cs.stage_macroblock(FileBlock("Well, unless you intentionally put them on top of each other... but that's cheating."), 1);
    cs.stage_macroblock(FileBlock("Now, notice- it's always the same number as the highest exponent in our equation."), 1);
    cs.stage_macroblock(FileBlock("This"), 1);
}
