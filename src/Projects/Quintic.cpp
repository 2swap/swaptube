#include "../Scenes/Math/ComplexPlotScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"

void render_video(){
    shared_ptr<ComplexPlotScene> cps = make_shared<ComplexPlotScene>(6);

    cps->stage_macroblock(SilenceBlock(8), 5);
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
    cps->state_manager.transition(MICRO, {{"roots_opacity",".5"}});
    cps->render_microblock();
    cps->stage_swap_roots_when_in_root_mode(MICRO, "1","0");
    cps->state_manager.transition(MICRO, {{"roots_opacity","0"}});
    cps->render_microblock();
    cps->stage_swap_roots_when_in_root_mode(MICRO, "2","1");
    cps->state_manager.transition(MICRO, {{"roots_opacity",".5"}});
    cps->render_microblock();
    cps->stage_swap_roots_when_in_root_mode(MICRO, "0","2");
    cps->state_manager.transition(MICRO, {{"roots_opacity","0"}});
    cps->render_microblock();

    CompositeScene cs;
    cs.add_scene(cps, "cps");
    shared_ptr<LatexScene> ls = make_shared<LatexScene>("x^3+ax^2+bx^1+cx^0", 1);
    cs.add_scene(ls, "ls");
    cs.stage_macroblock(SilenceBlock(2), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("This plot elucidates the relationship between a polynomial's roots and its coefficients."), 1);
    stage_macroblock(FileBlock("You might remember from Algebra class that a polynomial can be written in two forms-"), 1);
    stage_macroblock(FileBlock("It can be written out in an expanded form, where each exponent of x has an associated coefficient,"), 1);
    stage_macroblock(FileBlock("or alternatively, it can be written as the product of a term corresponding to each root."), 1);
    stage_macroblock(FileBlock("These points are the roots of the polynomial, plotted in the complex plane,"), 1);
    stage_macroblock(FileBlock("and these points are the coefficients of each power of the input."), 1);
    stage_macroblock(FileBlock("It's really a shame that, most likely, that algebra teacher never showed you this plot..."), 1);
    stage_macroblock(FileBlock("because with relatively little work, it'll let us peer right into the soul of the polynomial, and derive all kinds of important results in algebra, culminating in the Abel-Ruffini theorem, stating that there is no Quintic formula."), 1);
    stage_macroblock(FileBlock("The space we're looking at here is the complex plane, home to numbers like i and 3-2i."), 1);
    stage_macroblock(FileBlock("For each point in this plot, I have taken that complex number, and put it as an input to the polynomial,"), 1);
    stage_macroblock(FileBlock("and colored the associated pixel accordingly."), 1);
    stage_macroblock(FileBlock("For example, say I take the point 1+i- plugging that into the polynomial gives (?)"), 1);
    stage_macroblock(FileBlock("The brightness shows distance to the origin, where pure white is an output value of zero,"), 1);
    stage_macroblock(FileBlock("and the color is the angle of the output complex number."), 1);
    stage_macroblock(FileBlock("That means our output value gets assigned this (?) color."), 1);
    stage_macroblock(FileBlock("Doing this for every point, we can view what our complex polynomial looks like."), 1);
    stage_macroblock(FileBlock("Now some of you might be asking..."), 1);
    stage_macroblock(FileBlock("What's all this business with complex numbers? Why have we left the familiar land of the reals?"), 1);
    stage_macroblock(FileBlock("But in turn, I would ask,"), 1);
}
