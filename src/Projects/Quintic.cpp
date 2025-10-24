#include "../Scenes/Math/ComplexPlotScene.cpp"
#include "../Scenes/Math/ComplexArbitraryFunctionScene.cpp"
#include "../Scenes/Math/RealFunctionScene.cpp"
#include "../Scenes/Math/RootFractalScene.cpp"
#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/ThreeDimensionScene.cpp"
#include "../Scenes/Math/ManifoldScene.cpp"
#include "../Scenes/Math/AngularFractalScene.cpp"
#include "../Scenes/Common/CoordinateSceneWithTrail.cpp"
#include <regex>

// TODO I think I should talk during all of these title slides, since it's sort of awkward silence otherwise
shared_ptr<ThreeDimensionScene> get_part_1_title(){
    shared_ptr<LatexScene> ls = make_shared<LatexScene>("\\text{Linear Polynomials are Easy}", 1);
    shared_ptr<ThreeDimensionScene> title = make_shared<ThreeDimensionScene>();
    title->add_surface(Surface("ls"), ls);
    title->state.set({
        {"qi", ".01 {t} .5 * sin .01 * +"},
        {"qj", ".01 {t} .5 * cos .01 * +"},
        {"qk", ".04 {t} .5 * sin .01 * +"},
    });
    return title;
}
shared_ptr<ThreeDimensionScene> get_part_2_title(){
    shared_ptr<LatexScene> ls = make_shared<LatexScene>("\\text{Quadratics are Multi-Valued}", 1);
    shared_ptr<ThreeDimensionScene> title = make_shared<ThreeDimensionScene>();
    title->add_surface(Surface("ls"), ls);
    title->state.set({
        {"qi", ".01 {t} .5 * sin .01 * +"},
        {"qj", ".01 {t} .5 * cos .01 * +"},
        {"qk", ".04 {t} .5 * sin .01 * +"},
    });
    return title;
}
shared_ptr<ThreeDimensionScene> get_part_3_title(){
    shared_ptr<LatexScene> ls = make_shared<LatexScene>("\\text{Cubics have Commutators}", 1);
    shared_ptr<ThreeDimensionScene> title = make_shared<ThreeDimensionScene>();
    title->add_surface(Surface("ls"), ls);
    title->state.set({
        {"qi", ".01 {t} .5 * sin .01 * +"},
        {"qj", ".01 {t} .5 * cos .01 * +"},
        {"qk", ".04 {t} .5 * sin .01 * +"},
    });
    return title;
}
shared_ptr<ThreeDimensionScene> get_part_4_title(){
    shared_ptr<LatexScene> ls = make_shared<LatexScene>("\\text{Quartics, and Beyond}", 1);
    shared_ptr<ThreeDimensionScene> title = make_shared<ThreeDimensionScene>();
    title->add_surface(Surface("ls"), ls);
    title->state.set({
        {"qi", ".01 {t} .5 * sin .01 * +"},
        {"qj", ".01 {t} .5 * cos .01 * +"},
        {"qk", ".04 {t} .5 * sin .01 * +"},
    });
    return title;
}

string pole_r = "1";
string pole_i = "1";

string point_x_start = "<zradius> <ztheta> cos * (u) cos (v) sin * .02 * +";
string point_x_start_sqrt = "<sqrt_in_radius> <sqrt_in_theta> cos * (u) cos (v) sin * .02 * +";
string point_x_start_qrt = "<qrt_in_radius> <qrt_in_theta> cos * (u) cos (v) sin * .02 * +";
string point_y_start = "(v) cos .02 *";
string point_z_start = "<zradius> <ztheta> sin * (u) sin (v) sin * .02 * +";
string point_z_start_sqrt = "<sqrt_in_radius> <sqrt_in_theta> sin * (u) sin (v) sin * .02 * +";
string point_z_start_qrt = "<qrt_in_radius> <qrt_in_theta> sin * (u) sin (v) sin * .02 * +";

void part_0(CompositeScene& cs, shared_ptr<ComplexPlotScene> cps) {
    shared_ptr<RootFractalScene> rfs_intro = make_shared<RootFractalScene>();
    rfs_intro->global_identifier = "fractal";
    rfs_intro->state.set({{"terms", "17"}, {"coefficients_opacity", "0"}, {"ticks_opacity", "0"}});
    rfs_intro->stage_macroblock(CompositeBlock(FileBlock("These dots are the solutions of polynomials."), SilenceBlock(3)), 1);
    rfs_intro->state.set({
        {"coefficient0_r", "<coefficient0_i> 2 / exp"},
        {"coefficient0_i", "{t} .4 -"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"center_x", "-.8"},
        {"center_y", ".2"},
        {"zoom", "2"},
    });
    rfs_intro->render_microblock();

    rfs_intro->stage_macroblock(FileBlock("Aside from looking pretty, they illustrate polynomials' intrinsic complexity."), 2);
    rfs_intro->state.transition(MICRO, {
        {"coefficient0_r", "{t} 1 - 2 / cos"},
        {"coefficient0_i", "{t} 1 - 2 / sin"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"center_x", "0"},
        {"center_y", ".6"},
        {"zoom", "1.2"},
    });
    rfs_intro->render_microblock();
    rfs_intro->render_microblock();

    rfs_intro->stage_macroblock(FileBlock("Complexity suggesting that there is no simple formula to express their solutions."), 2);
    rfs_intro->state.transition(MICRO, {
        {"coefficient0_r", "{t} 4 / cos"},
        {"coefficient0_i", "{t} 4 / sin"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1"},
    });
    rfs_intro->render_microblock();
    rfs_intro->render_microblock();

    rfs_intro->stage_macroblock(SilenceBlock(1.5), 2);
    rfs_intro->state.set("spindist", ".03");
    rfs_intro->state.begin_timer("spindist_timer");
    rfs_intro->state.transition(MICRO, {
        {"coefficient0_r", "<spindist_timer> 3 * sin <spindist> * -1 *"},
        {"coefficient0_i", "<spindist_timer> 3 * cos <spindist> * -1 *"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "{t} 4 / sin .3 *"},
    });
    rfs_intro->render_microblock();
    rfs_intro->render_microblock();

    cs.add_scene(rfs_intro, "rfs_intro");

    vector<string> formulas = {
        "-\\frac{b}{a}",
        "\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a}",
        "\\phantom{+} \\sqrt[3]{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right) + \\sqrt{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right)^2 + \\left(\\frac{c}{3a} - \\frac{b^2}{9a^2}\\right)^3}} \\\\\\\\ + \\sqrt[3]{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right) - \\sqrt{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right)^2 + \\left(\\frac{c}{3a} - \\frac{b^2}{9a^2}\\right)^3}} \\\\\\\\ - \\frac{b}{3a}",
        "\\frac{-a}{4}\\pm\\frac{1}{2}{\\sqrt{\\frac{a^{2} }{4}\\pm\\frac{2b}{3}+\\frac{\\sqrt[3]{2}\\left(b^{2}-3ac+12d\\right)}{3{\\sqrt[3]{2b^{3}-9abc+27c^{2}+27a^{2}d-72bd+{\\sqrt{-4{\\left(b^{2}-3ac+12d\\right)}^{3}+{\\left(2b^{3}-9abc+27c^{2}+27a^{2}d-72bd\\right)}^{2} } }}} }+\\sqrt[3]{\\frac{ {2b^{3}-9abc+27c^{2}+27a^{2}d-72bd+{\\sqrt{-4{\\left(b^{2}-3ac+12d\\right)}^{3}+{\\left(2b^{3}-9abc+27c^{2}+27a^{2}d-72bd\\right)}^{2} } } } }{54}} } } \\\\\\\\ -\\frac{1}{2}{\\sqrt{\\frac{a^{2} }{2}-\\frac{4b}{3}-\\frac{2^{\\frac{1}{3} }\\left(b^{2}-3ac+12d\\right)}{3{\\sqrt[3]{2b^{3}-9abc+27c^{2}+27a^{2}d-72bd+{\\sqrt{-4{\\left(b^{2}-3ac+12d\\right)}^{3}+{\\left(2b^{3}-9abc+27c^{2}+27a^{2}d-72bd\\right)}^{2} } }}} }-\\sqrt[3]{\\frac{ {2b^{3}-9abc+27c^{2}+27a^{2}d-72bd+{\\sqrt{-4{\\left(b^{2}-3ac+12d\\right)}^{3}+{\\left(2b^{3}-9abc+27c^{2}+27a^{2}d-72bd\\right)}^{2} } } } }{54}}-\\frac{-a^{3}+4ab-8c}{4{\\sqrt{\\frac{a^{2} }{4}-\\frac{2b}{3}+\\frac{\\sqrt[3]{2}\\left(b^{2}-3ac+12d\\right)}{3{\\sqrt[3]{2b^{3}-9abc+27c^{2}+27a^{2}d-72bd+{\\sqrt{-4{\\left(b^{2}-3ac+12d\\right)}^{3}+{\\left(2b^{3}-9abc+27c^{2}+27a^{2}d-72bd\\right)}^{2} } }}} }+\\sqrt[3]{\\frac{ {2b^{3}-9abc+27c^{2}+27a^{2}d-72bd+{\\sqrt{-4{\\left(b^{2}-3ac+12d\\right)}^{3}+{\\left(2b^{3}-9abc+27c^{2}+27a^{2}d-72bd\\right)}^{2} } } } }{54}} } } } } }",
        "\\text{?}",
    };
    vector<string> titles = {
        "\\text{Linear Formula}",
        "\\text{Quadratic Formula}",
        "\\text{Cubic Formula}",
        "\\text{Quartic Formula}",
        "\\text{Quintic Formula}",
    };
    cs.stage_macroblock(FileBlock("For linear... quadratic... cubic... and quartic polynomials,"), 8);
    rfs_intro->state.transition(MACRO, "spindist", ".01 {t} .5 * sin 1 + .1 * +");
    vector<shared_ptr<LatexScene>> formula_scenes;
    vector<shared_ptr<LatexScene>> title_scenes;
    for(int i = 0; i < 5; i++) {
        shared_ptr<LatexScene> title_scene = make_shared<LatexScene>(titles[i], 1, 1, .25);
        title_scenes.push_back(title_scene);
        if(i != 4) cs.add_scene(title_scenes[i], "title_scene" + to_string(i), .5, .1);
        cs.state.set("title_scene" + to_string(i) + ".x", "1.5");

        shared_ptr<LatexScene> ls_formula = make_shared<LatexScene>(formulas[i], i==4?.6:1, i==3?5:1, 1);
        formula_scenes.push_back(ls_formula);
        if(i != 4) cs.add_scene(formula_scenes[i], "ls_formula" + to_string(i));
        cs.state.set("ls_formula" + to_string(i) + ".x", i==3?"3.5":"1.5");
    }
    cs.slide_subscene(MICRO, "title_scene0", -1, 0);
    cs.slide_subscene(MICRO, "ls_formula0", -1, 0);
    cs.render_microblock();
    cs.render_microblock();
    cs.slide_subscene(MICRO, "title_scene0", -1, 0);
    cs.slide_subscene(MICRO, "ls_formula0", -1, 0);
    cs.slide_subscene(MICRO, "title_scene1", -1, 0);
    cs.slide_subscene(MICRO, "ls_formula1", -1, 0);
    cs.render_microblock();
    cs.render_microblock();
    cs.slide_subscene(MICRO, "title_scene1", -1, 0);
    cs.slide_subscene(MICRO, "ls_formula1", -1, 0);
    cs.slide_subscene(MICRO, "title_scene2", -1, 0);
    cs.slide_subscene(MICRO, "ls_formula2", -1, 0);
    cs.render_microblock();
    cs.render_microblock();
    cs.slide_subscene(MICRO, "title_scene2", -1, 0);
    cs.slide_subscene(MICRO, "ls_formula2", -1, 0);
    cs.slide_subscene(MICRO, "title_scene3", -1, 0);
    cs.slide_subscene(MICRO, "ls_formula3", -1, 0);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("the formulas which yield their solutions are increasingly complex."), SilenceBlock(1)), 2);
    cs.slide_subscene(MACRO, "title_scene3", -4.5, 0);
    cs.slide_subscene(MACRO, "ls_formula3", -4.5, 0);
    cs.render_microblock();
    cs.fade_all_subscenes_except(MICRO, "rfs_intro", 0);
    cs.render_microblock();
    for(int i = 0; i < 4; i++) {
        cs.remove_subscene("title_scene" + to_string(i));
        cs.remove_subscene("ls_formula" + to_string(i));
    }

    cs.stage_macroblock(FileBlock("But the complexity of behavior that polynomials exhibit..."), 1);
    rfs_intro->state.transition(MICRO, {
        {"coefficient0_r", "{t} 4 / cos"},
        {"coefficient0_i", "{t} 4 / sin"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"zoom", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("grows faster than algebraic formulas are able to express."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->set_degree(3);
    cs.add_scene_fade_in(MICRO, cps, "cps");
    cps->coefficients_to_roots();
    cps->state.set({
        {"root0_r", "1"},
        {"root0_i", ".2"},
        {"root1_r", ".5"},
        {"root1_i", "1.2"},
        {"root2_r", "-1"},
        {"root2_i", "-.3"},
        {"coefficient0_opacity", "1"},
        {"coefficient1_opacity", "1"},
        {"coefficient2_opacity", "1"},
        {"coefficient3_opacity", "1"},
    });
    cs.render_microblock();
    cs.remove_all_subscenes_except("cps");

    cs.stage_macroblock(FileBlock("Today, we'll discover how permutational symmetries of solutions tell us about their polynomials,"), 4);
    cps->coefficients_to_roots();
    shared_ptr<CoordinateSceneWithTrail> trail1 = make_shared<CoordinateSceneWithTrail>();
    shared_ptr<CoordinateSceneWithTrail> trail2 = make_shared<CoordinateSceneWithTrail>();
    trail1->state.set({
        {"trail_x", "{cps.coefficient0_r}"},
        {"trail_y", "{cps.coefficient0_i}"},
    });
    trail2->state.set({
        {"trail_x", "{cps.coefficient1_r}"},
        {"trail_y", "{cps.coefficient1_i}"},
    });
    cs.add_scene(trail1, "trail1");
    cs.add_scene(trail2, "trail2");
    cs.state.set({
        {"trail1.opacity", ".3"},
        {"trail2.opacity", ".3"},
    });
    trail1->trail_color = 0xffff0000;
    trail2->trail_color = 0xffffff00;
    cps->stage_swap(MICRO, "0", "1", false, true);
    cs.render_microblock();
    trail1->trail_color = 0xff00ff00;
    trail2->trail_color = 0xff0000ff;
    cps->stage_swap(MICRO, "2", "0", false, true);
    cs.render_microblock();
    trail1->trail_color = 0xffff0000;
    trail2->trail_color = 0xffffff00;
    cps->stage_swap(MICRO, "1", "2", false);
    cs.render_microblock();
    trail1->trail_color = 0xff00ff00;
    trail2->trail_color = 0xff0000ff;
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and solve the mystery... of the missing quintic formula."), 2);
    cps->coefficients_to_roots();
    cps->state.transition(MACRO, {
        {"ab_dilation", "0"},
        {"dot_radius", "4"},
    });
    cps->begin_timer("ringspin");
    for(int i = 0; i < cps->get_degree(); i++) {
        cps->state.transition(MACRO, {
            {"root"+to_string(i)+"_r", "<ringspin> <ringspin> * .3333 " + to_string(i) + " * + 6.283 * sin"},
            {"root"+to_string(i)+"_i", "<ringspin> <ringspin> * .3333 " + to_string(i) + " * + 6.283 * cos"},
        });
    }
    cs.fade_subscene(MICRO, "trail1", 0);
    cs.fade_subscene(MICRO, "trail2", 0);
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.add_scene(title_scenes[4], "title_scene4", 1.5, .1);
    cs.add_scene(formula_scenes[4], "ls_formula4", 1.5);
    cs.state.transition(MACRO, {
        {"title_scene4.x", ".5 {t} 1000 * sin <ringspin> * .001 * +"},
        {"ls_formula4.x", ".5 {t} 1000 * sin <ringspin> * .002 * +"},
    });
    cps->state.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();
    cs.render_microblock();
    cps->construction.clear();
    cps->state.set("construction_opacity", "1");
    cs.remove_all_subscenes();

    cps->state.set({
        {"ab_dilation", "1"},
        {"dot_radius", "1"},
    });
    cs.add_scene_fade_in(MICRO, cps, "cps");
    cs.stage_macroblock(SilenceBlock(6), 1);
    cps->state.transition(MICRO, "center_y", ".6");
    for(int i = 0; i < cps->get_degree(); i++) {
        cps->state.set({
            {"root"+to_string(i)+"_r", "{t} .6 * 6.28 .3333 " + to_string(i) + " * * + sin"},
            {"root"+to_string(i)+"_i", "{t} .4 * 6.28 .3333 " + to_string(i) + " * * + cos .2 +"},
        });
    }
    cps->state.set({
        {"coefficient0_opacity", "1"},
        {"coefficient1_opacity", "1"},
        {"coefficient2_opacity", "1"},
        {"coefficient3_opacity", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This graph shows the relationship between a cubic polynomial's coefficients and its solutions."), 10);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
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

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state.transition(MICRO, {
        {"root0_r", ".4"},
        {"root0_i", "0"},
        {"root1_r", ".5"},
        {"root1_i", "1.2"},
        {"root2_r", "-1"},
        {"root2_i", "-.3"},
        {"center_y", ".6"},
    });
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
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But we want to find its solutions- the values of x for which the whole thing equals zero."), 4);
    //Highlight x's in the polynomial
    ls->begin_latex_transition(MICRO, latex_color(0xff333333, "a"+latex_color(OPAQUE_WHITE, "x")+"^3+b"+latex_color(OPAQUE_WHITE, "x")+"^2+c"+latex_color(OPAQUE_WHITE, "x")+"+d"));
    for(int i = 0; i < 2; i++) {
        cps->transition_root_rings(MICRO, 1);
        cs.render_microblock();
        cps->transition_root_rings(MICRO, 0);
        cs.render_microblock();
    }

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_subscene(MICRO, "ls", 0);
    cps->state.transition(MICRO, {{"center_y", "0"}});
    cs.render_microblock();
    cs.remove_subscene("ls");

    cs.stage_macroblock(CompositeBlock(FileBlock("We can move a solution and see its effect on the coefficients,"), SilenceBlock(1)), 4);
    cps->transition_coefficient_opacities(MICRO, 1);
    cps->state.set({
        {"root1_ring", "0"},
    });
    cps->state.transition(MICRO, {
        {"root1_ring", "1"},
    });
    cs.render_microblock();
    cps->state.set({
        {"root1_r", ".5 <spin_coefficient_r> +"},
        {"root1_i", "1.2 <spin_coefficient_i> +"},
        {"spin_coefficient_r", "{t} 2 * sin <spin_multiplier> *"},
        {"spin_coefficient_i", "{t} 3 * cos <spin_multiplier> *"},
        {"spin_multiplier", "0"},
    });
    cps->state.transition(MICRO, {
        {"spin_multiplier", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"spin_multiplier", "0"},
        {"root1_ring", "0"},
    });
    cs.render_microblock();

    cps->roots_to_coefficients();
    cs.stage_macroblock(CompositeBlock(FileBlock("and we can move a coefficient yielding some effect on the solutions."), SilenceBlock(1.5)), 4);
    cps->state.transition(MICRO, {
        {"coefficient0_ring", "1"},
    });
    cs.render_microblock();
    cps->state.set({
        {"coefficient0_r", cps->state.get_equation("coefficient0_r") + " <spin_coefficient_r> +"},
        {"coefficient0_i", cps->state.get_equation("coefficient0_i") + " <spin_coefficient_i> +"},
    });
    cps->state.transition(MICRO, {
        {"spin_multiplier", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"coefficient0_ring", "0"},
        {"spin_multiplier", "0"},
    });
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"ticks_opacity", "1"},
    });
    cps->state.transition(MACRO, {
        {"zoom", "-.5"},
    });
    cs.stage_macroblock(FileBlock("The polynomial lives in the complex plane, home to numbers like i and 2-i."), 5);
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 1), "i"));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(2, -1), "2-i"));
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state.transition(MICRO, {{"ticks_opacity", "0"}});
    cs.render_microblock();

    cps->state.transition(MICRO, "construction_opacity", "0");

    cps->coefficients_to_roots(); // Just to be safe
    cs.stage_macroblock(FileBlock("For each pixel on the screen, we can pass that complex number into the polynomial, and see what comes out..."), 3);
    cps->state.set({
        {"point_in_x", "0"},
        {"point_in_y", "0"},
    });
    cs.render_microblock();
    cps->construction.clear();
    cps->state.set("construction_opacity", "1");
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "in", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "out", .7, true));
    cps->state.set({
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
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(5), 1);
    cps->state.transition(MICRO, "zoom", "0");
    cps->state.transition(MICRO, {
        {"point_in_x", "{t} sin .9 * 1.2 *"},
        {"point_in_y", "{t} cos .8 * 1.2 *"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We color the input point depending on where the output lands."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state.transition(MICRO, "zero_crosshair_opacity", "1");
    cps->construction.add(GeometricLine(glm::vec2(0, 0), glm::vec2(0, 0), "vec", true));
    cps->state.set({
        {"line_vec_start_x", "0"},
        {"line_vec_start_y", "0"},
        {"line_vec_end_x", "<point_out_x>"},
        {"line_vec_end_y", "<point_out_y>"},
    });
    cs.render_microblock();

    cps->state.transition(MICRO, {
        {"ab_dilation", "0"},
    });
    cs.stage_macroblock(FileBlock("The brightness shows how far it is from the origin,"), 5);
    cps->state.transition(MICRO, "construction_opacity", "1");
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cps->coefficients_to_roots();
    cs.stage_macroblock(FileBlock("meaning that the white areas are solutions of the polynomial."), 6);
    for (int i = 0; i < 3; i++) {
        cps->state.transition(MICRO, {
            {"point_in_x", "<root" + to_string(i) + "_r>"},
            {"point_in_y", "<root" + to_string(i) + "_i>"},
        });
        cs.render_microblock();
        cs.render_microblock();
    }

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The color shows the angle of the output,"), 1);
    cps->state.transition(MICRO, {
        {"ab_dilation", "24"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("so, for example, pink means above the origin and green means below it."), 5);
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"point_in_x", "1"},
        {"point_in_y", "-.3"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"point_in_x", "-1"},
        {"point_in_y", ".6"},
    });
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state.transition(MICRO, "zero_crosshair_opacity", "0");
    cps->state.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();
    cps->construction.clear();
    cps->state.set("construction_opacity", "1");
    cps->state.remove({
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

    cs.stage_macroblock(FileBlock("Doing this for every complex number on the plane, we can graph our polynomial."), 2);
    cps->state.transition(MICRO, {
        {"ab_dilation", "1"},
    });
    cps->coefficients_to_roots();
    cps->state.transition(MACRO, {
        {"root0_r", "1.2"},
        {"root0_i", "0"},
        {"root1_r", "-.3"},
        {"root1_i", "0"},
        {"root2_r", "-1.2"},
        {"root2_i", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.remove_all_subscenes();

    ThreeDimensionScene tds;
    tds.global_identifier = "3d";
    tds.add_surface(Surface("cps"), cps);
    tds.stage_macroblock(SilenceBlock(2), /*2*/1);
    tds.state.transition(MICRO, {
        {"d", "1.5"},
    });
    //tds.render_microblock();

    shared_ptr<RealFunctionScene> rfs = make_shared<RealFunctionScene>();
    rfs->add_function("? 1.2 - ? .3 + ? 1.2 + * *", 0xffff0000);
    rfs->state.set("ticks_opacity", "0");
    tds.add_surface(Surface(glm::vec3(0, 0, 0), glm::vec3(.5, 0, 0), glm::vec3(0, 0, .5), "rfs"), rfs);
    tds.state.transition(MICRO, {
        {"qi", "-.3 {t} sin .07 * +"},
        {"qj", "{t} cos .025 *"},
    });
    tds.render_microblock();

    shared_ptr<LatexScene> f_complex = make_shared<LatexScene>("\\text{Complex Function}", 1, .5, .5);
    shared_ptr<LatexScene> f_real    = make_shared<LatexScene>("\\text{Real Function}", 1, .5, .5);
    tds.add_surface_fade_in(MICRO, Surface(glm::vec3(0,.25,-.05), glm::vec3(.25, 0, 0), glm::vec3(0, .25, 0), "f_complex"), f_complex);
    tds.add_surface_fade_in(MICRO, Surface(glm::vec3(0,.05,-.25), glm::vec3(.25, 0, 0), glm::vec3(0, 0, .25), "f_real"), f_real);
    tds.stage_macroblock(FileBlock("You might ask, what's all this complex number business? Why leave the familiar land of real numbers?"), 2);
    tds.render_microblock();
    tds.fade_subscene(MICRO, "f_real", 0);
    tds.fade_subscene(MICRO, "f_complex", 0);
    tds.fade_subscene(MICRO, "rfs", 0);
    tds.state.transition(MICRO, {
        {"d", "1"},
        {"qi", "0"},
        {"qj", "0"},
    });
    tds.render_microblock();
    tds.remove_all_subscenes();

    cps->stage_macroblock(FileBlock("In turn, I would ask _you_, who needs decimals or negatives either?"), 1);
    cps->roots_to_coefficients();
    cps->state.transition(MICRO, {
        {"coefficient0_r", "{t} .2 * cos"},
        {"coefficient0_i", "-5"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"coefficient2_r", "0"},
        {"coefficient2_i", "0"},
        {"coefficient3_r", "0.001"},
        {"coefficient3_i", "0"},
        {"ab_dilation", "1.5"},
    });
    cps->transition_coefficient_opacities(MICRO, 0);
    cps->render_microblock();

    cs.add_scene(cps, "cps");
    cs.stage_macroblock(FileBlock("Imagine there's nothing but natural numbers- 0, 1, 2, 3, sitting happily on the right side of the number line."), 12);
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
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("In such a world, there's a big problem..."), 1);
    cs.render_microblock();
    cps->set_degree(1);

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

    cs.stage_macroblock(FileBlock("But there's nothing we can write here for x that would make this equation come true."), 1);
    impossible->begin_latex_transition(MACRO, latex_color(0xff222222, "2"+latex_color(OPAQUE_WHITE, "x")+"+4=0"));
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    impossible->begin_latex_transition(MICRO, "2x+4=0");
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("Let's plot this polynomial."), SilenceBlock(2)), 1);
    cps->state.transition(MACRO, {
        {"construction_opacity", ".15"},
        {"coefficient1_r", "2"},
        {"coefficient1_i", "0"},
        {"coefficient1_opacity", "1"},
        {"coefficient0_r", "4"},
        {"coefficient0_i", "0"},
        {"coefficient0_opacity", "1"},
        {"ab_dilation", "1"},
    });
    cs.render_microblock();
    cps->set_degree(1);

    int flashes = 4;
    cs.stage_macroblock(FileBlock("The coefficients land on numbers in our number system,"), flashes*2);
    for(int i = 0; i < flashes; i++) {
        cps->transition_coefficient_rings(MICRO, 1);
        cs.render_microblock();
        cps->transition_coefficient_rings(MICRO, 0);
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("but the solution to the equation is -2... which isn't one of them!"), flashes*2+2);
    cps->state.transition(MACRO, "construction_opacity", ".4");
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

    cs.stage_macroblock(FileBlock("To solve this, we _need_ to invent -2..."), 2);
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(-2, 0), "-2"));
    cs.render_microblock();

    cps->state.transition(MICRO, {
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
        cps->state.transition(MICRO, {
            {"coefficient0_r", to_string(-x)},
            {"coefficient0_i", "0"},
        });
        cs.render_microblock();
        cps->construction.add(GeometricPoint(glm::vec2(x, 0), to_string(x)));
        cs.render_microblock();
    }

    cs.stage_macroblock(CompositeBlock(SilenceBlock(1), FileBlock("Sadly, our number system is still lacking.")), 1);
    cps->state.transition(MICRO, {
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
    tds2->global_identifier = "3d";
    cs.stage_macroblock(CompositeBlock(FileBlock("Surely our number system is finally complete!"), SilenceBlock(1.5)), 1);
    cs.add_scene(tds2, "tds2");
    rfs = make_shared<RealFunctionScene>();
    rfs->add_function("? 2 * 1 +", 0xffff0000);
    rfs->state.set("ticks_opacity", "0");
    tds2->add_surface(Surface("cps"), cps);
    shared_ptr<RealFunctionScene> rfs2 = make_shared<RealFunctionScene>();
    rfs2->state.set("ticks_opacity", "0");
    rfs2->add_function("1 ? ? * -", 0xffcc00ff);
    tds2->add_surface(Surface(glm::vec3(0, 0, 0), glm::vec3(0, .5, 0), glm::vec3(0, 0, .5), "rfs2"), rfs2);
    tds2->add_surface(Surface(glm::vec3(0, 0, 0), glm::vec3(.5, 0, 0), glm::vec3(0, 0, .5), "rfs"), rfs);
    tds2->state.transition(MICRO, {
        {"qi", "-.1 {t} 2 / sin .02 * +"},
        {"d", "1.5"},
    });
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("Well then, what about this equation?"), SilenceBlock(1)), 2);
    cps->set_degree(2);
    cps->state.transition(MACRO, {
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

    cs.stage_macroblock(FileBlock("and you can see it too- none of the solutions are on the number line."), 8);
    for(int flash = 0; flash < 4; flash++) {
        cps->transition_root_rings(MICRO, 1);
        cs.render_microblock();
        cps->transition_root_rings(MICRO, 0);
        cs.render_microblock();
    }

    cs.stage_macroblock(SilenceBlock(.3), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Just like before, our number line must be missing something..."), 1);
    tds2->state.transition(MICRO, {
        {"qi", "-.1 {t} 2 / sin .02 * +"},
        {"qj", "-.2 {t} 2 / cos .01 * +"},
        {"d", "1.5"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    tds2->state.transition(MICRO, {
        {"qi", "0"},
        {"qj", "0"},
        {"d", "1"},
    });
    tds2->fade_subscene(MICRO, "rfs", 0);
    tds2->fade_subscene(MICRO, "rfs2", 0);
    cs.render_microblock();
    tds2->remove_all_subscenes();
    cs.remove_all_subscenes();
    cs.add_scene(cps, "cps");

    cs.stage_macroblock(FileBlock("Let's call these solutions 'i' and '-i'."), 4);
    cs.render_microblock();
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 1), "i"));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, -1), "-i"));
    cs.render_microblock();

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
    for(int slice = -6; slice <= 6; slice++) {
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
    while(remaining_microblocks_in_macroblock) cs.render_microblock();

    cps->coefficients_to_roots();
    for(int i = 0; i < cps->get_degree(); i++) {
        cps->state.transition(MICRO, {
            {"root"+to_string(i)+"_r", "{t} 1.2 * 6.28 .3333 " + to_string(i) + " * * + sin"},
            {"root"+to_string(i)+"_i", "{t} .8 * 6.28 .3333 " + to_string(i) + " * * + cos"},
        });
    }

    cs.stage_macroblock(SilenceBlock(2), 1);
    cps->state.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();
    cps->construction.clear();
    cps->state.set("construction_opacity", "1");

    cs.stage_macroblock(FileBlock("Up until now, we've been playing whack-a-mole..."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("_You_ invent numbers, and _I_ make an equation that can't be solved without _even more numbers_."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But that game ends here!"), 1);
    cps->state.set({{"coefficient0_ring", "0"}, {"coefficient1_ring", "0"}});
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

    cs.stage_macroblock(FileBlock("This is so important, that it's called the Fundamental Theorem of Algebra."), 21);
    for(int i = 0; i < 7; i++) cs.render_microblock();
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
    shared_ptr<LatexScene> fta_title_3 = make_shared<LatexScene>("\\text{Theorem}", 1, .45, .25);
    cs.add_scene(fta_title_3, "fta_title_3", .82, .1);
    cs.render_microblock();
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
    cps->state.transition(MICRO, {
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
    cps->state.transition(MICRO, {
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
    cps->state.transition(MICRO, {
        {"root0_r", "1"},
        {"root0_i", "0"},
        {"root1_r", "-1"},
        {"root1_i", "0"},
        {"root2_r", "-1"},
        {"root2_i", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We call that a solution of multiplicity 2. It counts as 2 normal solutions."), 2);
    cps->state.transition(MICRO, {
        {"root2_ring", "1"},
        {"center_x", "-1"},
        {"zoom", ".7"},
        {"ab_dilation", "1.4"},
    });
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"root2_ring", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1.5), 1);
    cps->state.transition(MICRO, {
        {"center_x", "0"},
        {"zoom", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("You can actually identify the solutions' multiplicities from the graph."), 1);
    cps->state.transition(MICRO, {
        {"ticks_opacity", "1"},
        {"ab_dilation", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state.transition(MACRO, {
        {"center_x", "<root0_r>"},
        {"center_y", "<root0_i>"},
        {"zoom", "1.5"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Tracing around this normal solution,"), 1);
    cps->state.transition(MICRO, {
        {"ticks_opacity", "0"},
    });
    cps->state.transition(MICRO, {
        {"ab_dilation", "2"},
    });
    cps->state.begin_timer("theta");
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "", .7, true));
    cps->state.set({
        {"point__x", "<root0_r>"},
        {"point__y", "<root0_i>"},
    });
    cps->state.transition(MACRO, {
        {"point__x", "<root0_r> <theta> 2 * cos .1 * +"},
        {"point__y", "<root0_i> <theta> 2 * sin .1 * +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("we see red, then green, then blue."), 8);
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"theta", "3.4"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"theta", "4.4"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"theta", "5.4"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We make a spin around the color wheel."), 1);
    cps->state.transition(MICRO, {
        {"theta", "8.54"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    cps->state.begin_timer("theta2");
    cps->state.set({
        {"theta3", "<theta2> 1 -"},
    });
    cps->state.transition(MACRO, {
        {"point__x", "<root2_r> <theta3> 2 * cos .2 * +"},
        {"point__y", "<root2_i> <theta3> 2 * sin .2 * +"},
        {"center_x", "<root2_r>"},
        {"center_y", "<root2_i>"},
    });
    cps->state.transition(MICRO, {
        {"ab_dilation", "1.3"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Tracing around the multiplicity 2 solution,"), 1);
    cps->state.transition(MACRO, {
        {"ab_dilation", "1.8"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("we see red, green, blue, then red, green, blue again."), 19);
    cs.render_microblock();
    cs.render_microblock();
    for(int i = 0; i < 6; i++) {
        double angle = 3.14 * (i/6. + .275 + 1);
        if(i == 0) angle += .1;
        cps->state.transition(MICRO, "theta3", to_string(angle));
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
    cps->state.transition(MACRO, {
        {"zoom", "0"},
        {"center_x", "0"},
        {"center_y", "0"},
    });
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"ticks_opacity", "0"},
        {"ab_dilation", "1"},
    });
    cs.render_microblock();

    /*cs.stage_macroblock(FileBlock("Let's plot the output like before."), 2);
    cps->state.transition(MICRO, {
        {"point__x", "<root0_r> {t} 3 * sin .1 * +"},
        {"point__y", "<root0_i> {t} 3 * cos .1 * +"},
    });
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "out", .7, true));
    cps->state.set({
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
    cps->state.transition(MICRO, {
        {"zoom", ".7"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("As we trace around the multiplicity 1 solution, the output point follows the angle of the input point."), 1);
    cs.render_microblock();

    cps->state.transition(MICRO, {
        {"zoom", "0"},
    });
    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state.transition(MICRO, {
        {"point__x", "<root2_r> {t} 3 * sin .3 * +"},
        {"point__y", "<root2_i> {t} 3 * cos .3 * +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But tracing around the multiplicity 2 solution, the output point goes around twice as fast."), 1);
    cs.render_microblock();
    */

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();
    cps->construction.clear();
    cps->state.set("construction_opacity", "1");
    cps->state.remove({
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

    cs.stage_macroblock(FileBlock("The degree of the polynomial,"), 2);
    quadratic = make_shared<LatexScene>(latex_color(0xff333333, "1x^" + latex_color(OPAQUE_WHITE, "3") + "+1x^2-1x-1"), .7, 1, .5);
    cs.add_scene_fade_in(MICRO, quadratic, "quadratic", .5, .2);
    cps->roots_to_coefficients();
    cps->state.transition(MICRO, {
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
    cps->state.transition(MICRO, {
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
    cps->transition_root_rings(MICRO, 0);
    quadratic->begin_latex_transition(MICRO, latex_color(0xff333333, "1x^3+1x^2-1x-1"));
    cs.render_microblock();

    quadratic->begin_latex_transition(MICRO, latex_color(0xff333333, latex_color(OPAQUE_WHITE, "1")+"x^3+"+latex_color(OPAQUE_WHITE, "1")+"x^2-"+latex_color(OPAQUE_WHITE, "1")+"x-"+latex_color(OPAQUE_WHITE, "1")));
    cs.stage_macroblock(FileBlock("Another way of saying this is that there's always exactly one more coefficient than the number of solutions."), 3);
    cs.render_microblock();
    cs.render_microblock();
    quadratic->begin_latex_transition(MICRO, latex_color(0xff333333, "1x^3+1x^2+1x+1"));
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->transition_root_rings(MICRO, 0);
    cs.fade_subscene(MACRO, "quadratic", 0);
    cs.render_microblock();
    cs.remove_subscene("quadratic");

    cs.stage_macroblock(FileBlock("But, how do they relate to each other?"), 1);
    cps->transition_coefficient_opacities(MICRO, 1);
    cs.render_microblock();

    //move the roots around
    cps->roots_to_coefficients();
    for(int i = 0; i < cps->get_degree()+1; i++) {
        cps->state.transition(MICRO, {
            {"coefficient"+to_string(i)+"_r", "{t} 1.2 * 6.28 .47 " + to_string(i) + " * * + sin"},
            {"coefficient"+to_string(i)+"_i", "{t} .8 * 6.28 .36 " + to_string(i) + " * * + cos .2 +"},
        });
    }
    cs.stage_macroblock(CompositeBlock(SilenceBlock(1), FileBlock("Given the coefficients,")), 2);
    cps->transition_coefficient_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cs.render_microblock();

    cps->state.transition(MICRO, {
        {"coefficient3_r", "1"},
        {"coefficient3_i", "0"},
    });
    cps->transition_coefficient_opacities(MACRO, 0);
    cs.stage_macroblock(FileBlock("how do we know where the solutions should be?"), 2);
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("It's obvious since I'm plotting the graph for you..."), 1);
    //move roots offscreen
    /*cps->coefficients_to_roots();
    cps->state.transition(MICRO, {
        {"root0_r", "-6"},
        {"root0_i", "0"},
        {"root1_r", "4"},
        {"root1_i", "-6"},
        {"root2_r", "4"},
        {"root2_i", "6"},
    });*/
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but if I give you a, b, and c numerically, how do you find solutions 1 and 2?"), 12);
    shared_ptr<LatexScene> abc = make_shared<LatexScene>("\\begin{tabular}{ll} a=2+i \\phantom{x_1=?} & \\end{tabular}", .4);
    cs.add_scene_fade_in(MICRO, abc, "abc");
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{ll} a=2+i & \\phantom{x_1=?} \\\\\\\\ b=-i & \\phantom{x_2=?} \\end{tabular}");
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{ll} a=2+i & \\phantom{x_1=?} \\\\\\\\ b=-i & \\phantom{x_2=?} \\\\\\\\ c=1.5 &\\end{tabular}");
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{ll} a=2+i & x_1=? \\\\\\\\ b=-i & \\phantom{x_2=?} \\\\\\\\ c=1.5 & \\end{tabular}");
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "\\begin{tabular}{ll} a=2+i & x_1=? \\\\\\\\ b=-i & x_2=? \\\\\\\\ c=1.5 &\\end{tabular}");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Before I spoil this problem, let me illustrate its complexities."), 1);
    abc->begin_latex_transition(MICRO, "a=2+i \\\\\\\\ b=-i \\\\\\\\ c=1.5");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's take the simplest case imaginable- a polynomial with coefficients that are all 1 or -1."), 2);
    cps->roots_to_coefficients();
    cps->state.transition(MACRO, {
        {"coefficient3_r", new_coefficient_val},
        {"coefficient3_i", "0"},
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "-1"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "1"},
        {"coefficient0_i", "0"},
    });
    abc->begin_latex_transition(MICRO, "a=1 \\\\\\\\ b=-1 \\\\\\\\ c=1");
    cs.render_microblock();
    cps->roots_to_coefficients();
    cs.render_microblock();
    cps->set_degree(2);

    cs.stage_macroblock(CompositeBlock(SilenceBlock(.8), FileBlock("There's a few different options of which are 1s and which are -1s.")), 5);
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "a=1 \\\\\\\\ b=1 \\\\\\\\ c=-1");
    cps->state.transition(MICRO, {
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "-1"},
        {"coefficient0_i", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();
    abc->begin_latex_transition(MICRO, "a=1 \\\\\\\\ b=-1 \\\\\\\\ c=-1");
    cps->state.transition(MICRO, {
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "-1"},
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
    fracs->global_identifier = "fractal";
    cs.add_scene(fracs, "fracs", .5, .5, true);
    fracs->state.set({
        {"terms", "3"},
        {"coefficients_opacity", "0"},
        {"ticks_opacity", "0"},
        {"visibility_multiplier", "18"},
    });
    cps->transition_coefficient_opacities(MACRO, 1);
    for(int bit = 0; bit < flashes*8; bit++) {
        string a = (bit&4) ? "1" : "-1";
        string b = (bit&2) ? "1" : "-1";
        string c = (bit&1) ? "1" : "-1";
        abc->begin_latex_transition(MICRO, "a="+a+" \\\\\\\\ b="+b+" \\\\\\\\ c="+c);
        cps->state.transition(MICRO, {
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
    fracs->state.transition(MICRO, {
        {"visibility_multiplier", "15"},
        {"terms", "4"}, // degree 3 means 4 terms
    });
    fracs->render_microblock();

    fracs->stage_macroblock(FileBlock("degree 4,"), 1);
    fracs->state.transition(MICRO, {
        {"visibility_multiplier", "9"},
        {"terms", "5"},
    });
    fracs->render_microblock();

    fracs->stage_macroblock(CompositeBlock(FileBlock("and even higher,"), SilenceBlock(8)), 13);
    fracs->state.transition(MACRO, {
        {"visibility_multiplier", "1"},
        {"zoom", ".7"},
    });
    int i = 6;
    while(remaining_microblocks_in_macroblock) {
        fracs->state.transition(MICRO, {
            {"terms", to_string(i)},
        });
        i++;
        fracs->render_microblock();
    }

    fracs->stage_macroblock(FileBlock("We were letting the coefficients be either 1 or -1,"), 1);
    fracs->state.transition(MICRO, {
        {"coefficients_opacity", "1"},
    });
    fracs->render_microblock();

    fracs->stage_macroblock(FileBlock("but what happens if we change those options?"), 1);
    fracs->state.begin_timer("fractal_timer");
    fracs->state.transition(MICRO, {
        {"coefficient0_r", "-.01 <fractal_timer> 120 + 4 / cos *"},
        {"coefficient0_i", ".01 <fractal_timer> 120 + 4 / sin *"},
        {"zoom", "0"},
    });
    fracs->render_microblock();

    fracs->stage_macroblock(SilenceBlock(2), 1);
    fracs->state.transition(MICRO, {
        {"center_x", "-.5"},
        {"center_y", ".5"},
    });
    fracs->render_microblock();

    fracs->stage_macroblock(SilenceBlock(2), 1);
    fracs->state.transition(MICRO, {
        {"zoom", "2.5"},
    });
    fracs->render_microblock();

    fracs->stage_macroblock(SilenceBlock(40), 10);
    fracs->state.transition(MICRO, {
        {"coefficient0_r", "<fractal_timer> 4 - 8 / sin"},
        {"coefficient0_i", "<fractal_timer> 4 - 9 / sin"},
    });
    fracs->render_microblock();
    fracs->state.transition(MICRO, {
        {"center_x", "0"},
        {"center_y", "1"},
    });
    fracs->render_microblock();
    fracs->state.transition(MICRO, {
        {"zoom", "3"},
    });
    fracs->render_microblock();
    fracs->render_microblock();
    fracs->state.transition(MICRO, {
        {"zoom", ".3"},
    });
    fracs->render_microblock();
    fracs->state.begin_timer("spin");
    fracs->state.transition(MICRO, {
        {"center_x", "0"},
        {"center_y", "0"},
    });
    fracs->render_microblock();
    fracs->state.transition(MICRO, {
        {"coefficient0_r", "<spin> 2 / cos"},
        {"coefficient0_i", "<spin> 2 / sin"},
    });
    fracs->render_microblock();
    fracs->render_microblock();
    fracs->state.transition(MICRO, {
        {"coefficient0_r", "<spin> 2 / cos 5 *"},
        {"coefficient0_i", "<spin> 2 / sin 5 *"},
    });
    fracs->render_microblock();
    fracs->render_microblock();

    fracs->stage_macroblock(CompositeBlock(SilenceBlock(3), FileBlock("As we zoom into these point clouds,")), 1);
    fracs->state.transition(MACRO, {
        {"center_x", "-.48165"},
        {"center_y", "-.45802"},
        {"coefficient0_r", "-1.4"},
        {"coefficient0_i", ".4"},
    });
    fracs->render_microblock();

    cs.stage_macroblock(FileBlock("there are shapes resembling the Dragon Curve,"), 1);
    fracs->state.transition(MICRO, "zoom", "2.5");
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(4), 2);
    fracs->state.transition(MACRO, { {"zoom", "3"}, });
    cs.render_microblock();
    cs.fade_subscene(MICRO, "fracs", .5);
    int dragon_depth = 12;
    int dragon_size = 1 << dragon_depth;
    shared_ptr<AngularFractalScene> dragon = make_shared<AngularFractalScene>(dragon_size);
    dragon->state.set("zoom", "1");
    cs.add_scene_fade_in(MICRO, dragon, "dragon");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("That's the same shape that you get by folding up a strip of paper and letting it spring out."), dragon_depth);
    dragon->state.transition(MACRO, "zoom", "7.5");
    for(int i = 0; i < dragon_depth; i++) {
        int j = 0;
        int multiple = 1 << (dragon_depth-i);
        for(int angle = multiple >> 1; angle < dragon_size; angle += multiple) {
            if(angle == 0) continue;
            dragon->state.transition(MICRO, "angle_" + to_string(angle), "pi .998 *" + string((j%2)?" -1 *" : ""));
            j++;
        }
        cs.render_microblock();
    }

    dragon->state.transition(MACRO, "zoom", "8.5");
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    dragon->state.transition(MACRO, "zoom", "5.2");
    cs.stage_macroblock(SilenceBlock(4), dragon_depth);
    for(int i = dragon_depth-1; i >= 0; i--) {
        int j = 0;
        int multiple = 1 << (dragon_depth-i);
        for(int angle = multiple >> 1; angle < dragon_size; angle += multiple) {
            if(angle == 0) continue;
            dragon->state.transition(MICRO, "angle_" + to_string(angle), "pi .5 *" + string((j%2)?" -1 *" : ""));
            j++;
        }
        cs.render_microblock();
    }

    cs.fade_subscene(MACRO, "dragon", 0);
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();
    cs.remove_subscene("dragon");

    cs.stage_macroblock(FileBlock("The point is, the function mapping coefficients to solutions isn't simple- it must have a bit of magic to it."), 1);
    fracs->state.transition(MICRO, "zoom", "2");
    cs.render_microblock();
    cs.remove_all_subscenes_except("fracs");

    fracs->state.transition(MICRO, {
        {"coefficient0_r", "-1 {t} .2 * sin .2 * +"},
        {"coefficient0_i", "0 {t} .2 * cos .2 * +"},
    });
    cs.stage_macroblock(FileBlock("Even with just 1 and -1 as coefficients, we find ourselves in a zoo of emergent complexity."), 1);
    fracs->state.begin_timer("littlewood_timer");
    fracs->state.transition(MICRO, {
        {"center_x", "<littlewood_timer> 10 / cos -1.2 *"},
        {"center_y", "<littlewood_timer> 10 / sin 1.2 *"},
    });
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(8), FileBlock("What _can_ we say about the process which maps coefficients to solutions?")), 3);
    cs.render_microblock();
    cs.render_microblock();
    fracs->state.transition(MICRO, {
        {"zoom", "0"},
        {"center_x", "0"},
        {"center_y", "0"},
    });
    cs.fade_subscene(MICRO, "fracs", 0);
    cs.render_microblock();
}

void part_1(CompositeScene& cs, shared_ptr<ComplexPlotScene> cps) {
    cps->roots_to_coefficients();
    cps->state.set({
        {"coefficient2_r", "0"},
        {"coefficient2_i", "0"},
        {"coefficient2_opacity", "0"},
        {"coefficient1_opacity", "0"},
        {"coefficient0_opacity", "0"},
    });
    cps->set_degree(1);
    cs.add_scene_fade_in(MICRO, cps, "cps", .5, .5, .3, true);
    shared_ptr<ThreeDimensionScene> title1 = get_part_1_title();
    cs.add_scene_fade_in(MICRO, title1, "title1", .5, .4);
    cs.stage_macroblock(SilenceBlock(4), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "title1", 0);
    cs.fade_subscene(MICRO, "cps", 1);
    cs.render_microblock();
    cs.remove_subscene("title1");

    cs.stage_macroblock(FileBlock("Let's start off easy, with polynomials whose highest exponent is one."), 2);
    shared_ptr<LatexScene> linear = make_shared<LatexScene>(latex_color(0xff333333, "ax^"+latex_color(OPAQUE_WHITE, "1")+"+b = 0"), .6, 1, .5);
    cs.add_scene_fade_in(MICRO, linear, "linear", .5, .2);
    cps->roots_to_coefficients();
    cs.render_microblock();
    linear->begin_latex_transition(MICRO, "ax+b=0");
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->transition_coefficient_opacities(MICRO, 1);
    cps->state.transition(MICRO, {
        {"coefficient0_r", "{t} 1.5 * sin"},
        {"coefficient0_i", "{t} cos"},
        {"coefficient1_r", "{t} sin .5 +"},
        {"coefficient1_i", "{t} 2.1 * cos"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("With such a linear polynomial, there's only one solution."), 4);
    cs.render_microblock();
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This case is super easy."), 1);
    cps->state.transition(MICRO, {
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "0"},
        {"coefficient0_i", "-1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("To solve ax+b=0, we can just use algebra."), 2);
    linear->begin_latex_transition(MICRO, "ax=-b");
    cs.render_microblock();
    linear->begin_latex_transition(MICRO, "x=\\frac{-b}{a}");
    cs.render_microblock();
    cs.export_frame("Sample");

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state.transition(MICRO, {
        {"coefficient0_r", "{t} sin"},
        {"coefficient0_i", "{t} cos"},
        {"coefficient1_r", "{t} 1.5 * sin .5 +"},
        {"coefficient1_i", "{t} 2.1 * cos"},
    });
    cs.fade_subscene(MICRO, "linear", 0);
    cs.render_microblock();
    cs.remove_subscene("linear");

    cs.stage_macroblock(SilenceBlock(1.5), 1);
    cps->roots_to_coefficients();
    cps->state.transition(MICRO, {
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "0"},
        {"coefficient0_i", "1"},
    });
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.render_microblock();
}

void part_2(CompositeScene& cs, shared_ptr<ComplexPlotScene> cps, shared_ptr<ManifoldScene> ms) {
    cs.add_scene(cps, "cps");
    shared_ptr<ThreeDimensionScene> title2 = get_part_2_title();
    cs.stage_macroblock(CompositeBlock(FileBlock("But what happens when we go up a degree?"), SilenceBlock(4)), 3);
    cs.add_scene_fade_in(MICRO, title2, "title2", .5, .4);
    cs.fade_subscene(MICRO, "cps", .3);
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "title2", 0);
    cs.fade_subscene(MICRO, "cps", 1);
    cs.render_microblock();
    cs.remove_subscene("title2");

    cs.stage_macroblock(FileBlock("Jumping up to a polynomial with highest exponent 2,"), 1);
    cps->state.set({
        {"coefficient2_opacity", "0"},
        {"coefficient1_opacity", "0"},
        {"coefficient0_opacity", "0"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "0"},
        {"coefficient0_i", "1"},
    });
    cps->set_degree(2);
    StateSet preflip = {
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "1.7"},
        {"coefficient1_i", "1"},
        {"coefficient0_r", ".5"},
        {"coefficient0_i", "1"},
    };
    cps->state.transition(MICRO, preflip);
    cps->transition_coefficient_opacities(MICRO, 1);
    cs.render_microblock();

    shared_ptr<LatexScene> quadratic_formula = make_shared<LatexScene>("\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}", 1, .5, .5);
    cs.stage_macroblock(FileBlock("There's a certain formula you might have learned in school..."), 1);
    cs.add_scene(quadratic_formula, "quadratic_formula", 12345, .5);
    cs.state.begin_timer("quadratic_timer");
    cs.state.set("quadratic_formula.x", "<quadratic_timer> .1 * .25 -");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but pretend you didn't for a moment,"), 2);
    cs.state.transition(MICRO, "quadratic_formula.x", "-.7");
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
    shared_ptr<LatexScene> quadratic = make_shared<LatexScene>(latex_color(0xff333333, "x^2 + "+latex_color(OPAQUE_WHITE, "b")+"x + "+latex_color(OPAQUE_WHITE, "c")), .7, 1, .5);
    cs.add_scene_fade_in(MICRO, quadratic, "quadratic", .5, .15);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("If we switch b and c here, we don't have the same function anymore."), 3);
    cps->state.transition(MICRO, {
        {"coefficient0_ring", "1"},
        {"coefficient1_ring", "1"},
    });
    cs.render_microblock();
    quadratic->begin_latex_transition(MICRO, latex_color(0xff333333, "x^2 + "+latex_color(OPAQUE_WHITE, "c")+"x + "+latex_color(OPAQUE_WHITE, "b")));
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
    cps->state.transition(MICRO, {
        {"coefficient1_ring", "0"},
        {"coefficient0_ring", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The solutions, on the other hand, don't have a particular order to them."), 2);
    cps->state.transition(MACRO, {
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
    });
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("In the factored form of the polynomial, we can rearrange as we please,"), 2);
    shared_ptr<LatexScene> factored = make_shared<LatexScene>("(x - "+latex_color(OPAQUE_WHITE, "\\small{x_1}")+") (x - "+latex_color(OPAQUE_WHITE, "x_2")+")", .7, 1, .5);
    cs.add_scene_fade_in(MACRO, factored, "factored", .5, .2);
    cps->coefficients_to_roots();
    cs.render_microblock();
    factored->begin_latex_transition(MICRO, "(x - "+latex_color(OPAQUE_WHITE, "x_2")+") (x - "+latex_color(OPAQUE_WHITE, "\\small{x_1}")+")");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("since it doesn't matter what order we multiply the terms."), 2);
    cs.render_microblock();
    cs.fade_subscene(MICRO, "factored", 0);
    cs.render_microblock();
    cs.remove_subscene("factored");

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Swapping the solutions with each other, the coefficients go right back to where they started."), 4);
    shared_ptr<CoordinateSceneWithTrail> trail1 = make_shared<CoordinateSceneWithTrail>();
    trail1->state.set({
        {"trail_x", "{cps.coefficient0_r}"},
        {"trail_y", "{cps.coefficient0_i}"},
    });
    cs.add_scene(trail1, "trail1");
    cs.state.set({
        {"trail1.opacity", ".3"},
    });
    trail1->trail_color = 0xff00ff00;
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(.7), FileBlock("The coefficients are distinct from each other, but the solutions aren't.")), 4);
    cs.fade_subscene(MICRO, "trail1", 0);
    cps->transition_coefficient_rings(MICRO, 1);
    cs.render_microblock();
    cs.remove_subscene("trail1");
    cps->transition_coefficient_rings(MICRO, 0);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(1), FileBlock("This has some bizarre implications.")), 1);
    cps->state.transition(MICRO, "zoom", ".5");
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("First and foremost, there's no function of the coefficients that gives out _one solution_, and is continuous."), FileBlock("Here's why.")), 8);
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.render_microblock();
    shared_ptr<LatexScene> function = make_shared<LatexScene>("f(a, b, c) \\rightarrow x_1", 1, .75, .4);
    shared_ptr<LatexScene> details = make_shared<LatexScene>("f \\text{ continuous}", 1, .5, .4);
    cs.add_scene_fade_in(MICRO, function, "function", .5, .4);
    cs.render_microblock();
    function->begin_latex_transition(MICRO, "f("+latex_color(0xff4040ff, "a, b, c")+") \\rightarrow x_1");
    cs.render_microblock();
    function->begin_latex_transition(MICRO, "f(a, b, c) \\rightarrow "+latex_color(0xff40ff40, "x_1"));
    cs.render_microblock();
    function->begin_latex_transition(MICRO, "f(a, b, c) \\rightarrow x_1");
    cs.render_microblock();
    cs.add_scene_fade_in(MICRO, details, "details", .5, .65);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Assume we do have a continuous function that takes in a, b, and c,"), 4);
    cps->state.transition(MACRO, "zoom", "0");
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
    cps->state.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();
    cps->state.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();
    cps->state.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();
    cps->state.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();
    cps->state.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(.7), FileBlock("As we saw before, we can swap the solutions and the coefficients will return to their starting position.")), 3);
    cps->coefficients_to_roots();
    cs.render_microblock();
    shared_ptr<CoordinateSceneWithTrail> trail2 = make_shared<CoordinateSceneWithTrail>();
    trail2->state.set({
        {"trail_x", "{cps.coefficient0_r}"},
        {"trail_y", "{cps.coefficient0_i}"},
    });
    cs.add_scene(trail2, "trail2");
    cs.state.set("trail2.opacity", ".3");
    trail2->trail_color = 0xff00ff00;
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cs.fade_subscene(MICRO, "trail2", 0);
    cs.render_microblock();
    cs.remove_subscene("trail2");

    cs.stage_macroblock(FileBlock("But our 'continuous function' gives us a different solution now!"), 6);
    cs.render_microblock();
    cps->state.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();
    cps->state.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();
    cps->state.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();
    cps->state.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();
    cps->state.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Given this same a, b, and c,"), 3);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_coefficient_rings(MICRO, 0);
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("First it gave us this solution, but then it gave us that solution."), 8);
    cps->state.transition(MICRO, "root1_ring", "1");
    cs.render_microblock();
    cps->state.transition(MICRO, "root1_ring", "0");
    cs.render_microblock();
    cps->state.transition(MICRO, "root1_ring", "1");
    cs.render_microblock();
    cps->state.transition(MICRO, "root1_ring", "0");
    cs.render_microblock();
    cps->state.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();
    cps->state.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();
    cps->state.transition(MICRO, "root0_ring", "1");
    cs.render_microblock();
    cps->state.transition(MICRO, "root0_ring", "0");
    cs.render_microblock();


    cs.stage_macroblock(CompositeBlock(FileBlock("That's a contradiction! The function must not be continuous."), SilenceBlock(.8)), 1);
    cps->coefficients_to_roots();
    cps->state.transition(MICRO, {
        {"root0_r", "-.5"},
        {"root0_i", ".5"},
        {"root1_r", ".5"},
        {"root1_i", "-.5"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We could handle this by accepting a function which is discontinuous, like this."), 4);
    cps->state.transition(MICRO, "positive_quadratic_formula_opacity", "1");
    cs.render_microblock();
    cps->coefficients_to_roots();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Or alternatively, we could consider the two solutions together as a pair and give up on selecting a single one of the two."), 4);
    cps->state.transition(MICRO, "positive_quadratic_formula_opacity", "0");
    cps->transition_root_rings(MICRO, 1);
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.remove_all_subscenes_except("cps");
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    shared_ptr<LatexScene> continuous_operations = make_shared<LatexScene>("+, -, \\times, \\div", .5, 1, .3);
    shared_ptr<LatexScene> title = make_shared<LatexScene>("\\text{Basic Operators}", 1, .4, .3);
    cs.add_scene_fade_in(MICRO, continuous_operations, "continuous_operations", .5, .25);
    cs.add_scene_fade_in(MICRO, title, "title", .5, .1);
    cs.stage_macroblock(FileBlock("And that means a quadratic formula can't be written with just basic arithmetic!"), 1);
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Even if we use other functions, such as sines, (long pause), cosines, (long pause), and exponentials,"), 8);
    cs.render_microblock();
    cs.render_microblock();
    continuous_operations->begin_latex_transition(MICRO, "+, -, \\times, \\div, \\sin");
    shared_ptr<ComplexArbitraryFunctionScene> cafs = make_shared<ComplexArbitraryFunctionScene>();
    cafs->global_identifier = "cafs";
    cafs->state.set({{"ticks_opacity", "0"}, {"sqrt_coef", "0"}, {"sin_coef", "1"}});
    cs.fade_subscene(MICRO, "cps", 0);
    cs.add_scene_fade_in(MICRO, cafs, "cafs", .5, .5, 1, true);
    cs.render_microblock();
    cs.render_microblock();
    cafs->state.transition(MICRO, {{"sin_coef", "0"}, {"cos_coef", "1"}});
    continuous_operations->begin_latex_transition(MICRO, "+, -, \\times, \\div, \\sin, \\cos");
    cs.render_microblock();
    cs.render_microblock();
    cafs->state.transition(MICRO, {{"cos_coef", "0"}, {"exp_coef", "1"}});
    continuous_operations->begin_latex_transition(MICRO, "+, -, \\times, \\div, \\sin, \\cos, e^x");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("they're continuous and single-valued, so they can't track a solution of the quadratic."), SilenceBlock(1)), 2);
    cps->roots_to_coefficients();
    cps->state.set({
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "0"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "{t} .5 * sin .5 *"},
        {"coefficient0_i", "{t} .5 * cos .5 *"},
    });
    cs.render_microblock();
    cs.fade_all_subscenes(MICRO, 0);
    cs.fade_subscene(MICRO, "cps", 1);
    cs.render_microblock();
    cs.remove_subscene("cafs");
    cs.remove_subscene("title");
    cs.remove_subscene("continuous_operations");

    cs.stage_macroblock(CompositeBlock(SilenceBlock(.7), FileBlock("What about the quadratic formula!?")), 2);
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

    cs.stage_macroblock(SilenceBlock(.5), 1);
    cs.fade_subscene(MICRO, "quadratic_formula", 0);
    cs.render_microblock();
    cs.remove_subscene("quadratic_formula");

    cs.stage_macroblock(FileBlock("So what makes a square root different from those other functions?"), 1);
    shared_ptr<LatexScene> sqrt4 = make_shared<LatexScene>("\\sqrt{\\phantom{4}}", 1, .5, .3);
    cs.add_scene_fade_in(MICRO, sqrt4, "sqrt4");
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We usually say things like 'the square root of 4 is 2,'"), 2);
    sqrt4->begin_latex_transition(MICRO, "\\sqrt{4}");
    cs.render_microblock();
    sqrt4->begin_latex_transition(MICRO, "\\sqrt{4} = 2");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but what we're asking to begin with is,"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("what are the solutions of x^2=4?"), 1);
    sqrt4->begin_latex_transition(MICRO, "4 = x^2");
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.6), 1);
    cs.fade_subscene(MICRO, "sqrt4", 0);
    cs.render_microblock();
    cs.remove_subscene("sqrt4");

    cs.stage_macroblock(FileBlock("Plotting that equation, we of course see the result 2,"), 2);
    cps->set_degree(2);
    cps->roots_to_coefficients();
    cps->state.transition(MICRO, {
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "0"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "-4"},
        {"coefficient0_i", "0"},
    });
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(2, 0), "2"));
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("as well as the negative square root, -2."), 2);
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(-2, 0), "-2"));
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("By convention, we define the 'square root' to be positive, "), SilenceBlock(1)), 8);
    cps->state.transition(MICRO, "construction_opacity", "0");
    cps->coefficients_to_roots();
    cps->state.set({
        {"root0_r", "2"},
        {"root0_i", "0"},
        {"root1_r", "-2"},
        {"root1_i", "0"},
    });
    for(int i = 0; i < 8; i++) {
        cps->state.transition(MICRO, "root0_ring", to_string(1 - (i % 2)));
        cs.render_microblock();
    }
    cps->construction.clear();
    cps->state.set("construction_opacity", "1");

    cs.stage_macroblock(FileBlock("but what about the square root of -1?"), 1);
    cps->roots_to_coefficients();
    cps->state.transition(MICRO, {
        {"coefficient0_r", "1"},
        {"coefficient0_i", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We _define_ i as being precisely that,"), 1);
    cps->construction.add(GeometricPoint(glm::vec2(0, 1), "i"));
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but -i works too."), 1);
    cps->construction.add(GeometricPoint(glm::vec2(0, -1), "-i"));
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("It goes deeper than that though."), 1);
    cps->state.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();
    cps->construction.clear();
    cps->state.set("construction_opacity", ".3");

    cs.stage_macroblock(FileBlock("Watch as I move the coefficients of the polynomial along the real number line."), 2);
    cps->transition_coefficient_opacities(MICRO, 1);
    cps->state.transition(MICRO, {
        {"ab_dilation", ".25"},
        {"dot_radius", "3"},
    });
    cs.render_microblock();
    cps->roots_to_coefficients();
    cps->state.begin_timer("realline");
    cps->state.transition(MICRO, {
        {"coefficient0_r", "<realline> 1.0 * 0 + 2 / sin 3 *"},
        {"coefficient0_i", "0"},
        {"coefficient1_r", "<realline> 1.1 * 1 + 2 / sin 3 *"},
        {"coefficient1_i", "0"},
        {"coefficient2_r", "<realline> 1.2 * 2 + 2 / sin 3 *"},
        {"coefficient2_i", "0"},
    });
    cps->construction.add(GeometricLine(glm::vec2(-5, 0), glm::vec2(5, 0)));
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(4), 1);
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.render_microblock();

    int real_axis_degree = 6;
    cs.stage_macroblock(SilenceBlock(4), 1);//FileBlock("I'll jump up to degree " + to_string(real_axis_degree) + "..."), 1);
    cps->set_degree(real_axis_degree);
    for(int i = 0; i <= real_axis_degree; i++) {
        cps->state.transition(MICRO, "coefficient" + to_string(i) + "_r", "<realline> 1." + to_string(i) + " * " + to_string(i) + " + 2 / sin 3 *");
        cps->state.transition(MICRO, "coefficient" + to_string(i) + "_i", "0");
    }
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(4), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("With real-valued coefficients, the solutions are always vertically symmetrical."), 4);
    cps->transition_coefficient_opacities(MICRO, 1);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cps->transition_coefficient_opacities(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(3), 1);
    cps->state.transition(MICRO, {
        {"coefficient0_r", "1"},
        {"coefficient0_i", "0"},
        {"coefficient1_r", "0"},
        {"coefficient1_i", "0"},
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
    });
    for(int i = 3; i <= real_axis_degree; i++) {
        cps->state.transition(MICRO, "coefficient" + to_string(i) + "_r", "0.0001");
        cps->state.transition(MICRO, "coefficient" + to_string(i) + "_i", "0");
    }
    cps->state.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("So any definition for i written using the real numbers must simultaneously include -i!"), 2);
    cps->construction.clear();
    cps->state.set("construction_opacity", "1");
    cps->construction.add(GeometricPoint(glm::vec2(0, 1), "i"));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, -1), "-i"));
    cps->state.transition(MICRO, {
        {"ab_dilation", "1"},
        {"dot_radius", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("They're algebraically indistinguishable, so which gets to be the square root of -1 is a matter of mere notation."), 16);
    for(int blink = 0; blink < 4; blink++) {
        cps->state.transition(MICRO, {{"root0_ring", "1"}, {"root1_ring", "0"}});
        cs.render_microblock();
        cs.render_microblock();
        cps->state.transition(MICRO, {{"root0_ring", "0"}, {"root1_ring", "1"}});
        cs.render_microblock();
        cs.render_microblock();
    }

    cps->transition_root_rings(MICRO, 0);
    cs.stage_macroblock(SilenceBlock(.5), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("If we _do_ define the square root function with such an arbitrary choice, its graph is discontinuous like this..."), 3);
    cafs->state.set({
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
    cs.render_microblock();
    cafs->construction.add(GeometricLine(glm::vec2(-5, 0), glm::vec2(0, 0)));
    cafs->state.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();
    cps->state.set("positive_quadratic_formula_opacity", "0");

    cs.stage_macroblock(CompositeBlock(SilenceBlock(1), FileBlock("We can change our arbitrary choice, and that yucky discontinuity starts to move around.")), 1);
    cafs->state.transition(MICRO, "sqrt_branch_cut", "{t} 5 * sin 1 * {t} cos 4 * +");
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("In other words, there's not just one square root function, but a whole class of them depending on that choice."), SilenceBlock(1.5)), 3);
    cs.render_microblock();
    cs.render_microblock();
    cafs->state.transition(MICRO, "sqrt_branch_cut", to_string(M_PI));
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But discontinuities suck. With a bit of a change in perspective, we might be able to eliminate it!"), 3);
    ms->state.set({
        {"d", "1"},
        {"q1", ".92387953251"},
        {"qi", ".38268343236"},
        {"qj", "0"},
        {"qk", "0"},
    });
    // u is radius, v is angle
    ms->add_manifold("0",
        "(u) (v) cos *", "0", "(u) (v) sin *",
        "(u) .5 ^ (v) 2 / cos * -1 * 10 *", "(u) .5 ^ (v) 2 / sin * -1 * 10 *",
        "0", "1.5", "3000",
        "-3.14159", "3.14159", "11000"
    );
    cs.add_scene_fade_in(MICRO, ms, "ms");
    cs.render_microblock();
    cs.remove_all_subscenes_except("ms");
    ms->state.transition(MICRO, {
        {"d", "5"},
        {"q1", "1"},
        {"qi", "{t} .1 * sin .05 * .08 +"},
        {"qj", "{t} .1 * cos .15 *"},
        {"qk", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("We can imagine visualizing this function as jutting out of the complex plane to highlight the discontinuity."), SilenceBlock(1)), 1);
    ms->state.transition(MICRO, {
        {"manifold0_y", "(v) 2 / sin (u) .5 ^ * -0.5 *"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This is what it looks like if I change that arbitrary choice like before."), 1);
    ms->state.transition(MICRO, {
        {"d", "4.5"},
        {"manifold0_v_min", "6 -3.14159 +"},
        {"manifold0_v_max", "6 3.14159 +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(5), 1);
    ms->state.transition(MICRO, {
        {"qj", "{t} .1 * cos .1 * .05 +"},
        {"manifold0_v_min", "-30 -3.14159 +"},
        {"manifold0_v_max", "-30 3.14159 +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("Notice how we can extend this surface continuously until it meets itself."), SilenceBlock(1)), 1);
    ms->state.transition(MICRO, {
        {"manifold0_v_min", "-30 -6.28318 +"},
        {"manifold0_v_max", "-30 6.28318 +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This is called a Riemann surface."), 1);
    ms->state.transition(MICRO, {
        {"manifold0_v_min", "-6.28318"},
        {"manifold0_v_max", "6.28318"},
        {"qj", "{t} .1 * cos .1 * .5 +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    ms->state.transition(MICRO, {
        {"qi", "{t} .1 * sin .05 *"},
        {"qj", "{t} .1 * cos .1 * .05 +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Now that our surface has no sharp edges, we can make our square root function continuous on it."), 1);
    ms->add_manifold("1",
        "0", "0.01", "0",
        "0.00001", "0.00001",
        "-1.5", "1.5", "1500",
        "-1.5", "1.5", "1500"
    );
    ms->state.transition(MICRO, {
        {"manifold1_x", "(u)"},
        {"manifold1_z", "(v)"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Here's the complex plane."), 1);
    ms->state.transition(MICRO, "axes_length", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Picking some complex number, we can evaluate the square root by seeing where it intersects with the two sheets,"), 3);
    ms->state.set({
        {"zradius", "1.2"},
        {"ztheta", "3.8"},
    });
    ms->state.transition(MICRO, {
        {"manifold1_x", point_x_start},
        {"manifold1_y", point_y_start},
        {"manifold1_z", point_z_start},
        {"manifold1_u_min", "0"},
        {"manifold1_u_max", "3.14159"},
        {"manifold1_v_min", "-3.14159"},
        {"manifold1_v_max", "3.14159"},
        {"manifold1_u_steps", "200"},
        {"manifold1_v_steps", "1600"},
        {"qi", ".1"},
    });
    cs.render_microblock();
    ms->state.transition(MICRO, {
        {"manifold1_x", point_x_start},
        {"manifold1_y", point_y_start + " 40 *"},
        {"manifold1_z", point_z_start},
    });
    cs.render_microblock();
    // Add another identical manifold
    ms->add_manifold("2",
        ms->state.get_equation("manifold1_x"), ms->state.get_equation("manifold1_y"), ms->state.get_equation("manifold1_z"),
        "0.00001", "0.00001",
        "0", "3.14159", "200",
        "-3.14159", "3.14159", "1600"
    );
    ms->state.transition(MICRO, {
        {"manifold1_y", point_y_start + " <ztheta> 2 / sin <zradius> .5 ^ * -0.5 * +"},
        {"manifold2_y", point_y_start + " <ztheta> 2 / sin <zradius> .5 ^ * -0.5 * -"},
    });
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("and we can even track it continuously as we move around."), SilenceBlock(2)), 2);
    ms->state.transition(MICRO, {
        {"ztheta", "3"},
        {"zradius", "1.1"},
    });
    cs.render_microblock();
    ms->state.transition(MICRO, {
        {"ztheta", "-3"},
        {"zradius", "1.1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This should hopefully make the predicament clear-"), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We're either stuck with a discontinuous square root function which returns a single value,"), 2);
    ms->state.transition(MICRO, {
        {"manifold0_v_min", "-3.14159"},
        {"manifold0_v_max", "3.14159"},
        {"manifold0_y", "0"},
        {"manifold1_y", point_y_start},
        {"manifold2_y", point_y_start},
    });
    cs.render_microblock();
    ms->state.transition(MICRO, {
        {"q1", ".92387953251"},
        {"qi", ".38268343236"},
        {"qj", "0"},
        {"qk", "0"},
    });
    cs.render_microblock();

    ms->state.begin_timer("resheet");
    cs.stage_macroblock(FileBlock("or a continuous square root function which returns two that are always opposite each other!"), 1);
    ms->state.transition(MICRO, {
        {"q1", "1"},
        {"qi", ".05"},
        {"qj", "<resheet> .5 * 6.28 + sin .1 *"},
        {"qk", "0"},
        {"manifold0_y", "(v) 2 / sin (u) .5 ^ * -0.5 *"},
        {"manifold0_v_min", "-6.28318"},
        {"manifold0_v_max", "6.28318"},
        {"manifold1_y", point_y_start + " <ztheta> 2 / sin <zradius> .5 ^ * -0.5 * +"},
        {"manifold2_y", point_y_start + " <ztheta> 2 / sin <zradius> .5 ^ * -0.5 * -"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(3), 2);
    cs.render_microblock();
    cs.fade_subscene(MICRO, "ms", 0);
    cs.add_scene_fade_in(MICRO, cps, "cps", .5, .5, 1, true);
    cps->construction.clear();
    cps->roots_to_coefficients();
    cps->state.set({
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

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("And this makes sense! As we saw before, either our formula cannot transform continuously,"), 4);
    cps->set_degree(2);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cps->state.transition(MICRO, "positive_quadratic_formula_opacity", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("or otherwise we have to yield both solutions without distinguishing between them to achieve continuity."), 4);
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cps->coefficients_to_roots();
    cs.stage_macroblock(FileBlock("We've shown that other functions alone can't solve quadratics, but why can square roots?"), 1);
    cps->state.transition(MICRO, {
        {"ab_dilation", ".5"},
        {"dot_radius", ".5"},
        {"root0_r", "-1"},
        {"root0_i", "-.4"},
        {"root1_r", "0"},
        {"root1_i", "-.8"},
        {"center_x", "2"},
    });
    ms->state.set("w", ".5");
    cs.add_scene_fade_in(MICRO, ms, "ms", .75, .5);
    ms->state.begin_timer("solution_exchange");
    ms->state.set({
        {"ztheta", "3.14159"},
        {"ab_dilation", ".1"},
        {"dot_radius", ".1"},
        {"manifold0_r", "<zradius> <ztheta> cos * (u) 2 ^ (v) cos * -"},
        {"manifold0_i", "<zradius> <ztheta> sin * (u) 2 ^ (v) sin * -"},
        {"q1", "1"},
        {"qi", ".02"},
        {"qj", "<solution_exchange> .5 * sin .1 * -.6 +"},
        {"qk", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Remember: I can grab coefficients and make a loop which switches the solutions."), 3);
    cs.fade_subscene(MICRO, "ms", 0.2);
    cps->transition_coefficient_opacities(MICRO, 1);
    cs.render_microblock();
    cps->state.transition(MICRO, "coefficient0_ring", "1");
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(4), 3);
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->state.transition(MICRO, "coefficient0_ring", "0");
    cs.fade_subscene(MICRO, "ms", 1);
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("Now watch as I make a loop in the input value of the square root function..."), SilenceBlock(1)), 1);
    ms->add_manifold("cbrt_in",
        point_x_start, point_y_start, point_z_start,
        "0.00001", "0.00001",
        "0", "3.14159", "200",
        "-3.14159", "3.14159", "200"
    );
    ms->state.transition(MICRO, {
        {"ztheta", "-3.14159"},
        {"zradius", ".75"},
    });
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("the solutions exchange places!"), SilenceBlock(1)), 1);
    ms->state.transition(MICRO, {
        {"ztheta", "1"},
        {"zradius", ".75"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let me artificially stretch out that surface so it's clear what's going on..."), 1);
    ms->state.transition(MICRO, {
        {"q1", ".92387953251"},
        {"qi", ".38268343236"},
        {"qj", "0"},
        {"qk", "0"},
        {"ztheta", "0"},
        {"manifold0_x", "(u) (v) 2 / cos *"},
        {"manifold0_y", "0"},
        {"manifold0_z", "(u) (v) 2 / sin *"},
        {"axes_length", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.3), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    ms->state.transition(MICRO, {
        {"ztheta", to_string(3.14159 * 2)},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.3), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The behavior is just like our polynomial solutions!"), 1);
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("A loop of the input yields a swap of the two outputs."), 1);
    ms->state.transition(MICRO, {
        {"ztheta", to_string(3.14159 * 4)},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    ms->state.transition(MICRO, {
        {"ztheta", to_string(3.14159 * 6)},
    });
    cs.render_microblock();
    ms->state.set("ztheta", "0");

    cs.stage_macroblock(SilenceBlock(2), 1);
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cps->state.transition(MICRO, {
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
    cps->state.transition(MICRO, {
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "0"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "1"},
        {"coefficient0_i", "0"},
    });
    cs.render_microblock();
    cs.remove_subscene("quadratic_formula");

    /*shared_ptr<LatexScene> recap = make_shared<LatexScene>("\\begin{tabular}{ccc} \\text{\\huge{Degree}} & \\qquad \\qquad \\text{\\huge{Form}} \\qquad \\qquad & \\text{\\huge{Solutions}}", 1, 1, 1);
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
    cs.render_microblock();*/
}

void part_3(CompositeScene& cs, shared_ptr<ComplexPlotScene> cps, shared_ptr<ManifoldScene> ms) {
    cs.add_scene(cps, "cps");
    shared_ptr<ThreeDimensionScene> title3 = get_part_3_title();
    cs.stage_macroblock(CompositeBlock(SilenceBlock(2), FileBlock("Cubic polynomials are next.")), 3);
    cps->set_degree(3);
    cps->roots_to_coefficients();
    cps->state.transition(MACRO, {
        {"coefficient3_r", "{t} .2 * sin"},
        {"coefficient3_i", "{t} .3 * cos"},
        {"coefficient2_r", "{t} .49 * cos"},
        {"coefficient2_i", "{t} .23 * sin"},
        {"coefficient1_r", "{t} .33 * cos"},
        {"coefficient1_i", "{t} .27 * sin"},
        {"coefficient0_r", "{t} .54 * cos"},
        {"coefficient0_i", "{t} .47 * sin"},
    });
    cs.add_scene_fade_in(MICRO, title3, "title3", .5, .4);
    cs.fade_subscene(MICRO, "cps", .3);
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "title3", 0);
    cs.fade_subscene(MICRO, "cps", 1);
    cs.render_microblock();
    cs.remove_subscene("title3");

    cs.stage_macroblock(CompositeBlock(SilenceBlock(1), FileBlock("If there's a cubic formula, it also can't be written with simple operators, avoiding square roots.")), 2);
    cps->state.transition(MACRO, {
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
    cps->state.transition(MICRO, {
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
    cps->state.transition(MACRO, {
        {"ab_dilation", ".5"},
        {"dot_radius", ".5"},
        {"center_x", "2"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("First of all, a square root yields 2 values, but we need a function that gives 3."), 8);
    ms->state.begin_timer("cbrt_rotation");
    ms->state.transition(MACRO, {
        {"q1", "1"},
        {"qi", ".05"},
        {"qj", "<cbrt_rotation> .05 * sin"},
        {"qk", "0"},
        {"ztheta", "2"},
        {"manifold0_x", "(u) (v) cos *"},
        {"manifold0_y", "(v) 2 / sin (u) .5 ^ * -0.5 *"},
        {"manifold0_z", "(u) (v) sin *"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 1);
    cs.render_microblock();
    cps->transition_root_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(.8), FileBlock("In that case, how about a cube root?")), 2);
    cs.render_microblock();
    ms->state.transition(MICRO, {
        {"manifold0_x", "(u) (v) cos *"},
        {"manifold0_y", "(u) .333333 ^ (v) 3 / sin *"},
        {"manifold0_z", "(u) (v) sin *"},
        {"manifold0_r", "<zradius> <ztheta> cos * (u) 3 ^ (v) cos * -"},
        {"manifold0_i", "<zradius> <ztheta> sin * (u) 3 ^ (v) sin * -"},
        {"manifold0_v_min", to_string(-3 * M_PI)},
        {"manifold0_v_max", to_string(3 * M_PI)},
        {"manifold0_v_steps", "13500"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The associated Riemann surface has 3 sheets, and the cube root function is continuous on it."), 2);
    ms->state.transition(MICRO, "ztheta", to_string(3.14159 * 2.75));
    cs.render_microblock();
    ms->state.transition(MICRO, "ztheta", to_string(3.14159 * .75));
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    ms->state.transition(MICRO, "ztheta", to_string(3.14159 * 2.75));
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Stretching this manifold out so it's easy to see,"), 1);
    ms->state.transition(MICRO, {
        {"ztheta", "pi 2 *"},
        {"q1", ".92387953251"},
        {"qi", ".38268343236"},
        {"qj", "0"},
        {"qk", "0"},
        {"manifold0_x", "(u) (v) 3 / cos *"},
        {"manifold0_y", "0"},
        {"manifold0_z", "(u) (v) 3 / sin *"},
        {"manifold0_v_steps", "11000"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("we can cycle the 3 roots like this..."), 1);
    ms->state.transition(MICRO, {
        {"ztheta", to_string(3.14159 * 4)},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and we can move some coefficients to yield the same behavior on the solutions..."), 2);
    cps->state.transition(MICRO, {
        {"coefficient0_ring", "1"},
        {"coefficient1_ring", "1"},
    });
    cs.render_microblock();
    cps->coefficients_to_roots();
    cps->state.transition(MICRO, {
        {"root0_r", cps->state.get_equation("root1_r")},
        {"root0_i", cps->state.get_equation("root1_i")},
        {"root1_r", cps->state.get_equation("root2_r")},
        {"root1_i", cps->state.get_equation("root2_i")},
        {"root2_r", cps->state.get_equation("root0_r")},
        {"root2_i", cps->state.get_equation("root0_i")},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("That's a good sign, right?"), 1);
    cps->state.transition(MICRO, {
        {"coefficient0_ring", "0"},
        {"coefficient1_ring", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(3), FileBlock("Well, not so fast.")), 3);
    ms->state.transition(MICRO, "ztheta", to_string(3.14159 * 6));
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"root0_r", cps->state.get_equation("root1_r")},
        {"root0_i", cps->state.get_equation("root1_i")},
        {"root1_r", cps->state.get_equation("root2_r")},
        {"root1_i", cps->state.get_equation("root2_i")},
        {"root2_r", cps->state.get_equation("root0_r")},
        {"root2_i", cps->state.get_equation("root0_i")},
    });
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("As we already saw, we can also swap two solutions on the polynomial..."), 2);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but the cube root is only capable of cycling all three."), 2);
    ms->state.transition(MICRO, "ztheta", to_string(3.14159 * 2));
    ms->state.transition(MICRO, "zradius", ".45");
    cs.render_microblock();
    ms->state.transition(MICRO, "ztheta", to_string(3.14159 * 4));
    ms->state.transition(MICRO, "zradius", ".75");
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("Maybe we need a square root too, just to permit these swaps like before?"), SilenceBlock(1)), 2);
    cps->transition_coefficient_opacities(MICRO, 0);
    ms->state.set({
        {"xshift", "0"},
        {"yshift", "0"},
    });
    ms->state.set({
        {"manifold0_x", "(u) (v) 3 / cos * <xshift> +"},
        {"manifold0_y", "<yshift>"},
        {"manifoldcbrt_in_x", point_x_start + " <xshift> +"},
        {"manifoldcbrt_in_y", point_y_start + " <yshift> +"},
    });
    ms->add_manifold("sqrt",
        "(u) (v) 2 / cos * <xshift> -", "0.1", "(u) (v) 2 / sin * .01 +",
        "<sqrt_in_radius> <sqrt_in_theta> cos * (u) 2 ^ (v) cos * -", "<sqrt_in_radius> <sqrt_in_theta> sin * (u) 2 ^ (v) sin * -",
        "0", "1.5", "3000",
        "-6.28318", "6.28318", "11000"
    );
    ms->add_manifold("sqrt_in",
        point_x_start_sqrt + " <xshift> -", point_y_start + " <yshift> -", point_z_start_sqrt,
        "0.00001", "0.00001",
        "0", "3.14159", "200",
        "-3.14159", "3.14159", "200"
    );
    ms->state.set({
        {"sqrt_in_radius", "<zradius>"},
        {"sqrt_in_theta", "<ztheta>"},
    });
    ms->state.transition(MICRO, "axes_length", "0");
    cs.render_microblock();
    cs.slide_subscene(MICRO, "ms", -.25, 0);
    ms->state.transition(MICRO, {
        {"xshift", "2"},
        {"d", "7"},
        {"w", "1"},
        {"manifoldsqrt_y", "<yshift> -1 *"},
    });
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(1), FileBlock("To make matters worse, I'm gonna pick up those same coefficients again,")), 2);
    cs.slide_subscene(MICRO, "ms", 0, 1);
    StateSet reset = {
        {"root0_r", "0"},
        {"root0_i", "0.3"},
        {"root1_r", "1"},
        {"root1_i", "0.3"},
        {"root2_r", "2"},
        {"root2_i", "0.3"},
    };
    cps->state.transition(MICRO, reset);
    cps->state.transition(MICRO, "center_x", "1");
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"coefficient0_ring", "1"},
        {"coefficient1_ring", "1"},
        {"coefficient2_ring", "0"},
        {"coefficient3_ring", "0"},
        {"coefficient0_opacity", "1"},
        {"coefficient1_opacity", "1"},
        {"coefficient2_opacity", "0"},
        {"coefficient3_opacity", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and this time make a very special sequence of 4 loops."), 1);
    cps->transition_coefficient_rings(MICRO, 0);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("First, we move them in a way that swaps the left two solutions in a clockwise rotation,"), 1);
    cps->stage_swap(MICRO, "0", "1", false, true);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("then we swap the right two clockwise,"), 1);
    cps->stage_swap(MICRO, "2", "0", false, true);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("now swap the first two backwards,"), 1);
    cps->stage_swap(MICRO, "1", "2", false);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and un-swap the right two backwards."), 1);
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.5), 1);
    cs.render_microblock();

    shared_ptr<CoordinateSceneWithTrail> trail1 = make_shared<CoordinateSceneWithTrail>();
    shared_ptr<CoordinateSceneWithTrail> trail2 = make_shared<CoordinateSceneWithTrail>();
    trail1->state.set({
        {"trail_x", "{cps.coefficient0_r} 1 -"},
        {"trail_y", "{cps.coefficient0_i}"},
    });
    trail2->state.set({
        {"trail_x", "{cps.coefficient1_r} 1 -"},
        {"trail_y", "{cps.coefficient1_i}"},
    });
    cs.add_scene(trail1, "trail1");
    cs.add_scene(trail2, "trail2");
    cs.state.set({
        {"trail1.opacity", ".3"},
        {"trail2.opacity", ".3"},
    });

    cps->state.set(reset);
    cs.stage_macroblock(FileBlock("We did two things,"), 2);
    cps->stage_swap(MICRO, "0", "1", false, true);
    trail1->trail_color = 0xffff0000;
    trail2->trail_color = 0xffffff00;
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "0", false, true);
    trail1->trail_color = 0xff00ff00;
    trail2->trail_color = 0xff0000ff;
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("then undid them both."), 2);
    cps->stage_swap(MICRO, "1", "2", false);
    trail1->trail_color = 0xffff0000;
    trail2->trail_color = 0xffffff00;
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    trail1->trail_color = 0xff00ff00;
    trail2->trail_color = 0xff0000ff;
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.5), 1);
    cs.fade_subscene(MICRO, "trail1", 0);
    cs.fade_subscene(MICRO, "trail2", 0);
    cs.render_microblock();
    cs.remove_subscene("trail1");
    cs.remove_subscene("trail2");

    cps->state.set(reset);
    cps->state.set({
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
    cps->stage_swap(MICRO, "1", "2", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The solutions changed places, even though each loop we did was later un-done!"), 14);
    for(int i = 0; i < 7; i++) {
        string opacity = (i % 2 == 0) ? ".2" : "1";
        if(i == 6) opacity = "0";
        cps->state.transition(MICRO, "construction_opacity", opacity);
        cs.render_microblock();
        cs.render_microblock();
    }
    cps->construction.clear();

    cps->state.set(reset);
    //shared_ptr<LatexScene> commutator = make_shared<LatexScene>("x^{-1} \\phantom{y^{-1} x y}", 1, .5, .35);
    cs.stage_macroblock(FileBlock("This special way of cycling objects is called a commutator."), 8);
    cps->stage_swap(MICRO, "0", "1", false, true);
    //cs.add_scene_fade_in(MICRO, commutator, "commutator", .5, .2);
    cs.render_microblock();
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "0", false, true);
    //commutator->begin_latex_transition(MICRO, "x^{-1} y^{-1} \\phantom{x y}");
    cs.render_microblock();
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "2", false);
    //commutator->begin_latex_transition(MICRO, "x^{-1} y^{-1} x \\phantom{y}");
    cs.render_microblock();
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    //commutator->begin_latex_transition(MICRO, "x^{-1} y^{-1} x y");
    cs.render_microblock();
    cs.render_microblock();

    cps->state.transition(MICRO, "construction_opacity", "0");
    cs.stage_macroblock(FileBlock("But watch what happens if we make commutator loops on the input of the cube root function..."), 1);
    cps->transition_coefficient_opacities(MICRO, 0);
    cps->transition_coefficient_rings(MICRO, 0);
    cs.fade_subscene(MICRO, "commutator", 0);
    // TODO slide the two manifolds in one by one
    // TODO label solutions
    ms->state.set({
        {"loop1", "0"},
        {"loop2", "0"},
    });
    ms->state.set({
        {"sqrt_in_theta", "0"},
        {"sqrt_in_radius", ".7"},
        {"ztheta", "<loop1> 6.283 * <loop2> 6.283 * sin .3 * +"},
        {"zradius", "<loop2> 6.283 * cos .2 * .5 +"},
    });
    cs.slide_subscene(MICRO, "ms", 0, -1);
    cs.render_microblock();
    cs.remove_subscene("commutator");
    cps->construction.clear();
    cps->state.set("construction_opacity", "1");

    cs.stage_macroblock(FileBlock("Loop, loop, back, back..."), 5);
    ms->state.transition(MICRO, "loop1", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop2", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop1", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop2", "0");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("the solutions returned to where they started..."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(3), 5);
    ms->state.transition(MICRO, "loop1", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop2", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop1", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop2", "0");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("Let's try a different commutator on the square root..."), SilenceBlock(3)), 4);
    cs.render_microblock();
    ms->state.set({
        {"sqrt_in_theta", "<loop3> 6.283 * sin .2 * <loop4> 6.283 * +"},
        {"sqrt_in_radius", "<loop4> 6.283 * sin <loop3> 6.283 * cos + -.2 * .9 +"},
        {"loop3", "0"},
        {"loop4", "0"},
    });
    ms->state.transition(MICRO, "loop3", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop4", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop3", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop4", "0");
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("No matter what commutator the input follows, or which root we try it on, each output always spins back to where it was before the commutator."), 7);
    cs.render_microblock();
    ms->state.transition(MICRO, "loop1", "1");
    ms->state.transition(MICRO, "loop3", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop2", "1");
    ms->state.transition(MICRO, "loop4", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop1", "0");
    ms->state.transition(MICRO, "loop3", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop2", "0");
    ms->state.transition(MICRO, "loop4", "0");
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(1), FileBlock("In general, the roots do a cycle whenever our loop circles around the origin,")), 1);
    ms->state.set({
        {"ztheta", "<loop1> 6.283 * <loop2> 6.283 * sin .1 * +"},
        {"zradius", "<loop2> 6.283 * cos .2 * .5 +"},
        {"sqrt_in_theta", "<loop3> 6.283 * <loop4> 6.283 * sin .1 * +"},
        {"sqrt_in_radius", "<loop4> 6.283 * cos .2 * .5 +"},
    });
    ms->state.transition(MICRO, "loop1", "1");
    ms->state.transition(MICRO, "loop3", "1");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and don't do anything at all otherwise."), 1);
    ms->state.transition(MICRO, "loop2", "1");
    ms->state.transition(MICRO, "loop4", "1");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("but the commutator unwinds any accumulated rotation around the origin in the second half."), 2);
    ms->state.transition(MICRO, "loop1", "0");
    ms->state.transition(MICRO, "loop3", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop2", "0");
    ms->state.transition(MICRO, "loop4", "0");
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_subscene(MICRO, "ms", .1);
    cs.render_microblock();

    cps->state.set(reset);
    cs.stage_macroblock(FileBlock("The cubic polynomial can express permutations..."), 4);
    cps->coefficients_to_roots();
    cps->stage_swap(MICRO, "0", "1", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "0", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "2", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("that the square root and cube root are unable to emulate!"), SilenceBlock(1)), 5);
    cs.fade_subscene(MICRO, "ms", 1);
    cs.render_microblock();
    ms->state.transition(MICRO, "loop1", "1");
    ms->state.transition(MICRO, "loop3", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop2", "1");
    ms->state.transition(MICRO, "loop4", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop1", "0");
    ms->state.transition(MICRO, "loop3", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "loop2", "0");
    ms->state.transition(MICRO, "loop4", "0");
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("What kind of operator could be more expressive?"), SilenceBlock(1)), 1);
    cps->coefficients_to_roots();
    cps->state.transition(MICRO, {
        {"root0_r", "-1"},
        {"root0_i", "-4"},
        {"root1_r", "0"},
        {"root1_i", "-4"},
        {"root2_r", "1"},
        {"root2_i", "-4"},
    });
    ms->state.transition(MICRO, "axes_length", "0");
    cs.render_microblock();
}

void part_3p5(CompositeScene& cs, shared_ptr<ComplexPlotScene> cps, shared_ptr<ManifoldScene> ms) {
    cs.add_scene(cps, "cps");
    cs.add_scene(ms, "ms");
    ms->state.transition(MICRO, {
        {"manifold0_x", "(u) (v) 3 / cos * <xshift> +"},
        {"manifold0_y", "<yshift>"},
        {"manifold0_z", "(u) (v) 3 / sin *"},
        {"manifoldsqrt_x", "(u) (v) 2 / cos * <xshift> -"},
        {"manifoldsqrt_y", "<yshift> -1 *"},
        {"manifoldsqrt_z", "(u) (v) 2 / sin *"},
    });
    cs.stage_macroblock(FileBlock("Imagine a pole which you need to circle around to cause the outputs to cycle."), 4);
    ms->state.transition(MACRO, {
        {"q1", "1"},
        {"qi", ".1"},
        {"qj", "0"},
        {"qk", "0"},
    });
    cs.render_microblock();
    ms->state.set("scale_sqrtpole", "0");
    ms->state.set("scale_cbrtpole", "0");
    ms->add_manifold("sqrtpole",
        "(u) cos .02 * <scale_sqrtpole> * <xshift> -", "(v) <scale_sqrtpole> * <yshift> -", "(u) sin .02 * <scale_sqrtpole> *",
        pole_r, pole_i,
        "0", "6.28318", "400",
        "-.4", ".4", "400"
    );
    ms->add_manifold("cbrtpole",
        "(u) cos .02 * <scale_cbrtpole> * <xshift> +", "(v) <scale_cbrtpole> * <yshift> +", "(u) sin .02 * <scale_cbrtpole> *",
        pole_r, pole_i,
        "0", "6.28318", "400",
        "-.4", ".4", "400"
    );
    ms->state.transition(MICRO, "scale_sqrtpole", "1");
    ms->state.transition(MICRO, "scale_cbrtpole", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "sqrt_in_theta", "2 pi *");
    cs.render_microblock();
    ms->state.transition(MICRO, "ztheta", "2 pi *");
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We're gonna use it to construct a machine that has the same behavior as our cubic polynomial."), 1);
    ms->state.transition(MICRO, {
        {"q1", "1"},
        {"qi", ".1 {t} sin .01 * +"},
        {"qj", "{t} cos .01 *"},
        {"z", "-.1"},
        {"d", "4"},
        {"fov", ".5"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's take our square root function's output,"), 1);
    ms->state.transition(MICRO, {
        {"qi", "{t} sin .01 *"},
        {"yshift", ".5"},
        {"xshift", "0"},
        {"d", "2.2"},
    });
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("and use _that_ as the input of this cube root."), SilenceBlock(2)), 2);
    ms->state.set({
        {"sqrt_out_radius", "<sqrt_in_radius> .5 ^"},
        {"sqrt_out_theta", "<sqrt_in_theta> 2 /"},
        {"cbrt_in_real", "<sqrt_out_radius> <sqrt_out_theta> cos * <xshift> 2 * -"},
        {"cbrt_in_imag", "<sqrt_out_radius> <sqrt_out_theta> sin *"},
    });
    ms->add_manifold("tie",
        "<sqrt_out_radius> <sqrt_out_theta> cos * <xshift> -", "<yshift> -1 *", "<sqrt_out_radius> <sqrt_out_theta> sin *",
        "0.00001", "0.00001",
        "0", "6.28318", "100",
        "-1", "1", "600"
    );
    ms->state.transition(MACRO, "manifoldtie_y", "(v) <yshift> *");
    cs.render_microblock();
    ms->state.transition(MICRO, {
        {"ztheta", "<cbrt_in_imag> <cbrt_in_real> atan2"},
        {"zradius", "<cbrt_in_real> 2 ^ <cbrt_in_imag> 2 ^ + .5 ^"},
    });
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(.8), FileBlock("This still isn't enough to implement a functional commutator,")), 1);
    ms->state.transition(MICRO, "sqrt_in_theta", "12");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("because we're just accumulating and dissipating rotation between the two halves of the commutator."), 1);
    ms->state.transition(MICRO, "sqrt_in_theta", "-5");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("But here's the trick:"), 1);
    ms->state.transition(MICRO, "sqrt_in_theta", "-2");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("we skew the square root and cube root functions away from each other like this."), 1);
    ms->state.transition(MICRO, "xshift", ".25");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Now the two poles don't line up anymore."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Ready for the commutator?"), 1);
    ms->state.set({
        {"twist1", "0"},
        {"twist2", "0"},
    });
    ms->state.transition(MICRO,  {
        {"sqrt_in_theta", "<twist1> 6.283 * <twist2> 6.283 * sin +"},
        {"sqrt_in_radius", "<twist2> 6.283 * cos -.25 * .4 +"},
    });
    cs.render_microblock();

    // TODO highlight outputs somehow
    cs.stage_macroblock(FileBlock("Pay attention to the 3 outputs here."), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Loop, loop, undo, undo..."), 5);
    ms->state.transition(MICRO, "twist1", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist2", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist1", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist2", "0");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The three outputs did a cycle!"), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Let's try that again. This time, watch the bar tying them together."), 1);
    // TODO highlight the bar
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("One, two, undo, undo..."), 4);
    ms->state.transition(MICRO, "twist1", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist2", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist1", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist2", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("See how it only went around the bottom pole once?"), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(4), 4);
    ms->state.transition(MICRO, "twist1", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist2", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist1", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist2", "0");
    cs.render_microblock();

    shared_ptr<LatexScene> cubic_formula = make_shared<LatexScene>("\\sqrt[3]{\\phantom{\\phantom{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right) +} \\sqrt{\\phantom{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right)^2 + \\left(\\frac{c}{3a} - \\frac{b^2}{9a^2}\\right)^3}}}}", 1);
    cs.add_scene_fade_in(MICRO, cubic_formula, "cubic_formula");
    cs.fade_subscene(MICRO, "ms", .5);
    cs.stage_macroblock(FileBlock("This means, in order to support commutator cycles, we need to use a root as input to another root,"), 3);
    cs.render_microblock();
    cs.render_microblock();
    cubic_formula->begin_latex_transition(MICRO, "\\sqrt[3]{\\phantom{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right) +} \\sqrt{\\phantom{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right)^2 + \\left(\\frac{c}{3a} - \\frac{b^2}{9a^2}\\right)^3}}}");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("along with a term added to 'skew' their two poles apart."), 3);
    cs.render_microblock();
    cubic_formula->begin_latex_transition(MICRO, "\\sqrt[3]{\\phantom{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right)} + \\sqrt{\\phantom{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right)^2 + \\left(\\frac{c}{3a} - \\frac{b^2}{9a^2}\\right)^3}}}");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("And sure enough, this is exactly the tool used in the cubic formula."), 5);
    cs.fade_subscene(MICRO, "cps", 0);
    cs.render_microblock();
    cubic_formula->begin_latex_transition(MICRO, "\\sqrt[3]{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right) + \\sqrt{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right)^2 + \\left(\\frac{c}{3a} - \\frac{b^2}{9a^2}\\right)^3}}");
    cs.render_microblock();
    cs.render_microblock();
    cubic_formula->begin_latex_transition(MICRO, "\\sqrt[3]{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right) + \\sqrt{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right)^2 + \\left(\\frac{c}{3a} - \\frac{b^2}{9a^2}\\right)^3}} \\\\\\\\ + \\sqrt[3]{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right) - \\sqrt{\\left(\\frac{-b^3}{27a^3} + \\frac{bc}{6a^2} - \\frac{d}{2a}\\right)^2 + \\left(\\frac{c}{3a} - \\frac{b^2}{9a^2}\\right)^3}} \\\\\\\\ - \\frac{b}{3a}");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(3), 1);
    cps->set_degree(4);
    cps->state.set("coefficient4_r", "1");
    cs.fade_subscene(MICRO, "cubic_formula", 0);
    cs.fade_subscene(MICRO, "ms", 0);
    cs.fade_subscene(MICRO, "cps", 1);
    cs.render_microblock();
}

void part_4(CompositeScene& cs, shared_ptr<ComplexPlotScene> cps, shared_ptr<ManifoldScene> ms) {
    cs.stage_macroblock(CompositeBlock(SilenceBlock(2), FileBlock("Quartics have only a slightly harder trick up their sleeve.")), 3);
    cs.add_scene(cps, "cps");
    cps->coefficients_to_roots();
    cps->state.transition(MACRO, {
        {"root0_r", "-1.5"},
        {"root0_i", "0"},
        {"root1_r", "-.5"},
        {"root1_i", "0"},
        {"root2_r", ".5"},
        {"root2_i", "0"},
        {"root3_r", "1.5"},
        {"root3_i", "0"},
        {"center_x", "0"},
        {"center_y", "0"},
        {"ab_dilation", "1.5"},
        {"dot_radius", "1.5"},
    });
    shared_ptr<ThreeDimensionScene> title4 = get_part_4_title();
    cs.add_scene_fade_in(MICRO, title4, "title4", .5, .4);
    cs.fade_subscene(MICRO, "cps", .3);
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "title4", 0);
    cs.fade_subscene(MICRO, "cps", 1);
    cs.render_microblock();
    cs.remove_subscene("title4");

    cps->state.set({
        {"point_1_x", "<root0_r>"},
        {"point_1_y", "<root0_i>"},
        {"point_2_x", "<root1_r>"},
        {"point_2_y", "<root1_i>"},
        {"point_3_x", "<root2_r>"},
        {"point_3_y", "<root2_i>"},
        {"point_4_x", "<root3_r>"},
        {"point_4_y", "<root3_i>"},
    });
    cs.stage_macroblock(FileBlock("Watch me do a nested commutator on the roots."), 4);
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "1", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "2", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "3", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "4", .7, true));
    cs.render_microblock();

    // Nested Commutator on S4
    cs.stage_macroblock(FileBlock("Here's the first commutator,"), 5);
    cps->stage_swap(MICRO, "0", "1", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "0", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "2", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("here's the second,"), 5);
    cps->stage_swap(MICRO, "3", "1", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "3", "0", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "0", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "3", false);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("here's the first one getting undone,"), 5);
    cps->stage_swap(MICRO, "1", "3", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "3", "2", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "2", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "3", "1", false);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("and the second one getting undone."), 5);
    cps->stage_swap(MICRO, "3", "2", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "3", "0", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "0", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "3", false);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(SilenceBlock(1), FileBlock("This nested commutator has scrambled up the roots, just like a single commutator scrambled up 3 roots.")), 11);
    for(int i = 0; i < 11; i++) {
        cps->state.transition(MICRO, "construction_opacity", (i%2 == 0) ? "0" : "1");
        cs.render_microblock();
    }

    cs.stage_macroblock(CompositeBlock(FileBlock("They started like this."), SilenceBlock(1.5)), 2);
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "3", false, true);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("If we do any nested commutator on the machine we built earlier..."), 2);
    cps->construction.clear();
    ms->state.set({
        {"d", "4"},
    });
    cps->state.remove({
        "point_1_x",
        "point_1_y",
        "point_2_x",
        "point_2_y",
        "point_3_x",
        "point_3_y",
        "point_4_x",
        "point_4_y",
    });
    cs.add_scene_fade_in(MICRO, ms, "ms");
    cps->state.transition(MICRO, "center_y", "4");
    cps->state.transition(MICRO, {
        {"ab_dilation", "1.5"},
        {"dot_radius", "2"},
    });
    ms->state.set({
        {"twist1", "0"},
        {"twist2", "0"},
        {"twist3", "0"},
        {"twist4", "0"},
        {"sqrt_in_theta", "<twist1> 6.283 * <twist2> 6.283 * cos .4 * <twist3> 6.283 * <twist4> 6.283 * cos -.4 * + + +"},
        {"sqrt_in_radius", "<twist2> 6.283 * sin -.25 * <twist3> 6.283 * sin .2 * <twist4> 6.283 * sin -.2 * .4 + + +"},
    });
    cs.render_microblock();
    ms->state.transition(MICRO, {
        {"d", "2.2"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(10), 20);
    ms->state.transition(MICRO, "twist1", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist2", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist1", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist2", "0");
    cs.render_microblock();
    cs.render_microblock();
    ms->state.transition(MICRO, "twist3", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist4", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist3", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist4", "0");
    cs.render_microblock();
    cs.render_microblock();
    ms->state.transition(MICRO, "twist2", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist1", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist2", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist1", "0");
    cs.render_microblock();
    cs.render_microblock();
    ms->state.transition(MICRO, "twist4", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist3", "1");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist4", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "twist3", "0");
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("everything goes back to where it started!"), 1);
    // TODO track roots
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("The way to work around this is actually comically simple. Any guesses?"), 1);
    ms->state.transition(MICRO, {
        {"d", "3.5"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("Just add another root to the chain!"), 1);
    ms->state.set({
        {"qrt_in_theta", "0"},
        {"qrt_in_radius", ".5"},
        {"scale_qrtpole", "1"},
        {"manifoldqrt_in_x", point_x_start + " <xshift> 3 * +"},
        {"manifoldqrt_in_y", point_y_start + " <yshift> 3 * +"},
    });
    ms->add_manifold("qrt",
        "(u) (v) 4 / cos * <xshift> 3 * +", "30", "(u) (v) 4 / sin * .01 +",
        "<qrt_in_radius> <qrt_in_theta> cos * (u) 4 ^ (v) cos * -", "<qrt_in_radius> <qrt_in_theta> sin * (u) 4 ^ (v) sin * -",
        "0", "1.5", "3000",
        "-12.56637", "12.56637", "11000"
    );
    ms->add_manifold("qrt_in",
        point_x_start_qrt + " <xshift> 3 * +", point_y_start + " <yshift> 3 * +", point_z_start_qrt,
        "0.00001", "0.00001",
        "0", "3.14159", "200",
        "-3.14159", "3.14159", "200"
    );
    ms->add_manifold("qrtpole",
        "(u) cos .02 * <scale_qrtpole> * <xshift> 3 * +", "30", "(u) sin .02 * <scale_qrtpole> *",
        pole_r, pole_i,
        "0", "6.28318", "400",
        "-.4", ".4", "400"
    );
    ms->add_manifold("tie2",
        "<cbrt_out_radius> <cbrt_out_theta> cos * <cbrt_center_x> +", "(v) <cbrt_center_y> * <qrt_center_y> +", "<cbrt_out_radius> <cbrt_out_theta> sin *",
        "0.00001", "0.00001",
        "0", "6.28318", "100",
        "-1", "1", "600"
    );
    ms->state.set({
        {"hide_sqrt", "0"},
        {"hide_cbrt", "0"},
        {"hide_qrt", "0"},
        {"sqrt_center_x", "<xshift> -2 *"},
        {"sqrt_center_y", "<yshift> -2 * <hide_sqrt> 6 * +"},
        {"cbrt_center_x", "0"},
        {"cbrt_center_y", "<hide_cbrt> 6 *"},
        {"qrt_center_x", "<xshift> 2 *"},
        {"qrt_center_y", "<yshift> 2 * <hide_qrt> 6 * +"},
        {"cbrt_in_x", "<sqrt_out_radius> <sqrt_out_theta> cos * <xshift> +"},
        {"cbrt_in_y", "<sqrt_out_radius> <sqrt_out_theta> sin *"},
        {"cbrt_out_radius", "<cbrt_in_x> 2 ^ <cbrt_in_y> 2 ^ + .5 ^"},
        {"cbrt_out_theta", "<cbrt_in_y> <cbrt_in_x> atan2"},
    });
    ms->state.transition(MICRO, {
        {"d", "4"},

        {"manifoldsqrt_x", "(u) (v) 2 / cos * <sqrt_center_x> +"},
        {"manifoldsqrt_y", "<sqrt_center_y>"},
        {"manifoldsqrt_in_x", point_x_start_sqrt + " <sqrt_center_x> +"},
        {"manifoldsqrt_in_y", point_y_start + " <sqrt_center_y> +"},
        {"manifoldsqrtpole_x", "(u) cos .02 * <scale_sqrtpole> * <sqrt_center_x> +"},
        {"manifoldsqrtpole_y", "(v) <scale_sqrtpole> * <sqrt_center_y> +"},

        {"manifold0_x", "(u) (v) 3 / cos * <cbrt_center_x> +"},
        {"manifold0_y", "<cbrt_center_y>"},
        {"manifoldcbrt_in_x", point_x_start + " <cbrt_center_x> +"},
        {"manifoldcbrt_in_y", point_y_start + " <cbrt_center_y> +"},
        {"manifoldcbrtpole_x", "(u) cos .02 * <scale_cbrtpole> * <cbrt_center_x> +"},
        {"manifoldcbrtpole_y", "(v) <scale_cbrtpole> * <cbrt_center_y> +"},

        {"manifoldqrt_x", "(u) (v) 4 / cos * <qrt_center_x> +"},
        {"manifoldqrt_y", "<qrt_center_y>"},
        {"manifoldqrt_in_x", point_x_start_qrt + " <qrt_center_x> +"},
        {"manifoldqrt_in_y", point_y_start + " <qrt_center_y> +"},
        {"manifoldqrtpole_x", "(u) cos .02 * <scale_qrtpole> * <qrt_center_x> +"},
        {"manifoldqrtpole_y", "(v) <scale_qrtpole> * <qrt_center_y> +"},

        {"manifoldtie_x", "<sqrt_out_radius> <sqrt_out_theta> cos * <sqrt_center_x> +"},
        {"manifoldtie_y", "(v) <sqrt_center_y> * <cbrt_center_y> +"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("This trend continues inductively- each time we add a root, we can support another level of nested commutators."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    ms->state.transition(MICRO, {
        {"hide_sqrt", "1"},
        {"hide_cbrt", "1"},
        {"hide_qrt", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("With one square root operator, we can represent a single swap on 2 solutions."), 2);
    ms->state.transition(MICRO, "hide_sqrt", "0");
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"root0_r", "-1"},
        {"root0_i", "0"},
        {"root1_r", "1"},
        {"root1_i", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("With two nested root operators, we can represent a simple commutator on 3 solutions."), 2);
    ms->state.transition(MICRO, "hide_cbrt", "0");
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"root0_r", "-2"},
        {"root0_i", "0"},
        {"root1_r", "0"},
        {"root1_i", "0"},
        {"root2_r", "2"},
        {"root2_i", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("With three nested root operators, we can represent a commutator of commutators on 4 solutions."), 2);
    ms->state.transition(MICRO, "hide_qrt", "0");
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"root0_r", "-3"},
        {"root0_i", "0"},
        {"root1_r", "-1"},
        {"root1_i", "0"},
        {"root2_r", "1"},
        {"root2_i", "0"},
        {"root3_r", "3"},
        {"root3_i", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("And so on successively."), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("So, why can't we represent actions on quintic polynomials' 5 solutions, just by adding another root?"), 1);
    cs.render_microblock();

    // TODO fifth title scene here
    cs.fade_subscene(MICRO, "ms", 0);
    cs.stage_macroblock(FileBlock("Here's the key to the proof, the source of the impossibility:"), 1);
    cps->roots_to_coefficients();
    cps->construction.clear();
    cps->state.transition(MICRO, {
        {"dot_radius", "1"},
        {"ab_dilation", "1"},
        {"construction_opacity", "1"},
        {"coefficient4_r", ".001"},
        {"coefficient4_i", "0"},
        {"coefficient3_r", ".001"},
        {"coefficient3_i", "0"},
    });
    cs.render_microblock();
    cps->set_degree(2);
    cs.remove_all_subscenes();
}

void part_5(CompositeScene& cs, shared_ptr<ComplexPlotScene> cps, shared_ptr<ManifoldScene> ms) {
    cs.stage_macroblock(FileBlock("With two objects, we can make 0th-order commutators- that is, just a single swap, to leave them scrambled."), 2);
    cs.add_scene(cps, "cps");
    cps->set_degree(2);
    cps->roots_to_coefficients();
    cps->state.transition(MICRO, {
        {"coefficient2_r", "1"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "0"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "-1"},
        {"coefficient0_i", "0"},
        {"center_x", "0"},
        {"center_y", "0"},
    });
    cs.render_microblock();
    cps->transition_coefficient_opacities(MICRO, 0);
    cps->coefficients_to_roots();
    cps->stage_swap(MICRO, "0", "1", false, true);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 2);
    cps->state.set({
        {"root0_r", "-1"},
        {"root0_i", "0"},
        {"root1_r", "1"},
        {"root1_i", "0"},
        {"point_1_x", "{cps.root0_r}"},
        {"point_1_y", "{cps.root0_i}"},
        {"point_2_x", "{cps.root1_r}"},
        {"point_2_y", "{cps.root1_i}"},
    });
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "1", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "2", .7, true));
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("A commutator doesn't really make sense here because there's only one swap to choose from."), 4);
    cps->stage_swap(MICRO, "0", "1", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(3), 2);
    cps->state.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();
    cps->construction.clear();
    cps->state.set("construction_opacity", "1");
    cps->set_degree(3);
    cps->roots_to_coefficients();
    cps->state.transition(MICRO, {
        {"ab_dilation", "1.2"},
        {"dot_radius", "2.5"},
        {"coefficient3_r", "1"},
        {"coefficient3_i", "0"},
        {"coefficient2_r", "0"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "-4"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "0"},
        {"coefficient0_i", "0"},
    });
    cs.render_microblock();
    cps->coefficients_to_roots();
    cps->state.set({
        {"root0_r", "-2"},
        {"root0_i", "0"},
        {"root1_r", "0"},
        {"root1_i", "0"},
        {"root2_r", "2"},
        {"root2_i", "0"},
    });

    cps->state.set({
        {"point_3_x", "{cps.root2_r}"},
        {"point_3_y", "{cps.root2_i}"},
    });
    cs.stage_macroblock(CompositeBlock(FileBlock("With three objects, there are commutators which leave them scrambled,"), SilenceBlock(2)), 8);
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "1", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "2", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "3", .7, true));
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "0", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "2", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    cps->state.transition(MICRO, {
        {"root0_r", "-2"},
        {"root0_i", "0"},
        {"root1_r", "0"},
        {"root1_i", "0"},
        {"root2_r", "2"},
        {"root2_i", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("But any commutator of commutators does nothing."), SilenceBlock(1)), 20);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    cps->stage_swap(MICRO, "0", "1", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "0", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "2", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cps->stage_swap(MICRO, "2", "1", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "2", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "0", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "1", false);
    cs.render_microblock();

    cps->stage_swap(MICRO, "1", "2", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "0", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "2", false);
    cs.render_microblock();

    cps->stage_swap(MICRO, "2", "0", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "2", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "0", false);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(3), 5);
    cs.render_microblock();
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "1", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "2", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "3", .7, true));
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(3), 3);
    cps->state.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();
    cps->construction.clear();
    cps->state.set("construction_opacity", "1");
    cps->set_degree(4);
    cps->roots_to_coefficients();
    cps->state.transition(MICRO, {
        {"ab_dilation", "2"},
        {"dot_radius", "3"},
        {"coefficient4_r", "1"},
        {"coefficient4_i", "0"},
        {"coefficient3_r", "0"},
        {"coefficient3_i", "0"},
        {"coefficient2_r", "-10"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "0"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "9"},
        {"coefficient0_i", "0"},
    });
    cs.render_microblock();
    cps->coefficients_to_roots();
    cps->state.set({
        {"root0_r", "-3"},
        {"root0_i", "0"},
        {"root1_r", "-1"},
        {"root1_i", "0"},
        {"root2_r", "1"},
        {"root2_i", "0"},
        {"root3_r", "3"},
        {"root3_i", "0"},
    });
    cs.render_microblock();
    cps->state.set({
        {"point_4_x", "{cps.root3_r}"},
        {"point_4_y", "{cps.root3_i}"},
    });

    cs.stage_macroblock(SilenceBlock(.5), 1);
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("With four objects, there are commutators of commutators which leave them scrambled,"), SilenceBlock(2)), 20);
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "1", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "2", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "3", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "4", .7, true));
    cs.render_microblock();

    cps->stage_swap(MICRO, "0", "1", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "0", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "2", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "0", "1", false);
    cs.render_microblock();

    cps->stage_swap(MICRO, "3", "1", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "3", "0", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "0", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "3", false);
    cs.render_microblock();

    cps->stage_swap(MICRO, "1", "3", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "3", "2", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "1", "2", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "3", "1", false);
    cs.render_microblock();

    cps->stage_swap(MICRO, "3", "2", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "3", "0", false, true);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "0", false);
    cs.render_microblock();
    cps->stage_swap(MICRO, "2", "3", false);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(4), 6);
    cs.render_microblock();
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "1", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "2", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "3", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "4", .7, true));
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1.5), 1);
    cps->stage_swap(MICRO, "0", "1", false, true);
    cps->stage_swap(MICRO, "2", "3", false, true);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(.5), 1);
    // Photosensitivity Safety- decreasing ab_dilation and dot_radius
    cps->state.transition(MICRO, {
        {"ab_dilation", "1"},
        {"dot_radius", "1"},
    });
    cs.render_microblock();

    cs.stage_macroblock(CompositeBlock(FileBlock("But triply-nested commutators do not."), SilenceBlock(5)), 84);
    for(int i = 0; i < 20; i++) {
        cs.render_microblock();
    }
    for(int i = 0; i < 4; i++) {
        cps->stage_swap(MICRO, "0", "1", false, true);
        cs.render_microblock();
        cps->stage_swap(MICRO, "2", "0", false, true);
        cs.render_microblock();
        cps->stage_swap(MICRO, "1", "2", false);
        cs.render_microblock();
        cps->stage_swap(MICRO, "0", "1", false);
        cs.render_microblock();

        cps->stage_swap(MICRO, "3", "1", false, true);
        cs.render_microblock();
        cps->stage_swap(MICRO, "3", "0", false, true);
        cs.render_microblock();
        cps->stage_swap(MICRO, "1", "0", false);
        cs.render_microblock();
        cps->stage_swap(MICRO, "1", "3", false);
        cs.render_microblock();

        cps->stage_swap(MICRO, "1", "3", false, true);
        cs.render_microblock();
        cps->stage_swap(MICRO, "3", "2", false, true);
        cs.render_microblock();
        cps->stage_swap(MICRO, "1", "2", false);
        cs.render_microblock();
        cps->stage_swap(MICRO, "3", "1", false);
        cs.render_microblock();

        cps->stage_swap(MICRO, "3", "2", false, true);
        cs.render_microblock();
        cps->stage_swap(MICRO, "3", "0", false, true);
        cs.render_microblock();
        cps->stage_swap(MICRO, "2", "0", false);
        cs.render_microblock();
        cps->stage_swap(MICRO, "2", "3", false);
        cs.render_microblock();
    }

    cs.stage_macroblock(SilenceBlock(2), 6);
    cs.render_microblock();
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "1", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "2", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "3", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "4", .7, true));
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(3), 3);
    cps->state.transition(MICRO, "construction_opacity", "0");
    cs.render_microblock();
    cps->construction.clear();
    cps->state.set("construction_opacity", "1");
    cps->set_degree(5);
    cps->roots_to_coefficients();
    cps->state.transition(MICRO, {
        {"ab_dilation", ".75"},
        {"dot_radius", "1"},
        {"coefficient5_r", "1"},
        {"coefficient5_i", "0"},
        {"coefficient4_r", "0"},
        {"coefficient4_i", "0"},
        {"coefficient3_r", "0"},
        {"coefficient3_i", "0"},
        {"coefficient2_r", "0"},
        {"coefficient2_i", "0"},
        {"coefficient1_r", "-1"},
        {"coefficient1_i", "0"},
        {"coefficient0_r", "0"},
        {"coefficient0_i", "0"},
    });
    cs.render_microblock();
    cps->coefficients_to_roots();
    StateSet plussign = {
        {"root0_r", "0"},
        {"root0_i", "1"},
        {"root1_r", "-1"},
        {"root1_i", "0"},
        {"root2_r", "0"},
        {"root2_i", "0"},
        {"root3_r", "1"},
        {"root3_i", "0"},
        {"root4_r", "0"},
        {"root4_i", "-1"},
    };
    cps->state.set(plussign);
    cs.render_microblock();
    cps->state.set({
        {"point_5_x", "{cps.root4_r}"},
        {"point_5_y", "{cps.root4_i}"},
    });

    cs.stage_macroblock(FileBlock("5 objects are different."), 5);
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "1", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "2", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "3", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "4", .7, true));
    cs.render_microblock();
    cps->construction.add(GeometricPoint(glm::vec2(0, 0), "5", .7, true));
    cs.render_microblock();

    shared_ptr<CoordinateSceneWithTrail> trail1 = make_shared<CoordinateSceneWithTrail>();
    shared_ptr<CoordinateSceneWithTrail> trail2 = make_shared<CoordinateSceneWithTrail>();
    shared_ptr<CoordinateSceneWithTrail> trail3 = make_shared<CoordinateSceneWithTrail>();
    cs.add_scene(trail1, "trail1");
    cs.add_scene(trail2, "trail2");
    cs.add_scene(trail3, "trail3");
    cs.state.set({{"trail1.opacity", ".5"}, {"trail2.opacity", ".5"}, {"trail3.opacity", ".5"}});
    trail1->trail_color = trail2->trail_color = trail3->trail_color = 0xffff0000;
    trail1->state.set({{"trail_x", "{cps.root1_r}"}, {"trail_y", "{cps.root1_i}"}});
    trail2->state.set({{"trail_x", "{cps.root2_r}"}, {"trail_y", "{cps.root2_i}"}});
    trail3->state.set({{"trail_x", "{cps.root3_r}"}, {"trail_y", "{cps.root3_i}"}});

    cs.stage_macroblock(CompositeBlock(FileBlock("We can take a cycle of 3 like this..."), SilenceBlock(1)), 1);
    cps->roots_to_coefficients();
    StateSet ovals = {
        {"c1", "0"},
        {"c2", "0"},
        {"oval_hor", "0"},
        {"oval_ver", "0"},
        {"coefficient0_r", "<oval_hor> pi 2 * * sin 1 * <oval_ver> pi 4 * * sin .33 * +"},
        {"coefficient0_i", "<oval_ver> pi 2 * * sin 1 * <oval_hor> pi 4 * * sin .33 * +"},
    };
    cps->state.set(ovals);
    cps->state.transition(MICRO, "oval_hor", "1");
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.state.transition(MICRO, {{"trail1.opacity", "0"}, {"trail2.opacity", "0"}, {"trail3.opacity", "0"}});
    cs.render_microblock();
    cs.remove_all_subscenes_except("cps");

    shared_ptr<CoordinateSceneWithTrail> trail4 = make_shared<CoordinateSceneWithTrail>();
    shared_ptr<CoordinateSceneWithTrail> trail5 = make_shared<CoordinateSceneWithTrail>();
    shared_ptr<CoordinateSceneWithTrail> trail6 = make_shared<CoordinateSceneWithTrail>();
    cs.add_scene(trail4, "trail4");
    cs.add_scene(trail5, "trail5");
    cs.add_scene(trail6, "trail6");
    cs.state.set({{"trail4.opacity", ".5"}, {"trail5.opacity", ".5"}, {"trail6.opacity", ".5"}});
    trail4->trail_color = trail5->trail_color = trail6->trail_color = 0xff00ff00;
    trail4->state.set({{"trail_x", "{cps.root0_r}"}, {"trail_y", "{cps.root0_i}"}});
    trail5->state.set({{"trail_x", "{cps.root1_r}"}, {"trail_y", "{cps.root1_i}"}});
    trail6->state.set({{"trail_x", "{cps.root4_r}"}, {"trail_y", "{cps.root4_i}"}});

    cs.stage_macroblock(CompositeBlock(FileBlock("and another cycle of 3 like this..."), SilenceBlock(1)), 1);
    cps->state.transition(MICRO, "oval_ver", "1");
    cs.render_microblock();

    cs.add_scene_fade_in(MICRO, trail1, "trail1", .5, .5, .3);
    cs.add_scene_fade_in(MICRO, trail2, "trail2", .5, .5, .3);
    cs.add_scene_fade_in(MICRO, trail3, "trail3", .5, .5, .3);
    trail1->state.set({{"trail_x", "0"}, {"trail_y", "0"}});
    trail2->state.set({{"trail_x", "1"}, {"trail_y", "0"}});
    trail3->state.set({{"trail_x", "-1"}, {"trail_y", "0"}});
    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("That's two 3-cycles, and they only intersect in the center."), 8);
    for(int i = 0; i < 8; i++) {
        cps->state.transition(MICRO, "coefficient2_ring", i%2==0?"1":"0");
        cs.render_microblock();
    }

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_all_subscenes_except(MICRO, "cps", 0);
    cs.render_microblock();
    cs.remove_all_subscenes_except("cps");

    cs.stage_macroblock(CompositeBlock(FileBlock("Undoing to make a commutator,"), SilenceBlock(3)), 2);
    cps->state.transition(MICRO, "oval_hor", "0");
    cs.render_microblock();
    cps->state.transition(MICRO, "oval_ver", "0");
    cs.render_microblock();

    trail1->clear_trail();
    trail2->clear_trail();
    trail3->clear_trail();
    trail4->clear_trail();
    trail5->clear_trail();
    trail6->clear_trail();

    cs.add_scene(trail1, "trail1");
    cs.add_scene(trail2, "trail2");
    cs.add_scene(trail3, "trail3");
    cs.state.set({{"trail1.opacity", ".5"}, {"trail2.opacity", ".5"}, {"trail3.opacity", ".5"}});
    trail1->trail_color = trail2->trail_color = trail3->trail_color = 0xffff0000;
    trail1->state.set({{"trail_x", "{cps.root1_r}"}, {"trail_y", "{cps.root1_i}"}});
    trail2->state.set({{"trail_x", "{cps.root2_r}"}, {"trail_y", "{cps.root2_i}"}});
    trail3->state.set({{"trail_x", "{cps.root4_r}"}, {"trail_y", "{cps.root4_i}"}});

    cs.stage_macroblock(CompositeBlock(FileBlock("the final result was a cycle on these 3 items."), SilenceBlock(2)), 1);
    cps->coefficients_to_roots();
    cps->state.transition(MICRO, plussign);
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("If we instead order our commutator in reverse..."), 1);
    cs.fade_all_subscenes_except(MICRO, "cps", 0);
    cs.render_microblock();
    cs.remove_all_subscenes_except("cps");

    cs.stage_macroblock(SilenceBlock(5), 4);
    cps->roots_to_coefficients();
    cps->state.set(ovals);
    for(int i = 0; i < 4; i++) {
        string which_oval = (i % 2 == 0) ? "oval_hor" : "oval_ver";
        cps->state.transition(MICRO, which_oval, i>1?"0":"-1");
        cs.render_microblock();
    }

    cps->coefficients_to_roots();
    cps->state.transition(MICRO, plussign);
    cs.add_scene(trail4, "trail4");
    cs.add_scene(trail5, "trail5");
    cs.add_scene(trail6, "trail6");
    cs.state.set({{"trail4.opacity", ".5"}, {"trail5.opacity", ".5"}, {"trail6.opacity", ".5"}});
    trail4->trail_color = trail5->trail_color = trail6->trail_color = 0xff00ff00;
    trail4->state.set({{"trail_x", "{cps.root0_r}"}, {"trail_y", "{cps.root0_i}"}});
    trail5->state.set({{"trail_x", "{cps.root2_r}"}, {"trail_y", "{cps.root2_i}"}});
    trail6->state.set({{"trail_x", "{cps.root3_r}"}, {"trail_y", "{cps.root3_i}"}});
    cs.stage_macroblock(CompositeBlock(FileBlock("we see a cycle on another set of 3 items."), SilenceBlock(1)), 1);
    cs.render_microblock();

    cs.add_scene_fade_in(MICRO, trail1, "trail1", .5, .5, .3);
    cs.add_scene_fade_in(MICRO, trail2, "trail2", .5, .5, .3);
    cs.add_scene_fade_in(MICRO, trail3, "trail3", .5, .5, .3);
    trail1->state.set({{"trail_x", "-1"}, {"trail_y", "0"}});
    trail2->state.set({{"trail_x", "0"}, {"trail_y", "0"}});
    trail3->state.set({{"trail_x", "0"}, {"trail_y", "-1"}});
    cs.stage_macroblock(FileBlock("Just like before, these two cycles share just one item of intersection."), 8);
    for(int i = 0; i < 8; i++) {
        cps->state.transition(MICRO, "coefficient2_ring", i%2==0?"1":"0");
        cs.render_microblock();
    }

    cs.stage_macroblock(FileBlock("So, we can just make a commutator with those."), 1);
    cs.render_microblock();

    cps->coefficients_to_roots();
    cs.stage_macroblock(SilenceBlock(4), 4);
    cps->state.transition(MICRO, {
        {"root1_r", "0"},
        {"root1_i", "0"},
        {"root2_r", "0"},
        {"root2_i", "-1"},
        {"root4_r", "-1"},
        {"root4_i", "0"},
    });
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"root0_r", "1"},
        {"root0_i", "0"},
        {"root1_r", "0"},
        {"root1_i", "1"},
        {"root3_r", "0"},
        {"root3_i", "0"},
    });
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"root2_r", "0"},
        {"root2_i", "0"},
        {"root3_r", "-1"},
        {"root3_i", "0"},
        {"root4_r", "0"},
        {"root4_i", "-1"},
    });
    cs.render_microblock();
    cps->state.transition(MICRO, {
        {"root0_r", "0"},
        {"root0_i", "1"},
        {"root1_r", "0"},
        {"root1_i", "0"},
        {"root2_r", "1"},
        {"root2_i", "0"},
    });
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(1), 1);
    cs.fade_all_subscenes_except(MICRO, "cps", 0);
    cs.render_microblock();
    cs.remove_all_subscenes_except("cps");

    cs.stage_macroblock(FileBlock("See what we get? The exact same loop we started with!"), 1);
    trail1->clear_trail();
    trail2->clear_trail();
    trail3->clear_trail();
    trail4->clear_trail();
    trail5->clear_trail();
    trail6->clear_trail();
    cs.add_scene(trail1, "trail1");
    cs.add_scene(trail2, "trail2");
    cs.add_scene(trail3, "trail3");
    cs.state.set({{"trail1.opacity", ".5"}, {"trail2.opacity", ".5"}, {"trail3.opacity", ".5"}});
    trail1->trail_color = trail2->trail_color = trail3->trail_color = 0xffff0000;
    trail1->state.set({{"trail_x", "{cps.root1_r}"}, {"trail_y", "{cps.root1_i}"}});
    trail2->state.set({{"trail_x", "{cps.root2_r}"}, {"trail_y", "{cps.root2_i}"}});
    trail3->state.set({{"trail_x", "{cps.root3_r}"}, {"trail_y", "{cps.root3_i}"}});
    cps->roots_to_coefficients();
    cps->state.set(ovals);
    cps->state.transition(MICRO, "oval_hor", "-1");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("We can infinitely nest commutators like this, guaranteeing swaps on the 5 objects."), 1);
    cs.render_microblock();

    cs.stage_macroblock(SilenceBlock(2), 1);
    cs.fade_all_subscenes_except(MICRO, "cps", 0);
    cs.render_microblock();
    cs.remove_all_subscenes_except("cps");

    cs.add_scene(ms, "ms");
    ms->state.set({
        {"hide_sqrt", "1"},
        {"hide_cbrt", "1"},
        {"hide_qrt", "1"},
    });
    cs.stage_macroblock(FileBlock("So, with any finite number of nested square roots, cube roots, you name it,"), 4);
    cs.render_microblock();
    ms->state.transition(MICRO, "hide_sqrt", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "hide_cbrt", "0");
    cs.render_microblock();
    ms->state.transition(MICRO, "hide_qrt", "0");
    cs.render_microblock();

    cs.stage_macroblock(FileBlock("there's always a deeper commutator of 5 solutions we simply cannot express."), 1);
    cs.render_microblock();
}

void ending(CompositeScene& cs, shared_ptr<ComplexPlotScene> cps){
    cs.add_scene(cps, "cps");
    cs.fade_subscene(MICRO, "cps", 0);
    shared_ptr<RootFractalScene> fracs = make_shared<RootFractalScene>();
    fracs->global_identifier = "fractal";
    cs.add_scene_fade_in(MICRO, fracs, "fracs");
    fracs->state.set({
        {"terms", "17"},
        {"coefficients_opacity", "0"},
        {"ticks_opacity", "0"},
    });

    cs.stage_macroblock(FileBlock("So there you have it! There is no general approach to solving polynomials. Hopefully now it makes sense why we get these beautiful fractals as a result."), 1);

    cs.stage_macroblock(FileBlock("I wanted to make an interactive demo that you could use to explore these fractals yourself... but frontend web development isn't my forte, so I used Lovable to make one."), 1);

    cs.stage_macroblock(FileBlock("To make the page, I just described what I wanted it to look like, and it spit out a working interactive demo on the spot!"), 1);

    cs.stage_macroblock(FileBlock("You can visit this page in the description to play with the parameters yourself!"), 1);
    cs.stage_macroblock(FileBlock("It was extremely quick and easy to make... or at least it would have been, but I stayed up all night playing with different options."), 1);
    cs.stage_macroblock(FileBlock("For example, I was curious what would happen if we allowed for 3 legal coefficients instead of just 2."), 1);
    cs.stage_macroblock(FileBlock("All I had to do was ask!"), 1);
    cs.stage_macroblock(FileBlock("I also got Lovable to add a second mode for rendering coefficients of polynomials with only positive coefficients which add up to less than some value."), 1);
    cs.stage_macroblock(FileBlock("I challenge you to use the demo to drag around the coefficients and try to find ways to swap different roots among themselves!"), 1);
    cs.stage_macroblock(FileBlock("I've also been thinking about making a store to sell 2swap t-shirts, mugs, posters, and stickers."), 1);
    cs.stage_macroblock(FileBlock("Just to get a feel for what that might look like, I used Lovable to quickly generate a prototype store."), 1);
    cs.stage_macroblock(FileBlock("It immediately made a working, beautiful online storefront using the channel art which I provided."), 1);
    cs.stage_macroblock(FileBlock("Lovable helps you easily create interactive web pages and online stores without needing to know any code."), 1);
    cs.stage_macroblock(FileBlock("Go to lovable.dev to start building today and use my code XXXXXXX for 20 percent off!"), 1);
    cs.stage_macroblock(FileBlock("Thanks to Lovable for sponsoring this video!"), 1);
    cs.stage_macroblock(FileBlock("This has been 2swap, with music by 6884."), 1);
}

void render_video(){
    keys_to_record = {
        "cps.coefficient0_r",
        "cps.coefficient0_i",
        "cps.coefficient0_opacity",
        "cps.coefficient0_ring",
        "cps.coefficient1_r",
        "cps.coefficient1_i",
        "cps.coefficient1_opacity",
        "cps.coefficient1_ring",
        "cps.coefficient2_r",
        "cps.coefficient2_i",
        "cps.coefficient2_opacity",
        "cps.coefficient2_ring",
        "cps.coefficient3_r",
        "cps.coefficient3_i",
        "cps.coefficient3_opacity",
        "cps.coefficient3_ring",
        "cps.coefficient4_r",
        "cps.coefficient4_i",
        "cps.coefficient4_opacity",
        "cps.coefficient4_ring",
        "cps.coefficient5_r",
        "cps.coefficient5_i",
        "cps.coefficient5_opacity",
        "cps.coefficient5_ring",
        "cps.root0_r",
        "cps.root0_i",
        "cps.root0_ring",
        "cps.root1_r",
        "cps.root1_i",
        "cps.root1_ring",
        "cps.root2_r",
        "cps.root2_i",
        "cps.root2_ring",
        "cps.root3_r",
        "cps.root3_i",
        "cps.root3_ring",
        "cps.root4_r",
        "cps.root4_i",
        "cps.root4_ring",
        "fractal.coefficient0_r",
        "fractal.coefficient0_i",
        "fractal.coefficient1_r",
        "fractal.coefficient1_i",
        "cafs.sqrt_branch_cut",
        "3d.d",
        "3d.q1",
        "3d.qi",
        "3d.qj",
        "3d.qk",
    }; // Output for 6884

    CompositeScene cs;
    shared_ptr<ComplexPlotScene> cps = make_shared<ComplexPlotScene>(1);
    cps->global_identifier = "cps";
    shared_ptr<ManifoldScene> ms = make_shared<ManifoldScene>();
    ms->global_identifier = "3d";

    //FOR_REAL = false;
    part_0(cs, cps);
    cs.remove_all_subscenes();
    part_1(cs, cps);
    cs.remove_all_subscenes();
    part_2(cs, cps, ms);
    cs.remove_all_subscenes();
    part_3(cs, cps, ms);
    cs.remove_all_subscenes();
    part_3p5(cs, cps, ms);
    cs.remove_all_subscenes();
    part_4(cs, cps, ms);
    cs.remove_all_subscenes();
    //FOR_REAL = true;
    part_5(cs, cps, ms);
    cs.remove_all_subscenes();
    //ending(cs, cps);
}
