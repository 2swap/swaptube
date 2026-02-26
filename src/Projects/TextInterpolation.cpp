#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/StateSliderScene.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Common/ExposedPixelsScene.h"

void render_video(){
    FOR_REAL = false;
    string begin_latex = "a=bc";
    string end_latex = "\\frac{a}{c}=b";

    CompositeScene cs;

    LatexScene what_have(latex_text("What we have:"), 1, 0.4, 0.2);
    cs.add_scene_fade_in(&what_have, "what_have", 0.25, 0.1);
    cs.stage_macroblock_and_render(SilenceSegment(1));

    double iw = 0.4;
    double ih = 0.3;
    LatexScene img1(begin_latex, 0.5, iw, ih);
    LatexScene label1(latex_text("First Image"), 1, 0.4, 0.1);
    cs.add_scene_fade_in(&img1, "img1", 0.25, 0.35);
    cs.add_scene_fade_in(&label1, "label1", 0.25, 0.5);
    cs.stage_macroblock_and_render(SilenceSegment(1));

    LatexScene img2(begin_latex, 0.5, iw, ih);
    LatexScene label2(latex_text("Second Image"), 1, 0.4, 0.1);
    img2.jump_latex(end_latex);
    cs.add_scene_fade_in(&img2, "img2", 0.25, 0.75);
    cs.add_scene_fade_in(&label2, "label2", 0.25, 0.9);
    cs.stage_macroblock_and_render(SilenceSegment(1));

    LatexScene what_want(latex_text("What we want:"), 1, 0.4, 0.2);
    cs.add_scene_fade_in(&what_want, "what_want", 0.75, 0.1);
    cs.stage_macroblock_and_render(SilenceSegment(1));

    LatexScene vid(begin_latex, 0.5, iw, ih);
    vid.override_transition_end = true;
    cs.add_scene_fade_in(&vid, "vid", 0.75, 0.55);
    cs.stage_macroblock_and_render(SilenceSegment(1));
    vid.state.set({
        {"transparency_profile", "[interp]"},
    });
    cs.state.set({
        {"interp", "{t} sin 1 + 2 /"},
    });
    vid.begin_latex_transition(end_latex);
    cs.stage_macroblock_and_render(SilenceSegment(2));
    StateSliderScene ss("[interp]", "", 0, 1, .3, .05);
    cs.add_scene_fade_in(&ss, "ss", 0.75, 0.7); 
    cs.stage_macroblock_and_render(SilenceSegment(2));
    LatexScene label3(latex_text("Visual Interpolation"), 1, 0.4, 0.1);
    cs.add_scene_fade_in(&label3, "label3", 0.75, 0.8);
    cs.stage_macroblock_and_render(SilenceSegment(2));
    cs.state.microblock_transition({
        {"interp", "1"},
    });
    cs.stage_macroblock_and_render(SilenceSegment(1));


    cs.state.set({
        {"img1.opacity", "1"},
        {"img2.opacity", "1"},
    });
    cs.state.microblock_transition({
        {"vid.opacity", "0"},
        {"what_want.opacity", "0"},
        {"label1.opacity", "0"},
        {"label2.opacity", "0"},
        {"label3.opacity", "0"},
        {"ss.opacity", "0"},
        {"what_have.opacity", "0"},
        {"img1.x", ".33"},
        {"img1.y", ".5"},
        {"img2.x", ".67"},
        {"img2.y", ".5"},
    });
    LatexScene step1(latex_text("Step 1: Find a shared component"), 1, 1, 0.2);
    cs.add_scene_fade_in(&step1, "step1", 0.5, 0.1);
    cs.stage_macroblock_and_render(SilenceSegment(2));
    cs.remove_scene(&what_have);
    cs.remove_scene(&what_want);
    cs.remove_scene(&vid);
    cs.remove_scene(&label1);
    cs.remove_scene(&label2);
    cs.remove_scene(&label3);
    cs.stage_macroblock_and_render(SilenceSegment(2));

    FOR_REAL = true;
    Pixels p1 = img1.get_copy_p1();
    Pixels p2 = img2.get_copy_p1();
    cs.state.set({
        {"x_frac", ".5"},
        {"y_frac", ".5"},
    });
    cs.state.microblock_transition({
        {"img1.x", ".5 .5 <x_frac> - " + to_string(iw) + " * .5 * +"},
        {"img1.y", ".5 .5 <y_frac> - " + to_string(ih) + " * .5 * +"},
        {"img2.x", ".5"},
        {"img2.y", ".5"},
    });
    int xmax, ymax;
    Pixels cmap = convolve_map(p1, p2, xmax, ymax);
    ExposedPixelsScene eps(static_cast<double>(cmap.w)/VIDEO_WIDTH, static_cast<double>(cmap.h)/VIDEO_HEIGHT);
    eps.exposed_pixels = cmap;
    cs.add_scene_fade_in(&eps, "eps", 0.5, 0.9);
    cs.stage_macroblock_and_render(SilenceSegment(2));
    cs.state.microblock_transition({
        {"x_frac", to_string((static_cast<double>(xmax) + p2.w - 1)/cmap.w)},
        {"y_frac", to_string((static_cast<double>(ymax) + p2.h - 1)/cmap.h)},
    });
    cs.stage_macroblock_and_render(SilenceSegment(2));
    cs.stage_macroblock_and_render(SilenceSegment(2));
}
