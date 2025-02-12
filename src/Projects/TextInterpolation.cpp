#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"

void render_video(){
    string begin_latex = "a=bc";
    string end_latex = "\\frac{a}{c}=b";

    CompositeScene cs;

    LatexScene what_have(latex_text("What we have:"), 1, 0.4, 0.2);
    cs.add_scene_fade_in(&what_have, "what_have", 0.25, 0.1);
    cs.inject_audio_and_render(SilenceSegment(1));

    LatexScene img1(begin_latex, 0.5, 0.4, 0.3);
    cs.add_scene_fade_in(&img1, "img1", 0.25, 0.35);
    cs.inject_audio_and_render(SilenceSegment(1));

    LatexScene img2(end_latex, 1, 0.4, 0.3);
    cs.add_scene_fade_in(&img2, "img2", 0.25, 0.75);
    cs.inject_audio_and_render(SilenceSegment(1));

    LatexScene what_want(latex_text("What we want:"), 1, 0.4, 0.2);
    cs.add_scene_fade_in(&what_want, "what_want", 0.75, 0.1);
    cs.inject_audio_and_render(SilenceSegment(1));

    LatexScene vid(begin_latex, 0.5, 0.4, 0.3);
    cs.add_scene_fade_in(&vid, "vid", 0.75, 0.55);
    cs.inject_audio_and_render(SilenceSegment(1));
    vid.begin_latex_transition(end_latex);
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.inject_audio_and_render(SilenceSegment(1));

    cs.state_manager.set({
        {"img1.opacity", "1"},
        {"img2.opacity", "1"},
    });
    cs.state_manager.microblock_transition({
        {"vid.opacity", "0"},
        {"what_want.opacity", "0"},
        {"what_have.opacity", "0"},
        {"img1.x", ".33"},
        {"img1.y", ".5"},
        {"img2.x", ".67"},
        {"img2.y", ".5"},
    });
    LatexScene step1(latex_text("Step 1: ???"), 1, 1, 0.2);
    cs.add_scene_fade_in(&step1, "step1", 0.5, 0.1);
    cs.inject_audio_and_render(SilenceSegment(2));
    cs.remove_scene(&what_have);
    cs.remove_scene(&what_want);
    cs.remove_scene(&vid);
    cs.inject_audio_and_render(SilenceSegment(2));
}
