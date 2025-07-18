#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"
#include "../Scenes/Media/Mp4Scene.cpp"
#include "../Scenes/Media/PngScene.cpp"

void promo0(){
    Mp4Scene ms = Mp4Scene("Sequence_VisualAlgebra_250616");
    ms.stage_macroblock(FileBlock("It's no secret that I prefer a visual approach to understanding mathematical concepts."), 1);
    ms.render_microblock();

    {
        shared_ptr<Mp4Scene> csms = make_shared<Mp4Scene>("Sequence_Calc_250616");
        CompositeScene cs;
        cs.add_scene(csms, "csms");
        cs.stage_macroblock(FileBlock("Math doesn't have to feel like a jumble of fancy greek letters- there's always an underlying intuition."), 5);
        cs.render_microblock();
        cs.render_microblock();
        cs.render_microblock();
        cs.render_microblock();
        cs.fade_all_subscenes(MICRO, 0);
        cs.render_microblock();
    }

    ms = Mp4Scene(vector<string>{"Logo Animation_mp4", "Digital Circuits Course Sequence 250616"});
    ms.stage_macroblock(FileBlock("That's why Brilliant's lessons in math, computer science, and physics all involve visualizations, games, and examples which don't just teach you those messy equations, but rather the ideas which inspire them."), 1);
    ms.render_microblock();

    ms = Mp4Scene(vector<string>{"CC_TheForLoop_cropped_v01", "Correct to Streak"});
    ms.stage_macroblock(FileBlock("You don't just solve problems, but you can play with the systems to really grok how they work to begin with."), 1);
    ms.render_microblock();

    ms = Mp4Scene("Content Footage_How AI Works_Digit Checker");
    ms.stage_macroblock(FileBlock("Brush up on the basics, or dive into the cutting edge mathematics behind neural networks and quantum mechanics."), 1);
    ms.render_microblock();

    ms = Mp4Scene("Logo header to content_Math_Koji_25Q2");
    ms.stage_macroblock(FileBlock("Math doesn't have to be hard, or boring. Learn to appreciate its beauty with Brilliant!"), 1);
    ms.render_microblock();

    CompositeScene cs;
    shared_ptr<Mp4Scene> csms = make_shared<Mp4Scene>("main");
    cs.add_scene(csms, "csms");
    shared_ptr<PngScene> ps = make_shared<PngScene>("QR Code - 2swap", .4, .4);
    cs.add_scene(ps, "ps", -.25, .7);
    cs.stage_macroblock(FileBlock("You can try all of Brilliant's features for free for 30 days by visiting brilliant.org/2swap, and get 20% off in the long term on an annual subscription."), 5);
    cs.render_microblock();
    cs.slide_subscene(MICRO, "ps", .42, 0);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
}
