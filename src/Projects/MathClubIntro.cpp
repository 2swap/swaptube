#include "../Scenes/Math/MandelbrotScene.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"

void render_video() {
    CompositeScene cs;
    shared_ptr<MandelbrotScene> ms = make_shared<MandelbrotScene>();
    cs.add_scene(ms, "ms");
    ms->manager.begin_timer("zoom10x");
    ms->manager.set("zoom", "<zoom10x> 14 /");
    ms->manager.set("max_iterations", "1000");
    ms->manager.set("seed_c_r", "-0.5914800382174565");
    ms->manager.set("seed_c_i", "0.619993772449449");

    stage_macroblock(FileBlock("Welcome to Math Club!"), 1);
    cs.render_microblock();

    shared_ptr<PngScene> ps = make_shared<PngScene>("2swap");
    cs.add_scene_fade_in(MICRO, ps, "ps");

    stage_macroblock(FileBlock("In case you don't know me, I am 2swap! On my main YouTube channel, I post animated videos about various mathematical systems which I like."), 5);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "ps", 0);
    cs.render_microblock();

    stage_macroblock(FileBlock("With all the friends I have made along the way, I created a discord server called Math Club. It's a tight-knit, focused community of experts in various technical fields."), 1);
    cs.render_microblock();

    shared_ptr<PngScene> ps2 = make_shared<PngScene>("guidelines");
    cs.add_scene_fade_in(MICRO, ps2, "ps2");

    stage_macroblock(FileBlock("The main feature of the server is that usually about 2 or 3 times a week, a presenter will give a talk about some subject which they are an expert in, and everyone else is free to ask questions and discuss."), 5);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("The subject is usually about math, but it can really be about any topic which is technical in nature. Computer science, physics, and linguistics talks have also been common."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("If you like, we can record your talk as well, and post it here or give it to you for your own use. However, this is not a requirement, and usually only about three quarters of presenters choose to do so."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("Most talks are in the range of 20 to 90 minutes."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("These are all just suggestions- there are no hard requirements about length, topic, etc."), 1);
    cs.render_microblock();

    cs.fade_subscene(MICRO, "ps2", 0);

    stage_macroblock(FileBlock("So how do you join?"), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("There's lots of demand, so to keep the group small and focused, I will be gatekeeping entry a bit."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("To join, you must give a talk on some subject which you are familiar with."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("If you have some credentials in that field, let me know! If not, that's fine too. Just don't be surprised when you receive a lot of open-ended questions about the topic and how it ties into other subfields of math."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("For normal talks, we don't require any pretty slides or other preparation for the talk. Just some rough planning is usually fine."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("However, for new members wishing to join the server, I will vet the talks with much higher scrutiny, and accordingly, I will require that you show me some preparation before I send you an invite."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("Usually this means a presentation slide deck. But an interactive demo, a thorough outline, some code you will use, some diagrams, or something along those lines would do as well."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("Just send it to me on discord- my username is 2swap."), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("I look forward to seeing you there!"), 1);
    cs.render_microblock();
}
