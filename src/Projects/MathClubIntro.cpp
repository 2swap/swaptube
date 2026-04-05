#include "../Scenes/Math/MandelbrotScene.h"

void render_video() {
    MandelbrotScene ms;
    ms.manager.begin_timer("zoom10x");
    ms.manager.set("zoom", "<zoom10x> 10 / 4 +");
    ms.manager.set("max_iterations", "1000");
    ms.manager.set("seed_c_r", "-0.20729917287826538");
    ms.manager.set("seed_c_i", "-0.7939999401569362");

    stage_macroblock(FileBlock("Welcome to Math Club!"), 1);
    ms.render_microblock();

    stage_macroblock(FileBlock("In case you don't know me, I am 2swap. This is my second channel, dedicated to long-form discussions of technical topics."), 1);
    ms.render_microblock();

    stage_macroblock(FileBlock("The content you see on this channel is mostly taken from the Math Club discord server, which has been private until now."), 1);
    ms.render_microblock();

    stage_macroblock(FileBlock("Once or twice a week, we have a presenter talk about some subject which they're familiar with, and everyone else asks questions and discusses."), 1);
    ms.render_microblock();

    stage_macroblock(FileBlock("The subject is usually about math, but it can really be about any topic which is technical in nature."), 1);
    ms.render_microblock();

    stage_macroblock(FileBlock("We are now allowing new members to join the server. However, in order to keep the group small and focused, we are gonna gatekeep entry a little bit."), 1);
    ms.render_microblock();

    stage_macroblock(FileBlock("To join, you have to give a talk on some subject of your choice. Usually, we don't require any preparation, with talks sometimes being completely off the cuff, or about a problem which we've not yet solved."), 1);
    ms.render_microblock();

    stage_macroblock(FileBlock("However, for entry to the server, I will be vetting the talks a little bit more, and accordingly, I will require that you show me some preparation before I send you an invite."), 1);
    ms.render_microblock();

    stage_macroblock(FileBlock("A google slides link would work, a math paper you wrote in the form of a PDF, a moderately thorough outline, some code you will demonstrate, or something along those lines would do."), 1);
    ms.render_microblock();

    stage_macroblock(FileBlock("If you are interested, add me as a friend on Discord (the username being 2swap, written with a 2), send me your preparation, and if it looks good, I'll send you an invite."), 1);
    ms.render_microblock();

    stage_macroblock(FileBlock("I look forward to seeing you there!"), 1);
    ms.render_microblock();
}
