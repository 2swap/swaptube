#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/StateSliderScene.cpp"

void render_video() {
    CompositeScene cs;
    StateSliderScene mac_f("<macroblock_fraction>", "macro fraction", 0, 1  , .25, .1);
    StateSliderScene mic_f("<microblock_fraction>", "micro fraction", 0, 1  , .25, .1);
    StateSliderScene mac_n("<macroblock_number>"  , "macro number"  , 0, 10 , .25, .1);
    StateSliderScene mic_n("<microblock_number>"  , "micro number"  , 0, 10 , .25, .1);
    StateSliderScene frame("<frame_number>"       , "frame number"  , 0, 100, .25, .1);
    StateSliderScene timer("<t>"                  , "t"             , 0, 10 , .25, .1);
    cs.add_scene(&mac_f, "mac_f", 0.25, 0.1);
    cs.add_scene(&mic_f, "mic_f", 0.25, 0.2);
    cs.add_scene(&mac_n, "mac_n", 0.75, 0.1);
    cs.add_scene(&mic_n, "mic_n", 0.75, 0.2);
    cs.add_scene(&frame, "frame", 0.5, 0.1);
    cs.add_scene(&timer, "timer", 0.5, 0.2);
    cs.stage_macroblock_and_render(SilenceSegment(2));
    cs.stage_macroblock(SilenceSegment(5), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
}

