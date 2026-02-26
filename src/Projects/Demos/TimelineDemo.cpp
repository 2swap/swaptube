#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/StateSliderScene.h"

void render_video() {
    CompositeScene cs;
    shared_ptr<StateSliderScene> mac_f = make_shared<StateSliderScene>("{macroblock_fraction}", "\\text{macro fraction}", 0, 1  , .25, .1);
    shared_ptr<StateSliderScene> mic_f = make_shared<StateSliderScene>("{microblock_fraction}", "\\text{micro fraction}", 0, 1  , .25, .1);
    shared_ptr<StateSliderScene> mac_n = make_shared<StateSliderScene>("{macroblock_number}"  , "\\text{macro number}"  , 0, 10 , .25, .1);
    shared_ptr<StateSliderScene> mic_n = make_shared<StateSliderScene>("{microblock_number}"  , "\\text{micro number}"  , 0, 10 , .25, .1);
    shared_ptr<StateSliderScene> frame = make_shared<StateSliderScene>("{frame_number}"       , "\\text{frame number}"  , 0, 100, .25, .1);
    shared_ptr<StateSliderScene> timer = make_shared<StateSliderScene>("{t}"                  , "\\text{t}"             , 0, 10 , .25, .1);
    cs.add_scene(mac_f, "mac_f", 0.25, 0.1);
    cs.add_scene(mic_f, "mic_f", 0.25, 0.2);
    cs.add_scene(mac_n, "mac_n", 0.75, 0.1);
    cs.add_scene(mic_n, "mic_n", 0.75, 0.2);
    cs.add_scene(frame, "frame", 0.5, 0.1);
    cs.add_scene(timer, "timer", 0.5, 0.2);
    stage_macroblock(CompositeBlock(SilenceBlock(1), SilenceBlock(1)), 5);
    for(int i = 0; i < 5; i++) cs.render_microblock();
    stage_macroblock(SilenceBlock(1), 5);
    for(int i = 0; i < 5; i++) cs.render_microblock();
    stage_macroblock(SilenceBlock(1), 2);
    for(int i = 0; i < 2; i++) cs.render_microblock();
}

