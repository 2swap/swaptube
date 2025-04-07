#include "../Scenes/Media/LoopAnimationScene.cpp"
void render_video(){
    LoopAnimationScene las1({"f1", "f2", "f3"});
    las1.stage_macroblock_and_render(SilenceSegment(3));
}
