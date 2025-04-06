#include "../Scenes/Media/Mp4Scene.cpp"
void render_video(){
    Mp4Scene ms("test");
    ms.stage_macroblock_and_render(SilenceSegment(4));
}
