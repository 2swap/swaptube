#include "../Scenes/Media/Mp4Scene.cpp"
void render_video(){
    Mp4Scene ms("test");
    ms.inject_audio_and_render(SilenceSegment(4));
}
