void render_video(){
    LatexScene latex("a=b");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_transition("a=b=c");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.inject_audio_and_render(AudioSegment(1));
}
