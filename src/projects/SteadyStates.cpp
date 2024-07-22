using namespace std;
#include "../scenes/Media/latex_scene.cpp"
#include "../scenes/Connect4/c4_scene.cpp"
#include "../scenes/Common/composite_scene.cpp"
#include "../scenes/Common/3d_scene.cpp"
#include "../scenes/Connect4/c4_graph_scene.cpp"
#include "../scenes/Media/png_scene.cpp"
#include "../scenes/Common/exposed_pixels_scene.cpp"
#include "../scenes/Common/twoswap_scene.cpp"

void beginning() {
    C4Scene c4("444");
    c4.set_annotations("   456...345671234567123456712345671234567");
    c4.inject_audio_and_render(AudioSegment(3));
}

void render_video() {
    FOR_REAL = true;
    PRINT_TO_TERMINAL = true;
    beginning();
}
