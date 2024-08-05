using namespace std;
#include "../Scenes/Media/latex_scene.cpp"
#include "../Scenes/Connect4/c4_scene.cpp"
#include "../Scenes/Common/composite_scene.cpp"
#include "../Scenes/Common/3d_scene.cpp"
#include "../Scenes/Connect4/c4_graph_scene.cpp"
#include "../Scenes/Media/png_scene.cpp"
#include "../Scenes/Common/exposed_pixels_scene.cpp"
#include "../Scenes/Common/twoswap_scene.cpp"

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
