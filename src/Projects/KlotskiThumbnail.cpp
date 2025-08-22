#include "../Scenes/KlotskiScene.cpp"
#include "../DataObjects/KlotskiBoard.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/PauseScene.cpp"
#include "../Scenes/Connect4/Connect4GraphScene.cpp"
#include "../Scenes/Math/GraphScene.cpp"
#include "../Scenes/Media/LoopAnimationScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Media/Mp4Scene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"
#include "../Scenes/Common/ExposedPixelsScene.cpp"

void render_video() {
    KlotskiBoard kb(6, 6, "............bb......................", true);

    kb.stage_macroblock(SilenceBlock(1), 1);
    kb.render_microblock();
}
