#include "../Scenes/Math/ComplexPlotScene.cpp"
#include "../Scenes/Math/ComplexArbitraryFunctionScene.cpp"
#include "../Scenes/Math/RealFunctionScene.cpp"
#include "../Scenes/Math/RootFractalScene.cpp"
#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/ThreeDimensionScene.cpp"
#include "../Scenes/Math/ManifoldScene.cpp"
#include "../Scenes/Math/AngularFractalScene.cpp"
#include "../Scenes/Common/CoordinateSceneWithTrail.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Media/Mp4Scene.cpp"
#include <regex>


void render_video(){
    VIDEO_BACKGROUND_COLOR = 0xff000000;

    RootFractalScene rfs_intro;
    rfs_intro.stage_macroblock(SilenceBlock(1), 1);
    rfs_intro.state.set({
        {"coefficient0_r", "pi .4 + cos"},
        {"coefficient0_i", "pi .4 + sin"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"center_x", ".6"},
        {"center_y", ".27"},
        {"zoom", "3.5"},
        {"terms", "25"},
        {"coefficients_opacity", "0"},
        {"ticks_opacity", "0"},
        {"visibility_multiplier", ".25"},
        {"brightness", "0"},
    });
    /*
    rfs_intro.state.set({
        {"coefficient0_r", "1 cos"},
        {"coefficient0_i", "1 sin"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"center_x", "pi 3 / 2 * cos"},
        {"center_y", "pi 3 / 2 * sin"},
        {"zoom", "5.5"},
        {"terms", "26"},
        {"coefficients_opacity", "0"},
        {"ticks_opacity", "0"},
        {"visibility_multiplier", ".2"},
        {"brightness", "0"},
    });
    */
    /*
    rfs_intro.state.set({
        {"coefficient0_r", "1.465 cos"},
        {"coefficient0_i", "1.465 sin"},
        {"coefficient1_r", "1"},
        {"coefficient1_i", "0"},
        {"center_x", "pi 4 / 2 * cos"},
        {"center_y", "pi 4 / 2 * sin"},
        {"zoom", "5"},
        {"terms", "26"},
        {"coefficients_opacity", "0"},
        {"ticks_opacity", "0"},
        {"visibility_multiplier", ".2"},
        {"brightness", "0"},
    });
    */
    rfs_intro.render_microblock();
    rfs_intro.export_frame("THUMB");
}
