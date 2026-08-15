#include "../Scenes/Math/TwoDAlgebraScene.h"
// #include "../Core/State/StateTester.h"
// #include "../Scenes/Math/RealFunctionScene.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Core/Smoketest.h"
#include "../IO/Writer.h"

void render_video(){
    TwoDAlgebraScene td;

    td.manager.set({
        {"diagram_opacity", "255"},
        {"dragger_type", "2"},
        {"dragger_x", "1"},
        {"dragger_y", "0"},
        {"xx_x", "1"},
        {"xx_y", "0"},
        {"xy_x", "0"},
        {"xy_y", "1"},
        {"yy_x", "-1"},
        {"yy_y", "0"},
    });

    stage_macroblock(SilenceBlock(6), 6);
    td.render_microblock();

    td.manager.transition(MICRO, "dragger_y", "1");
    td.render_microblock();

    td.manager.transition(MICRO, "yy_x", "0");
    td.render_microblock();
    
    td.manager.transition(MICRO, "xx_y", "1");
    td.render_microblock();

    td.manager.transition(MICRO, "dragger_x", "-1");
    td.render_microblock();

    td.manager.transition(MICRO, "xy_x", "-1");
    td.render_microblock();
}

