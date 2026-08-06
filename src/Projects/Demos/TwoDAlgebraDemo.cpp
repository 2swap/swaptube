
#include "../Scenes/Math/TwoDAlgebraScene.h"
// #include "../Core/State/StateTester.h"
// #include "../Scenes/Math/RealFunctionScene.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Core/Smoketest.h"
#include "../IO/Writer.h"





void plane_demo(CompositeScene& cs){

    shared_ptr<TwoDAlgebraScene> td = make_shared<TwoDAlgebraScene>();
    cs.add_scene(td, "td");

    td->manager.set({
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

    stage_macroblock(FileBlock(""), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock(""), 1);
    td->manager.transition(MICRO, {
        {"dragger_y", "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock(""), 1);
    td->manager.transition(MICRO, {
        {"yy_x", "0"},
    });
    cs.render_microblock();
    

    stage_macroblock(FileBlock(""), 1);
    td->manager.transition(MICRO, {
        {"xx_y", "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock(""), 1);
    td->manager.transition(MICRO, {
        {"dragger_x", "-1"},
    });
    cs.render_microblock();


    stage_macroblock(FileBlock(""), 1);
    td->manager.transition(MICRO, {
        {"xy_x", "-1"},
    });
    cs.render_microblock();


    cs.remove_subscene("td");
}


void render_video() {
    CompositeScene cs;


    plane_demo(cs);


}


