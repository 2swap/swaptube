#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Math/MandelbrotScene.h"

void render_video() {
    CompositeScene cs;
    stage_macroblock(SilenceBlock(5));
    shared_ptr<MandelbrotScene> ms = make_shared<MandelbrotScene>();
    cs.add_scene(ms, "ms");

    cs.render_microblock();

    ms->manager.transition(MICRO, "w", "{t} 3 * sin .3 * .5 +");
    ms->manager.transition(MICRO, "h", "{t} 5 * cos .3 * .5 +");
    cs.render_microblock();

    ms->manager.transition(MICRO, "w", ".5");
    ms->manager.transition(MICRO, "h", ".5");
    cs.render_microblock();

    cs.manager.transition(MICRO, "ms.x", "{t} sin .5 * .5 +");
    cs.manager.transition(MICRO, "ms.y", "{t} cos .5 * .5 +");
    cs.render_microblock();

    cs.manager.transition(MICRO, "ms.angle", "{t}");
    cs.render_microblock();
}
