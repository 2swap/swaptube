#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"

void render_video() {
    GraphScene gs;

    gs.manager.set({
        {"physics_multiplier", "1"},
    });

    stage_macroblock(SilenceBlock(10), 2);
    for(int i = 0; i < 9; i++) {
        gs.graph->add_node(i);
        if(i%3>0) gs.graph->add_edge(i, i-1);
        if(i>2) gs.graph->add_edge(i, i-3);
    }
    gs.render_microblock();
    gs.render_microblock();
}
