#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"

void render_video() {
    GraphScene gs;

    gs.manager.set({
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"d", "900"},
        {"physics_multiplier", "40"},
    });

    stage_macroblock(SilenceBlock(10), 2);
    int graph_size = 50;
    for(int i = 0; i < graph_size; i++) {
        for(int j = 0; j < graph_size; j++) {
            int hash = i+j*graph_size;
            gs.graph->add_node(hash);
            if(i>0) gs.graph->add_edge(hash, hash-1);
            if(j>0) gs.graph->add_edge(hash, hash-graph_size);
        }
    }
    gs.render_microblock();
    gs.render_microblock();
}
