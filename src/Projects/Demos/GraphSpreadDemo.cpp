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
        {"d", "10"},
        {"physics_multiplier", "0"},
    });

    stage_macroblock(SilenceBlock(5), 2);
    double a_hash = gs.graph->add_node(.1);
    double b_hash = gs.graph->add_node(.2);
    double c_hash = gs.graph->add_node(.3);
    double d_hash = gs.graph->add_node(.4);
    gs.graph->add_edge(a_hash, b_hash);
    gs.graph->add_edge(c_hash, d_hash);
    gs.graph->add_edge(a_hash, d_hash);
    gs.graph->add_edge(b_hash, c_hash);
    gs.render_microblock();
    gs.render_microblock();
}
