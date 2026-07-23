#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/RubiksGraphScene.h"
#include "../Core/Smoketest.h"

void render_video() {
    RubiksGraphScene rgs;

    rgs.manager.set({
        {"physics_multiplier", "40"},
        {"decay", ".8"},
        {"dimensions", "2"},
        {"d", "25"},
        {"qi", "{t} 5 * sin .2 *"},
        {"qj", "{t} 5 * cos .2 *"},
    });

    stage_macroblock(SilenceBlock(2), 1);
    rgs.add_cube("", true);
    rgs.render_microblock();

    // print the size of the graph
    cout << "Graph size: " << rgs.gs->graph->size() << " nodes";

    stage_macroblock(SilenceBlock(2), 1);
    rgs.add_children();
    rgs.render_microblock();

    // print the size of the graph
    cout << "Graph size: " << rgs.gs->graph->size() << " nodes";

    stage_macroblock(SilenceBlock(2), 1);
    rgs.add_children();
    rgs.render_microblock();

    // print the size of the graph
    cout << "Graph size: " << rgs.gs->graph->size() << " nodes";

    stage_macroblock(SilenceBlock(2), 1);
    rgs.add_children();
    rgs.render_microblock();

    // print the size of the graph
    cout << "Graph size: " << rgs.gs->graph->size() << " nodes";

    stage_macroblock(SilenceBlock(2), 1);
    rgs.add_children();
    rgs.render_microblock();

    // print the size of the graph
    cout << "Graph size: " << rgs.gs->graph->size() << " nodes";

    stage_macroblock(SilenceBlock(2), 1);
    rgs.add_children();
    rgs.render_microblock();

    // print the size of the graph
    cout << "Graph size: " << rgs.gs->graph->size() << " nodes";

    stage_macroblock(SilenceBlock(2), 1);
    rgs.add_children();
    rgs.render_microblock();

    // print the size of the graph
    cout << "Graph size: " << rgs.gs->graph->size() << " nodes";

    stage_macroblock(SilenceBlock(2), 1);
    rgs.add_children();
    rgs.render_microblock();

    // print the size of the graph
    cout << "Graph size: " << rgs.gs->graph->size() << " nodes";

    stage_macroblock(SilenceBlock(2), 1);
    rgs.add_children();
    rgs.render_microblock();

    // print the size of the graph
    cout << "Graph size: " << rgs.gs->graph->size() << " nodes";

    stage_macroblock(SilenceBlock(2), 1);
    rgs.add_children();
    rgs.render_microblock();

    // print the size of the graph
    cout << "Graph size: " << rgs.gs->graph->size() << " nodes";

    stage_macroblock(SilenceBlock(2), 1);
    rgs.add_children();
    rgs.render_microblock();

    // print the size of the graph
    cout << "Graph size: " << rgs.gs->graph->size() << " nodes";
}
