#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/RubiksGraphScene.h"
#include "../Core/Smoketest.h"

void render_video() {
    RubiksGraphScene rgs;

    int d = 10;
    rgs.manager.set({
        {"physics_multiplier", "50"},
        {"decay", ".6"},
        {"dimensions", "3"},
        {"d", to_string(d)},
        {"qi", "{t} 5 * sin .2 *"},
        {"qj", "{t} 5 * cos .2 *"},
    });

    stage_macroblock(SilenceBlock(2), 1);
    rgs.add_cube("", false);
    rgs.render_microblock();

    for(int i = 0; i < 5; i++) {
        stage_macroblock(SilenceBlock(2), 1);
        rgs.manager.transition(MICRO, "d", to_string(d));
        d*=4;
        rgs.add_children();
        rgs.render_microblock();
        // print the size of the graph
        cout << "Graph size: " << rgs.gs->graph->size() << " nodes";
    }
}
