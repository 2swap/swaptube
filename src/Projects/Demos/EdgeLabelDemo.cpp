#include "../DataObjects/HashableString.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Core/Smoketest.h"

void render_video() {
    shared_ptr<Graph> g = make_shared<Graph>();
    shared_ptr<GraphScene> gs = make_shared<GraphScene>(g);

    gs->manager.set({
        {"q1", "1"},
        {"qi", "{t} .13 * sin .13 *"},
        {"qj", "{t} .11 * cos .13 *"},
        {"qk", "0"},
        {"d", "4"},
        {"edge_weights_size", "1.2"},
        {"points_radius_multiplier", "3"},
    });

    stage_macroblock(SilenceBlock(5), 4);
    double a_hash = g->add_node(new HashableString("a"));
    g->move_node(a_hash, vec4(-1, 0, 0, 0));
    double b_hash = g->add_node(new HashableString("b"));
    g->move_node(b_hash, vec4(1, 0, 0, 0));
    g->add_edge(a_hash, b_hash);
    gs->config->transition_edge_label(MICRO, a_hash, b_hash, "abcdef");
    gs->render_microblock();

    gs->transition_node_position(MICRO, a_hash, vec4(0, 1, 0, 0));
    gs->transition_node_position(MICRO, b_hash, vec4(0, -1, 0, 0));
    gs->render_microblock();

    gs->transition_node_position(MICRO, a_hash, vec4(1, 0, 0, 0));
    gs->transition_node_position(MICRO, b_hash, vec4(-1, 0, 0, 0));
    gs->render_microblock();

    gs->transition_node_position(MICRO, a_hash, vec4(0, -1, 0, 0));
    gs->transition_node_position(MICRO, b_hash, vec4(0, 1, 0, 0));
    gs->render_microblock();
}
