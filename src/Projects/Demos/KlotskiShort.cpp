#include "../Scenes/KlotskiScene.cpp"
#include "../DataObjects/KlotskiBoard.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Math/GraphScene.cpp"

void demo(){
    //FOR_REAL = false;
    //PRINT_TO_TERMINAL = false;

    for(KlotskiBoard kb : {apk}){
        KlotskiScene ks(kb, .5, .5*VIDEO_WIDTH/VIDEO_HEIGHT);
        CompositeScene cs;
        cs.add_scene(&ks, "ks", 0.25, 0.25*VIDEO_WIDTH/VIDEO_HEIGHT);

        Graph g;
        g.add_to_stack(new KlotskiBoard(ks.copy_board()));

        int g_size;
        {
            Graph g2;
            g2.add_to_stack(new KlotskiBoard(ks.copy_board()));
            g2.expand_completely();
            g_size = g2.size();
        }
        GraphScene gs(&g);
        gs.state_manager.set({
            {"q1", "0"},
            {"qi", "<t> .1 * cos"},
            {"qj", "<t> .1 * sin"},
            {"qk", "0"}, // Camera orientation quaternion.
            {"d", "1"},
            {"physics_multiplier","5"}, // How many times to iterate the graph-spreader
            {"decay",".8"},
            {"surfaces_opacity","0"}, // 0 meaning do not render the corresponding board on top of each node.
            {"points_opacity","1"}, // Whether to draw nodes.
            {"lines_opacity","1"}, // Whether to draw edges.
        });
        cs.add_scene(&gs, "gs");

        // Let a random agent make move around on the board and build the graph as it does, for 50 moves
        cs.stage_macroblock(SilenceSegment(5), 50); // Stage a macroblock with 50 microblocks
        while(cs.microblocks_remaining()) {
            ks.stage_random_move();
            // Add the new node
            g.add_to_stack(new KlotskiBoard(ks.copy_board()));
            g.add_missing_edges(true);
            // Highlight the node of the board on the state-space graph
            gs.next_hash = ks.copy_board().get_hash();
            cs.render(); // Render a microblock
        }

        // Expand the graph until all nodes are present
        cs.stage_macroblock(SilenceSegment(2), (g_size-g.size())*1.2); // Stage a macroblock with more microblocks than graph nodes
        while(cs.microblocks_remaining()) {
        }
    }
}
