#include "../Scenes/Math/KlotskiScene.cpp"
#include "../DataObjects/KlotskiBoard.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Math/GraphScene.cpp"

void render_video(){
    // Check that the video width and height are mobile 9:16 ration
    if(VIDEO_WIDTH * 16 != VIDEO_HEIGHT * 9){
        throw std::runtime_error("VIDEO_WIDTH and VIDEO_HEIGHT must be in 9:16 ratio");
    }
    for(KlotskiBoard kb : {simple_4x4}){
        shared_ptr<KlotskiScene> ks = make_shared<KlotskiScene>(kb, .5, .5*VIDEO_WIDTH/VIDEO_HEIGHT);
        CompositeScene cs;
        cs.add_scene(ks, "ks", 0.25, 0.25*VIDEO_WIDTH/VIDEO_HEIGHT);

        Graph g;
        g.add_to_stack(new KlotskiBoard(ks->copy_board()));

        int g_size;
        {
            Graph g2;
            g2.add_to_stack(new KlotskiBoard(ks->copy_board()));
            g2.expand();
            g_size = g2.size();
        }
        shared_ptr<GraphScene> gs = make_shared<GraphScene>(&g, false);
        gs->state.set({
            {"q1", "0"},
            {"qi", "{t} .1 * cos"},
            {"qj", "{t} .1 * sin"},
            {"qk", "0"}, // Camera orientation quaternion.
            {"d", "1"},
            {"physics_multiplier","5"}, // How many times to iterate the graph-spreader
            {"decay",".8"},
            {"surfaces_opacity","0"}, // 0 meaning do not render the corresponding board on top of each node.
            {"points_opacity","1"}, // Whether to draw nodes.
            {"lines_opacity","1"}, // Whether to draw edges.
        });
        cs.add_scene(gs, "gs");

        // Let a random agent make move around on the board and build the graph as it does, for 50 moves
        cs.stage_macroblock(SilenceBlock(5), 50); // Stage a macroblock with 50 microblocks
        while(remaining_microblocks_in_macroblock) {
            ks->stage_random_move();
            // Add the new node
            g.add_to_stack(new KlotskiBoard(ks->copy_board()));
            g.add_missing_edges();
            // Highlight the node of the board on the state-space graph
            gs->next_hash = ks->copy_board().get_hash();
            cs.render_microblock(); // Render a microblock
        }

        // Expand the graph until all nodes are present
        cs.stage_macroblock(SilenceBlock(2), (g_size-g.size())*1.2); // Stage a macroblock with more microblocks than graph nodes
        while(remaining_microblocks_in_macroblock) {
            g.expand(1);
            cs.render_microblock(); // Render a microblock
        }
    }
}
