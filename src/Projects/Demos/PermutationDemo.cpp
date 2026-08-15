#include "../Scenes/Math/PermutationScene.h"

// create a permutation scene ps, and show it
 void render_video() {
     PermutationScene ps("io_in/permutation_example_0");

     stage_macroblock(SilenceBlock(3), 1);
     ps.move("left");
     ps.render_microblock();

     stage_macroblock(SilenceBlock(3), 1);
     ps.move("right");
     ps.render_microblock();

     stage_macroblock(SilenceBlock(3), 1);
     ps.manager.transition(MACRO, {
         {"C.y", "1.5"},
     });
     ps.move("left_inverse");
     ps.render_microblock();

     stage_macroblock(SilenceBlock(3), 1);
     ps.move("right_inverse");
     ps.render_microblock();

     stage_macroblock(SilenceBlock(3), 1);
     ps.render_microblock();
 }