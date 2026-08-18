#include "../Scenes/Math/PermutationScene.h"
#include <iostream>
#include <fstream>

 void render_video() {
    const std::string file_path = "io_in/permutation_example_0";

    // if no file, render a blank scene and print a warning
    if (!std::ifstream(file_path).is_open()) {
        std::cerr << "[Warning] Fichier " << file_path 
                  << " introuvable. Rendu d'une scène vide pour la démo.\n";

        PermutationScene ps("");
        stage_macroblock(SilenceBlock(1), 1);
        ps.render_microblock();
        return;
    }


    PermutationScene ps(file_path);

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