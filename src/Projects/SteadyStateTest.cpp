#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Connect4/C4Scene.h"
#include "../Scenes/Connect4/C4GraphScene.h"
#include "../DataObjects/Connect4/TreeValidator.h"
#include "../DataObjects/Connect4/SteadyState.h"

void render_video() {
    CompositeScene cs;

    Graph g;

    {
        string variation = "4261";
        if(variation.size()%2 != 0) throw runtime_error("Variation must be even length");
        C4Board board(FULL, variation);
        board.print();
        shared_ptr<SteadyState> ss = modify_child_suggestion(
           make_shared<SteadyState>(array<string, 6>{"       ",
                                                     "       ",
                                                     "       ",
                                                     "       ",
                                                     "       ",
                                                     "    !  ",}), board);
        if(ss->validate(board, true) == false) throw runtime_error("Unit Test 1 Failed");
    }

    {
        string variation = "4261";
        if(variation.size()%2 != 0) throw runtime_error("Variation must be even length");
        C4Board board(FULL, variation);
        board.print();
        shared_ptr<SteadyState> ss = modify_child_suggestion(
           make_shared<SteadyState>(array<string, 6>{"       ",
                                                     "       ",
                                                     "       ",
                                                     "       ",
                                                     "       ",
                                                     "       ",}), board);
        if(ss->validate(board, true) == true) throw runtime_error("Unit Test 2 Failed");
    }

    string variation = "4265664544655645222246";
    if(variation.size()%2 != 0) throw runtime_error("Variation must be even length");
    C4Board board(FULL, variation);
    board.print();
    shared_ptr<SteadyState> ss = modify_child_suggestion(
       make_shared<SteadyState>(array<string, 6>{"| = + !",
                                                 "| =    ",
                                                 "| =   !",
                                                 "  =    ",
                                                 "  =   !",
                                                 "- =    ",}), board);
    if(ss->validate(board, true) == false) throw runtime_error("Failed to validate correct board");
}
