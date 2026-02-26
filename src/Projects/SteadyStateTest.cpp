#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Connect4/Connect4Scene.h"
#include "../Scenes/Connect4/Connect4GraphScene.h"
#include "../DataObjects/Connect4/TreeValidator.h"

void render_video() {
    SAVE_FRAME_PNGS = false;

    CompositeScene cs;

    Graph g;
    string variation = "4261";
    if(variation.size()%2 != 0) throw runtime_error("Variation must be even length");
    C4Board board(variation);
    board.print();
    shared_ptr<SteadyState> ss = modify_child_suggestion(
       make_shared<SteadyState>(array<string, 6>{"       ",
                                                 "       ",
                                                 "       ",
                                                 "       ",
                                                 "       ",
                                                 "    !  ",}), board);

    ss->print();
    cout << "Validated? " << ss->validate(board, true) << endl;
}
