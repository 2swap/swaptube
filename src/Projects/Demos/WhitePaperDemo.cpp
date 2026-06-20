#include "../Scenes/Media/WhitePaperScene.h"

void render_video() {
    WhitePaperScene ps("sample", "Author", vector<int>{1, 2, 3, 5});
    stage_macroblock(SilenceBlock(2), 3);
    ps.manager.transition(MICRO, "completion", "1");
    ps.render_microblock();
    ps.manager.set("which_page", "2");
    ps.manager.transition(MICRO, "page_focus", "1");
    ps.render_microblock();
    ps.manager.transition(MICRO, {
        {"crop_top", ".3"},
        {"crop_bottom", ".6"},
        {"crop_left", ".1"},
        {"crop_right", ".5"},
    });
    ps.render_microblock();
}
