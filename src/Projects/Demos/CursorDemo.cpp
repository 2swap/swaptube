#include "../Scenes/Media/CursorScene.h"

void render_video() {
    CursorScene cursor;

    stage_macroblock(SilenceBlock(4));
    cursor.render_microblock();
    cursor.manager.transition(MICRO, "cursor_x", "0.2");
    cursor.render_microblock();
    cursor.manager.transition(MICRO, "cursor_y", "0.8");
    cursor.render_microblock();
    cursor.manager.transition(MICRO, {{"cursor_x", "0.6"},{"cursor_y", "0.3"}});
    cursor.render_microblock();
    cursor.render_microblock();


    stage_macroblock(SilenceBlock(3));
    cursor.manager.transition(MICRO, "cursor_size", "0.05");
    cursor.render_microblock();
    cursor.manager.transition(MICRO, "cursor_size", "0.3");
    cursor.render_microblock();
    cursor.manager.transition(MICRO, "cursor_size", "0.1");
    cursor.render_microblock();
    cursor.render_microblock();


    stage_macroblock(SilenceBlock(4));
    cursor.manager.transition(MICRO, "fill_g", "128");
    cursor.render_microblock();
    cursor.manager.transition(MICRO, {{"stroke_r", "128"},{"stroke_b", "128"}});
    cursor.render_microblock();
    cursor.manager.transition(MICRO, "fill_a", "128");
    cursor.render_microblock();
    cursor.manager.transition(MICRO, "fill_a", "0");
    cursor.render_microblock();
    cursor.manager.transition(MICRO, "fill_a", "255");
    cursor.render_microblock();
    cursor.render_microblock();
}
