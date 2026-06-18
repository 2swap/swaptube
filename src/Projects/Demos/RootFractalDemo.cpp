#include "../Scenes/Math/RootFractalScene.h"
#include "../IO/Writer.h"

void render_video() {
    RootFractalScene rfs;
    rfs.manager.set({
        {"coefficient0_r", "2.2 {t} - 3 * 7 - 8 / sin"},
        {"coefficient0_i", "2.2 {t} - 3 * 7 - 9 / sin"},
        {"center_x", "-.3"},
        {"center_y", ".6"},
        {"zoom", "2.5"},
        {"terms", "20"},
        {"coefficients_opacity", "0"},
        {"ticks_opacity", "0"},
    });
    stage_macroblock(SilenceBlock(1.5), 1);
    rfs.render_microblock();
}
