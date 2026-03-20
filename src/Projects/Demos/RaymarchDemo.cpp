#include "../Scenes/Math/RaymarchScene.h"

void render_video() {
    RaymarchScene rs;
    rs.manager.set({
        {"camera_x", "{t} pi * 8 / sin 2 *"},
        {"camera_y", "1"}, 
        {"camera_z", "{t} pi * 8 / cos -2 *"}});
    stage_macroblock(SilenceBlock(2), 1);
    rs.render_microblock();
}