#include "../Scenes/Math/AngularFractalScene.cpp"

void render_video() {
    int depth = 12;
    int dragon_size = 1 << depth;
    AngularFractalScene dragon(dragon_size);
    dragon.manager.set("zoom", "1");

    dragon.manager.transition(MACRO, "zoom", "7.5");
    dragon.stage_macroblock(SilenceBlock(4), depth);
    for(int i = 0; i < depth; i++) {
        int j = 0;
        int multiple = 1 << (depth-i);
        for(int angle = multiple >> 1; angle < dragon_size; angle += multiple) {
            if(angle == 0) continue;
            dragon.manager.transition(MICRO, "angle_" + to_string(angle), "pi .999 *" + string((j%2)?" -1 *" : ""));
            j++;
        }
        dragon.render_microblock();
    }

    dragon.manager.transition(MACRO, "zoom", "8.5");
    dragon.stage_macroblock(SilenceBlock(1), 1);
    dragon.render_microblock();

    dragon.manager.transition(MACRO, "zoom", "5.5");
    dragon.stage_macroblock(SilenceBlock(4), depth);
    for(int i = depth-1; i >= 0; i--) {
        int j = 0;
        int multiple = 1 << (depth-i);
        for(int angle = multiple >> 1; angle < dragon_size; angle += multiple) {
            if(angle == 0) continue;
            dragon.manager.transition(MICRO, "angle_" + to_string(angle), "pi .5 *" + string((j%2)?" -1 *" : ""));
            j++;
        }
        dragon.render_microblock();
    }
}
