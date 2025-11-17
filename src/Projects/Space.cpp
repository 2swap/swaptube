#include "../Scenes/Math/GeodesicScene.cpp"

void render_video() {
    GeodesicScene gs;

    gs.manager.set({{"fov", "2"}, {"z", "5"}, {"manifold_opacity", "0"}, {"step_size", "0.01"}, {"step_count", "0"},
            {"floor_y", "<step_size> <step_count> * -1 *"},
            {"ceiling_y", "<step_size> <step_count> *"},
    });

    gs.stage_macroblock(FileBlock("Space is a place where there are things."), 2);
    gs.manager.transition(MICRO, {
        {"step_count", "5000"},
        {"q1", ".2"},
        {"qj", "1"},
    });
    gs.render_microblock();
    gs.manager.transition(MICRO, {
        {"q1", "1"},
        {"qj", "0"},
    });
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("For example, this floor or this ceiling."), 4);
    gs.render_microblock();
    gs.manager.transition(MICRO, "floor_y", "-1");
    gs.render_microblock();
    gs.manager.transition(MICRO, "ceiling_y", "3");
    gs.render_microblock();
    gs.manager.transition(MICRO, "grid_opacity", "0");
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("Floors and ceilings are usually flat..."), 1);
    gs.manager.transition(MICRO, {
        {"q1", "1"},
        {"qj", "{t} .3 * cos .1 *"},
    });
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("but they don't have to be."), 1);
    gs.manager.transition(MICRO, {
        {"qi", "{t} .5 * sin .1 *"},
    });
    gs.manager.transition(MICRO, "floor_distort", "1");
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("We can make them curve like this or like that..."), 1);
    gs.manager.transition(MICRO, "floor_distort", "-1");
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("meaning, they don't follow the straight lines of some underlying coordinate system."), 1);
    gs.manager.transition(MICRO, {
        {"floor_distort", "0"},
        {"grid_opacity", "1"},
    });
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("Lines are great! They follow beside you when you walk straight."), 2);
    gs.manager.transition(MICRO, {
        {"grid_opacity", "0"},
        {"zaxis", "1"},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
    });
    gs.render_microblock();
    gs.manager.transition(MICRO, "z", "-5");
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("They always point the same way, so if I follow one, I won't go astray."), 1);
    gs.manager.transition(MICRO, "z", "5");
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("For an ant on a sheet who wants to go from A to B, there's just one straight path, and it's always the shortest."), 3);
    gs.manager.transition(MICRO, {
        {"zaxis", "0"},
        {"manifold_opacity", "1"},
    });
    gs.render_microblock();
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("...except when there's not!"), 2);
    gs.manager.transition(MICRO, {
        {"manifold_z", "(u) sin (v) sin +"},
        {"intensity_sin", "1"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("If we bend the space that the ant lives on, if it walks straight..."), 2);
    gs.manager.transition(MICRO, {
        {"manifold_z", "1 (u) (u) * (v) (v) * + .5 + /"},
        {"intensity_sin", "0"},
        {"intensity_witch", "1"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("the path it takes is anything but!"), 2);
    gs.render_microblock();
    gs.render_microblock();
}
