#include "../Scenes/Math/GeodesicScene.cpp"

void render_video() {
    GeodesicScene gs;

    FOR_REAL = false;
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
        {"manifold_z", "(a) sin (b) sin +"},
        {"intensity_sin", "1"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("If we bend the space that the ant lives on, if it walks straight..."), 2);
    gs.manager.transition(MICRO, {
        {"manifold_z", "1 (a) (a) * (b) (b) * + .5 + /"},
        {"intensity_sin", "0"},
        {"intensity_witch", "1"},
    });
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("the path it takes is anything but!"), 2);
    gs.render_microblock();
    gs.render_microblock();

    FOR_REAL = true;
    gs.manager.set({
        {"manifold_z", "(a) sin (b) sin +"},
        {"step_count", "0"},
        {"manifold_opacity", ".3"},
        {"num_geodesics", "800"},
        {"spread_angle", ".1"},
        {"geodesics_start_du", ".1"},
        {"manifold_d", "20"},
    });
    gs.manager.transition(MICRO, {
        {"geodesic_steps", "100"},
    });
    gs.stage_macroblock(SilenceBlock(2), 1);
    gs.render_microblock();

    return;
    gs.stage_macroblock(FileBlock("The ant is walking straight because, in its perspective, over the course of its journey, it never turns left or turns right."), 1);
    gs.stage_macroblock(FileBlock("You might say that the ant's line isn't straight,"), 1);
    gs.stage_macroblock(FileBlock("but what would you know? You don't live in its space!"), 1);

    gs.stage_macroblock(FileBlock("What's more, there are now several straight lines from A to B!"), 2);
    gs.stage_macroblock(FileBlock("If the ant starts walking in any given direction,"), 2);
    gs.stage_macroblock(FileBlock("it's no longer easy to tell where it will end up."), 1);
    gs.stage_macroblock(FileBlock("Let's follow every path that the ant could take."), 1);
    gs.stage_macroblock(FileBlock("Since light follows a straight line to get to the ant,"), 1);
    gs.stage_macroblock(FileBlock("a straight walk of the ant follows the same paht as its line of sight."), 1);
    gs.stage_macroblock(FileBlock("If the ant were in a room,"), 1);
    gs.stage_macroblock(FileBlock("by looking in this direction,"), 1);
    gs.stage_macroblock(FileBlock("it would see the wall behind it!"), 1);
    gs.stage_macroblock(FileBlock("Its entire field of view would be warped out of shape."), 1);
    gs.stage_macroblock(FileBlock("Something like this."), 1);
    gs.stage_macroblock(FileBlock("By changing how its space is bent, we change the nature of geometry in that space."), 1);
    gs.stage_macroblock(FileBlock("Straight lines are no longer straight, parallel lines intersect, three right turns get you back to where you began..."), 1);

    gs.stage_macroblock(FileBlock("But, you say, we haven't changed space itself at all!"), 1);
    gs.stage_macroblock(FileBlock("We've just warped the ant's ground into a higher dimension."), 1);
    gs.stage_macroblock(FileBlock("It still implicitly exists within our usual three-dimensional world."), 1);

    gs.stage_macroblock(FileBlock("Hold that thought."), 1);
    gs.stage_macroblock(FileBlock("Imagine a set of points, and only when two are connected, they have a distance of 1."), 1);
    gs.stage_macroblock(FileBlock("If we make a fabric of these points, it's just like a flat plane."), 1);
    gs.stage_macroblock(FileBlock("But if we start adding more connections..."), 1);
    gs.stage_macroblock(FileBlock(""), 1);
    gs.stage_macroblock(FileBlock(""), 1);
    gs.stage_macroblock(FileBlock(""), 1);
    gs.stage_macroblock(FileBlock(""), 1);
    gs.stage_macroblock(FileBlock(""), 1);
    gs.stage_macroblock(FileBlock(""), 1);
}
