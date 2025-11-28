#include "../Scenes/Math/GeodesicScene.cpp"

void render_video() {
    FOR_REAL = false;
    GeodesicScene gs;

    gs.manager.set({
        {"pov_fov", "2"},
        {"pov_z", "5"},
        {"pov_manifold_opacity", "0"},
        {"pov_step_size", "0.01"},
        {"pov_step_count", "0"},
        {"pov_floor_y", "<pov_step_size> <pov_step_count> * -1 *"},
        {"pov_ceiling_y", "<pov_step_size> <pov_step_count> *"},
    });

    gs.stage_macroblock(FileBlock("Space is a place where there are things."), 2);
    gs.manager.transition(MICRO, {
        {"pov_step_count", "2000"},
        {"pov_q1", ".2"},
        {"pov_qj", "1"},
    });
    gs.render_microblock();
    gs.manager.transition(MICRO, {
        {"pov_q1", "1"},
        {"pov_qj", "0"},
    });
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("For example, this floor or this ceiling."), 4);
    gs.render_microblock();
    gs.manager.transition(MICRO, "pov_floor_y", "-1");
    gs.render_microblock();
    gs.manager.transition(MICRO, "pov_ceiling_y", "1");
    gs.render_microblock();
    gs.manager.transition(MICRO, "pov_grid_opacity", "0");
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("Floors and ceilings are usually flat..."), 1);
    gs.manager.transition(MICRO, {
        {"pov_q1", "1"},
        {"pov_qj", "{t} .3 * cos .1 *"},
    });
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("but they don't have to be."), 1);
    gs.manager.transition(MICRO, {
        {"pov_qi", "{t} .5 * sin .1 *"},
    });
    gs.manager.transition(MICRO, "pov_floor_distort", "1");
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("We can make them curve like this or like that..."), 1);
    gs.manager.transition(MICRO, "pov_floor_distort", "-1");
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("meaning, they don't follow the straight lines of some underlying coordinate system."), 1);
    gs.manager.transition(MICRO, {
        {"pov_floor_distort", "0"},
        {"pov_grid_opacity", "1"},
    });
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("Lines are great! They follow beside you when you walk straight."), 2);
    gs.manager.transition(MICRO, {
        {"pov_grid_opacity", "0"},
        {"pov_zaxis", "1"},
        {"pov_q1", "1"},
        {"pov_qi", "0"},
        {"pov_qj", "0"},
    });
    gs.render_microblock();
    gs.manager.transition(MICRO, "pov_z", "-5");
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("They always point the same way, so if I follow one, I won't go astray."), 1);
    gs.manager.transition(MICRO, "pov_z", "5");
    gs.render_microblock();

    gs.manager.set({
        {"pov_step_count", "0"},
    });

    gs.manager.set({
        {"geodesics_start_u", "-2.5"},
        {"geodesics_start_v", ".5"},
        {"geodesics_start_du", ".1"},
        {"geodesics_start_dv", "0"},
        {"geodesics_count", "1"},
        {"geodesics_spread_angle", ".1"},
    });

    gs.stage_macroblock(FileBlock("For an ant on a sheet walking from A to B, there's just one straight path, and it's always the shortest."), 3);
    gs.manager.transition(MICRO, {
        {"pov_zaxis", "0"},
        {"manifold_opacity", ".4"},
        {"geodesics_steps", "50"},
    });
    gs.render_microblock();
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(1), 1);
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("...except when there's not!"), 1);
    gs.manager.transition(MICRO, {
        {"manifold_z", "(a) sin (b) sin +"},
        {"geodesics_steps", "200"},
        {"pov_intensity_sin", "1"},
    });
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(1), 1);
    gs.manager.set({
        {"manifold_z", "(a) <warp> * sin (b) <warp> * sin + <warp> /"},
        {"warp", "1"},
    });
    gs.manager.transition(MICRO, {
        {"geodesics_steps", "300"},
        {"pov_intensity_sin", "1"},
        {"warp", "3"},
    });
    gs.render_microblock();

    gs.stage_macroblock(SilenceBlock(1), 1);
    gs.manager.transition(MICRO, {
        {"warp", "5"},
    });
    gs.render_microblock();

    StateSet undo = gs.manager.transition(MICRO, {
        {"manifold_q1", "1.0"},
        {"manifold_qi", "0"},
        {"manifold_qj", "0"},
        {"manifold_qk", "0"},
    });
    gs.stage_macroblock(SilenceBlock(1), 1);
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("Bending the space that the ant lives on, if it walks straight..."), 2);
    gs.manager.transition(MICRO, {
        {"manifold_z", "1 (a) (a) * (b) (b) * + .5 + /"},
        {"pov_intensity_sin", "0"},
        {"pov_intensity_witch", "1"},
    });
    gs.manager.transition(MICRO, undo);
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("its path is anything but!"), 2);
    gs.render_microblock();
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("We can imagine all of the straight paths that it could take."), 2);
    gs.manager.transition(MICRO, {
        {"manifold_z", "0"},
        {"geodesics_steps", "0"},
    });
    gs.render_microblock();
    gs.manager.set({
        {"geodesics_count", "10000"},
        {"geodesics_opacity", ".03"},
        {"geodesics_spread_angle", "pi 2 *"},
    });
    gs.manager.transition(MICRO, {
        {"geodesics_steps", "100"},
    });
    gs.render_microblock();

    gs.stage_macroblock(FileBlock("and you'll notice strange effects between them when space is curved."), 1);
    gs.manager.transition(MICRO, "manifold_z", "1 (a) (a) * (b) (b) * + .5 + /");
    gs.render_microblock();

    FOR_REAL = true;
    gs.stage_macroblock(SilenceBlock(5), 3);
    gs.manager.transition(MICRO, "manifold_z", "0");
    gs.render_microblock();
    gs.manager.transition(MICRO, "manifold_z", "(a) <warp> * sin (b) <warp> * sin + <warp> /");
    gs.render_microblock();
    gs.render_microblock();

    gs.manager.transition(MICRO, {
        {"geodesics_spread_angle", "pi .2 *"},
        {"geodesics_count", "1000"},
    });
    return;

    gs.stage_macroblock(FileBlock("Since light follows a straight line to get to the ant,"), 1);
    gs.stage_macroblock(FileBlock("a straight walk of the ant is the same path as its line of sight."), 1);

    // Now, this is all cool, but of course it depends on us embedding this curved space into our usual three-dimensional world.
    // Luckily for us three-dimensionalites, our world can't be bent around in three-dimensional space, since it already is three-dimensional space. Right?
    // (Voice) Mwahahaha!
    // *Shriek* Who are you!? What did you do to my floor and my ceiling?!
    // (Voice) You think you're so special with your voluminous third dimension. You're no different than that ant! I've come from the fourth dimension to teach you a thing or two!

    // echo??
    gs.stage_macroblock(FileBlock("If the ant were in a room,"), 1);
    gs.stage_macroblock(FileBlock("by looking in this direction,"), 1);
    gs.stage_macroblock(FileBlock("it would see the wall behind it!"), 1);
    gs.stage_macroblock(FileBlock("Its entire field of view would be warped out of shape."), 1);

    gs.stage_macroblock(FileBlock("What's more, there are now several straight lines from A to B!"), 1);
    gs.stage_macroblock(FileBlock("If the ant starts walking in any given direction,"), 1);
    gs.stage_macroblock(FileBlock("it's no longer easy to tell where it will end up."), 1);
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
