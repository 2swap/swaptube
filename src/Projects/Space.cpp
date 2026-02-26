#include "../Scenes/Math/GeodesicScene.h"

void render_video() {
    GeodesicScene gs;

    gs.manager.set({
        {"pov_fov", "2"},
        {"pov_z", "5"},
        {"manifold_opacity", "0"},
        {"pov_grid_thickness", "0.1"},
        {"pov_floor_y", "<pov_max_dist> -1 *"},
        {"pov_ceiling_y", "<pov_max_dist>"},
    });

    stage_macroblock(FileBlock("Space is a place where there are things."), 2);
    gs.manager.transition(MICRO, {
        {"pov_max_dist", "20"},
        {"pov_q1", ".2"},
        {"pov_qj", "1"},
    });
    gs.render_microblock();
    gs.manager.transition(MICRO, {
        {"pov_q1", "1"},
        {"pov_qj", "0"},
    });
    gs.render_microblock();

    stage_macroblock(FileBlock("For example, this floor or this ceiling."), 4);
    gs.render_microblock();
    gs.manager.transition(MICRO, "pov_floor_y", "-1");
    gs.render_microblock();
    gs.manager.transition(MICRO, "pov_ceiling_y", "1");
    gs.render_microblock();
    gs.manager.transition(MICRO, "pov_grid_thickness", "0");
    gs.render_microblock();

    stage_macroblock(FileBlock("Floors and ceilings are usually flat..."), 1);
    gs.manager.transition(MICRO, {
        {"pov_q1", "1"},
        {"pov_qj", "{t} .3 * cos .1 *"},
    });
    gs.render_microblock();

    stage_macroblock(FileBlock("meaning, they follow the straight lines of some underlying coordinate system."), 1);
    gs.manager.transition(MICRO, {
        {"pov_grid_thickness", ".1"},
    });
    gs.render_microblock();

    stage_macroblock(FileBlock("Lines are great! They follow beside you when you walk straight."), 2);
    gs.manager.transition(MICRO, {
        {"pov_q1", "1"},
        {"pov_qi", "0"},
        {"pov_qj", "0"},
    });
    gs.render_microblock();
    gs.manager.transition(MICRO, "pov_z", "-5");
    gs.render_microblock();

    stage_macroblock(FileBlock("They always point the same way, so if I follow one, I won't go astray."), 1);
    gs.manager.transition(MICRO, "pov_z", "5");
    gs.render_microblock();

    gs.manager.set({
        {"pov_max_dist", "0"},
    });

    gs.manager.set({
        {"geodesics_count", "1"},
        {"geodesics_spread_angle", ".1"},
    });

    stage_macroblock(FileBlock("For an ant on a sheet walking from A to B, there's just one straight path, and it's always the shortest."), 3);
    gs.manager.transition(MICRO, {
        {"manifold_opacity", ".4"},
        {"geodesics_steps", "50"},
    });
    gs.render_microblock();
    gs.render_microblock();
    gs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    gs.render_microblock();

    stage_macroblock(FileBlock("...except when there's not!"), 1);
    gs.manager.transition(MICRO, {
        {"space_w", "(a) sin (b) sin +"},
        {"geodesics_steps", "200"},
    });
    gs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    gs.manager.set({
        {"space_w", "(a) <warp> * sin (b) <warp> * sin + <warp> /"},
        {"warp", "1"},
    });
    gs.manager.transition(MICRO, {
        {"geodesics_steps", "300"},
        {"warp", "3"},
    });
    gs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    gs.manager.transition(MICRO, {
        {"warp", "5"},
    });
    gs.render_microblock();

    StateSet undo = gs.manager.transition(MICRO, {
        {"pov_q1", "1.0"},
        {"pov_qi", "0"},
        {"pov_qj", "0"},
        {"pov_qk", "0"},
    });
    stage_macroblock(SilenceBlock(1), 1);
    gs.render_microblock();

    stage_macroblock(FileBlock("Bending the space that the ant lives on, if it walks straight..."), 2);
    gs.manager.transition(MICRO, {
        {"space_w", "1 (a) (a) * (b) (b) * + .5 + /"},
    });
    gs.manager.transition(MICRO, undo);
    gs.render_microblock();
    gs.render_microblock();

    stage_macroblock(FileBlock("its path is anything but!"), 2);
    gs.render_microblock();
    gs.render_microblock();

    stage_macroblock(FileBlock("We can imagine all of the straight paths that it could take."), 2);
    gs.manager.transition(MICRO, {
        {"space_w", "0"},
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

    stage_macroblock(FileBlock("and you'll notice strange effects between them when space is curved."), 1);
    gs.manager.transition(MICRO, "space_w", "1 (a) (a) * (b) (b) * + .5 + /");
    gs.render_microblock();

    stage_macroblock(SilenceBlock(5), 3);
    gs.manager.transition(MICRO, "space_w", "0");
    gs.render_microblock();
    gs.manager.transition(MICRO, "space_w", "(a) <warp> * sin (b) <warp> * sin + <warp> /");
    gs.render_microblock();
    gs.render_microblock();

    gs.manager.transition(MICRO, {
        {"geodesics_spread_angle", "pi .2 *"},
        {"geodesics_count", "1000"},
    });
    return;

    stage_macroblock(FileBlock("Since light follows a straight line to get to the ant,"), 1);
    stage_macroblock(FileBlock("a straight walk of the ant is the same path as its line of sight."), 1);

    // Now, this is all cool, but of course it depends on us embedding this curved space into our usual three-dimensional world.
    // Luckily for us three-dimensionalites, our world can't be bent around in three-dimensional space, since it already is three-dimensional space. Right?
    // (Voice) Mwahahaha!
    // *Shriek* Who are you!? What did you do to my floor and my ceiling?!
    // (Voice) You think you're so special with your voluminous third dimension. You're no different than that ant! I've come from the fourth dimension to teach you a thing or two!

    // echo??
    stage_macroblock(FileBlock("If the ant were in a room,"), 1);
    stage_macroblock(FileBlock("by looking in this direction,"), 1);
    stage_macroblock(FileBlock("it would see the wall behind it!"), 1);
    stage_macroblock(FileBlock("Its entire field of view would be warped out of shape."), 1);

    stage_macroblock(FileBlock("What's more, there are now several straight lines from A to B!"), 1);
    stage_macroblock(FileBlock("If the ant starts walking in any given direction,"), 1);
    stage_macroblock(FileBlock("it's no longer easy to tell where it will end up."), 1);
    stage_macroblock(FileBlock("Something like this."), 1);
    stage_macroblock(FileBlock("By changing how its space is bent, we change the nature of geometry in that space."), 1);
    stage_macroblock(FileBlock("Straight lines are no longer straight, parallel lines intersect, three right turns get you back to where you began..."), 1);

    stage_macroblock(FileBlock("But, you say, we haven't changed space itself at all!"), 1);
    stage_macroblock(FileBlock("We've just warped the ant's ground into a higher dimension."), 1);
    stage_macroblock(FileBlock("It still implicitly exists within our usual three-dimensional world."), 1);

    stage_macroblock(FileBlock("Hold that thought."), 1);
    stage_macroblock(FileBlock("Imagine a set of points, and only when two are connected, they have a distance of 1."), 1);
    stage_macroblock(FileBlock("If we make a fabric of these points, it's just like a flat plane."), 1);
    stage_macroblock(FileBlock("But if we start adding more connections..."), 1);
    stage_macroblock(FileBlock(""), 1);
    stage_macroblock(FileBlock(""), 1);
    stage_macroblock(FileBlock(""), 1);
    stage_macroblock(FileBlock(""), 1);
    stage_macroblock(FileBlock(""), 1);
    stage_macroblock(FileBlock(""), 1);
}
