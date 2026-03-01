#include "../Scenes/Math/GeodesicScene.h"

void render_video() {
    GeodesicScene gs;

    gs.manager.set({
        {"scrunch", "{t} sin .5 * 1 +"},
    });

    gs.manager.set(unordered_map<string, string>{
        {"space_x", "(a)"},
        {"space_y", "(b)"},
        {"space_z", "(c)"},
        {"space_w", "(c) <scrunch> * sin (a) cos (b) cos + +"}
    });

// *knock knock knock*
    gs.manager.transition(MICRO, {
        {"pov_fov", "1.5"},
        {"pov_q1", ".2"},
        {"pov_qj", "1"},
    });
    stage_macroblock(SilenceBlock(1), 1);
    gs.render_microblock();

    stage_macroblock(FileBlock("Interdimensional FBI! Open up!"), 1);
    gs.render_microblock();

    // Turn towards door

    stage_macroblock(FileBlock("*softly* Oh crap."), 1);
    gs.render_microblock();

    stage_macroblock(FileBlock("One moment!"), 1);
    gs.render_microblock();

// (2swap slams kill switch on space warping device, surfaces begin to flatten out)
// [Agent] Open the door, now!
// [2swap] Coming, coming!
// [Agent] *Bursts open the door* Put your hands where I can see them!
// [Agent] Do you live here?
// [2swap] Yes officer.
// [Agent] Our instruments show a large non-zero gaussian curvature in this room. What on earth are you doing in here?
// [2swap] It wasn't me, I swear!
// [Agent] Are you in possession of any counterfeit items?
// [2swap] Like, drugs?
// [Agent] No! Tesseracts, Klein bottles, Möbius strips...
// [2swap] Oh, I think I have seen my roommate playing with those before... (points at door)
// *Agent walks toward door*
// *2swap turns device back on*
// *Agent begins to yell, but voice is warped and cut off as space around him is warped and disconnected into a small bubble*
// [2swap] *muttering* How in the heck did they figure out...
// *2swap dials a number on his phone while fiddling with the device and warping space to and fro*
// [Dealer] Hello?
// [2swap] Dude, you told me this thing was legal!
// [Dealer] In the USA, no legislation has been passed regarding manipulating the fabric of space.
// [2swap] Then why is the Interdimensional FBI after me?
// [Dealer] Oh, yeah, I don't think there's any diplomatic relationship between the US and extradimensional law enforcement. Those guys make up their own laws. It's more like an activist group than anything else, really.
// [2swap] So, what? What they say doesn't matter?
// [Dealer] Oh, no, you're probably screwed. Manipulating space is punishable by death in the 4th dimension.
// [2swap] WHAT?
// [Dealer] ...
// [2swap] Why does an interdimensional activist group want me dead for using it!?!
// [Dealer] It's a party toy that manipulates the fabric of space. They can be used to create some problems...
// [2swap] What kind of problems?
// [Dealer] Things like interrupting radio communications by curving the space around the signal, or lensing large amounts of light to a single point and burning holes in things. Oh, and also making black holes and destroying the universe.
// [2swap] ...
// [Dealer] See the red button on the side?
// *2swap presses the red button*
// [Dealer] Whatever you do, just don't press that one, and you're probably fine. *black hole beginning to form*
// *2swap repeating 'oh no', spamming kill switch. Black hole shrinks away.*
// [Dealer] Are you alright?
// [2swap] *takes a moment to gather composure*
// [2swap] Let's start from the beginning. How exactly does this thing work?
// [Dealer] You've heard of space-time, right? Einstein conceived of space and time as being part of the same fabric, and that fabric can be manipulated by mass and energy.
// [Dealer] In my lab, we found a special kind of exotic matter which divorces the time component from the space components, allowing us to manipulate space alone.
// [2swap] What exactly do you mean by manipulating space? It, like, warps the things around me?
// [Dealer] No, it warps space itself. Think of it like this- Alaska and Norway look really far apart on a map. But when we curve that flat map into a globe, they end up close together. Furthermore, the straight line which connects the two is different on the globe than it is on the flat map. By manipulating space, geometry behaves differently.
// [Dealer] A map is two-dimensional, and by raising it into a three-dimensional space, we can curve it.
// [Dealer] That device allows you to change the curvature of your room as though you were raising it into the fourth dimension.
// [2swap] Are you saying my room is entering and leaving the fourth dimension?
// [Dealer] Well, maybe. As three dimensional beings, it's easy to imagine a curved two-dimensional space by embedding it into our three-dimensional world.
// [Dealer] This is an extrinsic perspective, since we perceive that curved space as it is embedded in our higher dimension.
// [Dealer] However, there is an equivalent point of view called the intrinsic perspective, where we can understand the geometry of a curved space from local measurements of angles, distances, and so on, without needing to embed it within a higher dimension at all.
// [Dealer] For example, when crocheting, one can create a flat sheet by making a regular grid of stitches. By altering the amount of stitches in different areas, we can create a curved surface.
// [Dealer] But distances in this surface don't depend on its embedding in three-dimensional space: all you need to know is the length of string present between two points.
// [Dealer] This is the intrinsic perspective. The notion of geometry is intrinsic to a particular space, independent of its representation.
// [Dealer] In differential geometry, this is known as Gauss's Theorema Egregium.
// [Dealer] Gauss showed that these two perspectives are mathematically equivalent. The device merely displays the extrinsic perspective to make it easier to visualize the curvature at play.
// [Dealer] At our lab, we still haven't discovered whether or not there is truly a higher-dimensional embedding being manipulated from a physical perspective, since both interpretations are predictively equivalent.
// [2swap] Ok... so if I'm understanding you right, the device manipulates the curvature of this space, within which my walls live, but it doesn't actually move the walls themselves. So, why do my walls look like they're moving when I turn it on?
// [Dealer] Remember how the straight line on the map is different from the straight line on the globe? It's the same idea here. Light itself follows straight lines. So if the geometry of space is warped, the straight line that light follows before reaching your eyes is different, and thus the image of the walls that you see is warped as well.
// [2swap] How can a straight line not be straight?
// [Dealer] You're right, it's not really a suitable word. The idea of straightness only really makes sense in a flat geometry, that is, a geometry we call Euclidean. In a non-Euclidean geometry, we call these straight lines geodesics.
// [Dealer] If you sail the arctic ocean from Alaska to Norway, to you the path seems straight, but on the globe, you are actually curving downwards since the Earth isn't flat. A geodesic is a curve like this- when you're stuck on a curvy space and walk straight without turning left or right, you are following a geodesic.
// [Dealer] The light inside your room follows a geodesic curve in non-euclidean space.

}

void old() {
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
