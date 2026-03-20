#include "../Scenes/Math/GeodesicScene.h"
#include "../Scenes/Math/ManifoldScene.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Core/Smoketest.h"

void first_half() {
    shared_ptr<GeodesicScene> gs = make_shared<GeodesicScene>();

    gs->manager.set({
        {"scrunch_x", "2"},
        {"scrunch_y", "2"},
        {"scrunch_z", "2"},
        {"amp", "0"},
        {"pov_qj", "1"},
    });

    gs->manager.set({
        {"space_x", "(a)"},
        {"space_y", "(b)"},
        {"space_z", "(c)"},
        {"space_w", "(a) <scrunch_x> * sin (b) <scrunch_y> * sin (c) <scrunch_z> * sin + + <amp> *"},
    });
    gs->manager.transition(MACRO, {
        {"pov_q1", "-.5"},
        {"pov_qj", "1"},
    });
    stage_macroblock(SilenceBlock(2), 2);
    gs->manager.transition(MICRO, {
        {"amp", ".25"}
    });
    gs->render_microblock();
    gs->manager.transition(MICRO, {
        {"amp", "0"}
    });
    gs->render_microblock();

    gs->manager.set({
        {"scrunch_x", "10 {t} sin +"},
        {"scrunch_y", "10 {t} cos +"},
        {"scrunch_z", "10"},
    });

    stage_macroblock(SilenceBlock(1), 1);
    gs->manager.transition(MICRO, {
        {"amp", ".05"}
    });
    gs->render_microblock();

    stage_macroblock(FileBlock("Woooahhh!"), 2);
    gs->render_microblock();
    gs->manager.transition(MICRO, {
        {"pov_x", "1"},
    });
    gs->render_microblock();

    stage_macroblock(FileBlock("*knock knock knock* Interdimensional FBI! Open up!"), 1);
    gs->render_microblock();

    // Turn towards door
    gs->manager.transition(MICRO, {
        {"pov_qj", "0"},
        {"pov_q1", "1"},
    });

    stage_macroblock(FileBlock("(softly) Oh crap."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Open up now!"), 1);
    gs->manager.transition(MICRO, {
        {"pov_x", "0"},
        {"pov_z", "2"},
    });
    // (2swap slams kill switch on space warping device, surfaces begin to flatten out)
    gs->manager.transition(MICRO, {
        {"space_w", "0"},
    });
    gs->render_microblock();

    stage_macroblock(FileBlock("(Door bursts open) Put your hands where I can see them!"), 2);
    gs->manager.set("sphere_radius", ".1 {voice} * .5 +");
    gs->manager.transition(MICRO, "sphere_z", "3");
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Do you live here?"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Y-yes!"), 4);
    // Nod head
    gs->manager.transition(MICRO, {
        {"pov_qi", "{t} 15 * sin .05 *"},
    });
    gs->render_microblock();
    gs->render_microblock();
    gs->manager.transition(MICRO, {
        {"pov_qi", "0"},
    });
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Our instruments show a large non-zero gaussian curvature in this room. What on earth are you doing in here?"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("It wasn't me, I swear!"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Are you in possession of any contraband items?"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("What, like, drugs?"), 1);
    gs->render_microblock();

    CompositeScene cs;
    cs.add_scene(gs, "gs");
    shared_ptr<ManifoldScene> ms = make_shared<ManifoldScene>(vec2(.5, .5));
    cs.add_scene(ms, "ms");
    ms->add_manifold("",
        "0", "0", "0",
        ".5", ".5",
        "0", "0", "5000",
        "0", "0", "5000"
    );

    ms->manager.set({
        {"q1", "1"},
        {"qi", "{t} .7 * sin .2 *"},
        {"qj", "{t} .5 * cos .2 *"},
        {"qk", "0"},
    });

    StateSet plane1{
        {"manifold_x", "(a) 16 / 1 -"},
        {"manifold_y", "(b)"},
        {"manifold_z", "0"},
        {"manifold_a_min", "0"},
        {"manifold_a_max", "32"},
        {"manifold_a_steps", "5000"},
        {"manifold_b_min", "-1"},
        {"manifold_b_max", "1"},
        {"manifold_b_steps", "5000"},
    };
    StateSet plane2{
        {"manifold_x", "(a)"},
        {"manifold_y", "(b)"},
        {"manifold_z", "0"},
        {"manifold_a_min", "-1"},
        {"manifold_a_max", "1"},
        {"manifold_a_steps", "5000"},
        {"manifold_b_min", "-1"},
        {"manifold_b_max", "1"},
        {"manifold_b_steps", "5000"},
    };

    ms->manager.set("d", "8");

    string radius = "(a) 1.57079632 - sin 10 ^ .3 +";
    string curve = "(a) .2 + sin 10 ^ 2 * (a) -";
    string x_skeleton = "(a) sin 10 ^";
    string y_skeleton = "(a) 2 * .78539816339 + sin 2 *";
    StateSet klein{
        {"manifold_x", radius + " (b) sin " + curve + " cos * * " + x_skeleton + " +"},
        {"manifold_y", radius + " (b) sin " + curve + " sin * * " + y_skeleton + " +"},
        {"manifold_z", radius + " (b) cos *"},
        {"manifold_a_min", "-1.57079632"},
        {"manifold_a_max", "1.57079632"},
        {"manifold_a_steps", "5000"},
        {"manifold_b_min", "-3.14159265359"},
        {"manifold_b_max", "3.14159265359"},
        {"manifold_b_steps", "5000"},
    };

    StateSet mobius{
        {"manifold_x", "2 (a) * sin (b) (a) cos * +"},
        {"manifold_y", "2 (a) * cos (b) (a) sin * +"},
        {"manifold_z", "(b) (a) cos *"},
        {"manifold_a_min", "-1.57079632"},
        {"manifold_a_max", "1.57079632"},
        {"manifold_a_steps", "5000"},
        {"manifold_b_min", "-.5"},
        {"manifold_b_max", ".5"},
        {"manifold_b_steps", "5000"},
    };

    StateSet tesseract_points{};
    for(int i = 0; i < 16; i++) {
        string idx = to_string(i);
        int _x = ((i/1)%2) * 2 - 1;
        int _y = ((i/2)%2) * 2 - 1;
        int _z = ((i/4)%2) * 2 - 1;
        int _w = ((i/8)%2) * 2 - 1;
        tesseract_points["tess_point" + idx + "_w"] = to_string(_w) + " {t} sin * " + to_string(_x) + " {t} cos * +";
        string div = " <tess_point" + idx + "_w> 2 + /";
        tesseract_points["tess_point" + idx + "_x"] = to_string(_x) + " {t} sin * " + to_string(_w) + " {t} cos * -" + div;
        tesseract_points["tess_point" + idx + "_y"] = to_string(_y) + div;
        tesseract_points["tess_point" + idx + "_z"] = to_string(_z) + div;// + " {t} sin * " + to_string(_x) + " {t} cos * +" + div;
    }

    ms->manager.set(tesseract_points);

    string mx = "";
    string my = "";
    string mz = "";
    // For each of the 32 edges in 4-cube,
    for(int i = 0; i < 32; i++) {
        string idx = to_string(i);
        string nidx = to_string(i+1);
        int axis = 1 << (i / 8); // The bit on which src and dst differ
        int axis_blur = axis | (axis << 1) | (axis << 2) | (axis << 3) | (axis << 4);
        int which_edge = i % 8;
        int base_mask = (which_edge & ~axis_blur) | ((which_edge & axis_blur) << 1);
        int src_i = base_mask;
        int dst_i = base_mask | axis;
        string src = to_string(src_i);
        string dst = to_string(dst_i);
        mx += "(a) " + idx + " >= (a) " + nidx + " < * <tess_point" + src + "_x> <tess_point" + dst + "_x> (a) " + idx + " - lerp * ";
        my += "(a) " + idx + " >= (a) " + nidx + " < * <tess_point" + src + "_y> <tess_point" + dst + "_y> (a) " + idx + " - lerp * ";
        mz += "(a) " + idx + " >= (a) " + nidx + " < * <tess_point" + src + "_z> <tess_point" + dst + "_z> (a) " + idx + " - lerp * ";
    }

    // Add trailing sum
    for(int i = 0; i < 31; i++) {
        mx += "+ ";
        my += "+ ";
        mz += "+ ";
    }

    mx += "(b) sin .05 * +";
    my += "(b) cos .05 * +";
    mz += "(b) .2 + cos .05 * +";

    StateSet tesseract{
        {"manifold_x", mx},
        {"manifold_y", my},
        {"manifold_z", mz},
        {"manifold_a_min", "0"},
        {"manifold_a_max", "31.999999"},
        {"manifold_a_steps", "40000"},
        {"manifold_b_min", "0"},
        {"manifold_b_max", "6.28"},
        {"manifold_b_steps", "100"},
    };

    stage_macroblock(FileBlock("No! Tesseracts, Möbius strips, or perhaps... a Klein bottle?"), 17);
    ms->manager.set(plane1);
    cs.render_microblock();
    ms->manager.transition(MICRO, tesseract);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    ms->manager.transition(MICRO, plane1);
    cs.render_microblock();
    ms->manager.set(plane2);
    ms->manager.transition(MICRO, mobius);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    ms->manager.transition(MICRO, klein);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("Oh, I think I saw my roommate playing with those before... (points at door)"), 1);
    gs->manager.transition(MICRO, {
        {"pov_q1", "1"},
        {"pov_qj", "-1"},
    });
    gs->render_microblock();

    stage_macroblock(FileBlock("*Agent walks toward door*"), 1);
    gs->manager.transition(MICRO, {
        {"sphere_x", "3"},
        {"sphere_z", "0"},
    });
    gs->render_microblock();

    return;

    // *Agent walks toward door*
    // *2swap turns device back on*

    stage_macroblock(FileBlock("[Agent begins to yell]"), 1);
    gs->render_microblock();

    // but voice is warped and cut off as space around him is warped and disconnected into a small bubble*

    stage_macroblock(FileBlock("How in the heck did they figure out..."), 1);
    gs->manager.set("sphere_radius", "0");
    gs->render_microblock();

    // 2swap dials a number on his phone while fiddling with the device and warping space to and fro*

    stage_macroblock(FileBlock("Hello?"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Dude, you told me this thing was legal!"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("In the USA, no legislation has been passed regarding manipulating the fabric of space."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Then why is the Interdimensional FBI after me?"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Oh, yeah, I don't think there's any diplomatic relationship between the US and extradimensional law enforcement. Those guys make up their own laws. It's more like an activist group than anything else, really."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("So, what? What they say doesn't matter?"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Oh, no, you're probably screwed. Manipulating space is punishable by death in the 4th dimension."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("WHAT?"), 1);
    gs->render_microblock();

// [Dealer] ...

    stage_macroblock(FileBlock("Why does an interdimensional activist group want me dead for using it!?!"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("It's a party toy that manipulates the fabric of space. They can be used to create some problems..."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("What kind of problems?"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Things like interrupting radio communications by curving the space around the signal, or lensing large amounts of light to a single point and burning holes in things. Oh, and also making black holes and destroying the universe."), 1);
    gs->render_microblock();

    // [2swap] ...
    stage_macroblock(FileBlock("See the red button on the side?"), 1);
    gs->render_microblock();

    // 2swap presses the red button

    stage_macroblock(FileBlock("Whatever you do, just don't press that one, and you're probably fine."), 1);
    gs->render_microblock();

    // *black hole beginning to form*
    stage_macroblock(FileBlock("oh no oh no oh no"), 1);
    gs->render_microblock();
    // 2swap spams kill switch. Black hole shrinks away.

    stage_macroblock(FileBlock("Are you alright?"), 1);
    gs->render_microblock();
    // 2swap takes a moment to gather composure

    stage_macroblock(FileBlock("Let's start from the beginning. How exactly does this thing work?"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("You've heard of space-time, right? Einstein conceived of space and time as being part of the same fabric, and that fabric can be manipulated by mass and energy."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("In my lab, we found a special kind of exotic matter which divorces the time component from the space components, allowing us to manipulate space alone."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("What exactly do you mean by manipulating space? It, like, warps the things around me?"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("No, it warps space itself."), 1);
    gs->render_microblock();
}

void globe() {
    shared_ptr<ManifoldScene> ms = make_shared<ManifoldScene>();

    StateSet sphere{
        {"manifold_x", "(a) cos (b) sin *"},
        {"manifold_y", "(b) cos"},
        {"manifold_z", "(a) sin (b) sin *"},
    };
    StateSet plane{
        {"manifold_x", "(a) .5 *"},
        {"manifold_y", "(b) -.5 * pi .25 * +"},
        {"manifold_z", "0"},
    };
    StateSet warpy_plane{
        {"manifold_x", "(a) .5 *"},
        {"manifold_y", "(a) 4 * sin (b) 4 * sin + .2 * (b) -.2 * pi .1 * + +"},
        {"manifold_z", "(b) -.5 * pi .25 * +"},
    };

    ms->add_manifold("",
        "0", "0", "0",
        "0", "0",
        "pi -1 *", "pi", "5000",
        "0", "pi", "5000"
    );

    ms->manager.set(plane);

    ms->manager.set({
        {"d", "3"},
    });

    Pixels pix;
    png_to_pix(pix, "map2");
    ms->set_texture(pix);

    stage_macroblock(FileBlock("Think of it like this- Alaska and Norway look really far apart on a map."), 1);
    ms->render_microblock();

    stage_macroblock(FileBlock("But when we curve that flat map into a globe, they end up close together."), 2);
    ms->manager.transition(MICRO, sphere);
    ms->manager.transition(MACRO, { {"q1", "1"}, {"qi", ".8"}, {"qj", "0"}, {"qk", "0"}, });
    ms->render_microblock();
    ms->manager.transition(MICRO, "d", "2");
    ms->render_microblock();

    stage_macroblock(FileBlock("Furthermore, the straight line which connects the two is different on the globe than it is on the flat map."), 5);
    ms->manager.transition(MICRO, "d", "3");
    ms->render_microblock();
    ms->render_microblock();
    ms->render_microblock();
    ms->render_microblock();
    ms->manager.transition(MICRO, plane);
    ms->manager.transition(MICRO, { {"q1", "1"}, {"qi", "0"}, {"qj", "0"}, {"qk", "0"}, });
    ms->render_microblock();

    stage_macroblock(FileBlock("A map is two-dimensional, but by raising it into a three-dimensional space, we can manipulate its internal geometry."), 4);
    ms->manager.transition(MICRO, warpy_plane);
    ms->render_microblock();
    ms->render_microblock();
    ms->manager.transition(MICRO, sphere);
    ms->render_microblock();
    ms->render_microblock();

}

void second_half() {
    shared_ptr<GeodesicScene> gs = make_shared<GeodesicScene>();

    stage_macroblock(FileBlock("That device allows you to change the curvature of your room as though you were raising it into the fourth dimension."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Are you saying my room is entering and leaving the fourth dimension?"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Well, maybe. As three dimensional beings, it's easy to imagine a curved two-dimensional space by embedding it into our three-dimensional world."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("This is an extrinsic perspective, since we perceive that curved space as it is embedded in our higher dimension. We call it extrinsic since we are percieving the curvature of the space from outside of that space."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("However, there is an equivalent point of view called the intrinsic perspective, where we can understand the geometry of a curved space from local measurements of angles, distances, and so on, without needing to embed it within a higher dimension at all."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("For example, when crocheting, one can create a flat sheet by making a regular grid of stitches. By altering the amount of stitches in different areas, we can create a curved surface."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("But distances in this surface don't depend on its embedding in three-dimensional space: the way that it curves in space is downstream of the length and angles of the strings holding it together."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("This reflects the intrinsic perspective, where geometry is defined within the fabric of a particular space, independent of its form."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("In differential geometry, this is known as Gauss's Theorema Egregium."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Gauss showed that these two perspectives are mathematically equivalent. The device merely displays the extrinsic embedding to make it easier to visualize the curvature at play."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("At our lab, we still haven't discovered whether or not there is truly a higher-dimensional embedding being manipulated from a physical perspective, since the intrinsic and extrinsic perspectives are predictively equivalent."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Ok... so if I'm understanding you right, the device manipulates the curvature of this space, within which my walls live, but it doesn't actually move the walls themselves. So, why do my walls look like they're moving when I turn it on?"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Remember how the straight line on the map is different from the straight line on the globe? It's the same idea here."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Light itself follows straight lines. So if the geometry of space is warped, the straight line that light follows before reaching your eyes is different, and thus the image of the walls that you see is warped as well."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("How can a straight line not be straight?"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("You're right, it's not really a suitable word. The idea of straightness only really makes sense in a flat geometry, that is, a geometry we call Euclidean. In a non-Euclidean geometry, we call these straight lines geodesics."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("If a boat sails the arctic ocean from Alaska to Norway, to the sailor, the path seems straight."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("But on the globe, you are actually curving downwards since the Earth isn't flat. A geodesic is a curve like this- when you're stuck on a curvy space and walk straight without turning left or right, even though the space underlying you might pull you some way or the other, you are following a geodesic."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("The light inside your room follows a geodesic curve in non-euclidean space."), 1);
    gs->render_microblock();
}

void render_video() {
    first_half();
    return;
    globe();
    second_half();
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
