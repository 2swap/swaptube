#include "../Scenes/Math/GeodesicScene.h"
#include "../Scenes/Math/ManifoldScene.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/Mp4Scene.h"
#include "../Core/Smoketest.h"
#include "../Scenes/Media/AlphaFilterScene.h"

void first_half() {
    //set_for_real(false);
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

    CompositeScene cs_killswitch;
    shared_ptr<Mp4Scene> killswitch = make_shared<Mp4Scene>(vector<string>{"greenscreen_killswitch"}, 1);
    shared_ptr<AlphaFilterScene> alpha_killswitch = make_shared<AlphaFilterScene>(killswitch, 0xff00ff00, true);
    cs_killswitch.add_scene(gs, "gs");
    cs_killswitch.add_scene(alpha_killswitch, "alpha");
    gs->manager.transition(MICRO, {
        {"space_w", "0"},
    });
    cs_killswitch.render_microblock();
    cs_killswitch.remove_all_subscenes();

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

    CompositeScene cs_handsup;
    shared_ptr<Mp4Scene> handsup = make_shared<Mp4Scene>(vector<string>{"greenscreen_wheel"}, 1);
    shared_ptr<AlphaFilterScene> alpha_handsup = make_shared<AlphaFilterScene>(handsup, 0xff00ff00, true);
    cs_handsup.add_scene(gs, "gs");
    cs_handsup.add_scene(alpha_handsup, "alpha");
    stage_macroblock(FileBlock("It wasn't me, I swear!"), 1);
    cs_handsup.render_microblock();
    cs_handsup.remove_all_subscenes();

    stage_macroblock(FileBlock("Are you in possession of any contraband items?"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("What, like, drugs?"), 1);
    gs->render_microblock();

    CompositeScene cs;
    cs.add_scene(gs, "gs");
    shared_ptr<ManifoldScene> ms = make_shared<ManifoldScene>(vec2(.5, .5));
    cs.add_scene_fade_in(MICRO, ms, "ms");
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

    stage_macroblock(FileBlock("No! Tesseracts, Möbius strips, or perhaps... a Klein bottle?"), 13);
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
    cs.fade_all_subscenes(MICRO, 1);
    cs.render_microblock();
    cs.remove_all_subscenes();

    CompositeScene cs_pointdoor;
    shared_ptr<Mp4Scene> pointdoor = make_shared<Mp4Scene>(vector<string>{"greenscreen_pointdoor"}, 1);
    shared_ptr<AlphaFilterScene> alpha_pointdoor = make_shared<AlphaFilterScene>(pointdoor, 0xff00ff00, true);
    cs_pointdoor.add_scene(gs, "gs");
    cs_pointdoor.add_scene(alpha_pointdoor, "alpha");
    stage_macroblock(FileBlock("Oh, I think I saw my roommate playing with those before... (points at door)"), 1);
    gs->manager.transition(MICRO, {
        {"pov_q1", "1"},
        {"pov_qj", "-1"},
    });
    cs_pointdoor.render_microblock();
    cs_pointdoor.remove_all_subscenes();

    stage_macroblock(FileBlock("Thanks, I'll take it from here."), 1);
    gs->manager.transition(MICRO, {
        {"sphere_x", "-3"},
        {"sphere_z", "0"},
    });
    gs->render_microblock();
    set_for_real(true);

    stage_macroblock(SilenceBlock(1), 1);
    gs->manager.transition(MICRO, {
        {"pov_q1", "1"},
        {"pov_qj", "0"},
        {"pov_x", "0"},
        {"pov_y", "0"},
        {"pov_z", "-2"},
    });
    gs->manager.transition(MICRO, {
        {"space_x", "(a) 3 - (b) (c) balloon_b (a) 3 - * (a) 3 - +"},
        {"space_y", "(a) 3 - (b) (c) balloon_b (b) * (b) +"},
        {"space_z", "(a) 3 - (b) (c) balloon_b (c) * (c) +"},
        {"space_w", "(a) 3 - (b) (c) balloon_z"},
    });
    gs->render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    gs->render_microblock();

    gs->manager.set({
        {"sphere_radius", "0"},
    });

    stage_macroblock(FileBlock("[Agent begins to yell]"), 1);
    gs->render_microblock();

    gs->manager.transition(MICRO, {
        {"space_x", "(a)"},
        {"space_y", "(b)"},
        {"space_z", "(c)"},
        {"space_w", "0"},
    });

    stage_macroblock(FileBlock("How in the heck did they figure out..."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Hello?"), 1);
    gs->render_microblock();

    // Paraboloid
    gs->manager.transition(MICRO, {
        {"space_x", "(a)"},
        {"space_y", "(b)"},
        {"space_z", "(c)"},
        {"space_w", "(a) ^2 (b) ^2 (c) ^2 + +"},
    });

    stage_macroblock(FileBlock("Dude, you told me this thing was legal!"), 1);
    gs->render_microblock();

    // Hyperbolic paraboloid
    gs->manager.transition(MICRO, {
        {"space_x", "(a)"},
        {"space_y", "(b)"},
        {"space_z", "(c)"},
        {"space_w", "(a) ^2 (b) ^2 (c) ^2 + -"},
    });

    stage_macroblock(FileBlock("In the USA, there's no legislation regarding manipulating the fabric of space."), 1);
    gs->render_microblock();

    // Ring
    gs->manager.set({
        {"ring_intensity", "5"},
    });
    gs->manager.transition(MICRO, {
        {"space_x", "(a)"},
        {"space_y", "(b)"},
        {"space_z", "(c)"},
        {"space_w", "(a) ^2 (b) ^2 + <ring_intensity> - ^2 (c) ^2 +"},
    });
    gs->manager.transition(MICRO, {
        {"ring_intensity", "-5"},
    });

    stage_macroblock(FileBlock("Then why is the Interdimensional FBI after me?"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Oh, yeah, I don't think there's any diplomatic relationship between the US and extradimensional law enforcement. Those guys make up their own rules. It's more like an activist group than anything else, really."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("So, what? What they say doesn't matter?"), 1);
    gs->render_microblock();

    gs->manager.transition(MICRO, {
        {"space_x", "(a)"},
        {"space_y", "(b)"},
        {"space_z", "(c)"},
        {"space_w", "(a) 2 * sin (b) 2 * sin (c) 2 * sin + + (a) ^2 (b) ^2 (c) ^2 + + * 100 /"},
    });

    stage_macroblock(FileBlock("Oh, no, you're probably screwed. Manipulating space is punishable by death in the 4th dimension."), 1);
    gs->render_microblock();

    gs->manager.transition(MICRO, {
        {"space_x", "(a)"},
        {"space_y", "(b)"},
        {"space_z", "(c)"},
        {"space_w", "0"},
    });

    stage_macroblock(FileBlock("WHAT?"), 1);
    gs->render_microblock();

// [Dealer] ...

    stage_macroblock(FileBlock("Why does an interdimensional activist group want me dead for using it!?!"), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("It may look like a party toy, but manipulating the fabric of space can create some real problems..."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("What kind of problems?"), 1);
    gs->render_microblock();

    gs->manager.transition(MICRO, {
        {"space_w", "0"},
    });

    stage_macroblock(FileBlock("See the button with the black circle in the top left?"), 1);
    gs->render_microblock();

    gs->manager.set({
        {"bh", "10"},
    });
    gs->manager.transition(MICRO, {
        {"space_x", "(a)"},
        {"space_y", "(b)"},
        {"space_z", "(c)"},
        {"space_w", "1 (a) ^2 (b) ^2 (c) ^2 <bh> + + + / "},
        {"bh", "-1"},
    });

    stage_macroblock(FileBlock("That's the black hole button. Whatever you do, just don't press that one, and you're probably fine."), 1);
    gs->render_microblock();

    gs->manager.transition(MICRO, {
        {"bh", "10"},
    });
    stage_macroblock(FileBlock("oh no oh no oh no"), 1);
    gs->render_microblock();

    gs->manager.set({
        {"space_w", "0"},
    });

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

    stage_macroblock(FileBlock("That device changes the curvature of your room, almost like we're raising it into a higher dimension."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("So, my room is entering and leaving the fourth dimension?"), 4);
    gs->render_microblock();
    gs->manager.transition(MICRO, "space_w", "(a) ^2 (b) ^2 (c) ^2 + + 5 /");
    gs->render_microblock();
    gs->manager.transition(MICRO, "space_w", "0");
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("Well, maybe. As three dimensional beings, it's easy to imagine a curvy two-dimensional space via its embedding in our three-dimensional world."), 5);
    gs->render_microblock();
    gs->manager.transition(MICRO, "subscreen_size", "1");
    gs->render_microblock();
    gs->manager.transition(MICRO, "space_w", "(a) ^2 (b) ^2 (c) ^2 + - 5 /");
    gs->render_microblock();
    gs->manager.transition(MICRO, "space_w", "0");
    gs->render_microblock();
    gs->render_microblock();

    stage_macroblock(FileBlock("This is an extrinsic perspective, since we perceive that curved space as it is embedded in our higher dimension. We call it extrinsic since we are percieving the curvature of the space from outside of that space."), 1);
    gs->manager.transition(MICRO, "pov_max_dist", "0");
    gs->render_microblock();
    gs->manager.set("a", "0");
    gs->manager.transition(MICRO, "a", "10");
    gs->manager.transition(MICRO, "space_w", "(a) ^2 (b) ^2 + <a> + sin");
    gs->render_microblock();
    gs->render_microblock();
    gs->manager.transition(MICRO, "space_w", "0");
    gs->render_microblock();
    gs->manager.transition(MICRO, "subscreen_size", "0.5");
    gs->render_microblock();

    stage_macroblock(FileBlock("However, there is an equivalent point of view called the intrinsic perspective, where we can understand the geometry of a curved space from local measurements of angles, distances, and so on, without needing to embed it within a higher dimension at all."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("For example, when crocheting, we can create a flat sheet by making a regular grid of stitches. By altering the amount of stitches in different areas, we can create a curved surface."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("But distances in this surface don't depend on its embedding in three-dimensional space: the way that it curves in space is the result of the length of the strings holding it together, not the cause."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("This reflects the intrinsic perspective, where geometry is defined within the fabric of a particular space, independent of the space it's placed in."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("In differential geometry, this is known as Gauss's Theorema Egregium."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("Gauss showed that these two perspectives are mathematically equivalent. The device merely displays the extrinsic embedding to make it easier to visualize the curvature at play."), 5);
    gs->render_microblock();
    gs->render_microblock();
    gs->render_microblock();
    gs->manager.transition(MICRO, "subscreen_size", "1");
    gs->render_microblock();
    gs->manager.transition(MICRO, "subscreen_size", "0.5");
    gs->render_microblock();

    stage_macroblock(FileBlock("Ok... so if I'm understanding you right, the device manipulates the curvature of this space, within which my walls live, but it doesn't actually move the walls themselves. So, why do my walls look like they're moving when I turn it on?"), 3);
    gs->render_microblock();
    gs->manager.transition(MICRO, "space_w", "(a) sin (b) sin +");
    gs->render_microblock();
    gs->manager.transition(MICRO, "space_w", "0");
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

    stage_macroblock(FileBlock("But on the globe, you are actually curving downwards since the Earth isn't flat. A geodesic is a curve like this- when you're stuck on a curvy surface and walk straight without turning left or right, even though the space underlying you might pull you some way or the other, you're following a geodesic."), 1);
    gs->render_microblock();

    stage_macroblock(FileBlock("The light inside your room follows a geodesic curve in non-euclidean space."), 1);
    gs->render_microblock();
}

void render_video() {
    //first_half();
    globe();
    return;
    second_half();
}
