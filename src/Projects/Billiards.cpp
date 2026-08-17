#include "../Scenes/Math/InnerBilliardsScene.h"
#include "../Scenes/Math/OuterBilliardsScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"

void render_video() {
    CompositeScene cs;

    shared_ptr<InnerBilliardsScene> ibs = make_shared<InnerBilliardsScene>();
    cs.add_scene(ibs, "ibs");

    StateSet simple_table({
        {"v0.x", "-2"}, {"v0.y", "-1"},
        {"v1.x",  "2"}, {"v1.y", "-1"},
        {"v2.x",  "2"}, {"v2.y",  "1"},
        {"v3.x", "-2"}, {"v3.y",  "1"},
    });
    ibs->manager.set(simple_table);
    ibs->manager.set("center_y", "-.5");

    ibs->manager.set({{"ball_start_x", "-0.4"}, {"ball_start_y", "-0.5"}});
    ibs->manager.set("ball_angle", "0.78539");
    ibs->manager.set("ball_distance", "-3");

    stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("Hit a billiards ball,"), SilenceBlock(2)), 3);
    ibs->manager.transition(MACRO, "ball_distance", "11.3137");
    cs.render_microblock();
    ibs->manager.transition(MICRO, "cue_opacity", "0");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("and it reflects off the walls before continuing its path."), 1);
    ibs->manager.set("ball_distance", "-2");
    ibs->manager.set("ball_angle", "0.8");
    ibs->manager.transition(MICRO, "path_length", "55");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(SilenceBlock(2), FileBlock("Its path is pretty interesting.")), 5);
    ibs->manager.transition(MICRO, "cue_opacity", ".4");
    ibs->manager.transition(MICRO, {{"ball_start_x", "-0.65"}, {"ball_start_y", "-0.3"}});
    cs.render_microblock();
    ibs->manager.transition(MICRO, {{"ball_start_x", "-0.8"}, {"ball_start_y", "-0.2"}});
    cs.render_microblock();
    ibs->manager.transition(MICRO, "ball_angle", ".85");
    cs.render_microblock();
    ibs->manager.transition(MICRO, "ball_angle", ".78");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(4.5), 4);
    ibs->manager.transition(MICRO, {{"v0.x", "-2.2"}, {"v0.y", "-.8"}});
    cs.render_microblock();
    ibs->manager.transition(MICRO, {{"ball_start_x", "-0.4"}, {"ball_start_y", "-0.4"}});
    cs.render_microblock();
    ibs->manager.transition(MICRO, {{"v1.x", "1.7"}, {"v1.y", "-1.2"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("This is _inner_ Billiards."), 1);
    ibs->manager.transition(MICRO, {{"v2.x", "2.2"}, {"v2.y", "1.2"}});
    shared_ptr<LatexScene> ls = make_shared<LatexScene>("\\text{Inner Billiards}", vec2(1, .2));
    cs.add_scene_fade_in(MICRO, ls, "ls", vec2(.5, .1));
    cs.render_microblock();

    stage_macroblock(FileBlock("but I want to show you _outer_ billiards."), 2);
    double bsx = 2.3, bsy = -1.6;
    ibs->manager.transition(MACRO, {{"ball_start_x", to_string(bsx)}, {"ball_start_y", to_string(bsy)}});
    ibs->manager.transition(MACRO, simple_table);
    ls->begin_latex_transition(MACRO, "\\text{Outer Billiards}");
    ibs->manager.transition(MICRO, {{"cue_opacity", "0"}, {"path_opacity", "0"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    shared_ptr<OuterBilliardsScene> obs = make_shared<OuterBilliardsScene>();
    obs->manager.set(simple_table);
    obs->manager.set({{"center_y", "-.5"}, {"ball_start_x", to_string(bsx)}, {"ball_start_y", to_string(bsy)}});
    cs.fade_subscene(MICRO, "ls", 0);
    cs.remove_subscene("ibs");
    cs.add_scene(obs, "obs");
    obs->manager.transition(MICRO, {{"zoom", "-.8001"}, {"center_y", "0"}});
    cs.render_microblock();

    shared_ptr<GraphScene> gs = make_shared<GraphScene>();
    gs->manager.set({
        {"d", "15"},
        {"physics_multiplier", "0"},
    });
    cs.add_scene(gs, "gs");

    stage_macroblock(FileBlock("The ball is hit so as to just graze the corner of the table,"), 2);
    cs.render_microblock();
    const OuterBilliards orbit_table({vec2(-2, -1), vec2(2, -1), vec2(2, 1), vec2(-2, 1)}, 0.0f);
        const vec2 p1 = obs->build_orbit_path(orbit_table, (double)1).back();
    {
        const vec2 p0 = obs->build_orbit_path(orbit_table, (double)0).back();
        gs->graph.add_node(0);
        gs->graph.add_node(1);
        gs->graph.move_node(0, vec4(p0.x, p0.y, 0, 0));
        const vec2 midpoint = (p0 + p1) * 0.5;
        gs->graph.move_node(1, vec4(midpoint.x, midpoint.y, 0, 0));
        gs->graph.add_edge(0, 1);
    }
    cs.render_microblock();

    stage_macroblock(FileBlock("going twice the length of its point of contact."), 2);
    cs.render_microblock();
    gs->transition_node_position(MICRO, 1, vec4(p1.x, p1.y, 0, 0));
    cs.render_microblock();
    // TODO maybe add edge lengths here

    stage_macroblock(FileBlock("Hit it again, grazing the next corner of the table."), 1);
    {
        const vec2 p2 = obs->build_orbit_path(orbit_table, (double)2).back();
        gs->graph.add_node(2);
        gs->graph.move_node(2, vec4(p2.x, p2.y, 0, 0));
        gs->graph.add_edge(1, 2);
    }
    cs.render_microblock();

    stage_macroblock(FileBlock("Keep doing this, and we get a very nice pattern."), 6);
    for (int i = 3; i < 8; i++) {
        const vec2 p = obs->build_orbit_path(orbit_table, (double)i).back();
        gs->graph.add_node(i);
        gs->graph.move_node(i, vec4(p.x, p.y, 0, 0));
        if (i > 0) gs->graph.add_edge(i - 1, i);
        cs.render_microblock();
    }
    gs->graph.add_edge(7, 0);
    cs.render_microblock();

    stage_macroblock(FileBlock("This ball, under these rules,"), 1);
    cs.fade_subscene(MICRO, "obs", 0.5);
    cs.render_microblock();

    stage_macroblock(FileBlock("follows a cycle of 8 positions."), 2);
    cs.fade_subscene(MICRO, "obs", 0);
    obs->manager.transition(MICRO, "table_opacity", "0");
    // TODO claude transition_node_position for each node, move them all to octagon
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    // TODO claude transition them all back to where they were
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    cs.fade_subscene(MICRO, "obs", 1);
    obs->manager.set("ball_distance", "0");
    obs->manager.set("path_length", "8");
    cs.fade_subscene(MICRO, "gs", 0);
    obs->manager.transition(MICRO, "table_opacity", "1");
    cs.render_microblock();
    cs.remove_subscene("gs");

    stage_macroblock(FileBlock("Move the starting point, and it makes a cycle of 4 positions."), 1);
    bsx = 2.8, bsy = -.5;
    obs->manager.transition(MACRO, {{"ball_start_x", to_string(bsx)}, {"ball_start_y", to_string(bsy)}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("So, what decides the length of the cycle?"), 1);
    obs->manager.transition(MICRO, {{"ball_start_x", "{t} 2 * sin 6 *"}, {"ball_start_y", "{t} 2 * cos 6 *"}});
    obs->manager.transition(MICRO, "zoom", "-1.5");
    obs->manager.set("path_length", "24");
    cs.render_microblock();

    stage_macroblock(FileBlock("Here's the pattern separating a length 8 cycle from a length 4 cycle?"), 1);
    obs->manager.transition(MACRO, "island_opacity", ".5");
    obs->manager.transition(MACRO, "singularity_depth", "100");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("From these green sections, it takes 4 iterations."), SilenceBlock(1)), 3);
    obs->manager.transition(MICRO, {{"ball_start_x", "{t} 3 * sin .5 * 4 +"}, {"ball_start_y", "{t} 2 * cos .5 *"}});
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("From red, it takes 8."), SilenceBlock(1)), 1);
    obs->manager.transition(MICRO, {{"ball_start_x", "{t} 3 * sin .5 * 4 +"}, {"ball_start_y", "{t} 2 * cos .5 * 2 +"}});
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("From purple, 12."), SilenceBlock(1)), 1);
    obs->manager.transition(MICRO, {{"ball_start_x", "{t} 3 * sin .5 * 4 +"}, {"ball_start_y", "{t} 2 * cos .5 * 4 +"}});
    cs.render_microblock();
    // TODO call out that the cycle hits every period-12 block

    stage_macroblock(SilenceBlock(1), 1);
    obs->manager.transition(MICRO, {/*{"ball_opacity", "0"}, */{"path_opacity", "0"}});
    cs.render_microblock();

    stage_macroblock(FileBlock("The regions all match the shape of the table,"), 2);
    obs->manager.transition(MICRO, {
        {"v0.x", "-1.5"}, {"v0.y", "-2.6"},
        {"v1.x",  "1.5"}, {"v1.y", "-2.6"},
        {"v2.x",  "1.5"}, {"v2.y",  "2.6"},
        {"v3.x", "-1.5"}, {"v3.y",  "2.6"},
    });
    cs.render_microblock();
    obs->manager.transition(MICRO, {
        {"v0.x", "-1"}, {"v0.y", "-1"},
        {"v1.x",  "1"}, {"v1.y", "-1"},
        {"v2.x",  "1"}, {"v2.y",  "1"},
        {"v3.x", "-1"}, {"v3.y",  "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Parallelograms tile the plane too, so they behave similarly."), 2);
    obs->manager.transition(MICRO, { // Parallelogram
        {"v0.x", "-1"}, {"v0.y", "-2"},
        {"v1.x",  "1"}, {"v1.y",  "0"},
        {"v2.x",  "1"}, {"v2.y",  "2"},
        {"v3.x", "-1"}, {"v3.y",  "0"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("Let's try other shapes!"), SilenceBlock(6)), 7);
    cs.render_microblock();
    obs->manager.transition(MICRO, { // Trapezoid
        {"v0.x", "-1"}, {"v0.y", "-2"},
        {"v1.x",  "1"}, {"v1.y", "-1"},
        {"v2.x",  "1"}, {"v2.y",  "1"},
        {"v3.x", "-1"}, {"v3.y",  "2"},
    });
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.transition(MICRO, { // Right triangle
        {"v0.x", "-1"}, {"v0.y", "-2"},
        {"v1.x",  "1"}, {"v1.y", "-0"},
        {"v2.x",  "1"}, {"v2.y",  "0"},
        {"v3.x", "-1"}, {"v3.y",  "2"},
    });
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.set({ // Right triangle fixed
        {"v0.x", "-1"}, {"v0.y", "-2"},
        {"v1.x",  "1"}, {"v1.y", "-0"},
        {"v2.x", "-1"}, {"v2.y",  "2"},
    });
    obs->manager.remove("v3.x");
    obs->manager.remove("v3.y");
    obs->manager.transition(MICRO, { // Equilateral triangle
        {"v0.x", "-.86602540378"}, {"v0.y", "1"},
        {"v1.x", "-.86602540378"}, {"v1.y", "-1"},
        {"v2.x",  "1"}, {"v2.y",  "0"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1), 1);
    obs->manager.set({ // Equilateral triangle
        {"v0.x", "-.86602540378"}, {"v0.y", "1"},
        {"v1.x", "-.86602540378"}, {"v1.y", "-1"},
        {"v2.x",  "1"}, {"v2.y",  "0"},
        {"v3.x",  "1"}, {"v3.y",  "0"},
    });
    obs->manager.transition(MICRO, { // Square again
        {"v0.x", "-1"}, {"v0.y",  "1"},
        {"v1.x", "-1"}, {"v1.y", "-1"},
        {"v2.x",  "1"}, {"v2.y", "-1"},
        {"v3.x",  "1"}, {"v3.y",  "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("What happens when a ball is right on the edge of two regions?"), 16);
    for(int i = 0; i < 10; i++) {
        cs.render_microblock();
    }
    for(int i = 0; i < 3; i++) {
        obs->manager.transition(MICRO, "singularity_opacity", "1");
        cs.render_microblock();
        obs->manager.transition(MICRO, "singularity_opacity", "0");
        cs.render_microblock();
    }

    stage_macroblock(FileBlock("Let's place it here."), 1);
    obs->manager.set("path_length", "0");
    obs->manager.set("ball_distance", "0");
    obs->manager.transition(MICRO, "path_opacity", "1");
    obs->manager.transition(MICRO, {{"ball_start_y", "-2"}, {"ball_start_x", "3"}});
    cs.render_microblock();

    stage_macroblock(FileBlock("After one iteration, it looks very normal..."), 1);
    obs->manager.set("path_length", "<ball_distance>");
    obs->manager.transition(MICRO, "ball_distance", "1");
    cs.render_microblock();

    stage_macroblock(FileBlock("After two, we have this "), 1);
    obs->manager.set("path_length", "<ball_distance>");
    obs->manager.transition(MICRO, "ball_distance", "2");
    cs.render_microblock();

    return;
    stage_macroblock(FileBlock("We know what direction to hit it to just graze the table,"), 1);
    stage_macroblock(FileBlock("and I _guess_ we could just hit it to mirror across the midpoint of that side of the table..."), 1);
    stage_macroblock(FileBlock("But with arbitrarily-shaped tables,"), 1);
    stage_macroblock(FileBlock("that stops making much sense."), 1);
    stage_macroblock(FileBlock("If this is our table,"), 1);
    stage_macroblock(FileBlock("then where's the midpoint?"), 1);
    stage_macroblock(FileBlock("New rule: a ball has to graze the corner of the table."), 1);
    stage_macroblock(FileBlock("None of this edge-parallel funny business."), 1);
    stage_macroblock(FileBlock("So if a ball starts right on this line, then it is no longer in play."), 1);
    stage_macroblock(FileBlock("Same for these other lines, corresponding to the other table edges."), 1);
    stage_macroblock(FileBlock("What if the ball starts on this line?"), 1);
    stage_macroblock(FileBlock("Still no good- we can hit it once, but then it lands on the _other line_..."), 1);
    stage_macroblock(FileBlock("One move delayed, sure, but it will still be forced to graze the wall."), 1);
    stage_macroblock(FileBlock("Let's color the lines to show how many moves a ball can last before crashing into a wall."), 1);
    stage_macroblock(FileBlock("1 move,"), 1);
    stage_macroblock(FileBlock("2,"), 1);
    stage_macroblock(FileBlock("3..."), 1);

    stage_macroblock(FileBlock("Ok. This is very cool. But what about the areas which are not caught in the singularity?"), 1);
    stage_macroblock(FileBlock("Let's take this section as a case study."), 1);
    stage_macroblock(FileBlock("Placing a lot of balls in it,"), 1);
    stage_macroblock(FileBlock("we can see that they all fall right into the next section over."), 1);
    stage_macroblock(FileBlock("This makes some intuitive sense when we look back at our cycle graph:"), 1);
    stage_macroblock(FileBlock("right at the center of each of them, we see a cycle of length 4."), 1);
    stage_macroblock(FileBlock("Let's color these regions."), 1);
    stage_macroblock(FileBlock("Darker means."), 1);
    stage_macroblock(FileBlock(""), 1);
    stage_macroblock(FileBlock(""), 1);
    stage_macroblock(FileBlock("One last treat: what happens if, instead of playing in euclidean space,"), 1);
    stage_macroblock(FileBlock("we play it in hyperbolic geometry?"), 1);
}
