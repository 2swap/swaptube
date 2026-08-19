#include "../Scenes/Math/InnerBilliardsScene.h"
#include "../Scenes/Math/OuterBilliardsScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"

void render_video() {
    CompositeScene cs;

    shared_ptr<InnerBilliardsScene> ibs = make_shared<InnerBilliardsScene>();
    cs.add_scene(ibs, "ibs");

    StateSet simple_table_hex({
        {"v0.x", "-2"}, {"v0.y", "-1"},
        {"v1.x",  "0"}, {"v1.y", "-1"},
        {"v2.x",  "2"}, {"v2.y", "-1"},
        {"v3.x",  "2"}, {"v3.y",  "1"},
        {"v4.x",  "0"}, {"v4.y",  "1"},
        {"v5.x", "-2"}, {"v5.y",  "1"},
    });
    StateSet simple_table({
        {"v0.x", "-2"}, {"v0.y", "-1"},
        {"v1.x",  "2"}, {"v1.y", "-1"},
        {"v2.x",  "2"}, {"v2.y",  "1"},
        {"v3.x", "-2"}, {"v3.y",  "1"},
    });
    ibs->manager.set(simple_table_hex);
    ibs->manager.set("center_y", "-.5");

    ibs->manager.set({{"ball_start_x", "-0.4"}, {"ball_start_y", "-0.5"}});
    ibs->manager.set("ball_angle", "0.78539");
    ibs->manager.set("ball_distance", "-3");

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("Hit a billiards ball,"), SilenceBlock(2)));
    ibs->manager.transition(MACRO, "ball_distance", "11.3137");
    cs.render_microblock();
    ibs->manager.transition(MICRO, "cue_opacity", "0");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("and it reflects off the walls before continuing its path."));
    ibs->manager.set("ball_distance", "-2");
    ibs->manager.set("ball_angle", "0.8");
    ibs->manager.transition(MICRO, "path_length", "405");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(SilenceBlock(2), FileBlock("Its path is pretty interesting.")));
    ibs->manager.transition(MICRO, "cue_opacity", ".4");
    ibs->manager.transition(MICRO, {{"ball_start_x", "-0.8"}, {"ball_start_y", "-0.2"}});
    cs.render_microblock();
    ibs->manager.transition(MICRO, "ball_angle", ".85");
    cs.render_microblock();
    ibs->manager.transition(MICRO, "ball_angle", ".78");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(4.5));
    ibs->manager.transition(MICRO, {{"v0.x", "-2.2"}, {"v0.y", "-.8"}, {"v1.x", ".67"}, {"v1.y", "-1.2"}, {"v5.x", "-2.2"}, {"v5.y", ".9"}});
    cs.render_microblock();
    ibs->manager.transition(MICRO, {{"ball_start_x", "-0.4"}, {"ball_start_y", "-0.4"}});
    cs.render_microblock();
    // Regular hexagon
    ibs->manager.transition(MICRO, "center_y", "0");
    StateSet regular_hexagon;
    for (int i = 0; i < 6; i++) {
        double theta = 3.5 + (double)i * (2.0 * M_PI / 6.0);
        regular_hexagon["v" + to_string(i) + ".x"] = to_string(1.7 * cos(theta));
        regular_hexagon["v" + to_string(i) + ".y"] = to_string(1.7 * sin(theta));
    }
    ibs->manager.transition(MICRO, regular_hexagon);
    cs.render_microblock();

    stage_macroblock(FileBlock("When you remove the pockets,"));
    ibs->manager.transition(MICRO, "pocket_size", "0");
    cs.render_microblock();

    stage_macroblock(FileBlock("this system is known to mathematicians as _inner_ Billiards."));
    ibs->manager.transition(MICRO, "ball_angle", "0.6");
    //shared_ptr<LatexScene> ls = make_shared<LatexScene>("\\text{Inner Billiards}", vec2(1, .2));
    //cs.add_scene_fade_in(MICRO, ls, "ls", vec2(.5, .1));
    cs.render_microblock();

    stage_macroblock(FileBlock("but I want to show you _outer_ billiards."));
    double bsx = 2.3, bsy = -1.6;
    ibs->manager.transition(MACRO, {{"ball_start_x", to_string(bsx)}, {"ball_start_y", to_string(bsy)}});
    ibs->manager.transition(MACRO, simple_table_hex);
    //ls->begin_latex_transition(MACRO, "\\text{Outer Billiards}");
    ibs->manager.transition(MICRO, {{"cue_opacity", "0"}, {"path_opacity", "0"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    shared_ptr<OuterBilliardsScene> obs = make_shared<OuterBilliardsScene>();
    obs->manager.set(simple_table);
    obs->manager.set({{"ball_start_x", to_string(bsx)}, {"ball_start_y", to_string(bsy)}});
    //cs.fade_subscene(MICRO, "ls", 0);
    cs.fade_subscene(MICRO, "ibs", 0);
    cs.add_scene(obs, "obs", vec2(.5, .5), true);
    ibs->manager.transition(MICRO, {{"zoom", "-.8001"}, {"center_y", "0"}});
    obs->manager.transition(MICRO, {{"zoom", "-.8001"}, {"center_y", "0"}});
    cs.render_microblock();
    cs.remove_subscene("ibs");

    shared_ptr<GraphScene> gs = make_shared<GraphScene>();
    gs->manager.set({
        {"d", "14.9"},
        {"physics_multiplier", "0"},
        {"points_radius_multiplier", "1.2"},
    });
    cs.add_scene(gs, "gs");

    stage_macroblock(FileBlock("The ball is hit so as to just graze the corner of the table,"));
    cs.render_microblock();
    const vec2 p1 = obs->build_orbit_path((double)1).back();
    {
        const vec2 p0 = obs->build_orbit_path((double)0).back();
        gs->graph.add_node(0);
        gs->graph.add_node(1);
        gs->graph.move_node(0, vec4(p0.x, p0.y, 0, 0));
        gs->graph.move_node(1, vec4(p0.x, p0.y, 0, 0));
        const vec2 midpoint = (p0 + p1) * 0.5;
        gs->transition_node_position(MICRO, 1, vec4(midpoint.x, midpoint.y, 0, 0));
        gs->graph.add_edge(0, 1);
        gs->config.set_edge_label(0, 1, "");
    }
    cs.render_microblock();

    stage_macroblock(FileBlock("and go twice the length of its point of contact."));
    gs->config.transition_edge_label(MICRO, 0, 1, "l");
    cs.render_microblock();
    cs.render_microblock();
    gs->transition_node_position(MICRO, 1, vec4(p1.x, p1.y, 0, 0));
    gs->config.transition_edge_label(MICRO, 0, 1, "2l");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();
    gs->config.transition_edge_label(MICRO, 0, 1, "");
    cs.render_microblock();

    stage_macroblock(FileBlock("Hit it again, grazing the next corner of the table."));
    {
        const vec2 p2 = obs->build_orbit_path((double)2).back();
        gs->graph.add_node(2);
        gs->graph.move_node(2, vec4(p1.x, p1.y, 0, 0));
        gs->graph.add_edge(1, 2);
        gs->config.set_edge_label(1, 2, "");
        gs->transition_node_position(MICRO, 2, vec4(p2.x, p2.y, 0, 0));
    }
    cs.render_microblock();

    stage_macroblock(FileBlock("Keep doing this, and we get a very nice pattern."));
    obs->manager.transition(MACRO, "ball_opacity", "0");
    for (int i = 3; i < 9; i++) {
        const vec2 p_pre = obs->build_orbit_path((double)i-1).back();
        const vec2 p_post= obs->build_orbit_path((double)i).back();
        gs->graph.add_node(i);
        gs->graph.move_node(i, vec4(p_pre.x, p_pre.y, 0, 0));
        if (i > 0){
            gs->graph.add_edge(i - 1, i);
            gs->config.set_edge_label(i - 1, i, "");
        }
        gs->transition_node_position(MICRO, i, vec4(p_post.x, p_post.y, 0, 0));
        cs.render_microblock();
    }

    gs->graph.remove_node(8);
    gs->graph.add_edge(7, 0);
    gs->config.set_edge_label(7, 0, "");

    stage_macroblock(FileBlock("This ball, under these rules,"));
    cs.render_microblock();

    stage_macroblock(FileBlock("follows a cycle of 8 positions."));
    obs->manager.transition(MICRO, "table_opacity", "0");
    for (int i = 0; i < 8; i++) {
        const double theta = i * (2.0 * M_PI / 8.0);
        gs->transition_node_position(MICRO, i, 4.0 * vec4(cos(theta), sin(theta), 0, 0));
    }
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    for (int i = 0; i < 8; i++) {
        const vec2 p = obs->build_orbit_path((double)i).back();
        gs->transition_node_position(MICRO, i, vec4(p.x, p.y, 0, 0));
    }
    obs->manager.transition(MICRO, "table_opacity", "1");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    obs->manager.set("ball_distance", "0");
    obs->manager.set("path_length", "8");
    obs->manager.transition(MICRO, "ball_opacity", "1");
    cs.render_microblock();
    cs.fade_subscene(MICRO, "gs", 0);
    cs.render_microblock();
    cs.remove_subscene("gs");

    stage_macroblock(FileBlock("Move the starting point, and it makes a cycle of length 4."));
    bsx = 2.8, bsy = -.5;
    obs->manager.transition(MACRO, {{"ball_start_x", to_string(bsx)}, {"ball_start_y", to_string(bsy)}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("So, what decides the length of the cycle?"));
    obs->manager.transition(MICRO, {{"ball_start_x", "{t} 2 * sin 4 *"}, {"ball_start_y", "{t} 2 * cos 2 *"}});
    obs->manager.transition(MICRO, "zoom", "-1.5");
    obs->manager.set("path_length", "24");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(FileBlock("Here's the pattern."));
    obs->manager.transition(MACRO, "island_opacity", "1");
    obs->manager.transition(MACRO, "singularity_depth", "100");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("From these green sections, it takes 4 iterations."), SilenceBlock(.3)), 3);
    obs->manager.transition(MICRO, {{"ball_start_x", "{t} 3 * sin .5 * 4 +"}, {"ball_start_y", "{t} 2 * cos .5 *"}});
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("From red, it takes 8."), SilenceBlock(.3)));
    obs->manager.transition(MICRO, {{"ball_start_x", "{t} 3 * sin .5 * 4 +"}, {"ball_start_y", "{t} 2 * cos .5 * 2 +"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("From purple, 12."), SilenceBlock(.3)));
    obs->manager.transition(MICRO, {{"ball_start_x", "{t} 3 * sin .5 * 4 +"}, {"ball_start_y", "{t} 2 * cos .5 * 4 +"}});
    cs.render_microblock();
    cs.render_microblock();
    // TODO call out that the cycle hits every period-12 block

    stage_macroblock(SilenceBlock(1));
    obs->manager.set("ball_opacity", "<path_opacity>");
    obs->manager.transition(MICRO, "path_opacity", "0");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("The regions all match the shape of the table,"), SilenceBlock(1)));
    obs->manager.transition(MICRO, {
        {"v0.x", "-1.5"}, {"v0.y", "-2.6"},
        {"v1.x",  "1.5"}, {"v1.y", "-2.6"},
        {"v2.x",  "1.5"}, {"v2.y",  "2.6"},
        {"v3.x", "-1.5"}, {"v3.y",  "2.6"},
    });
    cs.render_microblock();
    obs->manager.transition(MICRO, {
        {"v0.x", "-1.8"}, {"v0.y", "-.3"},
        {"v1.x",  "1.8"}, {"v1.y", "-.3"},
        {"v2.x",  "1.8"}, {"v2.y",  ".3"},
        {"v3.x", "-1.8"}, {"v3.y",  ".3"},
    });
    cs.render_microblock();
    obs->manager.transition(MICRO, {
        {"v0.x", "-1"}, {"v0.y", "-1"},
        {"v1.x",  "1"}, {"v1.y", "-1"},
        {"v2.x",  "1"}, {"v2.y",  "1"},
        {"v3.x", "-1"}, {"v3.y",  "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("Parallelograms tile the plane too, so they behave similarly."));
    obs->manager.transition(MICRO, { // Parallelogram
        {"v0.x","-.3"}, {"v0.y", "-4"},
        {"v1.x", ".8"}, {"v1.y", "-1"},
        {"v2.x", ".3"}, {"v2.y",  "4"},
        {"v3.x","-.8"}, {"v3.y",  "1"},
    });
    cs.render_microblock();
    obs->manager.transition(MICRO, { // Parallelogram
        {"v0.x", "-1"}, {"v0.y","-.9"},
        {"v1.x",  "2"}, {"v1.y", "-1"},
        {"v2.x",  "1"}, {"v2.y", ".9"},
        {"v3.x", "-2"}, {"v3.y",  "1"},
    });
    cs.render_microblock();
    obs->manager.transition(MICRO, { // Parallelogram
        {"v0.x", "-1"}, {"v0.y", "-2"},
        {"v1.x",  "1"}, {"v1.y",  "0"},
        {"v2.x",  "1"}, {"v2.y",  "2"},
        {"v3.x", "-1"}, {"v3.y",  "0"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("Let's try other shapes!"), SilenceBlock(6)));
    obs->manager.transition(MICRO, { // Trapezoid
        {"v0.x", "-1"}, {"v0.y", "-2"},
        {"v1.x",  "1"}, {"v1.y", "-1"},
        {"v2.x",  "1"}, {"v2.y",  "1"},
        {"v3.x", "-1"}, {"v3.y",  "2"},
    });
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.transition(MICRO, { // triangle
        {"v0.x", "-.86602540378"}, {"v0.y", "-1"},
        {"v1.x",  "1"}, {"v1.y", "-0"},
        {"v2.x",  "1"}, {"v2.y",  "0"},
        {"v3.x", "-.86602540378"}, {"v3.y",  "1"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    obs->manager.transition(MICRO, { // Square again
        {"v0.x", "-1"}, {"v0.y",  "1"},
        {"v1.x", "-1"}, {"v1.y", "-1"},
        {"v2.x",  "1"}, {"v2.y", "-1"},
        {"v3.x",  "1"}, {"v3.y",  "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("What happens when a ball is right on the edge of two regions?"));
    for(int i = 0; i < 10; i++) {
        cs.render_microblock();
    }
    for(int i = 0; i < 3; i++) {
        obs->manager.transition(MICRO, "singularity_opacity", "1");
        cs.render_microblock();
        obs->manager.transition(MICRO, "singularity_opacity", "0");
        cs.render_microblock();
    }

    stage_macroblock(FileBlock("Let's place it here."));
    obs->manager.set({{"path_length", "0"}, {"ball_distance", "0"}});
    obs->manager.transition(MICRO, "path_opacity", "1");
    obs->manager.transition(MICRO, {{"ball_start_y", "-2"}, {"ball_start_x", "3"}});
    cs.render_microblock();

    stage_macroblock(FileBlock("After one iteration, it looks very normal..."));
    obs->manager.set("path_length", "<ball_distance>");
    obs->manager.transition(MICRO, "ball_distance", "1");
    cs.render_microblock();

    stage_macroblock(FileBlock("But after two, we have this..."));
    obs->manager.transition(MICRO, "ball_distance", "2");
    cs.render_microblock();

    stage_macroblock(FileBlock("The ball traces along the edge of the table."));
    StateSet undo = obs->manager.transition(MICRO, {{"zoom", "0"}, {"center_x", "-.5"}});
    obs->manager.transition(MICRO, "ball_distance", "1.4");
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.transition(MICRO, undo);
    cs.render_microblock();

    stage_macroblock(FileBlock("Which corner defines the length we should jump by?"));
        cs.render_microblock();
        cs.render_microblock();
        cs.render_microblock();
        cs.render_microblock();
    for(int i = 0; i < 2; i++) {
        obs->manager.transition(MICRO, "ball_distance", "1.5");
        cs.render_microblock();
        cs.render_microblock();
        obs->manager.transition(MICRO, "ball_distance", "1.3");
        cs.render_microblock();
        cs.render_microblock();
    }

    stage_macroblock(FileBlock("It's undefined- we call it a singularity."));
    cs.render_microblock();

    stage_macroblock(FileBlock("Notice how this starting point is an immediate singularity,"));
    obs->manager.transition(MICRO, {{"ball_start_x", "4"}, {"ball_start_y", ".999"}, {"ball_distance", "1"}});
    cs.render_microblock();

    stage_macroblock(FileBlock("as is any point on one of these 4 lines."));
    obs->manager.transition(MICRO, {{"singularity_opacity", "1"}, {"island_opacity", "0.2"}});
    obs->manager.set("singularity_depth", "1");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("Our original starting point became indeterminate on the second iteration,"));
    obs->manager.transition(MICRO, {{"ball_start_y", "-2"}, {"ball_start_x", "3"}});
    cs.render_microblock();
    obs->manager.transition(MICRO, "ball_distance", "2");
    cs.render_microblock();

    stage_macroblock(FileBlock("like these points here."));
    obs->manager.transition(MICRO, "singularity_depth", "2");
    cs.render_microblock();
    obs->manager.set("ball_distance", "<singularity_depth>");

    stage_macroblock(FileBlock("Here are the depth 3 singularities,"));
    obs->manager.transition(MICRO, {{"ball_start_y", "-2"}, {"ball_start_x", "5"}, {"singularity_depth", "3"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("depth 4 singularities,"));
    obs->manager.transition(MICRO, {{"ball_start_y", "-4"}, {"ball_start_x", "7"}, {"singularity_depth", "4"}});
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.set("ball_distance", "4");

    // TODO rainbow color the lines by depth
    stage_macroblock(FileBlock("and so on."));
    obs->manager.transition(MICRO, "path_opacity", "0");
    obs->manager.transition(MICRO, "singularity_depth", "15");
    cs.render_microblock();
    obs->manager.set("ball_distance", "0");
    obs->manager.transition(MICRO, "singularity_depth", "100");
    cs.render_microblock();

    // Start to morph the table a bit
    stage_macroblock(SilenceBlock(1));
    obs->manager.transition(MICRO, {{"v0.x", "-1.5"}, {"v0.y", "-2.6"}});
    cs.render_microblock();

    // Transition to a regular pentagon
    stage_macroblock(SilenceBlock(1));
    // Dummy point to match v0
    obs->manager.set({{"v4.x", "-1.5"}, {"v4.y", "-2.6"}});
    StateSet pentagon;
    for (int i = 0; i < 5; i++) {
        double theta = (double)i * (2.0 * M_PI / 5.0);
        pentagon["v" + to_string(i) + ".x"] = to_string(2.0 * cos(theta));
        pentagon["v" + to_string(i) + ".y"] = to_string(2.0 * sin(theta));
    }
    obs->manager.transition(MACRO, pentagon);
    cs.render_microblock();

    stage_macroblock(SilenceBlock(10));
    obs->manager.transition(MICRO, "curvature", "-1");
    cs.render_microblock();

    return;
    stage_macroblock(FileBlock("Ok. This is very cool. But what about the areas which are not caught in the singularity?"));
    stage_macroblock(FileBlock("Let's take this section as a case study."));
    stage_macroblock(FileBlock("Placing a lot of balls in it,"));
    stage_macroblock(FileBlock("we can see that they all fall right into the next section over."));
    stage_macroblock(FileBlock("This makes some intuitive sense when we look back at our cycle graph:"));
    stage_macroblock(FileBlock("right at the center of each of them, we see a cycle of length 4."));
    stage_macroblock(FileBlock("Let's color these regions."));
    stage_macroblock(FileBlock("Darker means."));
    stage_macroblock(FileBlock(""));
    stage_macroblock(FileBlock(""));
    stage_macroblock(FileBlock("One last treat: what happens if, instead of playing in euclidean space,"));
    stage_macroblock(FileBlock("we play it in hyperbolic geometry?"));
}
