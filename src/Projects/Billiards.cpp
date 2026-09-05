#include "../Scenes/Math/InnerBilliardsScene.h"
#include "../Scenes/Math/OuterBilliardsScene.h"
#include "../Scenes/Math/OuterBilliardsVertexFlowScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Media/WhitePaperScene.h"
#include "../Core/State/BezierStateCurve.h"
#include "../Core/Smoketest.h"

StateSet regular_ngon(int n, double radius, double phase) {
    StateSet result;
    for (int i = 0; i < n; i++) {
        double theta = phase + (double)i * (2.0 * M_PI / (double)n);
        result["v" + to_string(i) + ".x"] = to_string(radius * cos(theta));
        result["v" + to_string(i) + ".y"] = to_string(radius * sin(theta));
    }
    return result;
}

void render_video() {
    set_for_real(false);
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

    ibs->manager.set({{"center_x", "<ball_start_x> .6 *"}, {"center_y", "<ball_start_y> .6 *"}});
    ibs->manager.set({{"ball_start_x", "-0.4"}, {"ball_start_y", "-0.5"}});
    set_global_state("billiards_ball_x", -0.4);
    set_global_state("billiards_ball_y", -0.5);
    ibs->manager.set("ball_angle", "0.78539");
    ibs->manager.set("ball_distance", "-3");

    stage_macroblock(CompositeBlock(FileBlock("Hit a billiards ball,"), SilenceBlock(2)));
    ibs->manager.transition(MICRO, {{"center_x", "{billiards_ball_x} .6 *"}, {"center_y", "{billiards_ball_y} .6 *"}});
    ibs->manager.transition(MACRO, "ball_distance", "11.3137");
    cs.render_microblock();
    cs.render_microblock();
    ibs->manager.transition(MICRO, "zoom", ".5");
    cs.render_microblock();
    ibs->manager.transition(MICRO, "cue_opacity", "0");
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("and it reflects off the walls of the table."));
    ibs->manager.transition(MICRO, {{"center_x", "0"}, {"center_y", "0"}});
    ibs->manager.set("ball_distance", "-2");
    ibs->manager.set("ball_angle", "0.8");
    ibs->manager.transition(MACRO, "path_length", "55");
    ibs->manager.transition(MICRO, "zoom", "0");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(4));
    ibs->manager.transition(MICRO, "path_length", "60");
    ibs->manager.transition(MICRO, "cue_opacity", ".4");
    ibs->manager.transition(MICRO, {{"ball_start_x", "-0.8"}, {"ball_start_y", "-0.2"}});
    cs.render_microblock();
    ibs->manager.transition(MICRO, "ball_angle", ".9");
    cs.render_microblock();
    ibs->manager.transition(MICRO, "ball_angle", ".67");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("With a non-rectangular table, its path is hard to predict."), SilenceBlock(2.5)));
    ibs->manager.transition(MICRO, {{"v0.x", "-2.5"}, {"v0.y", "-.8"}, {"v1.x", ".67"}, {"v1.y", "-1.2"}, {"v3.x", "1.7"}, {"v3.y", ".9"}, {"v5.x", "-1.9"}, {"v5.y", ".6"}});
    ibs->manager.transition(MICRO, "pocket_size", "0");
    cs.render_microblock();
    ibs->manager.transition(MICRO, {{"ball_start_x", "0.15"}, {"ball_start_y", "-0.2"}});
    cs.render_microblock();
    ibs->manager.transition(MICRO, "center_y", "0");
    StateSet regular_hexagon = regular_ngon(6, 1.7, 3.5);
    ibs->manager.transition(MICRO, regular_hexagon);
    cs.render_microblock();


    stage_macroblock(CompositeBlock(FileBlock("mathematicians call this _inner_ Billiards."), SilenceBlock(2.5)));
    ibs->manager.transition(MACRO, "ball_angle", "0.35");
    cs.render_microblock();
    cs.render_microblock();
    ibs->manager.transition(MICRO, "cue_opacity", "0");
    cs.render_microblock();

    stage_macroblock(FileBlock("but I want to show you _outer_ billiards."));
    double bsx = 2.3, bsy = -1.6;
    ibs->manager.transition(MACRO, {{"ball_start_x", to_string(bsx)}, {"ball_start_y", to_string(bsy)}});
    ibs->manager.transition(MACRO, simple_table_hex);
    //ls->begin_latex_transition(MACRO, "\\text{Outer Billiards}");
    ibs->manager.transition(MICRO, "path_opacity", "0");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    shared_ptr<OuterBilliardsScene> obs = make_shared<OuterBilliardsScene>();
    obs->manager.set(simple_table);
    obs->manager.set({{"ball0_start_x", to_string(bsx)}, {"ball0_start_y", to_string(bsy)}});
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

    stage_macroblock(FileBlock("Hit the ball, just grazing the corner of the table,"));
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

    stage_macroblock(FileBlock("That's the fundamental operation: hit the ball twice the way past the next available corner."));
    cs.render_microblock();
    {
        const vec2 p2 = obs->build_orbit_path((double)2).back();
        gs->graph.add_node(2);
        gs->graph.move_node(2, vec4(p1.x, p1.y, 0, 0));
        gs->graph.add_edge(1, 2);
        gs->config.set_edge_label(1, 2, "");
        gs->transition_node_position(MICRO, 2, vec4(p2.x, p2.y, 0, 0));
    }
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("Keep doing this, and we get a nice pattern."), SilenceBlock(1)));
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
    cs.render_microblock();
    cs.render_microblock();

    gs->graph.remove_node(8);
    gs->graph.add_edge(7, 0);
    gs->config.set_edge_label(7, 0, "");

    stage_macroblock(FileBlock("It's a cycle of 8 positions."));
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

    stage_macroblock(FileBlock("For this starting point, we get a cycle of length 4."));
    obs->manager.set("angley", ".8");
    obs->manager.transition(MICRO, {{"ball0_start_x", "<angley> 2 * sin 4 *"}, {"ball0_start_y", "<angley> 2 * cos 2 *"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.5));
    cs.render_microblock();

    stage_macroblock(FileBlock("What decides the length of the cycle?"));
    obs->manager.transition(MICRO, "angley", "5");
    obs->manager.transition(MICRO, "zoom", "-1.5");
    obs->manager.set("path_length", "24");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    obs->manager.transition(MICRO, "angley", "4");
    cs.render_microblock();

    stage_macroblock(FileBlock("Here's the pattern."));
    obs->manager.transition(MACRO, "island_opacity", ".5");
    obs->manager.transition(MACRO, "singularity_depth", "200");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("These 4 regions are stuck in a 4-iteration cycle."), SilenceBlock(.3)));
    obs->manager.set("cycle_highlight", "4");
    obs->manager.transition(MICRO, {{"ball0_start_x", "{t} 3 * sin .5 * 4 +"}, {"ball0_start_y", "{t} 2 * cos .5 *"}});
    obs->manager.transition(MICRO, "cycle_highlight_enable", "1");
    obs->manager.transition(MACRO, "island_opacity", "1");
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("These 8 are a little more complicated."), SilenceBlock(.3)));
    obs->manager.transition(MICRO, "cycle_highlight", "8");
    obs->manager.transition(MICRO, "cycle_highlight_enable", "1");
    obs->manager.transition(MICRO, {{"ball0_start_x", "{t} 3 * sin .5 * 4 +"}, {"ball0_start_y", "{t} 2 * cos .5 * 2 +"}});
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("Starting here, we have 12 blocks of period 12."), SilenceBlock(.3)));
    obs->manager.transition(MICRO, "cycle_highlight", "12");
    obs->manager.transition(MICRO, {{"ball0_start_x", "{t} 3 * sin .5 * 4 +"}, {"ball0_start_y", "{t} 2 * cos .5 * 4 +"}});
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.transition(MICRO, "cycle_highlight_enable", "0");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    obs->manager.set("ball_opacity", "<path_opacity>");
    obs->manager.transition(MICRO, "path_opacity", "0");
    cs.render_microblock();

    stage_macroblock(FileBlock("Instead of asking how many iterations it takes to return,"));
    obs->manager.set({{"ball_distance", "<path_length>"}, {"path_length", "0"}});
    obs->manager.transition(MICRO, "path_opacity", "1");
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_length", "12");
    cs.render_microblock();

    int num_iterations = 27*4;
    stage_macroblock(FileBlock("We can ask where it ends up after, say, " + to_string(num_iterations) + " iterations."));
    obs->manager.transition(MICRO, "path_length", to_string(num_iterations));
    cs.render_microblock();

    stage_macroblock(FileBlock("Here, I gave every point a unique color."));
    obs->manager.transition(MICRO, "path_opacity", "0");
    obs->manager.transition(MACRO, "periodicity_or_flow", "1");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("This is where every point lands after one iteration,"));
    obs->manager.transition(MICRO, "flow_depth", "1");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("two iterations,"));
    obs->manager.transition(MICRO, "flow_depth", "2");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("four iterations,"));
    obs->manager.transition(MICRO, "flow_depth", "4");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("and so on, up to " + to_string(num_iterations)), SilenceBlock(3)));
    obs->manager.transition(MICRO, "flow_depth", to_string(num_iterations));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
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

    stage_macroblock(CompositeBlock(FileBlock("Let's try other shapes!"), SilenceBlock(3)));
    // TODO this needs to be high flow depth (100)
    obs->manager.transition(MICRO, { // Trapezoid
        {"v0.x", "-1"}, {"v0.y", "-2"},
        {"v1.x",  "1"}, {"v1.y", "-1"},
        {"v2.x",  "1"}, {"v2.y",  "1"},
        {"v3.x", "-1"}, {"v3.y",  "2"},
    });
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.8));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(4));
    obs->manager.transition(MICRO, { // triangle
        {"v0.x", "-.86602540378"}, {"v0.y", "-1"},
        {"v1.x",  "1"}, {"v1.y", "-0"},
        {"v2.x",  "1"}, {"v2.y",  "0"},
        {"v3.x", "-.86602540378"}, {"v3.y",  "1"},
    });
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.8));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3));
    obs->manager.transition(MICRO, { // Square again
        {"v0.x", "-1"}, {"v0.y", "-1"},
        {"v1.x",  "1"}, {"v1.y", "-1"},
        {"v2.x",  "1"}, {"v2.y",  "1"},
        {"v3.x", "-1"}, {"v3.y",  "1"},
    });
    obs->manager.transition(MICRO, "island_opacity", ".4");
    cs.render_microblock();

    stage_macroblock(FileBlock("What happens when a ball is right on the edge of two regions?"));
    for(int i = 0; i < 7; i++) {
        cs.render_microblock();
    }
    for(int i = 0; i < 2; i++) {
        obs->manager.transition(MICRO, "singularity_opacity", ".2");
        cs.render_microblock();
        obs->manager.transition(MICRO, "singularity_opacity", "0");
        cs.render_microblock();
    }

    stage_macroblock(FileBlock("Let's place it here."));
    obs->manager.set({{"path_length", "0"}, {"ball_distance", "0"}});
    obs->manager.transition(MICRO, "path_opacity", "1");
    obs->manager.transition(MICRO, {{"ball0_start_y", "-2"}, {"ball0_start_x", "3"}});
    cs.render_microblock();

    stage_macroblock(FileBlock("After one iteration, it looks very normal..."));
    obs->manager.set("ball_distance", "<path_length>");
    obs->manager.transition(MICRO, "path_length", "1");
    cs.render_microblock();

    stage_macroblock(FileBlock("But after two, we have this..."));
    obs->manager.transition(MICRO, "path_length", "1.4");
    cs.render_microblock();

    stage_macroblock(FileBlock("The ball traces along the edge of the table."));
    StateSet undo = obs->manager.transition(MICRO, {{"zoom", "0"}, {"center_x", "-.5"}});
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
        obs->manager.transition(MICRO, "path_length", "1.5");
        cs.render_microblock();
        cs.render_microblock();
        obs->manager.transition(MICRO, "path_length", "1.3");
        cs.render_microblock();
        cs.render_microblock();
    }

    stage_macroblock(FileBlock("It's undefined- we call it a singularity."));
    obs->manager.transition(MICRO, "path_length", "1.4");
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_opacity", "0");
    cs.render_microblock();
    obs->manager.set("path_length", "0");

    stage_macroblock(SilenceBlock(1));
    obs->manager.transition(MICRO, {{"ball0_start_x", "4"}, {"ball0_start_y", "1"}, {"ball_opacity", "1"}});
    cs.render_microblock();

    stage_macroblock(FileBlock("This starting point is an immediate singularity,"));
    obs->manager.transition(MICRO, "path_opacity", "1");
    cs.render_microblock();
    obs->manager.set("ball_opacity", "<path_opacity>");
    obs->manager.transition(MICRO, "path_length", ".4");
    obs->manager.set("singularity_rainbow", "1");
    cs.render_microblock();

    stage_macroblock(FileBlock("just like any point on one of these lines."));
    obs->manager.transition(MICRO, {{"singularity_opacity", "1"}, {"island_opacity", "0.2"}, {"path_opacity", "0"}});
    obs->manager.set("singularity_depth", "1");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(FileBlock("This is a depth 2 singularity,"));
    obs->manager.set({{"ball0_start_y", "-6"}, {"ball0_start_x", "3"}});
    obs->manager.set("path_length", "0");
    obs->manager.transition(MICRO, "path_opacity", "1");
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_length", "1.45");
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_opacity", "0");
    obs->manager.transition(MICRO, "singularity_depth", "2");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.5));
    obs->manager.transition(MICRO, "path_opacity", "0");
    cs.render_microblock();

    stage_macroblock(FileBlock("here are depth 3 singularities,"));
    obs->manager.set({{"ball0_start_y", "-2"}, {"ball0_start_x", "5"}});
    obs->manager.transition(MICRO, "singularity_depth", "3");
    obs->manager.set("path_length", "0");
    obs->manager.transition(MICRO, "path_opacity", "1");
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_length", "2.4");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.5));
    obs->manager.transition(MICRO, "path_opacity", "0");
    cs.render_microblock();

    stage_macroblock(FileBlock("and depth 4."));
    obs->manager.set({{"ball0_start_y", "-4"}, {"ball0_start_x", "7"}});
    obs->manager.transition(MICRO, "singularity_depth", "4");
    obs->manager.set("path_length", "0");
    obs->manager.transition(MICRO, "path_opacity", "1");
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_length", "3.45");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("and so on."), SilenceBlock(1)));
    obs->manager.begin_timer("zoom_out_timer");
    obs->manager.transition(MACRO, "zoom", "-2.5 <zoom_out_timer> .15 * -");
    obs->manager.transition(MICRO, "path_opacity", "0");
    obs->manager.transition(MICRO, "island_opacity", "0");
    obs->manager.transition(MICRO, "singularity_depth", "100");
    cs.render_microblock();
    obs->manager.transition(MICRO, "singularity_depth", "200");
    cs.render_microblock();

    stage_macroblock(FileBlock("Let's check out the singularities of some unusual board shapes."));
    obs->manager.set("path_length", "0");
    obs->manager.transition(MICRO, "singularity_depth", "200");
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    // Start to morph the table a bit
    stage_macroblock(SilenceBlock(2));
    obs->manager.transition(MICRO, {{"zoom", "-1.5"}, {"singularity_rainbow", "0"}, {"island_opacity", "1"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(4));
    obs->manager.transition(MICRO, {{"v0.x", "-1.5"}, {"v0.y", "-2.6"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.5));
    cs.render_microblock();

    // Transition to a regular pentagon
    stage_macroblock(SilenceBlock(4));
    obs->add_dummy_point();
    StateSet pentagon = regular_ngon(5, 2.0, 3.1415 * 1.25);
    obs->manager.transition(MICRO, pentagon);
    cs.render_microblock();

    stage_macroblock(SilenceBlock(10));
    // TODO High flow depth
    obs->manager.transition(MICRO, "singularity_depth", "2000");
    obs->manager.begin_timer("spin");
    undo = obs->manager.transition(MICRO, {{"center_x", "<spin> .21 * sin 2.6 *"},
                                           {"center_y", "<spin> .21 * cos 2.6 *"}, {"zoom", "0.8"}});
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.transition(MICRO, "singularity_depth", "30000");
    obs->manager.transition(MICRO, {{"center_x", "<spin> .01 * 1.53 + sin 2.3 *"},
                                    {"center_y", "<spin> .01 * 1.53 + cos 2.3 *"}, {"zoom", "3"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(4));
    obs->manager.begin_timer("zoom_out_timer");
    obs->manager.transition(MACRO, "singularity_depth", "200");
    obs->manager.transition(MICRO, "zoom", "-2.2 <zoom_out_timer> .25 * -");
    cs.render_microblock();
    obs->manager.transition(MICRO, {{"center_x", "0"}, {"center_y", "0"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3));
    undo = obs->manager.transition(MICRO, "zoom", "-1");
    cs.render_microblock();

    StateSet warpy_pentagon;
    for (int i = 0; i < 5; i++) {
        double theta = (double)i * (2.0 * M_PI / 5.0) + 3.1415 * 1.25;
        string y_warp =  + " {t} " + to_string((i%3+5) * .1) + " * sin .3 * +";
        string x_warp =  + " {t} " + to_string((i  +5) * .1) + " * cos .3 * +";
        warpy_pentagon["v" + to_string(i) + ".x"] = to_string(2.0 * cos(theta)) + x_warp;
        warpy_pentagon["v" + to_string(i) + ".y"] = to_string(2.0 * sin(theta)) + y_warp;
    }
    stage_macroblock(SilenceBlock(10));
    obs->manager.transition(MACRO, warpy_pentagon);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("For these irregular board shapes, the singularities are tightly packed."));
    StateSet almost_square;
    for (int i = 0; i < 5; i++) {
        double theta = (double)i * (2.0 * M_PI / 4.2) + 3.1415 * 1.25;
        almost_square["v" + to_string(i) + ".x"] = to_string(2.0 * cos(theta));
        almost_square["v" + to_string(i) + ".y"] = to_string(2.0 * sin(theta));
    }
    obs->manager.transition(MICRO, almost_square);
    obs->manager.transition(MICRO, {{"center_x", "-2"}, {"center_y", "1"}});
    cs.render_microblock();

    stage_macroblock(FileBlock("Increasing the depth of our search for singularities, they fill increasingly more of the plane,"));
    obs->manager.set("singularity_depth", "<singularity_depth_log> exp");
    obs->manager.set("singularity_depth_log", "200 log");
    undo = obs->manager.transition(MICRO, {{"zoom", "3"}, {"singularity_depth_log", "30000 log"}, {"island_opacity", "0"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.3));
    cs.render_microblock();
    stage_macroblock(SilenceBlock(1.5));
    obs->manager.transition(MICRO, undo);
    cs.render_microblock();
    stage_macroblock(SilenceBlock(.3));
    cs.render_microblock();

    stage_macroblock(FileBlock("but some regions stay uninterrupted, no matter how deep we go."));
    obs->manager.transition(MICRO, {{"center_x", "-7"}, {"center_y", "-.3"}});
    cs.render_microblock();
    obs->manager.transition(MICRO, "zoom", "0");
    obs->manager.transition(MICRO, "island_opacity", "1");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(4));
    for(int i = 0; i < 2; i++) {
        obs->manager.transition(MICRO, "singularity_depth_log", "30000 log");
        cs.render_microblock();
        obs->manager.transition(MICRO, "singularity_depth_log", "500 log");
        cs.render_microblock();
    }
    cs.render_microblock();

    stage_macroblock(CompositeBlock(SilenceBlock(3), FileBlock("But what about the areas which are _not_ singularities?")));
    undo["zoom"] = "-1";
    obs->manager.transition(MACRO, undo);
    obs->manager.transition(MACRO, pentagon);
    cs.render_microblock();
    obs->manager.transition(MICRO, "singularity_opacity", "0");
    obs->manager.transition(MICRO, {{"center_x", "0"}, {"center_y", "0"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    int path_index = 0;
    int gridwidth = 20;
    /*
    for(int x = 0; x <= gridwidth; x++) {
        float xf = 2.6f + (float)x * 8.0 / (gridwidth+1);
        string xs = to_string(xf);
        for(int y = 0; y <= gridwidth; y++) {
            float yf = -2.8f + (float)y * 8.0 / (gridwidth+1);
            if(square(x-gridwidth/2) + square(y-gridwidth/2) > square(9.4f)) continue;
            string ys = to_string(yf);
            string pi = to_string(path_index);
            obs->manager.set({{"ball" + pi + "_start_x", xs}, {"ball" + pi + "_start_y", ys}});
            path_index++;
        }
    }
    */
    set_for_real(true);
    for(int i = 0; i < 10; i++) {
        float xf =  2.6f + 2.5f + sin(i*6.28/10) * .5f;
        float yf = -2.8f + 5.0f + cos(i*6.28/10) * .5f;
        string xs = to_string(xf);
        string ys = to_string(yf);
        string pi = to_string(path_index);
        obs->manager.set({{"ball" + pi + "_start_x", xs}, {"ball" + pi + "_start_y", ys}});
        path_index++;
    }
    for(int i = 0; i < 10; i++) {
        float xf =  2.6f + 5.5f + sin(i*6.28/10) * .5f;
        float yf = -2.8f + 5.0f + cos(i*6.28/10) * .5f;
        string xs = to_string(xf);
        string ys = to_string(yf);
        string pi = to_string(path_index);
        obs->manager.set({{"ball" + pi + "_start_x", xs}, {"ball" + pi + "_start_y", ys}});
        path_index++;
    }
    for(int i = 0; i < 20; i++) {
        float angle = (i-9.5) * 6.28/60 + 3 * 3.1415/2;
        float xf =  2.6f + 4.0f + cos(angle) * 2.0f;
        float yf = -2.8f + 4.0f + sin(angle) * 2.0f;
        string xs = to_string(xf);
        string ys = to_string(yf);
        string pi = to_string(path_index);
        obs->manager.set({{"ball" + pi + "_start_x", xs}, {"ball" + pi + "_start_y", ys}});
        path_index++;
    }
    cs.render_microblock();
    obs->manager.transition(MICRO, "ball_opacity", "1");
    cs.render_microblock();

    stage_macroblock(FileBlock("Within a given island, the operation of hitting a ball is an isometry. It preserves any shape."));
    obs->manager.set("path_opacity", "0");
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_length", "1");
    cs.render_microblock();
    undo = obs->manager.transition(MICRO, {{"path_opacity", "1"}, {"ball_opacity", "0.1"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(5));
    obs->manager.transition(MICRO, undo);
    for(int i = 2; i <= 5; i++) {
        obs->manager.transition(MICRO, "path_length", to_string(i));
        cs.render_microblock();
    }

    stage_macroblock(SilenceBlock(.5));
    obs->manager.transition(MICRO, "ball_opacity", "<path_opacity>");
    cs.render_microblock();

    for(int i = 1; i < path_index; i++) {
        string pi = to_string(i);
        obs->manager.remove(unordered_set<string>{"ball" + pi + "_start_x", "ball" + pi + "_start_y"});
    }
    obs->manager.set({{"ball0_start_x", "6.6"}, {"ball0_start_y", "1.2"}, {"path_length", "0"}});
    set_for_real(false);

    stage_macroblock(FileBlock("This means each island in a periodic cycle has the same shape."));
    obs->manager.transition(MICRO, {{"ball_opacity", "0"}, {"path_opacity", "0"}, {"zoom", "-.6"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();
    obs->manager.set({{"ball0_start_x", "2.03"}, {"ball0_start_y", "-.02"}, {"path_length", "0"}});
    obs->manager.set("cycle_highlight", "70");
    obs->manager.transition(MICRO, {{"ball_opacity", "1"}, {"path_opacity", "1"}});
    obs->manager.transition(MICRO, "cycle_highlight_enable", "1");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("This periodic cycle has 35 tiny little decagons."), SilenceBlock(5)));
    obs->manager.transition(MICRO, "path_length", "35", true);
    undo = obs->manager.transition(MICRO, {{"table_opacity", ".3"}, {"island_opacity", ".3"}});
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("But it's not the only cycle of period 35."), SilenceBlock(1)));
    obs->manager.transition(MICRO, "zoom", "-2");
    cs.render_microblock();

    stage_macroblock(FileBlock("But does every starting point even end up in a cycle to begin with?"));
    obs->manager.transition(MICRO, undo);
    obs->manager.transition(MICRO, "zoom", "-1");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    obs->manager.transition(MICRO, "cycle_highlight_enable", "0");
    obs->manager.transition(MICRO, {{"singularity_opacity", "0"}, {"path_opacity", "0"}, {"ball_opacity", "0"}});
    cs.render_microblock();

    stage_macroblock(FileBlock("This was the first shape proven to yield diverging orbits."));
    // Penrose kite has angles 72, 72, 72, 144 degrees.
    StateSet penrose_kite({{"v0.x", "-2"           }, {"v0.y", "0"},
                           {"v1.x", "1.2360679776" }, {"v1.y", "-2.3511410092"},
                           {"v2.x", "2"            }, {"v2.y", "0"},
                           {"v3.x", "1.2360679776" }, {"v3.y", "2.3511410092"},
                           {"v4.x", "<v3.x> <v0.x> + 2 /"}, {"v4.y", "<v3.y> <v0.y> + 2 /"}});
    obs->manager.transition(MACRO, penrose_kite);
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.transition(MICRO, "singularity_depth_log", "200 log"); // TODO needs to be at least 2000
    cs.render_microblock();
    obs->manager.remove(unordered_set<string>{"v4.x", "v4.y"});

    stage_macroblock(SilenceBlock(2));
    obs->manager.set({{"ball0_start_x", "7.416407865"}, {"ball0_start_y", "2.3511410092"}, {"path_length", "0"}});
    obs->manager.transition(MICRO, {{"ball_opacity", "1"}, {"path_opacity", "1"}, {"zoom", "-.8"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(4));
    obs->manager.transition(MICRO, "path_length", "<path_length_log> exp");
    obs->manager.set("path_length_log", "3 log");
    obs->manager.transition(MICRO, "ball_opacity", "0");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(12));
    obs->manager.transition(MACRO, "path_length_log", "20000 log");
    obs->manager.transition(MACRO, "zoom", "-5.18");
    obs->manager.set("path_opacity_log", "1 log");
    obs->manager.set("path_opacity", "<path_opacity_log> exp");
    obs->manager.transition(MACRO, "path_opacity_log", ".011 log");
    cs.render_microblock();

    stage_macroblock(FileBlock("Its unbounded orbit was first discovered by Richard Schwartz in 2006."));
    string title = "\\qquad \\qquad Schwartz, R. (2006) \\\\\\\\ \\tiny{Outer Billiards on the Penrose Kite: Compactification and Renormalization}";
    shared_ptr<WhitePaperScene> wps = make_shared<WhitePaperScene>("schwartz", title, vector<int>{1, 2, 3, 32});
    cs.add_scene_fade_in(MICRO, wps, "wps");
    wps->manager.set("which_page", "32");
    wps->manager.set("page_focus", "1");
    wps->manager.set("completion", "1");
    wps->manager.set({
        {"crop_top", ".28"},
        {"crop_bottom", ".7"},
        {"crop_left", ".23"},
        {"crop_right", ".75"},
    });
    cs.render_microblock();

    wps->manager.transition(MICRO, {
        {"crop_top", "0"},
        {"crop_bottom", "1"},
        {"crop_left", "0"},
        {"crop_right", "1"},
    });
    cs.render_microblock();
    wps->manager.transition(MICRO, "page_focus", "0");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    wps->manager.transition(MICRO, "completion", "0");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(5));
    obs->manager.transition(MACRO, "path_length_log", "0");
    cs.render_microblock();
    cs.remove_subscene("wps");
    obs->manager.transition(MICRO, "zoom", "-1");
    obs->manager.transition(MICRO, "singularity_opacity", "1");
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.transition(MICRO, {{"path_opacity", "0"}, {"ball_opacity", "0"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    // Reset everything above
    obs->manager.transition(MICRO, "singularity_depth", "125");
    cs.render_microblock();
    obs->manager.set("path_length", "0");
    obs->manager.remove(unordered_set<string>{"path_length_log", "path_opacity_log", "singularity_depth_log"});

    stage_macroblock(FileBlock("This shape is the Penrose Kite, which is used in aperiodic planar tilings."));
    // Set island and singularity opacity to 0 so we can see the tiling better.
    undo = obs->manager.transition(MICRO, {{"island_opacity", "0"}, {"singularity_opacity", "0"}});
    cs.render_microblock();
    cs.render_microblock();
    shared_ptr<PngScene> ps = make_shared<PngScene>("penrose");
    cs.add_scene_fade_in(MICRO, ps, "ps");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("But I still think my favorite shapes are the regular polygons."), SilenceBlock(2)));
    cs.fade_subscene(MICRO, "ps", 0);
    obs->manager.transition(MICRO, undo);
    cs.render_microblock();
    cs.remove_subscene("ps");
    obs->manager.transition(MICRO, regular_ngon(4, 2.0, 3.1415 * 1.25));
    cs.render_microblock();

    obs->manager.set(regular_ngon(8, 2.0, 3.1415 * 1.25));
    // Set odd index vertices to the midpoint of their neighbors, making a square.
    for(int i = 1; i < 8; i+=2) {
        string s_i = to_string(i);
        string s_ip1 = to_string((i+1)%8);
        string s_im1 = to_string(i-1);
        obs->manager.set({{"v" + s_i + ".x", "<v" + s_ip1 + ".x> <v" + s_im1 + ".x> + 2 /"},
                          {"v" + s_i + ".y", "<v" + s_ip1 + ".y> <v" + s_im1 + ".y> + 2 /"}});
    }
    // Transition to a regular octagon
    stage_macroblock(SilenceBlock(3));
    obs->manager.transition(MICRO, regular_ngon(8, 2.0, 3.1415 * 1.25));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    // Transition to a 12-gon
    stage_macroblock(SilenceBlock(3));
    // Construct a 12-gon with every 1st and 3rd vertex going around by pi/4, and then every 2nd vertex in between them. This makes an octagon ready to transition to a 12-gon.
    StateSet hack_12gon;
    double theta = 3.1415 * 1.25;
    for(int i = 0; i < 12; i++) {
        string s_i = to_string(i);
        string s_ip1 = to_string((i+1)%12);
        string s_im1 = to_string((i-1+12)%12);
        if(i%3 == 1) {
            // Midpoint of neighbors
            hack_12gon["v" + to_string(i) + ".x"] = "<v" + s_ip1 + ".x> <v" + s_im1 + ".x> + 2 /";
            hack_12gon["v" + to_string(i) + ".y"] = "<v" + s_ip1 + ".y> <v" + s_im1 + ".y> + 2 /";
        } else {
            hack_12gon["v" + to_string(i) + ".x"] = to_string(2.0 * cos(theta));
            hack_12gon["v" + to_string(i) + ".y"] = to_string(2.0 * sin(theta));
        }
        if(i%3 == 2) {
            theta += 3.141592653 / 4.0;
        } else {
            theta += 3.141592653 / 8.0;
        }
    }
    obs->manager.set(hack_12gon);
    obs->manager.transition(MICRO, regular_ngon(12, 2.0, 3.1415 * 1.25));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3));
    // Set alternate vertices to the midpoint of their neighbors, making a hexagon.
    for(int i = 1; i < 12; i+=2) {
        string s_i = to_string(i);
        string s_ip1 = to_string((i+1)%12);
        string s_im1 = to_string(i-1);
        obs->manager.transition(MICRO, {{"v" + s_i + ".x", "<v" + s_ip1 + ".x> <v" + s_im1 + ".x> + 2 /"},
                                        {"v" + s_i + ".y", "<v" + s_ip1 + ".y> <v" + s_im1 + ".y> + 2 /"}});
    }
    cs.render_microblock();
    obs->manager.set(regular_ngon(6, 2.0, 3.1415 * 1.25));
    obs->manager.remove(unordered_set<string>{"v6.x", "v6.y", "v7.x", "v7.y", "v8.x", "v8.y", "v9.x", "v9.y", "v10.x", "v10.y", "v11.x", "v11.y"});

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3));
    // Set alternate vertices to the midpoint of their neighbors, making a triangle.
    for(int i = 1; i < 6; i+=2) {
        string s_i = to_string(i);
        string s_ip1 = to_string((i+1)%6);
        string s_im1 = to_string(i-1);
        obs->manager.transition(MICRO, {{"v" + s_i + ".x", "<v" + s_ip1 + ".x> <v" + s_im1 + ".x> + 2 /"},
                                        {"v" + s_i + ".y", "<v" + s_ip1 + ".y> <v" + s_im1 + ".y> + 2 /"}});
    }
    cs.render_microblock();
    obs->manager.set(regular_ngon(3, 2.0, 3.1415 * 1.25));
    obs->manager.remove(unordered_set<string>{"v3.x", "v3.y", "v4.x", "v4.y", "v5.x", "v5.y"});

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3));
    obs->manager.set(regular_ngon(9, 2.0, 3.1415 * 1.25));
    // Set vertices not divisible by 3 to interpolate between their neighbors, making a triangle.
    for(int i = 0; i < 9; i++) {
        if(i % 3 == 0) continue;
        string s_i = to_string(i);
        string s_ip1 = to_string((i+1)%9);
        string s_ip2 = to_string((i+2)%9);
        string s_im1 = to_string((i-1+9)%9);
        string s_im2 = to_string((i-2+9)%9);
        if(i % 3 == 1) {
            // Weight towards the previous vertex
            obs->manager.set({{"v" + s_i + ".x", "<v" + s_ip2 + ".x> <v" + s_im1 + ".x> 2 * + 3 /"},
                              {"v" + s_i + ".y", "<v" + s_ip2 + ".y> <v" + s_im1 + ".y> 2 * + 3 /"}});
        } else {
            // Weight towards the next vertex
            obs->manager.set({{"v" + s_i + ".x", "<v" + s_ip1 + ".x> 2 * <v" + s_im2 + ".x> + 3 /"},
                              {"v" + s_i + ".y", "<v" + s_ip1 + ".y> 2 * <v" + s_im2 + ".y> + 3 /"}});
        }
    }
    obs->manager.transition(MICRO, regular_ngon(9, 2.0, 3.1415 * 1.25));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3));
    obs->manager.transition(MICRO, regular_ngon(7, 2.0, 3.1415 * 1.25));
    // Transition vertices 7 and 8 to be between 6 and 0.
    obs->manager.transition(MICRO, {{"v7.x", "<v6.x> 2 * <v0.x> + 3 /"}, {"v7.y", "<v6.y> 2 * <v0.y> + 3 /"},
                                    {"v8.x", "<v6.x> <v0.x> 2 * + 3 /"}, {"v8.y", "<v6.y> <v0.y> 2 * + 3 /"}});
    // TODO maybe all of the above should be slowly spinning?
    cs.render_microblock();
    // Remove vertices 7 and 8
    obs->manager.remove(unordered_set<string>{"v7.x", "v7.y", "v8.x", "v8.y"});

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    obs->manager.transition(MICRO, {{"singularity_opacity", "0"}, {"island_opacity", "0"}, {"path_opacity", "0"}});
    cs.render_microblock();

    obs->manager.set({{"singularity_opacity", "1"}, {"singularity_depth", "0"}, {"singularity_rainbow", "1"}});
    stage_macroblock(SilenceBlock(4));
    obs->manager.begin_timer("singdepth");
    obs->manager.transition(MACRO, "singularity_depth", "<singdepth> .5 * 2.3 ^ 2 +");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(12));
    obs->manager.transition(MICRO, "zoom", "-5");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(2));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3));
    obs->manager.transition(MICRO, {{"zoom", "-1"}, {"singularity_depth", "100"}, {"singularity_rainbow", "0"}, {"island_opacity", "1"}});
    cs.render_microblock();

    stage_macroblock(FileBlock("We've been plotting where a ball would land based on its starting position."));
    obs->manager.set({{"ball0_start_x", "4"}, {"ball0_start_y", "3"}});
    obs->manager.transition(MICRO, {{"ball_opacity", "1"}, {"path_opacity", "1"}, {"island_opacity", "0"}, {"singularity_opacity", "0"}});
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_length", "10");
    obs->manager.transition(MICRO, "ball_distance", "0");
    cs.render_microblock();

    stage_macroblock(FileBlock("That path changes based on the table."));
    obs->manager.transition(MACRO, {{"v4.x", "<v3.x> 3 * <v0.x> 1 * + 4 /"}, {"v4.y", "<v3.y> 3 * <v0.y> 1 * + 4 /"},
                                    {"v5.x", "<v3.x> 2 * <v0.x> 2 * + 4 /"}, {"v5.y", "<v3.y> 2 * <v0.y> 2 * + 4 /"},
                                    {"v6.x", "<v3.x> 1 * <v0.x> 3 * + 4 /"}, {"v6.y", "<v3.y> 1 * <v0.y> 3 * + 4 /"}});
    cs.render_microblock();
    obs->manager.transition(MICRO, regular_ngon(4, 1.0, 3.1415 * 1.25));
    // Transition extra vertices to the midpoint of their neighbors, making a square from a heptagon.
    cs.render_microblock();
    obs->manager.remove(unordered_set<string>{"v5.x", "v5.y", "v6.x", "v6.y"});

    stage_macroblock(FileBlock("So let's see what happens if we fix the ball,"));
    // Move around v4
    obs->manager.transition(MICRO, {{"v4.x", "{t} sin .5 * 3 -"}, {"v4.y", "{t} cos .5 *"}});
    cs.render_microblock();

    stage_macroblock(FileBlock("but we plot where it would go depending on where the last table corner is placed."));
    shared_ptr<OuterBilliardsVertexFlowScene> obvfs = make_shared<OuterBilliardsVertexFlowScene>();
    cs.add_scene_fade_in(MICRO, obvfs, "obvfs");
    obvfs->manager.set(regular_ngon(4, 1.0, 3.1415 * 1.25));
    obvfs->manager.set({{"ball_start_x", "4"}, {"ball_start_y", "3"}});
    obvfs->manager.set("zoom", "-1");

    obvfs->manager.set("flow_depth", "160");
    cs.render_microblock();
    cs.remove_subscene("obs");

    // Move the point around
    stage_macroblock(SilenceBlock(15));
    obvfs->manager.transition(MICRO, {{"ball_start_x", "{t} sin 1.41421 *"}, {"ball_start_y", "{t} cos 1.41421 *"}});
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.add_scene(obs, "obs", true);
    obs->manager.transition(MICRO, {{"ball_opacity", "0"}, {"path_opacity", "0"}, {"island_opacity", "1"}, {"singularity_opacity", "1"}});
    cs.fade_subscene(MICRO, "obvfs", 0);
    // Remove vertex 4
    obs->manager.remove(unordered_set<string>{"v4.x", "v4.y"});
    cs.render_microblock();
    cs.remove_subscene("obvfs");

    stage_macroblock(FileBlock("The rules of outer billiards seem simple, but the resulting dynamics still aren't fully understood."));
    cs.render_microblock();

    stage_macroblock(FileBlock("A largely untouched field asks,"));
    cs.render_microblock();

    stage_macroblock(FileBlock("What happens if we move from euclidean to hyperbolic geometry?"));
    obs->manager.transition(MICRO, "curvature", "-.1760");
    cs.render_microblock();
}
