#include "../Scenes/Math/InnerBilliardsScene.h"
#include "../Scenes/Math/OuterBilliardsScene.h"
#include "../Scenes/Math/OuterBilliardsVertexFlowScene.h"
#include "../Scenes/Math/GraphScene.h"
#include "../Scenes/Math/GeometryScene.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Media/WhitePaperScene.h"
#include "../Core/State/BezierStateCurve.h"
#include "../Core/Smoketest.h"

StateSet regular_ngon(int n, double radius, double phase, bool rotation = false) {
    StateSet result;
    for (int i = 0; i < n; i++) {
        double theta = phase + (double)i * (2.0 * M_PI / (double)n);
        float x = radius * cos(theta);
        float y = radius * sin(theta);
        string xs = to_string(x);
        string ys = to_string(y);
        if (rotation) {
            // Rotate the polygon by <rotation> radians
            result["v" + to_string(i) + ".x"] = xs + " <rotation> cos * " + ys + " <rotation> sin * -";
            result["v" + to_string(i) + ".y"] = xs + " <rotation> sin * " + ys + " <rotation> cos * +";
        }
        else {
            result["v" + to_string(i) + ".x"] = xs;
            result["v" + to_string(i) + ".y"] = ys;
        }
    }
    return result;
}

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

    ibs->manager.set({{"center_x", "<ball_start_x> .6 *"}, {"center_y", "<ball_start_y> .6 *"}});
    ibs->manager.set({{"ball_start_x", "-0.4"}, {"ball_start_y", "-0.5"}});
    set_global_state("billiards_ball_x", -0.4);
    set_global_state("billiards_ball_y", -0.5);
    ibs->manager.set("ball_angle", "0.78539");
    ibs->manager.set("ball_distance", "-3");

    stage_macroblock(CompositeBlock(FileBlock("Hit a billiards ball,"), SilenceBlock(2)));
    ibs->manager.transition(MICRO, {{"center_x", "{billiards_ball_x} .6 *"}, {"center_y", "{billiards_ball_y} .6 *"}});
    ibs->manager.transition(MACRO, "ball_distance", "11.3137");
    ibs->manager.transition(MICRO, "zoom", ".5");
    cs.render_microblock();
    cs.render_microblock();
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
    ibs->manager.transition(MACRO, "path_length", "80");
    ibs->manager.transition(MICRO, "zoom", "0");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(4));
    ibs->manager.transition(MICRO, "cue_opacity", ".7");
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
    ibs->manager.transition(MACRO, "ball_angle", "0.55");
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
    obs->stage_publish_to_global = {{"ball0_start_x", "ball0_start_x"}, {"ball0_start_y", "ball0_start_y"}, {"v0.x", "v0.x"}, {"v0.y", "v0.y"}, {"v1.x", "v1.x"}, {"v1.y", "v1.y"}, {"v2.x", "v2.x"}, {"v2.y", "v2.y"}, {"zoom", "zoom"}, {"center_x", "center_x"}, {"center_y", "center_y"}, {"ball_distance", "ball_distance"}, {"path_length", "path_length"}, {"cycle_highlight", "cycle_highlight"}, {"cycle_highlight_enable", "cycle_highlight_enable"}, {"island_opacity", "island_opacity"}, {"ball_opacity", "ball_opacity"}, {"path_opacity", "path_opacity"}, {"flow_depth", "flow_depth"}, {"singularity_depth", "singularity_depth"}, {"singularity_opacity", "singularity_opacity"}, {"singularity_rainbow", "singularity_rainbow"}, {"curvature", "curvature"}};
    obs->manager.set(simple_table);
    obs->manager.set({{"ball0_start_x", to_string(bsx)}, {"ball0_start_y", to_string(bsy)}});
    //cs.fade_subscene(MICRO, "ls", 0);
    cs.fade_subscene(MICRO, "ibs", 0);
    cs.add_scene(obs, "obs", vec2(.5, .5), true);
    ibs->manager.transition(MICRO, {{"zoom", "-.8001"}, {"center_y", "0"}});
    obs->manager.transition(MICRO, {{"zoom", "-.8001"}, {"center_y", "0"}});
    cs.render_microblock();
    cs.remove_subscene("ibs");

    {
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
        gs->config.transition_edge_label(MICRO, 0, 1, "a");
    }
    cs.render_microblock();

    stage_macroblock(FileBlock("and continue past its point of contact by the same distance."));
    cs.render_microblock();
    gs->transition_node_position(MICRO, 1, vec4(p1.x, p1.y, 0, 0));
    gs->config.transition_edge_label(MICRO, 0, 1, "2a");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();
    gs->config.transition_edge_label(MICRO, 0, 1, "");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("Do that again,"), SilenceBlock(2)));
    {
        const vec2 p2 = obs->build_orbit_path((double)2).back();
        gs->graph.add_node(2);
        gs->graph.move_node(2, vec4(p1.x, p1.y, 0, 0));
        const vec2 midpoint = (p1 + p2) * 0.5;
        gs->transition_node_position(MICRO, 2, vec4(midpoint.x, midpoint.y, 0, 0));
        gs->graph.add_edge(1, 2);
        gs->config.set_edge_label(1, 2, "");
        gs->config.transition_edge_label(MICRO, 1, 2, "b");
        cs.render_microblock();
        gs->transition_node_position(MICRO, 2, vec4(p2.x, p2.y, 0, 0));
        gs->config.transition_edge_label(MICRO, 1, 2, "2b");
        cs.render_microblock();
    }

    stage_macroblock(SilenceBlock(.5));
    cs.render_microblock();
    gs->config.transition_edge_label(MICRO, 0, 1, "");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("and again, and again... and we get a nice pattern."), SilenceBlock(1)));
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
    obs->manager.transition(MICRO, "table_opacity", "1");
    for (int i = 0; i < 8; i++) {
        const vec2 p = obs->build_orbit_path((double)i).back();
        gs->transition_node_position(MICRO, i, vec4(p.x, p.y, 0, 0));
    }
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    obs->manager.set("ball_distance", "0");
    obs->manager.set("path_length", "8");
    obs->manager.transition(MICRO, "ball_opacity", "1");
    cs.render_microblock();
    cs.fade_subscene(MICRO, "gs", 0);
    cs.render_microblock();
    cs.remove_subscene("gs");
    }

    stage_macroblock(FileBlock("For this starting point, we get a cycle of length 4."));
    obs->manager.set("angley", ".8 {t} sin .05 * +");
    obs->manager.transition(MICRO, {{"ball0_start_x", "<angley> 2 * sin 4 *"}, {"ball0_start_y", "<angley> 2 * cos 4 *"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.5));
    cs.render_microblock();

    stage_macroblock(FileBlock("What starting points have what cycle lengths?"));
    obs->manager.transition(MICRO, "angley", "5");
    obs->manager.transition(MICRO, "zoom", "-1.5");
    obs->manager.set("path_length", "24");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    obs->manager.transition(MICRO, "angley", "4");
    cs.render_microblock();

    stage_macroblock(FileBlock("Here's the pattern."));
    obs->manager.transition(MACRO, "island_opacity", ".5");
    obs->manager.transition(MACRO, "singularity_depth", "400");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(SilenceBlock(1), CompositeBlock(FileBlock("These 4 regions are stuck in a 4-move cycle."), SilenceBlock(.3))));
    obs->manager.set("cycle_highlight", "4");
    obs->manager.transition(MICRO, {{"ball0_start_x", "{t} 3 * sin .5 * 4 +"}, {"ball0_start_y", "{t} 2 * cos .5 *"}});
    obs->manager.transition(MICRO, "cycle_highlight_enable", "1");
    obs->manager.transition(MACRO, "island_opacity", "1");
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.transition(MICRO, "cycle_highlight", "6");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("These 8 are a little more complicated."), SilenceBlock(.3)));
    obs->manager.transition(MICRO, "cycle_highlight", "8");
    obs->manager.transition(MICRO, {{"ball0_start_x", "{t} 3 * sin .5 * 4 +"}, {"ball0_start_y", "{t} 2 * cos .5 * 2 +"}});
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.transition(MICRO, "cycle_highlight", "10");
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

    stage_macroblock(FileBlock("Instead of counting how many moves it takes to return,"));
    obs->manager.set({{"ball0_start_x", "4"}, {"ball0_start_y", "4"}});
    obs->manager.set({{"ball_distance", "<path_length>"}, {"path_length", "0"}});
    obs->manager.transition(MICRO, "path_opacity", "1");
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_length", "12");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("I gave every point in the plane a unique color."), SilenceBlock(1)));
    obs->manager.transition(MICRO, "path_opacity", "0");
    obs->manager.transition(MACRO, "periodicity_or_flow", "1");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("Now we can watch a ball's path after one move,"));
    cs.render_microblock();
    obs->manager.transition(MICRO, "flow_depth", "1");
    cs.render_microblock();

    stage_macroblock(FileBlock("two moves,"));
    obs->manager.transition(MICRO, "flow_depth", "2");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("four moves,"));
    obs->manager.transition(MICRO, "flow_depth", "4");
    cs.render_microblock();
    cs.render_microblock();

    int num_iterations = 51*4;
    stage_macroblock(FileBlock("and so on."));
    shared_ptr<OuterBilliardsScene> obs2 = make_shared<OuterBilliardsScene>();
    obs2->manager.set(simple_table);
    obs2->manager.set({{"zoom", "-1.5"}, {"flow_depth", to_string(num_iterations)}, {"singularity_depth", "400"}, {"singularity_opacity", "0"}, {"island_opacity", "1"}, {"periodicity_or_flow", "1"}, {"ball_opacity", "0"}});
    cs.add_scene_fade_in(MICRO, obs2, "obs2");
    cs.render_microblock();
    obs->manager.set("flow_depth", to_string(num_iterations));
    cs.remove_subscene("obs2");

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3));
    obs->manager.transition(MICRO, {
        {"v0.x", "-1.5"}, {"v0.y", "-2.6"},
        {"v1.x",  "1.5"}, {"v1.y", "-2.6"},
        {"v2.x",  "1.5"}, {"v2.y",  "2.6"},
        {"v3.x", "-1.5"}, {"v3.y",  "2.6"},
    });
    cs.render_microblock();
    obs->manager.transition(MICRO, {
        {"v0.x", "-1.8"}, {"v0.y", "-.4"},
        {"v1.x",  "1.8"}, {"v1.y", "-.4"},
        {"v2.x",  "1.8"}, {"v2.y",  ".4"},
        {"v3.x", "-1.8"}, {"v3.y",  ".4"},
    });
    cs.render_microblock();
    obs->manager.transition(MICRO, {
        {"v0.x", "-1"}, {"v0.y", "-1"},
        {"v1.x",  "1"}, {"v1.y", "-1"},
        {"v2.x",  "1"}, {"v2.y",  "1"},
        {"v3.x", "-1"}, {"v3.y",  "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("A parallelogram table tiles the plane just like a rectangle."));
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

    stage_macroblock(SilenceBlock(.7));
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("Let's try other shapes!"), SilenceBlock(3)));
    obs->manager.transition(MICRO, { // Trapezoid
        {"v0.x", "-1"}, {"v0.y", "-2"},
        {"v1.x",  "1"}, {"v1.y", "-1"},
        {"v2.x",  "1"}, {"v2.y",  "1"},
        {"v3.x", "-1"}, {"v3.y",  "2"},
    });
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.7));
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
    cs.render_microblock();

    stage_macroblock(FileBlock("What happens when a ball is right between two regions?"));
    obs->manager.transition(MACRO, "island_opacity", "0");
    obs->manager.transition(MICRO, "singularity_opacity", ".2");
    cs.render_microblock();
    num_iterations = 93*4;
    obs->manager.set("flow_depth", to_string(num_iterations));

    stage_macroblock(FileBlock("Let's place it here."));
    obs->manager.set({{"path_length", "0"}, {"ball_distance", "0"}});
    obs->manager.transition(MICRO, "island_opacity", "0.5");
    obs->manager.transition(MICRO, "path_opacity", "1");
    obs->manager.transition(MICRO, {{"ball0_start_y", "-2"}, {"ball0_start_x", "3"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("The first move looks normal..."));
    obs->manager.set("ball_distance", "<path_length>");
    obs->manager.transition(MICRO, "path_length", "1");
    cs.render_microblock();

    stage_macroblock(FileBlock("But on move two, the ball traces along the edge of the table."));
    obs->manager.transition(MICRO, "path_length", "1.4");
    StateSet undo = obs->manager.transition(MICRO, {{"zoom", "0"}, {"center_x", "-.5"}});
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.transition(MICRO, undo);
    cs.render_microblock();

    stage_macroblock(FileBlock("It's not clear which corner we jump by!"));
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

    stage_macroblock(FileBlock("We call this a singularity."));
    obs->manager.transition(MICRO, "singularity_opacity", "0");
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_length", "1.4");
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_opacity", "0");
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.set("path_length", "0");

    stage_macroblock(SilenceBlock(.5));
    obs->manager.set({{"ball0_start_x", "4"}, {"ball0_start_y", "1"}});
    cs.render_microblock();

    stage_macroblock(FileBlock("This starting point is an immediate singularity,"));
    obs->manager.transition(MICRO, "ball_opacity", "1");
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.set("ball_opacity", "<path_opacity>");
    obs->manager.transition(MICRO, "path_length", ".4");
    obs->manager.set("singularity_rainbow", "1");
    obs->manager.set("path_opacity", "1");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("just like any point on these lines."));
    obs->manager.transition(MICRO, {{"singularity_opacity", "1"}, {"island_opacity", "0.2"}, {"path_opacity", "0"}});
    obs->manager.set("singularity_depth", "1");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    obs->manager.set("path_length", "0");
    obs->manager.transition(MICRO, {{"ball0_start_y", "-6"}, {"ball0_start_x", "3"}});
    obs->manager.transition(MICRO, "path_opacity", "1");
    cs.render_microblock();

    stage_macroblock(FileBlock("This is a depth 2 singularity,"));
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_length", "1.45");
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_opacity", "0");
    obs->manager.transition(MICRO, "singularity_depth", "2");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.5));
    obs->manager.transition(MICRO, "path_opacity", "0");
    cs.render_microblock();

    stage_macroblock(FileBlock("depth 3 singularity,"));
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

    stage_macroblock(CompositeBlock(FileBlock("and beyond."), SilenceBlock(3)));
    obs->manager.begin_timer("zoom_out_timer");
    obs->manager.transition(MACRO, "zoom", "-3 <zoom_out_timer> .05 * -");
    obs->manager.transition(MICRO, "path_opacity", "0");
    obs->manager.transition(MICRO, "island_opacity", "0");
    obs->manager.set("singularity_depth_log", "3 log");
    obs->manager.set("singularity_depth", "<singularity_depth_log> exp");
    obs->manager.transition(MICRO, "singularity_depth_log", "400 log");
    cs.render_microblock();

    stage_macroblock(FileBlock("Let's check out the singularities of some different table shapes."));
    obs->manager.set("singularity_depth", "400");
    obs->manager.set("path_length", "0");
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

    stage_macroblock(SilenceBlock(.4));
    cs.render_microblock();

    // Transition to a regular pentagon
    stage_macroblock(SilenceBlock(4));
    obs->add_dummy_point();
    StateSet pentagon = regular_ngon(5, 2.0, 3.1415 * 1.25);
    obs->manager.transition(MICRO, pentagon);
    cs.render_microblock();

    stage_macroblock(SilenceBlock(8));
    obs->manager.transition(MICRO, "singularity_depth", "2000");
    obs->manager.begin_timer("spin");
    undo = obs->manager.transition(MICRO, {{"center_x", "<spin> .1575 * sin 2.6 *"},
                                           {"center_y", "<spin> .1575 * cos 2.6 *"}, {"zoom", "0.8"}});
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(4));
    obs->manager.begin_timer("spin2");
    obs->manager.transition(MICRO, "singularity_depth", "40000");
    obs->manager.transition(MICRO, {{"center_x", "<spin2> .01 * 7.53 + sin 2.3 *"},
                                    {"center_y", "<spin2> .01 * 7.53 + cos 2.3 *"}, {"zoom", "3"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(6));
    obs->manager.begin_timer("zoom_out_timer");
    obs->manager.transition(MACRO, "singularity_depth", "400");
    obs->manager.transition(MICRO, "zoom", "-1 <zoom_out_timer> .3 * -");
    cs.render_microblock();
    obs->manager.transition(MICRO, {{"center_x", "0"}, {"center_y", "0"}});
    cs.render_microblock();

    StateSet warpy_pentagon;
    for (int i = 0; i < 5; i++) {
        double theta = (double)i * (2.0 * M_PI / 5.0) + 3.1415 * 1.25;
        string y_warp =  + " {t} " + to_string((i%3+5) * .11) + " * sin .7 * +";
        string x_warp =  + " {t} " + to_string((i  +5) * .1) + " * cos .7 * +";
        warpy_pentagon["v" + to_string(i) + ".x"] = to_string(2.0 * cos(theta)) + x_warp;
        warpy_pentagon["v" + to_string(i) + ".y"] = to_string(2.0 * sin(theta)) + y_warp;
    }
    stage_macroblock(SilenceBlock(10));
    obs->manager.transition(MACRO, warpy_pentagon);
    cs.render_microblock();
    undo = obs->manager.transition(MICRO, "zoom", "-1");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("For irregular tables, the singularities are tightly packed."), SilenceBlock(1)));
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
    obs->manager.set("singularity_depth_log", "400 log");
    undo = obs->manager.transition(MICRO, {{"zoom", "3"}, {"singularity_depth_log", "30000 log"}, {"island_opacity", "0"}, {"singularity_rainbow", "1"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.3));
    cs.render_microblock();

    stage_macroblock(CompositeBlock(SilenceBlock(2), FileBlock("but some regions stay uninterrupted.")));
    undo["zoom"] = "0";
    obs->manager.transition(MACRO, undo);
    cs.render_microblock();
    obs->manager.transition(MICRO, {{"center_x", "-7"}, {"center_y", "-.3"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3.5));
    obs->manager.transition(MICRO, "singularity_depth_log", "20000 log");
    cs.render_microblock();
    obs->manager.transition(MICRO, "singularity_depth_log", "400 log");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("What decides the shape of these stable regions?"), SilenceBlock(1)));
    undo["zoom"] = "-2.5";
    obs->manager.transition(MACRO, undo);
    obs->manager.transition(MACRO, pentagon);
    obs->manager.transition(MICRO, {{"center_x", "0"}, {"center_y", "0"}});
    obs->manager.transition(MICRO, "singularity_opacity", "0");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("Hitting a lot of balls, their shape is quickly jumbled up."), SilenceBlock(3)));
    int path_index = 0;
    for(int i = 0; i < 20; i++) {
        float xf = sin(i*6.28/20) * .4f - 1.0f;
        float yf = cos(i*6.28/20) * .6f + 1.0f;
        string xs = to_string(xf*10);
        string ys = to_string(yf*10);
        string pi = to_string(path_index);
        obs->manager.set({{"ball" + pi + "_start_x", xs}, {"ball" + pi + "_start_y", ys}});
        path_index++;
    }
    for(int i = 0; i < 20; i++) {
        float xf = sin(i*6.28/20) * .4f + 1.0f;
        float yf = cos(i*6.28/20) * .6f + 1.0f;
        string xs = to_string(xf*10);
        string ys = to_string(yf*10);
        string pi = to_string(path_index);
        obs->manager.set({{"ball" + pi + "_start_x", xs}, {"ball" + pi + "_start_y", ys}});
        path_index++;
    }
    for(int i = 0; i < 25; i++) {
        float angle = (i-12) * 6.28/(25*3) + 3 * 3.1415/2;
        float xf = cos(angle) * 2.0f;
        float yf = (-1.2 - sin(angle)) * 2.0f;
        string xs = to_string(xf*10);
        string ys = to_string(yf*10);
        string pi = to_string(path_index);
        obs->manager.set({{"ball" + pi + "_start_x", xs}, {"ball" + pi + "_start_y", ys}});
        path_index++;
    }
    obs->manager.transition(MICRO, "ball_opacity", "1");
    cs.render_microblock();
    for(int i = 1; i <= 4; i++) {
        obs->manager.transition(MICRO, "path_length", to_string(i));
        cs.render_microblock();
    }

    stage_macroblock(SilenceBlock(1));
    obs->manager.transition(MICRO, "path_opacity", "0");
    obs->manager.transition(MICRO, "ball_opacity", "0");
    obs->manager.transition(MACRO, "zoom", "-1.5");
    cs.render_microblock();

    stage_macroblock(FileBlock("But within a given island, hitting a ball is an isometry. It preserves shapes."));
    obs->manager.set("path_length", "0");
    // Remove all the balls we just added
    for(int i = 0; i < path_index; i++) {
        string pi = to_string(i);
        obs->manager.remove(unordered_set<string>{"ball" + pi + "_start_x", "ball" + pi + "_start_y"});
    }
    path_index = 0;
    for(int i = 0; i < 10; i++) {
        float xf =  2.6f + 3.0f + sin(i*6.28/10) * .4f;
        float yf = -2.8f + 4.0f + cos(i*6.28/10) * .6f;
        string xs = to_string(xf);
        string ys = to_string(yf);
        string pi = to_string(path_index);
        obs->manager.set({{"ball" + pi + "_start_x", xs}, {"ball" + pi + "_start_y", ys}});
        path_index++;
    }
    for(int i = 0; i < 10; i++) {
        float xf =  2.6f + 5.0f + sin(i*6.28/10) * .4f;
        float yf = -2.8f + 4.0f + cos(i*6.28/10) * .6f;
        string xs = to_string(xf);
        string ys = to_string(yf);
        string pi = to_string(path_index);
        obs->manager.set({{"ball" + pi + "_start_x", xs}, {"ball" + pi + "_start_y", ys}});
        path_index++;
    }
    for(int i = 0; i < 14; i++) {
        float angle = (i-6.5) * 6.28/(14*3) + 3 * 3.1415/2;
        float xf =  2.6f + 4.0f + cos(angle) * 2.0f;
        float yf = -2.8f + 4.0f + sin(angle) * 2.0f;
        string xs = to_string(xf);
        string ys = to_string(yf);
        string pi = to_string(path_index);
        obs->manager.set({{"ball" + pi + "_start_x", xs}, {"ball" + pi + "_start_y", ys}});
        path_index++;
    }
    obs->manager.transition(MICRO, "ball_opacity", "1");
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_length", "1");
    cs.render_microblock();
    undo = obs->manager.transition(MICRO, {{"path_opacity", "1"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3));
    for(int i = 2; i <= 3; i++) {
        obs->manager.transition(MICRO, "path_length", to_string(i));
        cs.render_microblock();
    }

    stage_macroblock(SilenceBlock(.5));
    obs->manager.transition(MICRO, undo);
    cs.render_microblock();
    obs->manager.transition(MICRO, "ball_opacity", "<path_opacity>");
    cs.render_microblock();

    for(int i = 1; i < path_index; i++) {
        string pi = to_string(i);
        obs->manager.remove(unordered_set<string>{"ball" + pi + "_start_x", "ball" + pi + "_start_y"});
    }
    obs->manager.set({{"ball0_start_x", "6.6"}, {"ball0_start_y", "1.2"}, {"path_length", "0"}});

    stage_macroblock(FileBlock("So, each island in a periodic cycle has the same shape."));
    obs->manager.transition(MICRO, {{"ball_opacity", "0"}, {"path_opacity", "0"}, {"zoom", "-.6"}});
    obs->manager.set("cycle_highlight", "70");
    obs->manager.transition(MICRO, "cycle_highlight_enable", "1");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();
    obs->manager.set({{"ball0_start_x", "2.03"}, {"ball0_start_y", "-.02"}, {"path_length", "0"}});
    obs->manager.transition(MICRO, {{"ball_opacity", "1"}, {"path_opacity", "1"}});
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("This periodic cycle has 35 tiny little decagons."), SilenceBlock(3)));
    obs->manager.transition(MICRO, "path_length", "35");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("And so does this one!"), SilenceBlock(1)));
    obs->manager.transition(MICRO, "zoom", "-2");
    // fade out the path and ball
    obs->manager.transition(MICRO, {{"ball_opacity", "0"}, {"path_opacity", "0"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.5));
    obs->manager.set({{"ball0_start_x", "13.9"}, {"ball0_start_y", "-.2"}, {"path_length", "0"}});
    obs->manager.transition(MICRO, {{"ball_opacity", "1"}, {"path_opacity", "1"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3));
    obs->manager.transition(MICRO, {{"ball_opacity", "1"}, {"path_opacity", "1"}});
    // Transition to path length 35
    obs->manager.transition(MICRO, "path_length", "35");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.8));
    obs->manager.transition(MICRO, {{"ball_opacity", "0"}, {"path_opacity", "0"}});
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("Do all starting points have cyclic orbits?"), SilenceBlock(1)));
    undo["path_opacity"] = "0";
    obs->manager.transition(MICRO, undo);
    obs->manager.transition(MICRO, regular_ngon(5, 2.0, 3.1415*1.3));
    obs->manager.transition(MICRO, "zoom", "-.5");
    obs->manager.transition(MICRO, "cycle_highlight_enable", "0");
    cs.render_microblock();

    shared_ptr<GeometryScene> gs = make_shared<GeometryScene>();
    cs.add_scene(gs, "gs");
    gs->manager.set({
        {"zoom", "-.5"},
        {"ticks_opacity", "0"},
    });

    const float phi = (1.f + std::sqrt(5.f)) / 2.f;
    auto vtx = [](float radius, float angle_deg) {
        const float a = angle_deg * float(M_PI) / 180.f;
        return vec2(radius * std::cos(a), radius * std::sin(a));
    };
    // Intersection of the infinite line through (p,q) with the line through (u,v).
    auto meet = [](vec2 p, vec2 q, vec2 u, vec2 v) {
        const vec2 d1 = q - p, d2 = v - u;
        const float t = ((u.x - p.x) * d2.y - (u.y - p.y) * d2.x) / (d1.x * d2.y - d1.y * d2.x);
        return p + d1 * t;
    };

    // Outer pentagon: A at the top, then B C D E clockwise.
    const float R = 2.0f;
    const vec2 A = vtx(R,  90), B = vtx(R,  18), C = vtx(R, -54), D = vtx(R, -126), E = vtx(R, 162);

    // Inner pentagon (the diagonal intersections): a at the bottom, then b c d e clockwise.
    const float r = R / (phi * phi);
    const vec2 a = vtx(r, -90), b = vtx(r, -162), c = vtx(r, 126), d = vtx(r, 54), e = vtx(r, -18);

    // --- Outer pentagon: each vertex drops in with the edge that reaches it ---
    stage_macroblock(CompositeBlock(FileBlock("Here's a counterexample."), SilenceBlock(3.5)));
    gs->construction.add(GeometricPoint(A, "A", 1.0f, false, ""));
    gs->construction.add(GeometricPoint(B, "B", 1.0f, false, ""));
    gs->construction.add(GeometricLine(A, B, "AB"));
    gs->construction.add(GeometricPoint(C, "C", 1.0f, false, ""));
    gs->construction.add(GeometricLine(B, C, "BC"));
    gs->construction.add(GeometricPoint(D, "D", 1.0f, false, ""));
    gs->construction.add(GeometricLine(C, D, "CD"));
    gs->construction.add(GeometricPoint(E, "E", 1.0f, false, ""));
    gs->construction.add(GeometricLine(D, E, "DE"));
    gs->construction.add(GeometricLine(E, A, "EA"));
    cs.render_microblock();

    // --- The five diagonals, drawn as one continuous pentagram stroke ---
    gs->construction.add(GeometricLine(A, C, "AC"));
    gs->construction.add(GeometricLine(C, E, "CE"));
    gs->construction.add(GeometricPoint(d, "d", 0.7f, false, ""));
    gs->construction.add(GeometricLine(E, B, "EB"));
    gs->construction.add(GeometricPoint(e, "e", 0.7f, false, ""));
    gs->construction.add(GeometricPoint(a, "a", 0.7f, false, ""));
    gs->construction.add(GeometricLine(B, D, "BD"));
    gs->construction.add(GeometricPoint(b, "b", 0.7f, false, ""));
    gs->construction.add(GeometricPoint(c, "c", 0.7f, false, ""));
    gs->construction.add(GeometricLine(D, A, "DA"));
    cs.render_microblock();

    // --- Extend ce to outer edge BC (point y), and ca to outer edge CD (point z) ---
    const vec2 y = meet(c, e, B, C);
    const vec2 z = meet(c, a, C, D);
    gs->construction.add(GeometricPoint(y, "y", 0.7f, false, ""));
    gs->construction.add(GeometricLine(c, y, "cy"));
    gs->construction.add(GeometricPoint(z, "z", 0.7f, false, ""));
    gs->construction.add(GeometricLine(c, z, "cz"));
    cs.render_microblock();

    // --- Connect y-z; w is where it crosses diagonal CE ---
    const vec2 w = meet(y, z, C, E);
    gs->construction.add(GeometricLine(y, z, "yz"));
    gs->construction.add(GeometricPoint(w, "w", 0.7f, false, ""));
    cs.render_microblock();

    // --- Connect a-y and w-e; P is their intersection ---
    const vec2 P = meet(a, y, w, e);
    gs->construction.add(GeometricLine(a, y, "ay"));
    gs->construction.add(GeometricLine(w, e, "we"));
    gs->construction.add(GeometricPoint(P, "P", 0.75f, false, ""));
    gs->construction.add(GeometricLine(A, d, "Ad"));
    gs->construction.add(GeometricLine(E, a, "Ea"));
    cs.render_microblock();

    // --- Strip everything back to lines Ad, ad, EA, Ea and the point P ---
    const vector<string> dead_lines = {"AB","BC","CD","DE","AC","CE","EB","BD","DA","cy","cz","yz","ay","we"};
    const vector<string> dead_points = {"A","B","C","D","E","a","b","c","d","e","y","z","w"};
    for (const string& id : dead_lines)  gs->construction.fade_line(id);
    for (const string& id : dead_points) gs->construction.fade_point(id);
    gs->construction.add(GeometricLine(a, d, "ad"));
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("This was the first shape proven to feature diverging orbits."));

    StateSet penrose_kite_setup({{"v2.x", to_string(d.x)}, {"v2.y", to_string(d.y)}, {"v1.x", to_string(a.x)}, {"v1.y", to_string(a.y)},
            {"v0.x", "<v1.x> <v4.x> + 2 /"},
            {"v0.y", "<v1.y> <v4.y> + 2 /"}});
    obs->manager.transition(MACRO, penrose_kite_setup);
    obs->manager.set({{"ball0_start_x", to_string(P.x)}, {"ball0_start_y", to_string(P.y)}, {"path_length", "0"}});
    obs->manager.transition(MICRO, {{"ball_opacity", "1"}, {"path_opacity", "1"}});
    cs.render_microblock();
    cs.fade_subscene(MICRO, "gs", 0);
    cs.render_microblock();
    cs.remove_subscene("gs");

    stage_macroblock(CompositeBlock(SilenceBlock(1), FileBlock("Here's one of them.")));
    obs->manager.set("v0.x", to_string(obs->manager.get_local_value("v4.x")));
    obs->manager.set("v0.y", to_string(obs->manager.get_local_value("v4.y")));
    obs->manager.remove(unordered_set<string>{"v4.x", "v4.y"});
    // Penrose kite has angles 72, 72, 72, 144 degrees.
    StateSet penrose_kite({{"v0.x", "-2"           }, {"v0.y", "0"},
                           {"v1.x", "1.2360679776" }, {"v1.y", "-2.3511410092"},
                           {"v2.x", "2"            }, {"v2.y", "0"},
                           {"v3.x", "1.2360679776" }, {"v3.y", "2.3511410092"}});
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.transition(MICRO, "singularity_depth_log", "200 log");
    cs.render_microblock();
    obs->manager.remove(unordered_set<string>{"v4.x", "v4.y"});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(16));
    obs->manager.transition(MACRO, penrose_kite);
    obs->manager.transition(MACRO, {{"ball0_start_x", "2.4721359550"}, {"ball0_start_y", "-2.3511410092"}});
    obs->manager.transition(MICRO, "path_length", "<path_length_log> exp 12 +");
    obs->manager.set("path_length_log", "-2");
    obs->manager.set("path_opacity_log", "1 log");
    obs->manager.transition(MACRO, "path_opacity", "<path_opacity_log> exp");
    obs->manager.transition(MACRO, "path_length_log", "20000 log");
    obs->manager.set("zoom_follow", "-2.5");
    obs->manager.transition(MACRO, "zoom", "<zoom_follow>");
    obs->manager.transition(MACRO, "zoom_follow", "-5");
    obs->manager.transition(MACRO, "path_opacity_log", ".02 log");
    obs->manager.transition(MICRO, "ball_opacity", "0");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("It was first discovered by Richard Schwartz in 2007."), SilenceBlock(1)));
    string title = "Schwartz, R. (2007) \\\\\\\\ \\tiny{Unbounded Orbits for Outer Billiards.}";
    shared_ptr<WhitePaperScene> wps = make_shared<WhitePaperScene>("schwartz2007", title, vector<int>{1, 2, 3, 39});
    cs.add_scene_fade_in(MICRO, wps, "wps");
    wps->manager.set("which_page", "39");
    wps->manager.set("page_focus", "1");
    wps->manager.set("completion", "1");
    wps->manager.set({
        {"crop_top", ".41"},
        {"crop_bottom", ".79"},
        {"crop_left", ".15"},
        {"crop_right", ".83"},
    });
    cs.render_microblock();
    cs.render_microblock();
    wps->manager.transition(MICRO, {
        {"crop_top", "0"},
        {"crop_bottom", "1"},
        {"crop_left", "0"},
        {"crop_right", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    wps->manager.transition(MICRO, "page_focus", "0");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.5));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    wps->manager.transition(MICRO, "completion", "0");
    cs.render_microblock();
    cs.remove_subscene("wps");

    stage_macroblock(SilenceBlock(6));
    obs->manager.transition(MACRO, "path_length_log", "-2");
    obs->manager.transition(MACRO, "path_length", "<path_length_log> exp -2 exp -");
    obs->manager.transition(MACRO, "zoom", "-1");
    obs->manager.transition(MACRO, {{"path_opacity_log", "1 log"}, {"ball_opacity", "0"}});
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    obs->manager.transition(MICRO, "singularity_opacity", "1");
    cs.render_microblock();
    obs->manager.set("singularity_depth", "200");
    obs->manager.remove(unordered_set<string>{"path_length_log", "path_opacity_log", "singularity_depth_log"});

    stage_macroblock(SilenceBlock(.3));
    obs->manager.set("path_length", "0");
    obs->manager.set("path_opacity", "1");
    cs.render_microblock();

    stage_macroblock(FileBlock("This shape is the Penrose Kite, which is used in aperiodic planar tilings."));
    // Set island and singularity opacity to 0 so we can see the tiling better.
    undo = obs->manager.transition(MICRO, {{"island_opacity", "0"}, {"singularity_opacity", "0"}});
    cs.render_microblock();
    shared_ptr<PngScene> ps = make_shared<PngScene>("penrose");
    cs.add_scene_fade_in(MICRO, ps, "ps");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("But my favorite tables are the regular polygons."));
    obs->manager.set("flow_depth", "68");
    obs->manager.set("singularity_depth", "68");
    cs.fade_subscene(MICRO, "ps", 0);
    obs->manager.transition(MICRO, undo);
    cs.render_microblock();
    cs.remove_subscene("ps");

    stage_macroblock(SilenceBlock(1));
    obs->manager.begin_timer("rotation_timer");
    obs->manager.set("rotation", "<rotation_timer> .04 *");
    obs->manager.transition(MICRO, regular_ngon(4, 2.0, 3.1415 * 1.25, true));
    cs.render_microblock();

    obs->manager.set(regular_ngon(8, 2.0, 3.1415 * 1.25, true));
    // Set odd index vertices to the midpoint of their neighbors, making a square.
    for(int i = 1; i < 8; i+=2) {
        string s_i = to_string(i);
        string s_ip1 = to_string((i+1)%8);
        string s_im1 = to_string(i-1);
        obs->manager.set({{"v" + s_i + ".x", "<v" + s_ip1 + ".x> <v" + s_im1 + ".x> + 2 /"},
                          {"v" + s_i + ".y", "<v" + s_ip1 + ".y> <v" + s_im1 + ".y> + 2 /"}});
    }
    // Transition to a regular octagon
    stage_macroblock(SilenceBlock(8));
    obs->manager.transition(MICRO, regular_ngon(8, 2.0, 3.1415 * 1.25, true));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    // Transition to a 12-gon
    stage_macroblock(SilenceBlock(8));
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
            string xs = to_string(2.0 * cos(theta));
            string ys = to_string(2.0 * sin(theta));
            hack_12gon["v" + to_string(i) + ".x"] = xs + " <rotation> cos * " + ys + " <rotation> sin * -";
            hack_12gon["v" + to_string(i) + ".y"] = xs + " <rotation> sin * " + ys + " <rotation> cos * +";
        }
        if(i%3 == 2) {
            theta += 3.141592653 / 4.0;
        } else {
            theta += 3.141592653 / 8.0;
        }
    }
    obs->manager.set(hack_12gon);
    obs->manager.transition(MICRO, regular_ngon(12, 2.0, 3.1415 * 1.25, true));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(8));
    // Set alternate vertices to the midpoint of their neighbors, making a hexagon.
    for(int i = 1; i < 12; i+=2) {
        string s_i = to_string(i);
        string s_ip1 = to_string((i+1)%12);
        string s_im1 = to_string(i-1);
        obs->manager.transition(MICRO, {{"v" + s_i + ".x", "<v" + s_ip1 + ".x> <v" + s_im1 + ".x> + 2 /"},
                                        {"v" + s_i + ".y", "<v" + s_ip1 + ".y> <v" + s_im1 + ".y> + 2 /"}});
    }
    cs.render_microblock();
    obs->manager.set(regular_ngon(6, 2.0, 3.1415 * 1.25, true));
    obs->manager.remove(unordered_set<string>{"v6.x", "v6.y", "v7.x", "v7.y", "v8.x", "v8.y", "v9.x", "v9.y", "v10.x", "v10.y", "v11.x", "v11.y"});

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(8));
    // Set alternate vertices to the midpoint of their neighbors, making a triangle.
    for(int i = 1; i < 6; i+=2) {
        string s_i = to_string(i);
        string s_ip1 = to_string((i+1)%6);
        string s_im1 = to_string(i-1);
        obs->manager.transition(MICRO, {{"v" + s_i + ".x", "<v" + s_ip1 + ".x> <v" + s_im1 + ".x> + 2 /"},
                                        {"v" + s_i + ".y", "<v" + s_ip1 + ".y> <v" + s_im1 + ".y> + 2 /"}});
    }
    cs.render_microblock();
    obs->manager.set(regular_ngon(3, 2.0, 3.1415 * 1.25, true));
    obs->manager.remove(unordered_set<string>{"v3.x", "v3.y", "v4.x", "v4.y", "v5.x", "v5.y"});

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(8));
    obs->manager.set(regular_ngon(9, 2.0, 3.1415 * 1.25, true));
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
    obs->manager.transition(MICRO, regular_ngon(9, 2.0, 3.1415 * 1.25, true));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(8));
    obs->manager.transition(MICRO, regular_ngon(7, 2.0, 3.1415 * 1.25, true));
    // Transition vertices 7 and 8 to be between 6 and 0.
    obs->manager.transition(MICRO, {{"v7.x", "<v6.x> 2 * <v0.x> + 3 /"}, {"v7.y", "<v6.y> 2 * <v0.y> + 3 /"},
                                    {"v8.x", "<v6.x> <v0.x> 2 * + 3 /"}, {"v8.y", "<v6.y> <v0.y> 2 * + 3 /"}});
    cs.render_microblock();
    // Remove vertices 7 and 8
    obs->manager.remove(unordered_set<string>{"v7.x", "v7.y", "v8.x", "v8.y"});

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    obs->manager.transition(MICRO, {{"singularity_opacity", "0"}, {"island_opacity", "0"}, {"path_opacity", "0"}});
    cs.render_microblock();

    /*
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
    */

    stage_macroblock(FileBlock("We've been plotting where a ball would land based on its starting position,"));
    obs->manager.set({{"ball0_start_x", "4"}, {"ball0_start_y", "3"}});
    obs->manager.transition(MICRO, {{"ball_opacity", "1"}, {"path_opacity", "1"}, {"island_opacity", "0"}, {"singularity_opacity", "0"}});
    cs.render_microblock();
    obs->manager.transition(MICRO, "path_length", "10");
    obs->manager.set("ball_distance", "0");
    cs.render_microblock();

    stage_macroblock(FileBlock("and that path depends on the table."));
    obs->manager.transition(MACRO, {{"v4.x", "<v3.x> 3 * <v0.x> 1 * + 4 /"}, {"v4.y", "<v3.y> 3 * <v0.y> 1 * + 4 /"},
                                    {"v5.x", "<v3.x> 2 * <v0.x> 2 * + 4 /"}, {"v5.y", "<v3.y> 2 * <v0.y> 2 * + 4 /"},
                                    {"v6.x", "<v3.x> 1 * <v0.x> 3 * + 4 /"}, {"v6.y", "<v3.y> 1 * <v0.y> 3 * + 4 /"}});
    cs.render_microblock();
    obs->manager.transition(MICRO, regular_ngon(4, 1.0, 3.1415 * 1.25));
    // Transition extra vertices to the midpoint of their neighbors, making a square from a heptagon.
    cs.render_microblock();
    obs->manager.remove(unordered_set<string>{"v5.x", "v5.y", "v6.x", "v6.y"});

    stage_macroblock(FileBlock("So let's fix the ball,"));
    cs.render_microblock();
    obs->manager.transition(MICRO, "ball_opacity", "0");
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("but plot where it would go depending on where we place some additional table corner."), SilenceBlock(2)));
    obs->manager.begin_timer("spin1");
    obs->manager.transition(MICRO, {{"v4.x", "<spin1> sin 2 *"}, {"v4.y", "<spin1> cos 2 *"}});
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(2));
    shared_ptr<OuterBilliardsVertexFlowScene> obvfs = make_shared<OuterBilliardsVertexFlowScene>();
    cs.add_scene_fade_in(MICRO, obvfs, "obvfs");
    obvfs->manager.set(regular_ngon(4, 1.0, 3.1415 * 1.25));
    obvfs->manager.set({{"ball_start_x", "4"}, {"ball_start_y", "3"}});
    obvfs->manager.set("zoom", "-1");
    obvfs->manager.set("flow_depth", "160");
    cs.render_microblock();

    // Move the point around
    stage_macroblock(SilenceBlock(13));
    cs.remove_subscene("obs");
    obvfs->manager.begin_timer("spinny");
    obvfs->manager.transition(MICRO, {{"ball_start_x", "<spinny> sin 5 *"}, {"ball_start_y", "<spinny> cos 5 *"}});
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    obvfs->manager.transition(MICRO, {{"ball_start_x", "<spinny> sin"}, {"ball_start_y", "<spinny> cos"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3));
    obvfs->manager.transition(MICRO, "spinny", "17.5");
    cs.render_microblock();

    stage_macroblock(FileBlock("A section of this graph is always black,"));
    cs.render_microblock();
    cs.render_microblock();
    obvfs->manager.transition(MICRO, "black_stripes", "1");
    cs.render_microblock();
    obvfs->manager.transition(MICRO, "black_stripes", "0");
    cs.render_microblock();

    stage_macroblock(FileBlock("because adding an extra vertex there would land the ball inside the table."));
    shared_ptr<OuterBilliardsScene> obscure = make_shared<OuterBilliardsScene>();
    obscure->manager.set("zoom", "-1");
    obscure->manager.set("ball_opacity", "0");
    cs.add_scene_fade_in(MICRO, obscure, "obscure");
    obscure->manager.set(regular_ngon(4, 1.0, 3.1415 * 1.25));
    obscure->manager.set({{"v4.x", "{t} .7 * sin 13 ^ 2 * 5 -"}, {"v4.y", "{t} cos 13 ^ 2 *"}});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.fade_subscene(MICRO, "obscure", 0);
    cs.render_microblock();
    cs.remove_subscene("obscure");

    stage_macroblock(CompositeBlock(SilenceBlock(.4), FileBlock("Note the uniformly colored ring surrounding the table.")));
    obvfs->manager.transition(MICRO, "zoom", "0");
    obvfs->manager.transition(MICRO, "spinny", "11");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.5));
    cs.render_microblock();

    stage_macroblock(FileBlock("Its outline is the ball's natural path."));
    shared_ptr<OuterBilliardsScene> obs_ring = make_shared<OuterBilliardsScene>();
    cs.add_scene(obs_ring, "obs_ring");
    obs_ring->manager.set(regular_ngon(4, 1.0, 3.1415 * 1.25));
    obs_ring->manager.set({{"ball0_start_x", "11 sin"}, {"ball0_start_y", "11 cos"}, {"ball_opacity", "1"}, {"path_opacity", "1"}});
    obs_ring->manager.transition(MACRO, "path_length", "4");
    cs.fade_subscene(MICRO, "obvfs", .3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.fade_subscene(MICRO, "obvfs", 1);
    cs.render_microblock();

    stage_macroblock(FileBlock("Placing a new vertex on the inside won't interrupt that path, so we get a ring of uniform color."));
    obvfs->manager.transition(MICRO, "flow_opacity", ".3");
    obvfs->manager.set({{"v4.x", "-.707106"}, {"v4.y", ".707106"}});
    obvfs->manager.set({{"v3.x", "0"}, {"v3.y", ".707106"}});
    obvfs->manager.transition(MICRO, {{"v3.x", "-.2 {t} 1.1 * sin .3 * +"}, {"v3.y", ".95 {t} 1.3 * sin .08 * +"}});
    obs_ring->manager.transition(MICRO, "path_opacity", "0");
    cs.render_microblock();
    obs_ring->manager.set({{"path_length", "0"}, {"path_opacity", "1"}});
    obs_ring->manager.transition(MICRO, "path_length", "4");
    cs.render_microblock();
    obvfs->manager.transition(MICRO, "flow_opacity", "1");
    cs.fade_subscene(MICRO, "obs_ring", 0);
    cs.render_microblock();
    cs.remove_subscene("obs_ring");

    stage_macroblock(SilenceBlock(4));
    obvfs->manager.transition(MICRO, {{"v3.x", "0"}, {"v3.y", ".707106"}});
    obvfs->manager.transition(MACRO, {{"zoom", "-1"}, {"ball_start_x", "11 sin 2.8284 *"}, {"ball_start_y", "11 cos 3.60555 *"}});
    cs.render_microblock();
    obvfs->manager.transition(MICRO, {{"v3.x", "0"}, {"v3.y", ".707106"}});
    cs.render_microblock();
    obvfs->manager.set({{"v3.x", "-.707106"}, {"v3.y", ".707106"}});
    obvfs->manager.remove(unordered_set<string>{"v4.x", "v4.y"});

    stage_macroblock(FileBlock("With the ball slightly further away, there are more lines in the path to bound the ring."));
    cs.add_scene(obs_ring, "obs_ring");
    obs_ring->manager.set({{"zoom", "-1"}, {"ball0_start_x", "11 sin 2.8284 *"}, {"ball0_start_y", "11 cos 3.60555 *"}, {"path_length", "0"}, {"path_opacity", "1"}});
    cs.render_microblock();
    obs_ring->manager.transition(MICRO, "path_length", "8");
    cs.render_microblock();
    cs.fade_subscene(MICRO, "obs_ring", 0);
    cs.render_microblock();
    cs.remove_subscene("obs_ring");

    stage_macroblock(SilenceBlock(2));
    obvfs->manager.transition(MACRO, "zoom", "-1");
    obvfs->manager.transition(MACRO, {{"ball_start_x", "11 sin 3.60555 *"}, {"ball_start_y", "11 cos 3.60555 *"}});
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(6));
    obvfs->manager.set({{"v4.x", "-.707106"}, {"v4.y", ".707106"}});
    obvfs->manager.set({{"v3.x", "0"}, {"v3.y", ".707106"}});
    obvfs->manager.set({{"v5.x", "-.707106"}, {"v5.y", "0"}});
    obvfs->manager.transition(MACRO, regular_ngon(6, 1.0, 3.1415 * 1.25));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(10));
    obvfs->manager.set({{"ball_start_x", "<spin_transition> sin 3.60555 *"}, {"ball_start_y", "<spin_transition> cos 3.60555 *"}, {"spin_transition", "11"}});
    obvfs->manager.transition(MICRO, "spin_transition", "14");
    cs.render_microblock();

    stage_macroblock(FileBlock("If hexagons tile the euclidean plane,"));
    cs.add_scene(obs, "obs");
    obs->manager.set({{"flow_depth", "50"}, {"singularity_depth", "<flow_depth>"}});
    obs->manager.set({{"ball_opacity", "0"}, {"path_opacity", "0"}});
    obs->manager.set(regular_ngon(6, 1.0, 3.1415 * 1.25));
    obs->manager.transition(MICRO, {{"island_opacity", "1"}, {"singularity_opacity", "1"}});
    cs.render_microblock();
    cs.remove_subscene("obvfs");

    stage_macroblock(FileBlock("then what if we use pentagons..."));
    // Transition to a pentagon, occluding vertex 5 in between vertices 4 and 0.
    obs->manager.transition(MICRO, regular_ngon(5, 1.0, 3.1415 * 1.25));
    obs->manager.transition(MICRO, {{"v5.x", "<v4.x> <v0.x> + 2 /"}, {"v5.y", "<v4.y> <v0.y> + 2 /"}});
    cs.render_microblock();
    obs->manager.remove(unordered_set<string>{"v5.x", "v5.y"});

    stage_macroblock(CompositeBlock(FileBlock("...to tile the hyperbolic plane?"), SilenceBlock(4)));
    obs->manager.transition(MICRO, "curvature", "-0.472135955"); // Pentagonal tiling of hyperbolic space
    obs->manager.transition(MICRO, "zoom", ".3");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(.6));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(4));
    obs->manager.transition(MACRO, regular_ngon(5, 3.0, 3.1415 * 1.25));
    obs->manager.set("curvature", "<curvature_log> exp -1 *");
    obs->manager.set("curvature_log", "0.472135955 log");
    obs->manager.transition(MACRO, "curvature_log", "0.01 log");
    obs->manager.transition(MACRO, "zoom", "-1.5");
    obs->manager.transition(MICRO, "island_opacity", "0");
    cs.render_microblock();
    obs->manager.set("flow_depth", "1600");
    obs->manager.transition(MICRO, "island_opacity", "1");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(20));
    StateSet hack_3gon = regular_ngon(3, 3.0, 3.1415 * 1.25);
    hack_3gon["v4.x"] = hack_3gon["v2.x"];
    hack_3gon["v4.y"] = hack_3gon["v2.y"];
    // Now remove v2
    hack_3gon.erase("v2.x");
    hack_3gon.erase("v2.y");
    obs->manager.transition(MACRO, hack_3gon);
    obs->manager.transition(MACRO, {{"v2.x", "<v1.x>"}, {"v2.y", "<v1.y>"}, {"v3.x", "<v4.x>"}, {"v3.y", "<v4.y>"}});
    obs->manager.transition(MICRO, "singularity_opacity", "0");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("Other than polygonal tilings,", SilenceBlock(5)));
    obs->manager.set(regular_ngon(3, 3.0, 3.1415 * 1.25));
    obs->manager.remove(unordered_set<string>{"v3.x", "v3.y", "v4.x", "v4.y"});
    obs->manager.transition(MICRO, "singularity_opacity", ".4");
    obs->manager.transition(MICRO, "curvature", "-0.01348004"); // Trioctagonal tiling (3,8,3,8) with circumradius 3: curvature = -(3*sqrt(2)-4)/2/9 ~= -0.01348004
    cs.render_microblock();

    /*
    stage_macroblock(SilenceBlock(5));
    obs->manager.transition(MICRO, "curvature", "-0.00845160165"); // Triheptagonal tiling (3,7,3,7), triangle circumradius 3: -(2cos(2pi/7)-1)/(2+2cos(2pi/7))/9
    cs.render_microblock();
    */

    stage_macroblock(SilenceBlock(.5));
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("hyperbolic outer billiards are still largely shrouded in mystery."), SilenceBlock(20)));
    obs->manager.set({{"v3.x", "<v2.x>"}, {"v3.y", "<v2.y>"}});
    StateSet wobbly_square = regular_ngon(4, 3.0, 3.1415 * 1.25);
    /*
    for(int i = 0; i < 4; i++) {
        string s_i = to_string(i);
        string speed1 = to_string(.1 * (2.  + 0.3 * i));
        string speed2 = to_string(.1 * (1.6 + 0.4 * i));
        wobbly_square["v" + s_i + ".x"] += " {t} " + speed1 + " * cos .6 * +";
        wobbly_square["v" + s_i + ".y"] += " {t} " + speed2 + " * sin .6 * +";
    }
    */
    obs->manager.transition(MICRO, wobbly_square);
    cs.render_microblock();

    /*
    stage_macroblock(SilenceBlock(6));
    obs->manager.transition(MACRO, regular_ngon(4, 3.0, 3.1415 * 1.25));
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    */

    stage_macroblock(SilenceBlock(4));
    obs->manager.transition(MACRO, "curvature", "-.05");
    obs->manager.transition(MACRO, "zoom", "-2.5");
    cs.manager.transition(MACRO, "obs.x", ".25");
    obs->manager.transition(MACRO, "w", "2");
    obs->manager.transition(MACRO, "singularity_opacity", "1");
    obs->manager.transition(MICRO, regular_ngon(4, 0.005, 3.1415 * 1.25));
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("This has been 2swap,"));
    shared_ptr<LatexScene> ls = make_shared<LatexScene>("\\text{2swap}", vec2(.7, .7));
    shared_ptr<OuterBilliardsScene> cool_pattern = make_shared<OuterBilliardsScene>();
    cool_pattern->manager.set(regular_ngon(5, 2.0, 3.1415 * 1.25));
    cool_pattern->manager.set({{"flow_depth", "<singularity_depth> 1 -"}, {"singularity_depth", "1000"}, {"ball_opacity", "0"}, {"path_opacity", "0"}, {"zoom", "-.05"}, {"island_opacity", "1"}, {"singularity_opacity", "1"}, {"center_x", "-.72"}, {"center_y", "14.35"}, {"periodicity_or_flow", "1"}});
    cs.add_scene_fade_in(MACRO, cool_pattern, "cool_pattern", vec2(.5, .5), 0.2);
    cs.add_scene(ls, "ls", vec2(.6, .55));
    cs.manager.set("ls.opacity", "0");
    cs.manager.set("quadratic", "0");
    cs.manager.transition(MACRO, "quadratic", ".8");
    cs.manager.transition(MACRO, "ls.opacity", "<quadratic>");
    cs.render_microblock();

    stage_macroblock(FileBlock("with music by 6884."));
    shared_ptr<LatexScene> ls2 = make_shared<LatexScene>("\\text{Music by 6884}", vec2(.4, .4));
    cs.add_scene_fade_in(MICRO, ls2, "ls2", vec2(.55, .7));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    // Start to move cool pattern around with equal radius
    cool_pattern->manager.begin_timer("spin");
    cool_pattern->manager.set({{"center_x", "<spin> .02 * sin 14.5 *"}, {"center_y", "<spin> .02 * cos 14.5 *"}});
    cs.render_microblock();

    stage_macroblock(FileBlock("This video was made possible by David J. Romano, a Chemistry PhD candidate at Brown University."));
    cs.fade_all_subscenes_except(MICRO, "cool_pattern", 0);
    cs.fade_subscene(MICRO, "cool_pattern", 1);
    cs.render_microblock();
    cs.remove_all_subscenes_except("cool_pattern");
    shared_ptr<LatexScene> ls3 = make_shared<LatexScene>("\\text{What should I show here?}");
    cs.add_scene_fade_in(MICRO, ls3, "ls3");
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("He introduced me to this problem, and wrote much of the simulation code I used here."));
    ls3->begin_latex_transition(MICRO, "\\text{What about here?}");
    cs.render_microblock();

    stage_macroblock(FileBlock("A further thanks to Professor Richard Schwartz and Dr. Lael Edwards-Costa, Billiards researchers who generously reviewed this video!"));
    cs.render_microblock();

    stage_macroblock(FileBlock("Their work is linked in the description."));
    cs.render_microblock();
    return;
}
