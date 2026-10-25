
#include "../Scenes/Math/FourDAlgebraScene.h"
#include "../Scenes/Math/FourDPlaneScene.h"
#include "../Scenes/Math/ThreeDAlgebraScene.h"
#include "../Scenes/Math/TwoDAlgebraScene.h"
#include "../Scenes/Math/MultiplicationTableScene.h"

#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Media/CursorScene.h"
#include "../Scenes/Common/ThreeDimensionScene.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Common/BackgroundScene.h"

#include "../Core/State/BezierStateCurve.h"
#include "../Core/State/StateTester.h"
#include "../Core/Smoketest.h"
#include "../IO/Writer.h"


// spinny:   0  8 80
// squishy:  0 48 48
// flippy:  48  0 60


// TODO gauss filter


void set_global_draggers(float x_drag, float y_drag){
    set_global_state("dragger_x_1", x_drag);
    set_global_state("dragger_y_1", y_drag);
    set_global_state("dragger_x_2", x_drag);
    set_global_state("dragger_y_2", y_drag);
    set_global_state("dragger_x_3", x_drag);
    set_global_state("dragger_y_3", y_drag);
    set_global_state("dragger_x_4", x_drag);
    set_global_state("dragger_y_4", y_drag);
    set_global_state("dragger_x_5", x_drag);
    set_global_state("dragger_y_5", y_drag);
    set_global_state("dragger_x_6", x_drag);
    set_global_state("dragger_y_6", y_drag);
}

void set_global_number_line(float number_line_value){
    set_global_state("number_line_1", number_line_value);
    set_global_state("number_line_2", number_line_value);
    set_global_state("number_line_3", number_line_value);
    set_global_state("number_line_4", number_line_value);
    set_global_state("number_line_5", number_line_value);
    set_global_state("number_line_6", number_line_value);
}

void four_d_intro(CompositeScene& cs, StateSet& bg){
    shared_ptr<FourDAlgebraScene> fd = make_shared<FourDAlgebraScene>();
    cs.add_scene(fd, "fd");

    cs.manager.set({{"slider","-0.1"}});

    fd->manager.set(bg);
    fd->manager.set({
        {"brightness", "160.0"},
        {"offset_1", "0.33"},
        {"offset_2", "0.093"},
        {"scale", "10.0"},
        {"equation", "1"},
        {"pov_xz", "0.1"},
        {"pov_y", "0"},
        {"rotation_jk", "-0.83"},
        {"rotation_ik", "-0.7"},
        {"rotation_1k", "0.62"},
        {"slider_width", "0.006"},
    });

    stage_macroblock(CompositeBlock(FileBlock("You're looking at the four-dimensional sine function."), SilenceBlock(1.5)));
    vector<StateSet> waypoints;
    waypoints.push_back({
        {"rotation_jk", "-0.83"},
        {"rotation_ik", "-0.7"},
        {"rotation_1k", "0.62"},
    });
    waypoints.push_back({
        {"rotation_1k", "0.62"},
        {"rotation_ik", "-0.7"},
        {"rotation_jk", "-0.1"}
    });
    waypoints.push_back({
        {"rotation_1k", "0.62"},
        {"rotation_ik", "-1.2"},
        {"rotation_jk", "-0.1"}
    });
    waypoints.push_back({
        {"rotation_1k", "0.0"},
        {"rotation_ik", "-1.2"},
        {"rotation_jk", "-0.1"}
    });
    waypoints.push_back({
        {"rotation_1k", "0.0"},
        {"rotation_ik", "-1.2"},
        {"rotation_jk", "0.73"}
    });
    waypoints.push_back({
        {"rotation_1k", "0.0"},
        {"rotation_ik", "-1.2"},
        {"rotation_jk", "0.73"}
    });
    BezierStateCurve bsc(waypoints);
    bsc.run_curve(*fd,cs);

    stage_macroblock(CompositeBlock(FileBlock("And now you're looking at... the four-dimensional sine function."), SilenceBlock(1)));

    
    fd->manager.transition(MICRO, {
        {"brightness", "80.0"},
    });
    cs.manager.transition(MICRO, {
        {"slider", "1.1"},
    });
    cs.render_microblock();
    fd->manager.transition(MICRO, {
        {"brightness", "2.0"},
        {"pov_xz", "0.78"},
    });
    cs.render_microblock();
    fd->manager.transition(MICRO, {
        {"brightness", "60.0"},
        {"pov_xz", "1.57"},
        {"rotation_jk", "0.0"}
    });
    cs.render_microblock();



    stage_macroblock(CompositeBlock(FileBlock("I claim they're the same, but side-by-side they're clearly different."), SilenceBlock(1)));

    waypoints.clear();
    waypoints.push_back({
        {"rotation_1k", "0.0"},
        {"rotation_ik", "-1.2"},
        {"rotation_jk", "0.0"},
        {"offset_1", "0.33"},
    });
    waypoints.push_back({
        {"rotation_1k", "0.6"},
        {"rotation_ik", "-1.2"},
        {"rotation_jk", "0.0"},
        {"offset_1", "0.172"},
    });
    waypoints.push_back({
        {"rotation_1k", "0.6"},
        {"rotation_ik", "-1.2"},
        {"rotation_jk", "-0.8"},
        {"offset_1", "0.172"},
    });
    waypoints.push_back({
        {"rotation_1k", "0"},
        {"rotation_ik", "1.5"},
        {"rotation_jk", "-0.8"},
        {"offset_1", "0.172"},
    });

    BezierStateCurve bsc2(waypoints);
    vector<StateSet> _waypoints = bsc2.get_waypoints();

    fd->manager.set(_waypoints[0]);
    fd->manager.transition(MICRO, bsc2.pop_next_state_set());
    for(int i = 0; i < _waypoints.size()-2; i++) {
        if (i == 2)
            cs.manager.transition(MICRO, { {"slider", "0.5"}, });
        cs.render_microblock();
        fd->manager.set(bsc2.pop_next_state_set());
    }
    fd->manager.transition(MICRO, _waypoints[_waypoints.size()-1]);
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("They're calculated within different algebras."),SilenceBlock(0.6)));

    fd->manager.set({
        {"expy", "10 {microblock_fraction} .5 - 10 * ^"},
        {"lerpy", "<expy> 1 <expy> + / ^2"},
    });
    fd->manager.transition(MICRO,{
        {"brightness", "10.0"},
        {"scale", "12.0"},
        {"equation", "1 <lerpy> +"},
    });
    cs.render_microblock();
    fd->manager.set({
        {"equation", "2"},
    });
    fd->manager.transition(MICRO,{
        {"pov_xz", "0"},
    });
    cs.render_microblock();



    stage_macroblock(FileBlock("This graph uses quaternions,"));
    cs.manager.transition(MICRO, {{"slider", "1.1"},});

    fd->manager.transition(MICRO, {
        {"rotation_jk", "0.1"},
    });
    cs.render_microblock();
    fd->manager.transition(MICRO, {
        {"pov_xz", "-0.83"},
        {"pov_y", "0.1"},
    }); 
    cs.render_microblock();



    stage_macroblock(SilenceBlock(1.5));
    fd->manager.transition(MICRO, {
        {"rotation_ik", "0.6"},
        {"pov_y", "0.8"},
    }); 
    cs.render_microblock();



    stage_macroblock(CompositeBlock(FileBlock("while the other uses their evil twin."), SilenceBlock(0.6)));

    cs.manager.transition(MICRO, {
        {"slider", "-0.1"},
    });
    fd->manager.transition(MICRO, {
        {"rotation_ik", "0.93"},
        {"pov_y", "1.0"},
    }); 
    cs.render_microblock();
    fd->manager.transition(MICRO, {
        {"brightness", "30.0"},
        {"rotation_1k", "-0.78"},
        {"pov_xz", "-0.9"},
        {"pov_y", "0.5"},
    });
    cs.render_microblock();



    stage_macroblock(CompositeBlock(FileBlock("Each system is an algebra, a set of rules for how to add and multiply."), SilenceBlock(0.6)));

    fd->manager.transition(MICRO,{
        {"pov_xz", "-1.53"},
        {"pov_y", "-0.2"},
    });
    cs.manager.transition(MICRO, {
        {"slider", "0.5"},
    });
    cs.render_microblock();


    fd->manager.transition(MICRO, {
        {"brightness", "20.0"},
        {"equation", "3"},
        {"rotation_ik", "0"},
    });
    cs.render_microblock();

    fd->manager.transition(MICRO, {
        {"rotation_1k", "-1.2"},
        {"pov_y", "-0.7"},
    }); 
    cs.render_microblock();


    stage_macroblock(CompositeBlock(FileBlock("But these algebras multiply differently."), SilenceBlock(2)));

    fd->manager.transition(MICRO, {
        {"rotation_jk", "0.9"},
        {"rotation_1k", "-1.58"},
    }); 
    cs.render_microblock();
    fd->manager.transition(MICRO, {
        {"rotation_1k", "-2.3"},
        {"brightness", "2.0"},
        {"pov_xz", "-0.68"},
    }); 
    cs.render_microblock();
}


void one_dim_algebra(CompositeScene& cs){

    stage_macroblock(CompositeBlock(FileBlock("Let's start with 1-dimension."), SilenceBlock(0.5)));
    shared_ptr<TwoDAlgebraScene> td = make_shared<TwoDAlgebraScene>();
    td->manager.set({
        {"dragger_type", "0"},
        {"dragger_shown", "0"},
        {"number_line", "1"},
        {"dragger_x", "0"},
        {"dragger_y", "0"},
        {"xx_x", "1"},
        {"xx_y", "0"},
        {"xy_x", "0"},
        {"xy_y", "1"},
        {"yy_x", "0"},
        {"yy_y", "1"},
    });
    cs.render_microblock();
    cs.add_scene_fade_in(MICRO, td, "td", vec2(0.5,0.5), 1,false, 1);
    cs.manager.transition(MICRO,{{"bg_0_r", "0"},{"bg_0_g", "8"},{"bg_0_b", "80"},});
    cs.fade_subscene(MICRO,"fd",0);
    cs.render_microblock();
    cs.render_microblock();
    cs.remove_subscene("fd");
    cs.manager.set({{"slider", "1.1"},});


    stage_macroblock(CompositeBlock(FileBlock("Addition translates the number line."), SilenceBlock(2)));
    td->manager.set("dragger_type", "1");
    td->manager.transition(MICRO, {
        {"dragger_shown", "1"},
    });
    cs.render_microblock();

    shared_ptr<Scene> cursor = cs.get_subscene_pointer("cursor");
    cs.remove_subscene("cursor");
    cs.add_scene(cursor, "cursor");
    cursor->manager.transition(MICRO,{{"cursor_x", "{drag_x}"},{"cursor_y", "{drag_y}"},});
    cs.render_microblock();

    td->manager.transition(MICRO, {
        {"dragger_x", "1.5"},
    });
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"dragger_x", "-.5"},
    });
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"dragger_x", "0"},
    });
    cs.render_microblock();

    cursor->manager.set({{"cursor_x", "0.5"},{"cursor_y", "0.5"},});

    stage_macroblock(SilenceBlock(1));

    td->manager.transition(MICRO, {
        {"dragger_shown", "0"}
    });
    cursor->manager.transition(MICRO,{{"cursor_x", ".55"},{"cursor_y", ".6"},});
    cs.render_microblock();
    td->manager.set({
        {"dragger_type", "2"},
        {"dragger_x", "1"},
    });
    set_global_draggers(1,0);


    stage_macroblock(CompositeBlock(FileBlock("Multiplication, on the other hand, scales it."), SilenceBlock(2)));
    td->manager.transition(MICRO, {
        {"dragger_shown", "1"},
    });
    cs.render_microblock();

    cursor->manager.transition(MICRO,{{"cursor_x", "{drag_x}"},{"cursor_y", "{drag_y}"},});
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"dragger_x", "0.5"},
    });
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"dragger_x", "2"},
    });
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"dragger_x", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();


    stage_macroblock(CompositeBlock(FileBlock("This is a 1-dimensional algebra."), SilenceBlock(.8)));
    td->manager.transition(MICRO, {
        {"dragger_shown", "0"},
    });
    cursor->manager.transition(MICRO,{{"cursor_x", "0.57"},{"cursor_y", "0.53"},});
    cs.render_microblock();
    cs.render_microblock();

    td->manager.set({
        {"update_active", "0"},
    });



}


void two_dim_algebra(CompositeScene& cs, StateSet& bg){

    shared_ptr<TwoDAlgebraScene> td = make_shared<TwoDAlgebraScene>();
    shared_ptr<Scene> cursor = cs.get_subscene_pointer("cursor");

    set_global_draggers(0,0);
    set_global_number_line(0);
    td->manager.set(bg);
    td->manager.set({
        {"dragger_type", "1"},
        {"dragger_x", "0"},
        {"dragger_y", "0"},
        {"dragger_shown", "0"},
        {"xx_x", "1"},
        {"xx_y", "0"},
        {"xy_x", "0"},
        {"xy_y", "0"},
        {"yy_x", "0"},
        {"yy_y", "1"},
        
    });


    stage_macroblock(CompositeBlock(FileBlock("In 2-dimensions, we can add by translating each dimension independently."), SilenceBlock(2)));

    cs.add_scene_fade_in(MICRO, td, "td2", vec2(0.5,0.5), 1, false, 2);
    cs.move_to_front("cursor");

    cs.render_microblock();
    cs.remove_subscene("td");

    td->manager.transition(MICRO, {
        {"dragger_shown", "1"},
    });
    cs.render_microblock();

    cursor->manager.transition(MICRO,{{"cursor_x", "{drag_x}"},{"cursor_y", "{drag_y}"},});
    cs.render_microblock();

    td->manager.begin_timer("add_axes");
    td->manager.transition(MICRO, {
        {"dragger_x", "<add_axes> .5 * 3 + sin 2 *"},
        {"dragger_y", "<add_axes> .7 * cos 2 *"},
    });
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"dragger_x", "0"},
        {"dragger_y", "0"},
    });
    cs.render_microblock();



    td->manager.transition(MICRO, {{"dragger_shown", "0"},});
    cursor->manager.transition(MICRO, {{"cursor_x", "0.55"},{"cursor_y", "0.46"},});
    cs.render_microblock();
    


    stage_macroblock(CompositeBlock(FileBlock("For multiplication, we could scale independently, squishing the space."),SilenceBlock(3)));

    cs.manager.transition(MICRO, {{"bg_1_r", "0"},{"bg_1_g", "48"},{"bg_1_b", "48"},});
    td->manager.set({
        {"dragger_type", "2"},
        {"dragger_x", "1"},
        {"dragger_y", "1"},
    });
    set_global_draggers(1,1);

    td->manager.transition(MICRO, {
        {"dragger_shown", "1"},
    });

    cursor->manager.transition(MICRO,{{"cursor_x", "{drag_x}"},{"cursor_y", "{drag_y}"},});
    cs.render_microblock();

    td->manager.begin_timer("mult_axes");
    td->manager.transition(MICRO, {
        {"dragger_x", "<mult_axes> .7 * sin .8 * 1 +"},
        {"dragger_y", "<mult_axes> 1. * cos .8 * 1 +"},
    });
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"dragger_x", "1"},
        {"dragger_y", "1"},
    });
    cs.render_microblock();



    td->manager.transition(MICRO, {
        {"xx_x", "1"},
        {"xx_y", "0"},
        {"xy_x", "0"},
        {"xy_y", "1"},
        {"yy_x", "-1"},
        {"yy_y", "0"},
    });


    stage_macroblock(FileBlock("Alternatively, we can scale uniformly and spin."));
    
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "8"},{"bg_1_b", "80"},});
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"dragger_x", "1.6"},
        {"dragger_y", "1.6"},
    });
    cs.render_microblock();
    
    td->manager.transition(MICRO, {
        {"dragger_y", "1.6"},
        {"dragger_x", "0"},
    });
    cs.render_microblock();


    stage_macroblock(SilenceBlock(0.6));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    td->manager.transition(MICRO, {{"dragger_shown", "0"},});
    cursor->manager.transition(MICRO, {{"cursor_x", "0.45"},{"cursor_y", "-0.1"},});
    cs.render_microblock();
    



    td->manager.set({{"point_size", "1.6"},});
    stage_macroblock(FileBlock("Here, a point is some amount of x plus some amount of y."));
    td->manager.transition(MICRO, {
        {"brightness", "32"},
        {"point_count", "1"},
    });

    cs.render_microblock();



	shared_ptr<LatexScene> latex_x = make_shared<LatexScene>("x",vec2(0.181,0.432));
	cs.add_scene_fade_in(MICRO,latex_x , "latex_x",vec2(0.5,0.35));
	cs.slide_subscene(MICRO,"latex_x",vec2(-0.075,0));

    td->manager.transition(MICRO, {
        {"point_x", "-0.7"},
    });
    cs.render_microblock();


	
	shared_ptr<LatexScene> latex__1 = make_shared<LatexScene>("+",vec2(0.2,0.3));
	cs.add_scene_fade_in(MICRO,latex__1 , "latex__1",vec2(0.63,0.35));
	shared_ptr<LatexScene> latex_y = make_shared<LatexScene>("y",vec2(0.168,0.432));
	cs.add_scene_fade_in(MICRO,latex_y , "latex_y",vec2(0.74,0.37));

	cs.slide_subscene(MICRO,"latex_x",vec2(-0.035,0));
	cs.slide_subscene(MICRO,"latex__1",vec2(-0.13,0));
	cs.slide_subscene(MICRO,"latex_y",vec2(-0.13,0));

    td->manager.transition(MICRO, {
        {"point_y", "-1.2"},
    });
    cs.render_microblock();



    stage_macroblock(SilenceBlock(1.2));
    cs.render_microblock();



    stage_macroblock(FileBlock("When we multiply two points,"));

    td->manager.transition(MICRO, {{"point_opacity", "0"},});

	shared_ptr<LatexScene> latex_x2 = make_shared<LatexScene>("x",vec2(0.181,0.432));
	cs.add_scene(latex_x2 , "latex_x2",vec2(0.39,0.35));
	shared_ptr<LatexScene> latex__2 = make_shared<LatexScene>("+",vec2(0.2,0.3));
	cs.add_scene(latex__2 , "latex__2",vec2(0.5,0.35));
	shared_ptr<LatexScene> latex_y2 = make_shared<LatexScene>("y",vec2(0.168,0.412));
	cs.add_scene(latex_y2 , "latex_y2",vec2(0.61,0.37));
	
	latex_x->manager.transition(MICRO,{{"w","0.143"},{"h","0.346"}});
	latex_x->begin_latex_transition(MICRO, "x");
	cs.slide_subscene(MICRO,"latex_x",vec2(-0.11,0));
	latex__1->manager.transition(MICRO,{{"w","0.16"},{"h","0.24"}});
	latex__1->begin_latex_transition(MICRO, "+");
	cs.slide_subscene(MICRO,"latex__1",vec2(-0.14,0));
	latex_y->manager.transition(MICRO,{{"w","0.13"},{"h","0.346"}});
	latex_y->begin_latex_transition(MICRO, "y");
	cs.slide_subscene(MICRO,"latex_y",vec2(-0.17,-0.005));
	latex_x2->manager.transition(MICRO,{{"w","0.143"},{"h","0.346"}});
	latex_x2->begin_latex_transition(MICRO, "x");
	cs.slide_subscene(MICRO,"latex_x2",vec2(0.17,0));
	latex__2->manager.transition(MICRO,{{"w","0.16"},{"h","0.24"}});
	latex__2->begin_latex_transition(MICRO, "+");
	cs.slide_subscene(MICRO,"latex__2",vec2(0.14,0));
	latex_y2->manager.transition(MICRO,{{"w","0.13"},{"h","0.346"}});
	latex_y2->begin_latex_transition(MICRO, "y");
	cs.slide_subscene(MICRO,"latex_y2",vec2(0.11,-0.005));
	
	shared_ptr<LatexScene> latex__5 = make_shared<LatexScene>("(",vec2(0.099,0.289));
	cs.add_scene_fade_in(MICRO, latex__5 , "latex__5",vec2(0.35,0.35));
	cs.slide_subscene(MICRO,"latex__5",vec2(-0.12,0));
	shared_ptr<LatexScene> latex__6 = make_shared<LatexScene>(")",vec2(0.099,0.289));
	cs.add_scene_fade_in(MICRO, latex__6 , "latex__6",vec2(0.65,0.35));
	cs.slide_subscene(MICRO,"latex__6",vec2(-0.165,0));
	shared_ptr<LatexScene> latex__7 = make_shared<LatexScene>("(",vec2(0.099,0.289));
	cs.add_scene_fade_in(MICRO, latex__7 , "latex__7",vec2(0.35,0.35));
	cs.slide_subscene(MICRO,"latex__7",vec2(0.165,0));
	shared_ptr<LatexScene> latex__8 = make_shared<LatexScene>(")",vec2(0.099,0.289));
	cs.add_scene_fade_in(MICRO, latex__8 , "latex__8",vec2(0.65,0.35));
	cs.slide_subscene(MICRO,"latex__8",vec2(0.11,0));
	
    cs.render_microblock();
    cs.render_microblock();


    stage_macroblock(FileBlock("3 types of things can come out:"));

	shared_ptr<LatexScene> latex_2 = make_shared<LatexScene>("2",vec2(0.15,0.36));
	cs.add_scene_fade_in(MICRO, latex_2 ,"latex_2",vec2(0.5,0.328));

	
	latex_x->manager.transition(MICRO,{{"w","0.357"},{"h","0.346"}});
	latex_x->begin_latex_transition(MICRO, "x²");
	cs.slide_subscene(MICRO,"latex_x",vec2(0.01,-0.026));
	cs.slide_subscene(MICRO,"latex__1",vec2(0.015,0.01));
	cs.slide_subscene(MICRO,"latex_y",vec2(0.12,0.01));
	cs.slide_subscene(MICRO,"latex_x2",vec2(-0.054,0.01));
	cs.slide_subscene(MICRO,"latex__2",vec2(-0.01,0.01));
	
	latex_y2->manager.transition(MICRO,{{"w","0.364"},{"h","0.346"}});
	latex_y2->begin_latex_transition(MICRO, "y²");
	cs.slide_subscene(MICRO,"latex_y2",vec2(0,-0.026));
	cs.slide_subscene(MICRO,"latex_2",vec2(-0.048,0.01));

	cs.slide_subscene(MICRO,"latex__5",vec2(0.02,0.01));
	cs.slide_subscene(MICRO,"latex__6",vec2(0.01,0.01));
	cs.slide_subscene(MICRO,"latex__7",vec2(-0.01,0.01));
	cs.slide_subscene(MICRO,"latex__8",vec2(-0.02,0.01));
	
	cs.fade_subscene(MICRO,"latex__8",0);
	cs.fade_subscene(MICRO,"latex__7",0);
	cs.fade_subscene(MICRO,"latex__6",0);
	cs.fade_subscene(MICRO,"latex__5",0);

    cs.render_microblock();
    cs.render_microblock();
	cs.remove_subscene("latex__8");
	cs.remove_subscene("latex__7");
	cs.remove_subscene("latex__6");
	cs.remove_subscene("latex__5");
	
    


    stage_macroblock(FileBlock("x squared, xy, and y squared.")); 
    cs.render_microblock();
	cs.slide_subscene(MICRO,"latex_x",vec2(0,-0.022));
	cs.fade_subscene(MICRO,"latex__1",0.15);
	cs.fade_subscene(MICRO,"latex_y",0.15);
	cs.fade_subscene(MICRO,"latex_x2",0.15);
	cs.fade_subscene(MICRO,"latex__2",0.15);
	cs.fade_subscene(MICRO,"latex_y2",0.15);
	cs.fade_subscene(MICRO,"latex_2",0.15);
    cs.render_microblock();
    cs.render_microblock();
	cs.fade_subscene(MICRO,"latex_x",0.15);
	cs.slide_subscene(MICRO,"latex_y",vec2(0,-0.022));
	cs.fade_subscene(MICRO,"latex_y",1);
	cs.slide_subscene(MICRO,"latex_x2",vec2(0,-0.022));
	cs.fade_subscene(MICRO,"latex_x2",1);
    cs.render_microblock();
    cs.render_microblock();
	cs.fade_subscene(MICRO,"latex_y",0.15);
	cs.fade_subscene(MICRO,"latex_x2",0.15);
	cs.slide_subscene(MICRO,"latex_y2",vec2(0,-0.022));
	cs.fade_subscene(MICRO,"latex_y2",1);
    cs.render_microblock();
    cs.render_microblock();





    td->manager.set({
        {"force_complex", "1"},
        {"xx_x", "-1.65"},
        {"xx_y", "0.85"},
        {"xy_x", "0"},
        {"xy_y", "0.85"},
        {"yy_x", "1.65"},
        {"yy_y", "0.85"},
    });


    stage_macroblock(FileBlock("The answer to the multiplication depends on what these things equal."));
    cs.render_microblock();
    cs.render_microblock();

	cs.slide_subscene(MICRO,"latex_x",vec2(0.02,0));
	cs.slide_subscene(MICRO,"latex_x2",vec2(-0.03,0));
	cs.slide_subscene(MICRO,"latex_y",vec2(-0.03,0));
	cs.slide_subscene(MICRO,"latex_y2",vec2(-0.02,0));

	cs.fade_subscene(MICRO,"latex_x",1);
	cs.fade_subscene(MICRO,"latex__1",0);
	cs.fade_subscene(MICRO,"latex_y",1);
	cs.fade_subscene(MICRO,"latex_x2",1);
	cs.fade_subscene(MICRO,"latex__2",0);
	cs.fade_subscene(MICRO,"latex_2",0);
    cs.render_microblock();
    cs.render_microblock();


    stage_macroblock(FileBlock("For example, we can define x squared to equal this point"));
        
    cs.render_microblock();
    cs.fade_subscene(MICRO,"latex_xx",0);
    cs.manager.transition(MICRO, {{"latex_x.x", "0.68"},{"latex_x.y", "0.5"},});
    latex_x->manager.transition(MICRO,{{"w","0.08"},{"h","0.11"}});
    latex_x->begin_latex_transition(MICRO, "x^2");
	cs.fade_subscene(MICRO,"latex_x",0);
    td->manager.transition(MICRO, {
        {"xx_opacity", "255"},
        {"xx_x", "1.6"},
        {"xx_y", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();



    stage_macroblock(FileBlock("xy to equal this point"));


    cs.fade_subscene(MICRO,"latex_xy",0);
    cs.manager.transition(MICRO, {{"latex_x2.x", "0.495"},{"latex_x2.y", "0.168"},});
    latex_x2->manager.transition(MICRO,{{"w","0.07"},{"h","0.1"}});
    latex_x2->begin_latex_transition(MICRO, "x");
	cs.fade_subscene(MICRO,"latex_x2",0);
    cs.manager.transition(MICRO, {{"latex_y.x", "0.508"},{"latex_y.y", "0.168"},});
    latex_y->manager.transition(MICRO,{{"w","0.07"},{"h","0.1"}});
    latex_y->begin_latex_transition(MICRO, "y");
	cs.fade_subscene(MICRO,"latex_y",0);
    td->manager.transition(MICRO, {
        {"xy_opacity", "255"},
        {"xy_x", "0"},
        {"xy_y", "1.6"},
    });
    cs.render_microblock();
    cs.render_microblock();



    stage_macroblock(FileBlock("and y squared to equal this point"));

    cs.fade_subscene(MICRO,"latex_yy",0);
    cs.manager.transition(MICRO, {{"latex_y2.x", "0.32"},{"latex_y2.y", "0.5"},});
    latex_y2->manager.transition(MICRO,{{"w","0.08"},{"h","0.11"}});
    latex_y2->begin_latex_transition(MICRO, "y^2");
	cs.fade_subscene(MICRO,"latex_y2",0);
    td->manager.transition(MICRO, {
        {"yy_opacity", "255"},
        {"yy_x", "-1.6"},
        {"yy_y", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();

	cs.remove_subscene("latex_x");
	cs.remove_subscene("latex__1");
	cs.remove_subscene("latex_y");
	cs.remove_subscene("latex_x2");
	cs.remove_subscene("latex__2");
	cs.remove_subscene("latex_2");
	cs.remove_subscene("latex_y2");




    stage_macroblock(CompositeBlock(FileBlock("This is the spinny algebra."), SilenceBlock(0.6))); 
    td->manager.transition(MACRO, {
        {"diagram_opacity", "255"},
        {"xx_x", "1"},{"xx_y", "0"},{"xy_x", "0"},
        {"xy_y", "1"},{"yy_x", "-1"},{"yy_y", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();
    shared_ptr<PngScene> spinny_png = make_shared<PngScene>("spinny",vec2(0.18,0.09));
    cs.add_scene(spinny_png, "spinny_png",vec2(0.09,-0.05)); 
    cs.slide_subscene(MICRO, "spinny_png",vec2(0,0.1)); 
    td->manager.transition(MICRO, {
        {"diagram_label", "1"},
    });
    cs.render_microblock();




    stage_macroblock(SilenceBlock(0.6)); 
    td->manager.transition(MICRO, {{"dragger_shown", "1"},{"brightness", "255"},});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1)); 
    cursor->manager.transition(MICRO,{{"cursor_x", "{drag_x}"},{"cursor_y", "{drag_y}"},});
    cs.render_microblock();



    stage_macroblock(SilenceBlock(4)); 
    td->manager.set({{"force_complex", "0"},});
    td->manager.transition(MICRO, {
        {"dragger_x", "-0.9"},
        {"dragger_y", "-0.1"},
    });
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"dragger_x", "-0.2"},
        {"dragger_y", "0.4"},
        {"grid_scale", "0.6"},
    });
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"dragger_x", "1"},
        {"dragger_y", "1"},
        {"grid_scale", "1"},
    });
    cs.render_microblock();




    stage_macroblock(FileBlock("We can move the definitions to make new algebras.")); 

    cursor->manager.transition(MICRO,{{"cursor_x", "{xy_pos_x}"},{"cursor_y", "{xy_pos_y}"},});
    cs.render_microblock();
    
    cs.slide_subscene(MICRO, "spinny_png",vec2(0,-0.1)); 

    td->manager.transition(MICRO, {
        {"diagram_label", "0"},
        {"xy_x", "-0.6"},
        {"xy_y", "0.8"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "48"},{"bg_1_b", "24"},});
    cs.render_microblock();
    cs.remove_subscene("spinny_png");

    td->manager.transition(MICRO, {
        {"xy_x", "0.2"},
        {"xy_y", "0.4"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "40"},{"bg_1_g", "16"},{"bg_1_b", "0"},});
    cs.render_microblock();
    

    stage_macroblock(SilenceBlock(3)); 

    cursor->manager.transition(MICRO,{{"cursor_x", "{yy_pos_x}"},{"cursor_y", "{yy_pos_y}"},});
    cs.render_microblock();

    td->manager.transition(MICRO, {
        {"yy_x", "-0.8"},
        {"yy_y", "-0.3"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "48"},{"bg_1_g", "0"},{"bg_1_b", "24"},});
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"yy_x", "-0.5"},
        {"yy_y", "0.4"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "24"},{"bg_1_g", "0"},{"bg_1_b", "60"},});
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"yy_x", "0"},
        {"yy_y", "1"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "24"},{"bg_1_b", "72"},});
    cs.render_microblock();


    
    stage_macroblock(FileBlock("Here's the squishy one.")); 

    cursor->manager.transition(MICRO,{{"cursor_x", "{xy_pos_x}"},{"cursor_y", "{xy_pos_y}"},});
    cs.render_microblock();
    
    shared_ptr<PngScene> squishy_png = make_shared<PngScene>("squishy",vec2(0.18,0.09));
    cs.add_scene(squishy_png, "squishy_png",vec2(0.09,-0.05)); 
    cs.slide_subscene(MICRO, "squishy_png",vec2(0,0.1)); 

    td->manager.transition(MICRO, {
        {"diagram_label", "1"},
        {"xy_x", "0"},
        {"xy_y", "0"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "48"},{"bg_1_b", "48"},});
    cs.render_microblock();
    cs.render_microblock();



    stage_macroblock(SilenceBlock(6)); 
    cursor->manager.transition(MICRO,{{"cursor_x", "{drag_x}"},{"cursor_y", "{drag_y}"},});
    cs.render_microblock();

    vector<StateSet> waypoints;
    waypoints.push_back({
        {"dragger_x", "1"},
        {"dragger_y", "1"},
        {"grid_scale", "1"},
    });
    waypoints.push_back({
        {"dragger_x", "0.4 {t} cos .1 * +"},
        {"dragger_y", "1.4 {t} sin .4 * +"},
        {"grid_scale", "0.7"},
    });
    waypoints.push_back({
        {"dragger_x", "0.8 {t} cos .1 * +"},
        {"dragger_y", "0.7 {t} sin .4 * +"},
        {"grid_scale", "0.9"},
    });
    waypoints.push_back({
        {"dragger_x", "1.3"},
        {"dragger_y", "0.3"},
        {"grid_scale", "0.7"},
    });
    BezierStateCurve bsc3(waypoints);
    bsc3.run_curve(*td,cs);

    cursor->manager.transition(MICRO,{{"cursor_x", "{xy_pos_x}"},{"cursor_y", "{xy_pos_y}"},});
    cs.render_microblock();



    stage_macroblock(CompositeBlock(SilenceBlock(0.6), FileBlock("And here's a flippy one."))); 

    cs.slide_subscene(MICRO, "squishy_png",vec2(0,-0.1)); 

    td->manager.set({{"grid_size", "9"},});
    td->manager.transition(MICRO, {
        {"xy_x", "0"},
        {"xy_y", "1"},
        {"grid_scale", "1"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "32"},{"bg_1_g", "24"},{"bg_1_b", "48"},});
    cs.render_microblock();

    cs.remove_subscene("squishy_png");

    cursor->manager.set({{"cursor_x", "{yy_pos_x}"},{"cursor_y", "{yy_pos_y}"},});

    shared_ptr<PngScene> flippy_png = make_shared<PngScene>("flippy",vec2(0.18,0.09));
    cs.add_scene(flippy_png, "flippy_png",vec2(0.085,-0.05)); 
    cs.slide_subscene(MICRO, "flippy_png",vec2(0,0.11)); 

    td->manager.transition(MICRO, {
        {"yy_x", "1"},
        {"yy_y", "0"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "48"},{"bg_1_g", "0"},{"bg_1_b", "60"},});
    cs.render_microblock();


    stage_macroblock(SilenceBlock(.8));

    cursor->manager.transition(MICRO,{{"cursor_x", "{drag_x}"},{"cursor_y", "{drag_y}"},});
    cs.render_microblock();


    stage_macroblock(SilenceBlock(4.2));
    td->manager.set("flippy", "0");
    td->manager.transition(MICRO, {
        {"dragger_x", "<flippy> cos 1.5 *"},
        {"dragger_y", "<flippy> sin 1.5 *"},
    });
    td->manager.transition(MICRO, "flippy", "3.14 1.1 *");
    cs.render_microblock();
    


    stage_macroblock(FileBlock("Multiplication behaves differently across different algebras."));

    td->manager.set({{"grid_size", "16"},});
    cursor->manager.transition(MICRO,{{"cursor_x", "{yy_pos_x}"},{"cursor_y", "{yy_pos_y}"},});
    cs.slide_subscene(MICRO, "flippy_png",vec2(0,-0.11)); 
    td->manager.transition(MICRO, {
        {"diagram_label", "0"},
    });
    cs.render_microblock();

    cs.manager.transition(MICRO,{{"bg_1_r", "56"},{"bg_1_g", "0"},{"bg_1_b", "32"},});
    td->manager.transition(MICRO, {
        {"yy_x", "0.5"},
        {"yy_y", "-0.8"},
        {"grid_scale", "0.9"},
    });
    cs.render_microblock();

    cs.manager.transition(MICRO,{{"bg_1_r", "60"},{"bg_1_g", "0"},{"bg_1_b", "16"},});
    td->manager.transition(MICRO, {
        {"yy_x", "-0.5"},
        {"yy_y", "-0.6"},
    });
    cs.render_microblock();
    cs.remove_subscene("flippy_png");



    
    stage_macroblock(SilenceBlock(4));


    td->manager.set({{"grid_size", "25"},});
    cursor->manager.transition(MICRO,{{"cursor_x", "{xy_pos_x}"},{"cursor_y", "{xy_pos_y}"},});
    cs.render_microblock();

    cs.manager.transition(MICRO,{{"bg_1_r", "16"},{"bg_1_g", "48"},{"bg_1_b", "0"},});
    td->manager.transition(MICRO, {
        {"xy_x", "-0.5"},
        {"xy_y", "0.3"},
        {"grid_scale", "0.8"},
    });
    cs.render_microblock();

    cursor->manager.transition(MICRO,{{"cursor_x", "{xx_pos_x}"},{"cursor_y", "{xx_pos_y}"},});
    cs.render_microblock();

    td->manager.set({
        {"xx_x", "<quarter> cos"},
        {"xx_y", "<quarter> sin"},
        {"quarter", "0"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "16"},{"bg_1_b", "64"},});
    td->manager.transition(MICRO, {
        {"quarter", "3.14 2 /"},
        {"grid_scale", "1"},
    });
    cs.render_microblock();

}


void two_dim_smallness(CompositeScene& cs, StateSet& bg){

    shared_ptr<Scene> cursor = cs.get_subscene_pointer("cursor");
    shared_ptr<TwoDAlgebraScene> td = make_shared<TwoDAlgebraScene>();

    td->manager.set(bg);
    td->manager.set({
        {"mode", "102"},
        {"update_active", "0"},
        {"diagram_opacity", "255"},
        {"xx_opacity", "255"}, 
        {"xy_opacity", "255"}, 
        {"yy_opacity", "255"}, 
        {"xx_x", "0"},
        {"xx_y", "1"},
        {"xy_x", "-0.5"},
        {"xy_y", "0.3"},
        {"yy_x", "-0.5"},
        {"yy_y", "-0.6"},
        {"zoom", "0"},
        
    });



    stage_macroblock(FileBlock("And this differing behavior extends to functions, like the square function."));
    cs.render_microblock();
    cs.fade_subscene(MICRO,"td2",0);
    cs.add_scene_fade_in(MICRO, td, "td3", vec2(0.5,0.5), 1, false, 2);
    cs.render_microblock();

    cs.remove_subscene("td2");

    cs.render_microblock(); 

    stage_macroblock(SilenceBlock(4));
    td->manager.transition(MICRO, {
        {"xx_x", "-0.1"},
        {"xx_y", "-0.4"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "8"},{"bg_1_g", "0"},{"bg_1_b", "60"},});
    cs.render_microblock(); 


    cursor->manager.transition(MICRO,{{"cursor_x", "{yy_pos_x}"},{"cursor_y", "{yy_pos_y}"},});
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"yy_x", "0.7"},
        {"yy_y", "0.8"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "40"},{"bg_1_g", "0"},{"bg_1_b", "32"},});
    cs.render_microblock();


    cursor->manager.transition(MICRO,{{"cursor_x", "{xy_pos_x}"},{"cursor_y", "{xy_pos_y}"},});
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"xy_x", "-0.1"},
        {"xy_y", "-0.8"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "24"},{"bg_1_g", "0"},{"bg_1_b", "48"},});
    cs.render_microblock();



    stage_macroblock(FileBlock("I've colored points red if they land on the x axis when you square them."));
    td->manager.transition(MICRO, {
        {"re_channel", "0"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "48"},{"bg_1_g", "0"},{"bg_1_b", "0"},});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("Green points land on the y axis."));
    td->manager.transition(MICRO, {
        {"re_channel", "1"},
        {"im_channel", "0"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "48"},{"bg_1_b", "0"},});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1.7));
    td->manager.transition(MICRO, {
        {"im_channel", "1"},
    });
    cs.render_microblock();




    stage_macroblock(FileBlock("We can cube each point,"));


    td->manager.transition(MICRO, {
        {"mode", "103"},
        {"zoom", "0.4"},
    });
    cs.render_microblock(); 


    stage_macroblock(SilenceBlock(4));
    cs.render_microblock(); 
    td->manager.transition(MICRO, {
        {"xy_x", "0.7"},
        {"xy_y", "0.4"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "40"},{"bg_1_b", "24"},});
    cs.render_microblock();

    cursor->manager.transition(MICRO,{{"cursor_x", "{yy_pos_x}"},{"cursor_y", "{yy_pos_y}"},});
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"yy_x", "0.5"},
        {"yy_y", "-0.7"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "16"},{"bg_1_b", "68"},});
    cs.render_microblock();

    cs.render_microblock(); 


    stage_macroblock(FileBlock("take the fourth power,"));
    td->manager.transition(MICRO, {
        {"mode", "104"},
        {"zoom", "0.8"},
    });
    cs.render_microblock(); 

    stage_macroblock(SilenceBlock(3));
    cs.render_microblock();

    td->manager.transition(MICRO, {
        {"yy_x", "-0.1"},
        {"yy_y", "0.7"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "40"},{"bg_1_g", "0"},{"bg_1_b", "52"},});
    cs.render_microblock();

    cursor->manager.transition(MICRO,{{"cursor_x", "{xx_pos_x}"},{"cursor_y", "{xx_pos_y}"},});
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"xx_x", "0.4"},
        {"xx_y", "0.1"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "64"},{"bg_1_g", "0"},{"bg_1_b", "16"},});
    cs.render_microblock();


    stage_macroblock(CompositeBlock(CompositeBlock(SilenceBlock(0.4), FileBlock("plot more complicated polynomials,")), SilenceBlock(0.5)));
    cs.render_microblock();
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"mode", "105"},
    });
    cs.render_microblock();
    cs.render_microblock(); 
    td->manager.transition(MICRO, {
        {"xx_x", "0.8"},
        {"xx_y", "-0.4"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "56"},{"bg_1_g", "24"},{"bg_1_b", "0"},});
    cs.render_microblock();



    stage_macroblock(SilenceBlock(3.5));
    cursor->manager.transition(MICRO,{{"cursor_x", "{yy_pos_x}"},{"cursor_y", "{yy_pos_y}"},});
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"yy_x", "-0.6"},
        {"yy_y", "-0.8"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "48"},{"bg_1_b", "16"},});
    cs.render_microblock();
    cs.render_microblock();


    shared_ptr<LatexScene> latex_poly = make_shared<LatexScene>("\\sin(p) \\approx p - \\frac{p^3}{3!} + \\frac{p^5}{5!} - \\frac{p^7}{7!} + \\frac{p^9}{9!}",vec2(0.6,0.18));
	cs.add_scene(latex_poly , "latex_poly",vec2(0.69,-0.08));
	cs.slide_subscene(MICRO,"latex_poly",vec2(0,0.23));


    stage_macroblock(CompositeBlock(FileBlock("And even use a Taylor polynomial to graph the sine function in all sorts of algebras."), SilenceBlock(4)));
    // Equation 105 is ~100x larger than the sine polynomial, so a linear blend only reveals the sine
    // in the last few frames. Shrink 105's weight geometrically instead, from 1 down to 1e-4.
    td->manager.set({
        {"sine_ease", "0 1 {microblock_fraction} smoothlerp"},
        {"sine_lerp", "1 10 <sine_ease> -4 * ^ - .9999 /"},
        {"mode", "105 <sine_lerp> +"},
    });
    td->manager.transition(MICRO, {
        {"zoom", "0"},
    });
    cs.render_microblock();
    td->manager.set({
        {"mode", "106"},
    });
    cs.render_microblock();

    td->manager.transition(MICRO, {
        {"yy_x", "0.3"},
        {"yy_y", "0.7"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "32"},{"bg_1_b", "48"},});
    cs.render_microblock();

    cursor->manager.transition(MICRO,{{"cursor_x", "{xy_pos_x}"},{"cursor_y", "{xy_pos_y}"},});
    cs.render_microblock();


    td->manager.transition(MICRO, {
        {"xy_x", "0.2"},
        {"xy_y", "-0.2"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "16"},{"bg_1_b", "68"},});
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"xy_x", "-0.8"},
        {"xy_y", "0.4"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "32"},{"bg_1_g", "0"},{"bg_1_b", "60"},});
    cs.render_microblock();

    cursor->manager.transition(MICRO,{{"cursor_x", "{xx_pos_x}"},{"cursor_y", "{xx_pos_y}"},});
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"xx_x", "-0.5"},
        {"xx_y", "-0.4"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "60"},{"bg_1_g", "0"},{"bg_1_b", "48"},});
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"xx_x", "0"},
        {"xx_y", "-1"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "68"},{"bg_1_g", "0"},{"bg_1_b", "24"},});
    cs.render_microblock();
    cursor->manager.transition(MICRO,{{"cursor_x", "{yy_pos_x}"},{"cursor_y", "{yy_pos_y}"},});
    cs.render_microblock();
    cs.render_microblock();





    stage_macroblock(FileBlock("We can also graph cosine!"));
    cs.render_microblock();
	cs.slide_subscene(MICRO,"latex_poly",vec2(0.01,0));
	latex_poly->begin_latex_transition(MICRO, "\\cos(p) \\approx 1 - \\frac{p^2}{2!} + \\frac{p^4}{4!} - \\frac{p^6}{6!} + \\frac{p^8}{8!}");
    td->manager.transition(MICRO, {
        {"mode", "107"},
        {"zoom", "0.2"},
    });
    cs.render_microblock();
    cs.render_microblock();


    td->manager.transition(MICRO, {
        {"yy_x", "0.6"},
        {"yy_y", "-0.1"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "72"},{"bg_1_g", "0"},{"bg_1_b", "0"},});
    cs.render_microblock();


    stage_macroblock(CompositeBlock(SilenceBlock(3), FileBlock("Though... not always.")));
    td->manager.transition(MICRO, {
        {"yy_x", "-0.8"},
        {"yy_y", "-0.5"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "60"},{"bg_1_g", "24"},{"bg_1_b", "0"},});
    cs.render_microblock();


    cursor->manager.transition(MICRO,{{"cursor_x", "{xy_pos_x}"},{"cursor_y", "{xy_pos_y}"},});
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"xy_x", "0.1"},
        {"xy_y", "-0.6"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "52"},{"bg_1_g", "32"},{"bg_1_b", "0"},});
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"xy_x", "0"},
        {"xy_y", "0.4"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "60"},{"bg_1_g", "0"},{"bg_1_b", "16"},});
    cs.render_microblock();


    stage_macroblock(FileBlock("The polynomial has a 1 term, ...   ...but what is 1?"));
	cs.manager.transition(MICRO,{{"latex_poly.x", "0.5"},{"latex_poly.y", "0.5"}});
    td->manager.transition(MICRO, {{"brightness", "64"},});
    cs.render_microblock(); 
    cs.render_microblock(); 
	latex_poly->begin_latex_transition(MICRO, "\\cos(p) \\approx {\\Huge 1} - \\frac{p^2}{2!} + \\frac{p^4}{4!} - \\frac{p^6}{6!} + \\frac{p^8}{8!}");
	latex_poly->manager.transition(MICRO,{{"w","0.64"}});
    cs.render_microblock(); 
    cs.render_microblock(); 
    cs.render_microblock(); 

    td->manager.set({{"update_active", "0"},});



}


void two_dim_wrapup(CompositeScene& cs, StateSet& bg){



    shared_ptr<TwoDAlgebraScene> td = make_shared<TwoDAlgebraScene>();
    cs.add_scene_fade_in(MICRO, td, "td4");
    cs.move_to_front("cursor");
    shared_ptr<Scene> cursor = cs.get_subscene_pointer("cursor");
    cursor->manager.transition(MICRO,{{"cursor_x", "{drag_x}"},{"cursor_y", "{drag_y}"},});

    td->manager.set(bg);
    td->manager.set({
        {"dragger_type", "2"},
        {"dragger_x", "1.4"},
        {"dragger_y", "0.6"},
        {"dragger_shown", "1"},
        {"diagram_opacity", "255"},
        {"xx_opacity", "255"}, 
        {"xy_opacity", "255"}, 
        {"yy_opacity", "255"}, 
        {"xx_x", "0"},
        {"xx_y", "-1"},
        {"xy_x", "0"},
        {"xy_y", "0.4"},
        {"yy_x", "-0.8"},
        {"yy_y", "-0.5"},
        {"div_scene", "1"},
        {"point_count", "1"},
        {"point_size", "3"},
        {"point_label", "1"},
        {"point_x", "1"},
        {"point_y", "0"},
        {"mode", "0"},
        {"grid_size", "18"},
    });

    set_global_draggers(1.4,0.6);


    stage_macroblock(CompositeBlock(SilenceBlock(0.8),FileBlock("This algebra has no neutral multiplication.")));
    cs.fade_subscene(MICRO,"latex_poly",0);
    cs.fade_subscene(MICRO,"latex_trig",0);
    cs.fade_subscene(MICRO,"latex_unit",0);
    cs.fade_subscene(MICRO,"td3",0);
    cs.render_microblock(); 

    cs.remove_subscene("td3");
    cs.remove_subscene("latex_poly");
    cs.remove_subscene("latex_trig");
    cs.remove_subscene("latex_unit");

    td->manager.set({{"spinny", "3.14 .3 *"},{"spinny_r", "1.3"},});
    td->manager.transition(MICRO, {
        {"dragger_x", "<spinny> cos <spinny_r> *"},
        {"dragger_y", "<spinny> sin <spinny_r> *"},
    });
    cs.render_microblock(); 


    stage_macroblock(CompositeBlock(FileBlock("We can't get x and y to their original positions."),SilenceBlock(5)));

    // Land exactly on (-0.1, -1.1): angle atan2(-1.1, -0.1) + 2pi ~= 1.471pi, radius sqrt(1.22)
    td->manager.transition(MACRO, {
        {"spinny", "pi 1.471 *"},
        {"spinny_r", "1.22 sqrt"},
    });
    cs.render_microblock(); 
    cs.render_microblock(); 
    cs.render_microblock(); 
    cs.render_microblock(); 
    cs.render_microblock(); 
    cs.render_microblock(); 
    td->manager.set("unit_targets", "1");
    cs.render_microblock(); 
    td->manager.set("unit_targets", "0");
    cs.render_microblock(); 
    td->manager.set("unit_targets", "1");
    cs.render_microblock(); 
    td->manager.set("unit_targets", "0");
    cs.render_microblock(); 
    td->manager.set("unit_targets", "1");
    cs.render_microblock(); 
    td->manager.set("unit_targets", "0");
    cs.render_microblock(); 
    td->manager.set("unit_targets", "1");
    cs.render_microblock(); 
    td->manager.set("unit_targets", "0");
    cs.render_microblock(); 
    td->manager.set("unit_targets", "1");
    cs.render_microblock(); 
    td->manager.set("unit_targets", "0");
    cs.render_microblock(); 
    cs.render_microblock(); 
    cs.render_microblock(); 
    cs.render_microblock(); 
    cs.render_microblock(); 

    td->manager.transition(MICRO, {
        {"dragger_x", "-0.1"},
        {"dragger_y", "-1.1"},
        {"grid_scale", "0.8"},
    });
    cs.render_microblock(); 


    stage_macroblock(FileBlock("This algebra doesn't have a 1. yuck."));
    cs.render_microblock(); 

    stage_macroblock(CompositeBlock(FileBlock("Let's define x to act as 1."),SilenceBlock(0.6)));
    // (dx*xx_x + dy*xy_x, dx*xx_y + dy*xy_y) = (0, -0.1*-1 + -1.1*0.4) = (0, -0.34),
    cursor->manager.transition(MICRO,{{"cursor_x", ".5"},{"cursor_y", ".5 .34 5 / +"},});
    cs.render_microblock(); 
    StateSet undo_cursor_click = cursor->manager.set("cursor_size", "{microblock_fraction} .8 > -.03 * .08 +");
    cs.render_microblock(); 
    td->manager.set({
        {"xpoint_label", "0"},
    });
    cursor->manager.set(undo_cursor_click);
    cs.render_microblock(); 
    cs.render_microblock(); 


    cursor->manager.transition(MICRO,{{"cursor_x", "{xx_pos_x}"},{"cursor_y", "{xx_pos_y}"},});

    td->manager.transition(MICRO, {
        {"point_opacity", "0"},
    });
    stage_macroblock(FileBlock("That means x squared is 1,")); 
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"xx_x", "1"},
        {"xx_y", "0"},
        {"x_label", "0"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "48"},{"bg_1_g", "0"},{"bg_1_b", "32"},});
    cs.render_microblock();
    td->manager.set({{"point_count", "0"},});

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    cursor->manager.transition(MICRO,{{"cursor_x", "{xy_pos_x}"},{"cursor_y", "{xy_pos_y}"},});
    stage_macroblock(FileBlock("and xy is simply y.")); 
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"xy_x", "0"},
        {"xy_y", "1"},
        {"xy_label", "0"},
        {"grid_scale", "1"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "24"},{"bg_1_g", "0"},{"bg_1_b", "60"},});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();
    cursor->manager.transition(MICRO,{{"cursor_x", "{yy_pos_x}"},{"cursor_y", "{yy_pos_y}"},});
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("Now y squared is the only free definition."), SilenceBlock(4)));
    cs.manager.set("siney", "0");
    td->manager.transition(MACRO, {
        {"yy_x", "0.6"},
    });
    td->manager.transition(MICRO, {
        {"yy_y", "[siney] sin"},
    });
    cs.manager.transition(MACRO, "siney", "3.14 1.2 *");
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "<siney> 1.2 / sin 50 *"},{"bg_1_b", "68 <siney> 1.2 / sin 38 * -"},});
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"yy_y", "-0.7"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "24"},{"bg_1_g", "0"},{"bg_1_b", "68"},});
    cs.render_microblock();



    stage_macroblock(CompositeBlock(FileBlock("When y squares to 1, we get the flippy algebra."), SilenceBlock(4)));
    td->manager.transition(MICRO, {
        {"yy_x", "1"},
        {"yy_y", "0"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "48"},{"bg_1_g", "0"},{"bg_1_b", "60"},});
    cs.render_microblock();

    shared_ptr<PngScene> flippy_png = make_shared<PngScene>("flippy",vec2(0.18,0.09));
    cs.add_scene(flippy_png, "flippy_png",vec2(0.09,-0.05)); 
    cs.slide_subscene(MICRO, "flippy_png",vec2(0,0.11)); 
    td->manager.transition(MICRO, {{"diagram_label", "1"},});
    cs.render_microblock();

    cursor->manager.transition(MICRO,{{"cursor_x", "{drag_x}"},{"cursor_y", "{drag_y}"},});
    cs.render_microblock();
    td->manager.set("flippy", "3.14 1.5 *");
    td->manager.transition(MICRO, "flippy", "3.14 1 *");
    td->manager.transition(MICRO, {
        {"dragger_x", "<flippy> cos 1.1 *"},
        {"dragger_y", "<flippy> sin 1.1 *"},
    });

    td->manager.set({{"grid_size", "12"}});
    
    cs.render_microblock();
    td->manager.transition(MICRO, "flippy", "3.14 .5 *");
    cs.render_microblock();





    stage_macroblock(FileBlock("These transformations look nice, but don't let them fool you."));
    td->manager.transition(MICRO, {
        {"flippy", "3.14 -.5 *"},
    });
    cs.render_microblock();


    stage_macroblock(FileBlock("Flipping is bad."));

    shared_ptr<PngScene> flipping_text = make_shared<PngScene>("flipping",vec2(0.7,0.7));
    cs.add_scene_fade_in(MICRO, flipping_text, "flipping_text",vec2(0.5,0.44)); 
    cs.slide_subscene(MICRO, "flipping_text", vec2(0,0.1));
    cs.fade_subscene(MICRO, "td4", 0.3);
    cs.fade_subscene(MICRO, "flippy_png", 0.3);
    cursor->manager.transition(MICRO, {{"fill_a", "75"}});
    cs.render_microblock();
    shared_ptr<PngScene> is_text = make_shared<PngScene>("is",vec2(0.7,0.7));
    cs.add_scene_fade_in(MICRO, is_text, "is_text",vec2(0.5,0.44)); 
    cs.slide_subscene(MICRO, "is_text", vec2(0,0.1));
    cs.render_microblock();
    shared_ptr<PngScene> bad_text = make_shared<PngScene>("bad",vec2(0.7,0.7));
    cs.add_scene_fade_in(MICRO, bad_text, "bad_text",vec2(0.5,0.44)); 
    cs.slide_subscene(MICRO, "bad_text", vec2(0,0.1));
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();


    stage_macroblock(SilenceBlock(1));
    cs.slide_subscene(MICRO, "flipping_text", vec2(0,0.05));
    cs.fade_subscene(MICRO, "flipping_text", 0);
    cs.slide_subscene(MICRO, "is_text", vec2(0,0.05));
    cs.fade_subscene(MICRO, "is_text", 0);
    cs.slide_subscene(MICRO, "bad_text", vec2(0,0.05));
    cs.fade_subscene(MICRO, "bad_text", 0);
    cs.fade_subscene(MICRO, "td4", 1);
    cs.fade_subscene(MICRO, "flippy_png", 1);
    cursor->manager.transition(MICRO, {{"fill_a", "255"}});
    cs.render_microblock();
    cs.render_microblock();


    stage_macroblock(FileBlock("If we only flip halfway, the plane collapses into a line."));
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"dragger_x", "-0.5"},
        {"dragger_y", "-0.5"},
    });
    cs.render_microblock();
    cs.render_microblock();

    cs.remove_subscene("flipping_text");
    cs.remove_subscene("is_text");
    cs.remove_subscene("bad_text");


    td->manager.set({
        {"div_scene", "2"},
        {"point_label","1"},
        {"point_x","1"},
        {"point_y","0"},
        {"point_size", "3.5"},
        {"point_opacity", "255"}
    });
    stage_macroblock(SilenceBlock(1));
    td->manager.transition(MICRO, {
        {"dragger_x", "0"},
        {"dragger_y", "-1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("1 and y are not the same point,"));
    td->manager.transition(MICRO, {{"point_count", "1"}});
    cs.render_microblock();
    cs.render_microblock();



    stage_macroblock(FileBlock("But a half flip collapses them together."));
    td->manager.transition(MICRO, {
        {"dragger_x", "-0.5"},
        {"dragger_y", "-0.5"},
    });
    cs.render_microblock();
    cs.render_microblock();



    stage_macroblock(SilenceBlock(0.6));
	cs.render_microblock();
	

    
    stage_macroblock(FileBlock("1 times a half flip is y times a half flip."));
	shared_ptr<LatexScene> latex_1 = make_shared<LatexScene>("1",vec2(0.118,0.274));
	cs.add_scene(latex_1 , "latex_1",vec2(0.901,0.704));
	shared_ptr<LatexScene> latex_h = make_shared<LatexScene>("h",vec2(0.118,0.274));
	cs.add_scene(latex_h , "latex_h",vec2(0.951,0.704));
	cs.slide_subscene(MICRO,"latex_1",vec2(-0.189,-0.002));
	cs.slide_subscene(MICRO,"latex_h",vec2(-0.189,-0.002));
	cs.manager.set({{"latex_1.opacity","0"},{"latex_h.opacity","0"}});
	cs.manager.transition(MICRO,{{"latex_1.opacity","1"},{"latex_h.opacity","1"}});
    cs.render_microblock();
	cs.render_microblock();

	shared_ptr<LatexScene> latex__3 = make_shared<LatexScene>("=",vec2(0.177,0.274));
	cs.add_scene(latex__3 , "latex__3",vec2(1.015,0.704));
	shared_ptr<LatexScene> latex_y = make_shared<LatexScene>("y",vec2(0.112,0.274));
	cs.add_scene(latex_y , "latex_y",vec2(1.128,0.734));
	shared_ptr<LatexScene> latex_h2 = make_shared<LatexScene>("h",vec2(0.118,0.274));
	cs.add_scene(latex_h2 , "latex_h2",vec2(1.176,0.704));
	cs.manager.set({{"latex__3.opacity","0"},{"latex_y.opacity","0"},{"latex_h2.opacity","0"}});

	cs.slide_subscene(MICRO,"latex_1",vec2(-0.14,0));
	cs.slide_subscene(MICRO,"latex_h",vec2(-0.14,0));
	cs.slide_subscene(MICRO,"latex__3",vec2(-0.306,0.002));
	cs.slide_subscene(MICRO,"latex_y",vec2(-0.329,0.002));
	cs.slide_subscene(MICRO,"latex_h2",vec2(-0.329,0.002));
	cs.manager.transition(MICRO,{{"latex__3.opacity","1"},{"latex_y.opacity","1"},{"latex_h2.opacity","1"}});

	cs.render_microblock();
	cs.render_microblock();
	 


    stage_macroblock(FileBlock("So we can't divide by the half flip. If we did, we'd get 1 equals y."));
  
	shared_ptr<LatexScene> latex_h3 = make_shared<LatexScene>("\\frac{~~~~}{h}",vec2(0.264,0.204));
	cs.add_scene(latex_h3 , "latex_h3",vec2(0.608,0.925));
	shared_ptr<LatexScene> latex_h4 = make_shared<LatexScene>("\\frac{~~~~}{h}",vec2(0.264,0.204));
	cs.add_scene(latex_h4 , "latex_h4",vec2(0.821,0.925));
	cs.manager.set({{"latex_h3.opacity","0"},{"latex_h4.opacity","0"}});
	

	latex_1->manager.transition(MICRO,{{"w","0.105"},{"h","0.244"}});
	latex_1->begin_latex_transition(MICRO, "1");
	cs.slide_subscene(MICRO,"latex_1",vec2(0.013,-0.068));
	latex_h->manager.transition(MICRO,{{"w","0.105"},{"h","0.244"}});
	latex_h->begin_latex_transition(MICRO, "h");
	cs.slide_subscene(MICRO,"latex_h",vec2(0.007,-0.068));
	latex_y->manager.transition(MICRO,{{"w","0.1"},{"h","0.244"}});
	latex_y->begin_latex_transition(MICRO, "y");
	cs.slide_subscene(MICRO,"latex_y",vec2(0,-0.081));
	latex_h2->manager.transition(MICRO,{{"w","0.105"},{"h","0.244"}});
	latex_h2->begin_latex_transition(MICRO, "h");
	cs.slide_subscene(MICRO,"latex_h2",vec2(-0.005,-0.081));
	
	cs.slide_subscene(MICRO,"latex_h3",vec2(0.006,-0.18));
	cs.manager.transition(MICRO,{{"latex_h3.opacity","1"}});
	cs.slide_subscene(MICRO,"latex_h4",vec2(0.006,-0.18));
	cs.manager.transition(MICRO,{{"latex_h4.opacity","1"}});
	
	cs.render_microblock();
	cs.render_microblock();


	latex_1->manager.transition(MICRO,{{"w","0.149"},{"h","0.346"}});
	latex_1->begin_latex_transition(MICRO, "1");
	cs.slide_subscene(MICRO,"latex_1",vec2(0.031,0.067));
	latex_h->manager.transition(MICRO,{{"w","0.083"},{"h","0.192"}});
	latex_h->begin_latex_transition(MICRO, "h");
	cs.slide_subscene(MICRO,"latex_h",vec2(0.014,0.084));
	cs.manager.transition(MICRO,{{"latex_h.opacity","0"}});
	latex_y->manager.transition(MICRO,{{"w","0.141"},{"h","0.346"}});
	latex_y->begin_latex_transition(MICRO, "y");
	cs.slide_subscene(MICRO,"latex_y",vec2(0.005,0.077));
	latex_h2->manager.transition(MICRO,{{"w","0.074"},{"h","0.171"}});
	latex_h2->begin_latex_transition(MICRO, "h");
	cs.slide_subscene(MICRO,"latex_h2",vec2(-0.003,0.097));
	cs.manager.transition(MICRO,{{"latex_h2.opacity","0"}});
	
	cs.slide_subscene(MICRO,"latex_h3",vec2(0.032,-0.032));
	cs.manager.transition(MICRO,{{"latex_h3.opacity","0"}});
	cs.slide_subscene(MICRO,"latex_h4",vec2(0.014,-0.038));
	cs.manager.transition(MICRO,{{"latex_h4.opacity","0"}});
	
	cs.render_microblock();
	cs.render_microblock();

	
	cs.slide_subscene(MICRO,"latex_1",vec2(0.5,0.5));
	cs.slide_subscene(MICRO,"latex_y",vec2(0.5,0.5));
	cs.slide_subscene(MICRO,"latex__3",vec2(0.5,0.5));



    stage_macroblock(FileBlock("And that's not true!"));
    td->manager.transition(MICRO, {
        {"dragger_x", "0"},
        {"dragger_y", "-1"},
    });
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("Flipping is bad, because it breaks division."),SilenceBlock(0.6)));
    td->manager.transition(MICRO, {
        {"point_count", "0"},
        {"dragger_x", "0.8"},
        {"dragger_y", "0.2"},
    });
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"point_count", "0"},
        {"dragger_x", "0.1"},
        {"dragger_y", "1.1"},
    });
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"dragger_x", "-1"},
        {"dragger_y", "0"},
    });
    cs.render_microblock();

	 
	cs.remove_subscene("latex_h4");
	cs.remove_subscene("latex_h3");
	cs.remove_subscene("latex_h2");
	cs.remove_subscene("latex_y");
	cs.remove_subscene("latex__3");
	cs.remove_subscene("latex_h");
	cs.remove_subscene("latex_1");
	


    stage_macroblock(CompositeBlock(FileBlock("We only get division in the spinny algebra."), SilenceBlock(1)));

    cursor->manager.transition(MICRO,{{"cursor_x", "{yy_pos_x}"},{"cursor_y", "{yy_pos_y}"},});
    cs.render_microblock();
    td->manager.set({{"grid_size","20"}});

    td->manager.transition(MICRO, {
        {"yy_x", "-1"},
        {"yy_y", "0"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "8"},{"bg_1_b", "80"},});

    shared_ptr<PngScene> spinny_png = make_shared<PngScene>("spinny",vec2(0.18,0.09));
    cs.add_scene(spinny_png, "spinny_png",vec2(0.09,-0.05)); 

    cs.slide_subscene(MICRO, "spinny_png",vec2(0,0.1)); 
    cs.slide_subscene(MICRO, "flippy_png",vec2(0,-0.11)); 
    cs.render_microblock();

    cursor->manager.transition(MICRO,{{"cursor_x", "{drag_x}"},{"cursor_y", "{drag_y}"},});
    cs.render_microblock();
    td->manager.begin_timer("spinny");


    td->manager.transition(MICRO, {
        {"dragger_x", "-0.4 <spinny> -.7 * sin *"},
        {"dragger_y", "0.4 <spinny> .7 * cos *"},
        {"grid_scale", "0.6"},
    });
    cs.render_microblock();

    cs.remove_subscene("flippy_png");


    stage_macroblock(FileBlock("Any spin preserves the structure of the grid."));
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("Unless you multiply by zero... but oh well, we'll have to make some sacrifices."), SilenceBlock(1)));
    td->manager.transition(MICRO, {
        {"dragger_x", "0"},
        {"dragger_y", "0"},
        {"grid_scale", "0.2"},
    });
    cs.render_microblock();
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"dragger_x", "1"},
        {"dragger_y", "0"},
        {"grid_scale", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("This algebra is the complex numbers."));
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"dragger_shown", "0"},
        {"brightness", "64"},
        {"diagram_opacity", "0"},
        {"diagram_label", "0"},
    });
    cursor->manager.transition(MICRO,{{"cursor_x", "0.75"},{"cursor_y", "1.1"},});
    cs.slide_subscene(MICRO, "spinny_png",vec2(0,-0.1)); 
    cs.render_microblock();



    stage_macroblock(FileBlock("y squares to negative 1. It's the imaginary unit i."));
    td->manager.transition(MICRO, {
        {"diagram_opacity", "0"},
    });
    StateSet undo = td->manager.transition(MICRO, {
        {"xx_opacity", "40"},
        {"xy_opacity", "40"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    td->manager.transition(MICRO, {
        {"y_label", "0"},
    });
    td->manager.transition(MICRO, undo);
    cs.render_microblock();

    
    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();



    
    stage_macroblock(FileBlock("Functions in this algebra can feature linear symmetry, rotational symmetry,"));
    cs.render_microblock();
    cs.render_microblock();
    shared_ptr<TwoDAlgebraScene> complex = make_shared<TwoDAlgebraScene>();
    cs.add_scene_fade_in(MICRO, complex, "complex");
    cs.fade_subscene(MICRO, "td4",0);

    complex->manager.set(bg);
    complex->manager.set({
        {"xx_x", "1"},
        {"xx_y", "0"},
        {"xy_x", "0"},
        {"xy_y", "1"},
        {"yy_x", "-1"},
        {"yy_y", "0"},
        {"mode", "3"},
    });
    cs.render_microblock();
    cs.remove_subscene("td4");
    cs.remove_subscene("spinny_png");
    cs.remove_subscene("cursor"); 
    cs.render_microblock();

    complex->manager.transition(MICRO, {
        {"mode", "4"},
    });
    cs.render_microblock();
    cs.render_microblock();


    stage_macroblock(CompositeBlock(FileBlock("and plenty of asymmetry too!"), SilenceBlock(8)));
    complex->manager.transition(MICRO, {
        {"mode", "5"},
    });

    cs.render_microblock();

    complex->manager.transition(MICRO, {
        {"mode", "6"},
    });
    cs.render_microblock();

    complex->manager.transition(MICRO, {
        {"mode", "7"},
    });
    cs.render_microblock();

    complex->manager.transition(MICRO, {
        {"mode", "8"},
    });
    cs.render_microblock();



    shared_ptr<MultiplicationTableScene> table = make_shared<MultiplicationTableScene>(vec2(1,1));
    cs.add_scene_fade_in(MICRO, table, "table",vec2(0.48,1.08));
    cs.slide_subscene(MICRO, "table", vec2(0,-0.6));
    cs.fade_subscene(MICRO, "complex", 0);

    table->manager.set(bg);
    table->manager.set({
        {"table_size", "2"},
        {"three_d", "1"},
        {"q_1j", "1"},{"q_ij", "1"},{"q_jj", "1"},
        {"triangle", "0"},
    });


    stage_macroblock(FileBlock("This multiplication table describes the algebra."));
    cs.render_microblock();
    cs.render_microblock();



    stage_macroblock(SilenceBlock(0.5));
    cs.render_microblock();


    stage_macroblock(FileBlock("Although, we don't need this i times 1,"));
    cs.render_microblock();
    cs.render_microblock();
    table->manager.transition(MICRO,{
        {"opacity_10", "0.1"},{"opacity_02", "0.1"},
        {"opacity_11", "0.1"},{"opacity_12", "0.1"},{"opacity_22", "0.1"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("since it's the same as 1 times i."));
    cs.render_microblock();
    table->manager.transition(MICRO,{
        {"opacity_01", "0.1"},{"opacity_20", "0.1"},
        {"opacity_10", "1"},{"opacity_02", "1"},
        {"triangle", "1"},

        {"opacity_12", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();



    stage_macroblock(SilenceBlock(0.6));
    cs.render_microblock();


    stage_macroblock(FileBlock("These 3 cells are our 3 definitions from before."));
    table->manager.transition(MICRO,{
        {"opacity_10", "0.1"},{"opacity_02", "0.1"},
        {"opacity_11", "1"},{"opacity_12", "1"},{"opacity_22", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();

    cs.manager.set({{"complex.x","0.3"},{"complex.opacity","1"}});
    complex->manager.set({
        {"brightness","0"},
        {"diagram_opacity","255"},{"diagram_label","0"},
        {"xx_opacity", "255"},{"xy_opacity", "255"},{"yy_opacity", "255"},
        {"x_label", "0"},{"xy_label", "0"},{"y_label", "0"},
    });
    cs.slide_subscene(MICRO,"complex",vec2(0.2,0));

    cs.render_microblock();
    cs.render_microblock();


    stage_macroblock(SilenceBlock(1));
    cs.render_microblock();


    stage_macroblock(FileBlock("So we've found this nice 2D algebra."));
    cs.slide_subscene(MICRO,"complex",vec2(-0.2,0));
    table->manager.transition(MICRO,{
        {"opacity_01", "1"},{"opacity_20", "1"},
        {"opacity_10", "1"},{"opacity_02", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.remove_subscene("complex");



    table->manager.set({
        {"opacity_13", "0"},{"opacity_23", "0"},{"opacity_33", "0"},
    });
    stage_macroblock(CompositeBlock(FileBlock("Let's try extending it to 3D."), SilenceBlock(1)));
    cs.render_microblock();
    table->manager.transition(MICRO,{
        {"table_size", "3"},
    });
    cs.render_microblock();
    cs.render_microblock();


    stage_macroblock(FileBlock("With a new unit j we have 3 new products to define."));
    table->manager.transition(MICRO,{
        {"opacity_01", "0.1"},{"opacity_11", "0.1"},
        {"opacity_02", "0.1"},{"opacity_12", "0.1"},{"opacity_22", "0.1"},
        {"opacity_10", "0.1"},{"opacity_20", "0.1"},{"opacity_30", "0.1"},
    });
    cs.render_microblock();
    table->manager.transition(MICRO,{
        {"opacity_10", "1"},{"opacity_20", "1"},{"opacity_30", "1"},
        {"opacity_13", "1"},{"opacity_23", "1"},{"opacity_33", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("1j should just be j."),SilenceBlock(0.6)));
    table->manager.transition(MICRO,{
        {"opacity_20", "0.1"},{"opacity_23", "0.1"},
        {"opacity_30", "0.1"},{"opacity_33", "0.1"},
    });
    cs.render_microblock();
    cs.render_microblock();
    table->manager.transition(MICRO,{{"q_1j", "0"},});
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("But the others have some wiggle room,"));
    table->manager.transition(MICRO,{
        {"opacity_10", "0.1"},{"opacity_13", "0.1"},
        {"opacity_20", "1"},{"opacity_23", "1"},
        {"opacity_30", "1"},{"opacity_33", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();



}



void three_dim(CompositeScene& cs, StateSet& bg){


    shared_ptr<BackgroundScene> boxBorder = make_shared<BackgroundScene>(vec2(0.23,0.43));
    cs.add_scene_fade_in(MICRO, boxBorder, "boxBorder", vec2(-0.42,0.175));
    boxBorder->manager.set(bg);
    boxBorder->manager.set({{"adder","40"}});
    shared_ptr<BackgroundScene> boxFill = make_shared<BackgroundScene>(vec2(0.218,0.412));
    cs.add_scene_fade_in(MICRO, boxFill, "boxFill", vec2(-0.42,0.175));
    boxFill->manager.set(bg);


    cs.manager.begin_timer("tx");
    cs.manager.set({
        // {"xz_x", "0"}, {"xz_y", ".1"}, {"xz_z", ".9"}, 
        {"timer", "<tx> 2.4 *"},
        {"yz_x", ".6"}, {"yz_y", "-.3 <timer> sin 0.4 * +"}, {"yz_z", "0"}, {"yz_w", "0"},
        {"zz_x", "-.6"}, {"zz_y", ".1"}, {"zz_z", ".2 <timer> 0.93 * sin 0.4 * +"}, {"zz_w", "0"},
        {"d", "16"},{"w_slider", "0"},
    });


    StateSet parentcontrol = {
        // {"xz_x", "[xz_x]"}, {"xz_y", "[xz_y]"}, {"xz_z", "[xz_z]"},
        {"yz_x", "[yz_x]"}, {"yz_y", "[yz_y]"}, {"yz_z", "[yz_z]"},{"yz_w", "[yz_w]"},
        {"zz_x", "[zz_x]"}, {"zz_y", "[zz_y]"}, {"zz_z", "[zz_z]"},{"zz_w", "[zz_w]"},
        {"d", "[d]"},{"w_slider", "[w_slider]"},
    };

    float line_distance = 0.9;
    // Builds the basis scene. Called again later to make an identical copy for fading in the w axis.
    auto make_basis = [&]() {
        shared_ptr<ThreeDimensionScene> b = make_shared<ThreeDimensionScene>(vec2(.32, .4));
        b->manager.set(parentcontrol);
        b->manager.set(bg);
        b->manager.begin_timer("tx");
        b->manager.set({
            {"point_r", "<bg_1_r>"},  {"point_g", "<bg_1_g>"},  {"point_b", "<bg_1_b>"},
            // {"point_r", "<bg_1_r> 1.5 * 40 +"},  {"point_g", "<bg_1_g> 1.5 * 40 +"},  {"point_b", "<bg_1_b> 1.5 * 40 +"},
            {"point_size", "3.2"},
            {"point_labels_size", "1.4"},

            {"point0.x", "1"},  {"point0.y", "0"},  {"point0.z", "0"},   // xx
            {"point1.x", "0"},  {"point1.y", "1"},  {"point1.z", "0"},   // xy
            // {"point2.x", "<xz_x>"},  {"point2.y", "<xz_y>"},  {"point2.z", "<xz_z>"},   // xz
            {"point2.x", "0"},  {"point2.y", "0"},  {"point2.z", "1"},   // xz
            {"point3.x", "-1"}, {"point3.y", "0"},  {"point3.z", "0"},   // yy
            {"point4.x", "<yz_w> 0.5 * <yz_x> +"}, {"point4.y", "<yz_w> 0.5 * <yz_y> +"}, {"point4.z", "<yz_w> 0.5 * <yz_z> +"},   // yz
            {"point5.x", "<zz_w> 0.5 * <zz_x> +"}, {"point5.y", "<zz_w> 0.5 * <zz_y> +"}, {"point5.z", "<zz_w> 0.5 * <zz_z> +"},   // zz

            {"point0.a", "0.4"},{"point1.a", "0.4"},{"point2.a", "0.4"},{"point3.a", "0.4"},{"point4.a", "1"},{"point5.a", "1"},

            {"points_radius_multiplier", "1.8"},
            {"lines_thickness_multiplier", "4"},

            {"slow", "<tx> .08 * 0.2 +"},
            {"d", "4"}, {"q1", "<slow> .7 * sin"}, {"qi", "<slow> sin"}, {"qj", "<slow> 1.3 * cos"},
        });
        b->point_names = {
            {0, "1"}, {1, "i"}, {2, "j"}, {3, "i^2"}, {4, "ij"}, {5, "j^2"},
        };

        b->add_line(Line(vec3(-line_distance, 0, 0), vec3(line_distance, 0, 0), 0xffff7777, 1.0f, false)); // x-axis
        b->add_line(Line(vec3(0, -line_distance, 0), vec3(0, line_distance, 0), 0xff77ff77, 1.0f, false)); // y-axis
        b->add_line(Line(vec3(0, 0, -line_distance), vec3(0, 0, line_distance), 0xff7777ff, 1.0f, false)); // z-axis
        return b;
    };
    shared_ptr<ThreeDimensionScene> basis = make_basis();

    
    shared_ptr<ThreeDAlgebraScene> abAssoc = make_shared<ThreeDAlgebraScene>(vec2(0.5, 0.8));
    abAssoc->brightest_lines = true; // TEMP: fixes grid flicker, revert after this video
    abAssoc->manager.set({
        {"associativity", "1"},{"a_x", "0"},{"a_y", "1"},{"b_x", "0"},{"b_z", "1"},
        {"reach", "3"},
    });
    abAssoc->manager.set(parentcontrol);
    cs.add_scene_fade_in(MICRO, basis, "basis", vec2(-.4, .2));
    cs.add_scene_fade_in(MICRO, abAssoc, "abAssoc",vec2(0,0.45), 1, false, 1);
    cs.slide_subscene(MICRO, "basis", vec2(0.5,0));
    cs.slide_subscene(MICRO, "boxFill", vec2(0.5,0));
    cs.slide_subscene(MICRO, "boxBorder", vec2(0.5,0));
    cs.slide_subscene(MICRO, "abAssoc", vec2(0.5,0));
    cs.fade_subscene(MICRO, "table", 0);
    cs.slide_subscene(MICRO, "table", vec2(0.5,0));

    
    stage_macroblock(FileBlock("so let's wiggle them!"));
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "32"},{"bg_1_b", "60"},});
    cs.render_microblock();
    cs.remove_subscene("table");


    
    stage_macroblock(SilenceBlock(4));

    cs.manager.transition(MICRO, {
        {"yz_x", "0.3"}, {"yz_y", "-0.6 <timer> sin 0.5 * +"}, {"yz_z", "0.3"},
        {"zz_x", "-0.8"}, {"zz_y", "-0.2"}, {"zz_z", "0.1 <timer> 0.93 * sin 0.5 * +"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "40"},{"bg_1_b", "48"},});
    cs.render_microblock();

    cs.manager.transition(MICRO, {
        {"yz_x", "0.3 <timer> sin 0.5 * +"}, {"yz_y", "-0.7 "}, {"yz_z", "0.3"},
        {"zz_x", "-0.8"}, {"zz_y", "-0.2 <timer> 0.93 * sin 0.5 * +"}, {"zz_z", "0.1 "},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "48"},{"bg_1_b", "16"},});
    cs.render_microblock();




    stage_macroblock(FileBlock("Here, the grid is being multiplied by ij"));
    cs.manager.transition(MICRO, {
        {"yz_x", "0.8"}, {"yz_y", "-0.5"}, {"yz_z", "0.2"},
        {"zz_x", "-0.5"}, {"zz_y", "-0.3"}, {"zz_z", "0.8"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "24"},{"bg_1_g", "48"},{"bg_1_b", "0"},});
    cs.render_microblock();

    shared_ptr<LatexScene> abLatex = make_shared<LatexScene>("\\times ij",vec2(0.25,0.25));
    cs.add_scene_fade_in(MICRO, abLatex, "abLatex", vec2(0.5,1));
    cs.slide_subscene(MICRO, "abLatex", vec2(0,-0.1));
    cs.render_microblock();
    

    stage_macroblock(SilenceBlock(3));
    cs.manager.transition(MICRO, {
        {"yz_x", "-0.6"}, {"yz_y", "-0.5"}, {"yz_z", "0.3"},
        {"zz_x", "0"}, {"zz_y", "0.7"}, {"zz_z", "0.6"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "48"},{"bg_1_g", "24"},{"bg_1_b", "0"},});
    cs.render_microblock();
    cs.manager.transition(MICRO, {
        {"yz_x", "-1"}, {"yz_y", "0.2"}, {"yz_z", "0"},
        {"zz_x", "0"}, {"zz_y", "1"}, {"zz_z", "-0.2"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "60"},{"bg_1_g", "0"},{"bg_1_b", "24"},});
    cs.render_microblock();

    shared_ptr<ThreeDAlgebraScene> eaAssoc = make_shared<ThreeDAlgebraScene>(vec2(0.5, 0.8));
    eaAssoc->brightest_lines = true; // TEMP: fixes grid flicker, revert after this video
    eaAssoc->manager.set({
        {"associativity", "0"},
        {"a_x", "1"}, {"a_y", "0"}, {"a_1", "0"},
        {"b_x", "1"}, {"b_y", "0"}, {"b_z", "0"},
        {"reach", "3"},
    });
    eaAssoc->manager.set(parentcontrol);

    cs.add_scene_fade_in(MICRO, eaAssoc, "eaAssoc" ,vec2(0.0,0.49), 1, false, 1);

    stage_macroblock(FileBlock("Strangely, we get a different result if we multiply by i and then j."));
    cs.slide_subscene(MICRO, "abAssoc", vec2(0.26,0.04));
    cs.slide_subscene(MICRO, "abLatex", vec2(0.26,0));
    cs.slide_subscene(MICRO, "eaAssoc", vec2(0.24,0));
    cs.slide_subscene(MICRO, "basis", vec2(0.4,-0.04));
    cs.slide_subscene(MICRO, "boxFill", vec2(0.415,-0.055)); 
    cs.slide_subscene(MICRO, "boxBorder", vec2(0.415,-0.055));
    boxBorder->manager.transition(MICRO,{{"w","0.2"}});
    boxFill->manager.transition(MICRO,{{"w","0.189"}});

    cs.render_microblock();



    shared_ptr<LatexScene> eaLatex = make_shared<LatexScene>("\\times i",vec2(0.25,0.25));
    cs.add_scene_fade_in(MICRO, eaLatex, "eaLatex", vec2(0.24,1));
    cs.slide_subscene(MICRO, "eaLatex", vec2(0,-0.1));

    eaAssoc->manager.transition(MICRO, {
        {"a_x", "0"},
        {"a_y", "1"},
    });
    cs.render_microblock();

    eaLatex->begin_latex_transition(MICRO, "\\times i \\times j");
    eaAssoc->manager.transition(MICRO, {
        {"b_x", "0"},
        {"b_z", "1"},
    });
    cs.render_microblock();




    stage_macroblock(FileBlock("This algebra is not associative, and moving the definitions around doesn't fix it."));
    shared_ptr<LatexScene> neqLatex = make_shared<LatexScene>("\\neq", vec2(0.4, 0.4));
    cs.add_scene_fade_in(MICRO, neqLatex, "neqLatex", vec2(0.5, 0.5));
    cs.render_microblock();
    cs.manager.transition(MICRO, {
        {"yz_x", "-0.7"}, {"yz_y", "0.2"}, {"yz_z", "0.4"},
        {"zz_x", "0.7"}, {"zz_y", "0.3"}, {"zz_z", "-0.4"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "68"},{"bg_1_g", "16"},{"bg_1_b", "0"},});
    cs.render_microblock();

     cs.manager.transition(MICRO, {
        {"yz_x", "-0.1"}, {"yz_y", "0.6"}, {"yz_z", "0.3"},
        {"zz_x", "0.1"}, {"zz_y", "-0.7"}, {"zz_z", "-0.5"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "52"},{"bg_1_g", "0"},{"bg_1_b", "16"},});
    cs.fade_subscene(MICRO, "neqLatex", 0);
    cs.render_microblock();
    cs.remove_subscene("neqLatex");




    stage_macroblock(SilenceBlock(1.5));
    cs.manager.transition(MICRO, {
        {"yz_x", "0.2"}, {"yz_y", "0.5"}, {"yz_z", "-0.6"},
        {"zz_x", "0.9"}, {"zz_y", "0.2"}, {"zz_z", "-0.6"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "48"},{"bg_1_g", "24"},{"bg_1_b", "0"},});
    cs.render_microblock();



    stage_macroblock(FileBlock("Unless... we move that pesky ij into a 4th dimension."));

    cs.render_microblock();
    cs.manager.transition(MICRO, {
        {"yz_x", "0"}, 
        {"yz_y", "0"}, 
        {"yz_z", "0"},
        {"yz_w", "1"},
    });

    // fade in an identical copy of basis that has the w axis, sharing its timer and position so they line up exactly
    const Line w_axis(vec3(-line_distance*0.6), vec3(line_distance*0.6), 0xffcccccc, 1.0f, false);
    shared_ptr<ThreeDimensionScene> basis_w = make_basis();
    basis_w->manager.set("tx", basis->manager.get_equation_string("tx"));
    basis_w->add_line(w_axis);
    cs.add_scene_fade_in(MICRO, basis_w, "basis_w");
    cs.manager.set({
        {"basis_w.x", cs.manager.get_equation_string("basis.x")},
        {"basis_w.y", cs.manager.get_equation_string("basis.y")},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "52"},{"bg_1_b", "0"},});
    cs.manager.transition(MICRO, {
        {"w_slider", "1"},
    });
    cs.render_microblock();

    // give originl basis a w axis too and yeet the duplicate
    basis->add_line(w_axis);
    cs.remove_subscene("basis_w");
    cs.render_microblock();




    stage_macroblock(SilenceBlock(1.2));
    cs.manager.transition(MICRO, {
        {"zz_x", "-0.7"}, {"zz_y", "0"}, {"zz_z", "0"}, {"zz_w", "0.7"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "16"},{"bg_1_g", "48"},{"bg_1_b", "16"},});
    cs.render_microblock();


    stage_macroblock(FileBlock("A fourth-dimensional algebra lets us restore associativity."));
    cs.manager.transition(MICRO, {
        {"zz_x", "-0.4"}, {"zz_y", "-0.3"}, {"zz_z", "0.6"}, {"zz_w", "0.2"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "32"},{"bg_1_b", "48"},});
    cs.render_microblock();

    cs.manager.transition(MICRO, {
        {"zz_x", "-0.3"}, {"zz_y", "0"}, {"zz_z", "0.3"}, {"zz_w", "-0.9"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "24"},{"bg_1_b", "68"},});
    cs.render_microblock();


    stage_macroblock(CompositeBlock(FileBlock("Now, the two sides always match."), SilenceBlock(.8)));
    cs.manager.transition(MICRO, {
        {"zz_x", "-1.1"}, {"zz_y", "-0.4"}, {"zz_z", "-0.2"}, {"zz_w", "-0.3"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "24"},{"bg_1_g", "16"},{"bg_1_b", "72"},});
    cs.render_microblock();

    cs.manager.transition(MICRO, {
        {"zz_x", "-0.3"}, {"zz_y", "0"}, {"zz_z", "0.1"}, {"zz_w", "1"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "40"},{"bg_1_g", "0"},{"bg_1_b", "52"},});
    cs.render_microblock();


    cs.manager.transition(MICRO, {
        {"zz_x", "-0.3"}, {"zz_y", "0"}, {"zz_z", "0.3"}, {"zz_w", "-0.9"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "24"},{"bg_1_g", "0"},{"bg_1_b", "68"},});
    cs.render_microblock();

}




void four_d_cosine(CompositeScene& cs, StateSet& bg){



    StateSet parentcontrol = {
        {"jj_1", "[jj_1]"}, {"jj_i", "[jj_i]"}, {"jj_j", "[jj_j]"},{"jj_ij", "[jj_ij]"}
    };

    cs.manager.set({
        {"jj_1", "-0.3"}, {"jj_i", "-0.7"}, {"jj_j", "0.4"}, {"jj_ij", "0.3"},
    });
    

    shared_ptr<FourDAlgebraScene> fd2 = make_shared<FourDAlgebraScene>();
    cs.add_scene_fade_in(MICRO, fd2, "fd2");

    fd2->manager.set(parentcontrol);
    fd2->manager.set({
        {"brightness", "12000.0"},
        {"offset_1", "0.03732"},
        {"offset_2", "-0.0677"},
        {"scale", "12.0"},
        {"equation", "10"},
        {"slider", "-0.1"},
        {"pov_xz", "-1"},
        {"pov_y", "0.6"},
        {"rotation_jk", "0.0"},
        {"rotation_ik", "-0.43"},
        {"rotation_1k", "0.0"},
    });

    cs.fade_subscene(MICRO,"eaAssoc",0);
    cs.fade_subscene(MICRO,"abAssoc",0);
    cs.fade_subscene(MICRO,"eaLatex",0);
    cs.fade_subscene(MICRO,"abLatex",0);
    cs.fade_subscene(MICRO,"basis",0);
    cs.fade_subscene(MICRO,"boxBorder",0);
    cs.fade_subscene(MICRO,"boxFill",0);
    
    stage_macroblock(FileBlock("Let's check out the 4-dimensional cosine."));
    fd2->manager.transition(MICRO, {
        {"scale", "9.0"},
        {"pov_xz", "-0.2"},
        {"brightness", "100000.0"},
    });
    cs.render_microblock();
    cs.remove_subscene("eaAssoc");
    cs.remove_subscene("abAssoc");
    cs.remove_subscene("eaLatex");
    cs.remove_subscene("abLatex");
    cs.remove_subscene("basis");
    cs.remove_subscene("boxBorder");
    cs.remove_subscene("boxFill");


    fd2->manager.transition(MICRO, {
        {"pov_y", "0.75"},
        {"pov_xz", "0.03"},
    });
    cs.render_microblock();



    stage_macroblock(SilenceBlock(4));
    fd2->manager.transition(MICRO, {
        {"brightness", "200000.0"},
        {"rotation_jk", "0.64"},
    });
    cs.render_microblock();
    fd2->manager.transition(MICRO, {
        {"rotation_1k", "-0.79"},
    });
    cs.render_microblock();

    fd2->manager.transition(MICRO, {
        {"rotation_ik", "0.0"},
    });
    cs.render_microblock();
    cs.render_microblock();



    stage_macroblock(CompositeBlock(SilenceBlock(0.5),FileBlock("Red denotes the i axis,")));
    fd2->manager.transition(MICRO, {
        {"brightness", "80000.0"},
        {"r_channel", "1.2"},
        {"g_channel", "0.1"},
        {"b_channel", "0.1"},
        {"rotation_1k", "-0.9"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "60"},{"bg_1_g", "0"},{"bg_1_b", "16"},});
    cs.render_microblock();
    cs.render_microblock();


    stage_macroblock(CompositeBlock(SilenceBlock(0.5),FileBlock("green is the j,")));
    fd2->manager.transition(MICRO, {
        {"r_channel", "0.1"},
        {"g_channel", "1.2"},
        {"b_channel", "0.1"},
        {"rotation_1k", "-1.1"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "48"},{"bg_1_b", "8"},});
    cs.render_microblock();
    cs.render_microblock();



    stage_macroblock(CompositeBlock(SilenceBlock(0.5),FileBlock("and blue is ij.")));
    fd2->manager.transition(MICRO, {
        {"r_channel", "0.1"},
        {"g_channel", "0.3"},
        {"b_channel", "1.2"},
        {"rotation_1k", "-1.2"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "8"},{"bg_1_g", "16"},{"bg_1_b", "72"},});
    cs.render_microblock();
    cs.render_microblock();



    stage_macroblock(SilenceBlock(4));
    fd2->manager.transition(MICRO, {
        {"brightness", "8000.0"},
        {"r_channel", "1"},
        {"g_channel", "1"},
        {"r_channel", "1"},
        {"rotation_1k", "0"},
        {"rotation_ik", "-0.63"},
        {"rotation_jk", "1.27"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "16"},{"bg_1_b", "72"},});
    cs.render_microblock();
    // TODO too jostly
    fd2->manager.transition(MICRO, {
        {"pov_xz", "0.1"},
        {"rotation_jk", "0.63"},
    });
    cs.render_microblock();

    fd2->manager.transition(MICRO, {
        {"brightness", "200000.0"},
        {"pov_xz", "0.2"},
        {"rotation_ik", "0"},
    });
    cs.render_microblock();



    stage_macroblock(FileBlock("We can explore this graph by moving the camera around,"));
    fd2->manager.transition(MICRO, {
        {"brightness", "8000.0"},
        {"pov_xz", "0"},
        {"pov_y", "0.3"},
    });
    cs.render_microblock();
    fd2->manager.transition(MICRO, {
        {"pov_y", "-0.83"},
        {"pov_xz", "-0.2"},
        {"rotation_jk", "0.83"},
    });
    cs.render_microblock();



    shared_ptr<BackgroundScene> boxBorder = make_shared<BackgroundScene>(vec2(0.27,0.43));
    cs.add_scene_fade_in(MICRO, boxBorder, "boxBorder", vec2(0.09,0.18));
    boxBorder->manager.set(bg);
    boxBorder->manager.set({{"adder","40"}});
    shared_ptr<BackgroundScene> boxFill = make_shared<BackgroundScene>(vec2(0.259,0.412));
    cs.add_scene_fade_in(MICRO, boxFill, "boxFill", vec2(0.09,0.18));
    boxFill->manager.set(bg);

    
    shared_ptr<ThreeDimensionScene> basis = make_shared<ThreeDimensionScene>(vec2(.32, .4));
    basis->manager.set(parentcontrol);
    basis->manager.set(bg);
    basis->manager.begin_timer("tx");
    basis->manager.set({
        {"point_r", "<bg_1_r>"},  {"point_g", "<bg_1_g>"},  {"point_b", "<bg_1_b>"},
        {"point_size", "3.2"},{"point_labels_size", "1.4"},{"points_radius_multiplier", "1.8"},

        {"point0.x", "1"},  {"point0.y", "0"},  {"point0.z", "0"},   // xx
        {"point1.x", "0"},  {"point1.y", "1"},  {"point1.z", "0"},   // xy
        {"point2.x", "0"},  {"point2.y", "0"},  {"point2.z", "1"},   // xz
        {"point3.x", "-1.05"}, {"point3.y", "-0.02"},  {"point3.z", "0"},   // yy
        {"point4.x", "0.5"}, {"point4.y", "0.5"}, {"point4.z", "0.5"},   // yz
        {"point5.x", "<jj_ij> 0.5 * <jj_1> +"}, {"point5.y", "<jj_ij> 0.5 * <jj_i> +"}, {"point5.z", "<jj_ij> 0.5 * <jj_j> +"},   // zz

        {"point0.a", "0.4"},{"point1.a", "0.4"},{"point2.a", "0.4"},{"point3.a", "0.4"},{"point4.a", "1"},{"point5.a", "1"},

        {"slow", "<tx> .08 *"},
        {"lines_thickness_multiplier", "4"},
        {"d", "4"}, {"q1", "<slow> sin"}, {"qi", "<slow> sin .2 *"}, {"qj", "<slow> cos"},
    });
    basis->point_names = {
        {0, "1"}, {1, "i"}, {2, "j"}, {3, "i^2"}, {4, "ij"}, {5, "j^2"},
    };

    float line_distance = 0.9;
    basis->add_line(Line(vec3(-line_distance, 0, 0), vec3(line_distance, 0, 0), 0xffff7777, 1.0f, false)); // x-axis
    basis->add_line(Line(vec3(0, -line_distance, 0), vec3(0, line_distance, 0), 0xff77ff77, 1.0f, false)); // y-axis
    basis->add_line(Line(vec3(0, 0, -line_distance), vec3(0, 0, line_distance), 0xff7777ff, 1.0f, false)); // z-axis
    basis->add_line(Line(vec3(-line_distance*0.6), vec3(line_distance*0.6), 0xffcccccc, 1.0f, false)); // w-axis

    cs.add_scene_fade_in(MICRO, basis, "basis", vec2(.11, .21));

    stage_macroblock(FileBlock("Or we could change the definition of j^2, to move through algebra space."));
    cs.render_microblock();

    cs.manager.transition(MICRO, {
        {"jj_1", "0.7"},
        {"jj_i", "0.2"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "32"},{"bg_1_b", "52"},});
    cs.render_microblock();



    stage_macroblock(SilenceBlock(8));
    fd2->manager.transition(MICRO, {
        {"rotation_1k", "0.63"},
        {"brightness", "12000.0"},
    });
    cs.manager.transition(MICRO, {
        {"jj_j", "-0.1"},
        {"jj_ij", "-0.9"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "8"},{"bg_1_g", "48"},{"bg_1_b", "24"},});
    cs.render_microblock();

    cs.manager.transition(MICRO, {
        {"jj_i", "-0.6"},
        {"jj_j", "-0.6"},
    });
    fd2->manager.transition(MICRO, {
        {"rotation_jk", "0.0"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "16"},{"bg_1_b", "68"},});
    cs.render_microblock();


    cs.manager.transition(MICRO, {
        {"jj_1", "-1"},
        {"jj_ij", "0"},
    });
    fd2->manager.transition(MICRO, {
        {"pov_xz", "0"},
        {"rotation_ik", "0"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "24"},{"bg_1_g", "0"},{"bg_1_b", "72"},});
    cs.render_microblock();


    cs.manager.transition(MICRO, {
        {"jj_i", "0"},
        {"jj_j", "0"},
    });
    fd2->manager.transition(MICRO, {
        {"rotation_1k", "-0.63"},
        {"offset_1", "0.0"},
        {"pov_xz", "0.23"},
    });
    cs.manager.transition(MICRO,{{"bg_1_r", "48"},{"bg_1_g", "0"},{"bg_1_b", "60"},});
    cs.render_microblock();

    stage_macroblock(FileBlock("Here, j squares to negative 1, just like the imaginary i."));

    fd2->manager.transition(MICRO, {
        {"pov_y", "0.1"},
        {"brightness", "80000.0"},
    });
    cs.render_microblock();

    fd2->manager.transition(MICRO, {
        {"pov_xz", "1.6"},
        {"pov_y", "-0.1"},
    });
    cs.render_microblock();



    // TODO camera looking nearly straight down the grid
    shared_ptr<ThreeDAlgebraScene> tess = make_shared<ThreeDAlgebraScene>(vec2(1,1));
    tess->manager.set({
        {"associativity", "1"},
        {"reach", "2"},
        {"a_x", "1"}, {"a_y", "0"}, {"a_z", "0"},
        {"b_x", "1"}, {"b_y", "0"}, {"b_z", "0"},
        {"yz_x", "0"}, {"yz_y", "0"}, {"yz_z", "0"}, {"yz_w", "1"},
        {"zz_x", "-1"}, {"zz_y", "0"}, {"zz_z", "0"}, {"zz_w", "0"},
        {"d", "20"},{"w_slider", "1"},

    });
    cs.add_scene_fade_in(MICRO, tess, "tess");
    cs.fade_subscene(MICRO, "fd2", 0);


    stage_macroblock(FileBlock("The grids can spin in the i dimension,"));
    
    cs.render_microblock();
    tess->manager.transition(MICRO, {
        {"a_x", "0"}, 
        {"a_y", "1"}, 
    });
    cs.render_microblock();

    tess->manager.transition(MICRO, {
        {"a_x", "-1"}, 
        {"a_y", "0"}, 
    });
    cs.render_microblock();




    stage_macroblock(SilenceBlock(0.6));
    cs.render_microblock();

    stage_macroblock(CompositeBlock(FileBlock("and in the j dimension."),SilenceBlock(1)));
    
    tess->manager.transition(MICRO, {
        {"a_x", "0"}, 
        {"a_z", "-1"}, 
    });
    cs.render_microblock();

    tess->manager.transition(MICRO, {
        {"a_x", "1"}, 
        {"a_z", "0"}, 
    });
    cs.render_microblock();



    shared_ptr<PngScene> dub_png = make_shared<PngScene>("bicomplex",vec2(0.24,0.12));
    cs.add_scene(dub_png, "dub_png",vec2(0.11,-0.06)); 
    


    stage_macroblock(FileBlock("This is the double spinny algebra!"));

    cs.slide_subscene(MICRO, "dub_png",vec2(0,0.16)); 
    cs.slide_subscene(MICRO, "basis",vec2(0,0.15)); 
    basis->manager.transition(MICRO, {
        {"point0.a", "1"}, 
        {"point1.a", "1"}, 
        {"point2.a", "1"}, 
        {"point3.a", "1"}, 
        {"point4.a", "1"}, 
        {"point5.a", "1"}, 
    });
    boxBorder->manager.transition(MICRO,{{"h","0.69"}});
    boxFill->manager.transition(MICRO,{{"h","0.672"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(FileBlock("But it doesn't just spin."));
    
    tess->manager.transition(MICRO, {
        {"a_x", "0"}, 
        {"a_y", "1"}, 
    });
    cs.render_microblock();

    tess->manager.transition(MICRO, {
        {"a_x", "-1"}, 
        {"a_y", "0"}, 
    });
    cs.render_microblock();



    stage_macroblock(FileBlock("It flips too!"));
    
    tess->manager.transition(MICRO, {
        {"a_x", "0"}, 
        {"a_w", "1"}, 
    });
    cs.render_microblock();

    tess->manager.transition(MICRO, {
        {"a_x", "-1"}, 
        {"a_w", "0"}, 
    });
    cs.render_microblock();


    stage_macroblock(FileBlock("And flipping is bad."));
    cs.render_microblock();
    cs.render_microblock();

    cs.fade_subscene(MICRO, "tess", 0.3);
    cs.fade_subscene(MICRO, "basis", 0.3);
    cs.fade_subscene(MICRO, "boxBorder", 0.3);
    cs.fade_subscene(MICRO, "dub_png", 0.3);

    shared_ptr<PngScene> flipping_text = make_shared<PngScene>("flipping",vec2(0.7,0.7));
    cs.add_scene_fade_in(MICRO, flipping_text, "flipping_text",vec2(0.5,0.44)); 
    cs.slide_subscene(MICRO, "flipping_text", vec2(0,0.1));
    cs.render_microblock();
    cs.render_microblock();
    shared_ptr<PngScene> is_text = make_shared<PngScene>("is",vec2(0.7,0.7));
    cs.add_scene_fade_in(MICRO, is_text, "is_text",vec2(0.5,0.44)); 
    cs.slide_subscene(MICRO, "is_text", vec2(0,0.1));
    cs.render_microblock();
    shared_ptr<PngScene> bad_text = make_shared<PngScene>("bad",vec2(0.7,0.7));
    cs.add_scene_fade_in(MICRO, bad_text, "bad_text",vec2(0.5,0.44)); 
    cs.slide_subscene(MICRO, "bad_text", vec2(0,0.1));
    cs.render_microblock();
    cs.render_microblock();


    dub_png->manager.transition(MICRO,{{"w","0.30"},{"h","0.14"}});
    cs.slide_subscene(MICRO, "tess", vec2(-0.5,0));
    cs.slide_subscene(MICRO, "basis", vec2(-0.5,0));
    cs.slide_subscene(MICRO, "boxBorder", vec2(-0.5,0));
    cs.slide_subscene(MICRO, "boxFill", vec2(-0.5,0));
    cs.slide_subscene(MICRO, "dub_png", vec2(0.4,0.05));
    cs.slide_subscene(MICRO, "flipping_text", vec2(-0.5,0));
    cs.slide_subscene(MICRO, "is_text", vec2(-0.5,0));
    cs.slide_subscene(MICRO, "bad_text", vec2(-0.5,0));


    cs.fade_subscene(MICRO, "tess", 0);
    cs.fade_subscene(MICRO, "basis", 0);
    cs.fade_subscene(MICRO, "boxBorder", 0);
    cs.fade_subscene(MICRO, "flipping_text", 0);
    cs.fade_subscene(MICRO, "is_text", 0);
    cs.fade_subscene(MICRO, "bad_text", 0);
    cs.fade_subscene(MICRO, "dub_png", 1);


}


void non_commutative(CompositeScene& cs, StateSet& bg){

    shared_ptr<MultiplicationTableScene> table = make_shared<MultiplicationTableScene>(vec2(0.8,0.7));
    cs.add_scene_fade_in(MICRO, table, "table", vec2(1,0.55));
    cs.slide_subscene(MICRO, "table", vec2(-0.5,0));

    table->manager.set(bg);


    stage_macroblock(CompositeBlock(SilenceBlock(1), FileBlock("Here's the multiplication table.")));
    cs.render_microblock();
    cs.render_microblock();
    cs.remove_subscene("tess");
    cs.remove_subscene("boxFill");
    cs.remove_subscene("basis");
    cs.remove_subscene("boxBorder");
    cs.remove_subscene("flipping_text");
    cs.remove_subscene("is_text");
    cs.remove_subscene("bad_text");
    
    
    shared_ptr<PngScene> flippy_png = make_shared<PngScene>("flippy",vec2(0.28,0.14));
    cs.add_scene(flippy_png, "flippy_png",vec2(0.51,-0.06)); 

    stage_macroblock(FileBlock("The space of just 1 and ij forms the flippy algebra,"));

    table->manager.transition(MICRO, {
        {"opacity_02", "0"},{"opacity_03", "0"},
        {"opacity_12", "0"},{"opacity_13", "0"},
        {"opacity_20", "0"},{"opacity_21", "0"},{"opacity_22", "0"},{"opacity_23", "0"},{"opacity_24", "0"},
        {"opacity_30", "0"},{"opacity_31", "0"},{"opacity_32", "0"},{"opacity_33", "0"},{"opacity_34", "0"},
        {"opacity_42", "0"},{"opacity_43", "0"},
    });
    cs.fade_subscene(MICRO, "dub_png", 0.1);
    cs.render_microblock();

    cs.slide_subscene(MICRO, "flippy_png",vec2(0,0.2)); 
    cs.slide_subscene(MICRO, "dub_png",vec2(0,-0.2)); 
    cs.render_microblock();
    cs.remove_subscene("dub_png");


    stage_macroblock(CompositeBlock(FileBlock("because ij squares to 1."),SilenceBlock(1)));
    // TODO will they remember the significance of this?

    table->manager.transition(MICRO, {
        {"opacity_01", "0.1"},{"opacity_10", "0.1"},
        {"opacity_11", "0.1"},{"opacity_14", "0.1"},{"opacity_41", "0.1"},
        {"opacity_02", "0.1"},{"opacity_03", "0.1"},
        {"opacity_12", "0.1"},{"opacity_13", "0.1"},
        {"opacity_20", "0.1"},{"opacity_21", "0.1"},{"opacity_22", "0.1"},{"opacity_23", "0.1"},{"opacity_24", "0.1"},
        {"opacity_30", "0.1"},{"opacity_31", "0.1"},{"opacity_32", "0.1"},{"opacity_33", "0.1"},{"opacity_34", "0.1"},
        {"opacity_42", "0.1"},{"opacity_43", "0.1"},
    });
    cs.render_microblock();
    cs.render_microblock();



    stage_macroblock(FileBlock("We can prove it, with some algebra."));

    cs.slide_subscene(MICRO, "table",vec2(-0.23,0)); 
    cs.slide_subscene(MICRO, "flippy_png",vec2(-0.23,0)); 
    
    cs.fade_subscene(MICRO,"table",0.7);
    cs.fade_subscene(MICRO,"flippy_png",0.7);


	shared_ptr<LatexScene> latex_2 = make_shared<LatexScene>("2",vec2(0.046,0.107));
	cs.add_scene(latex_2 , "latex_2",vec2(0.94,0.165));
	cs.manager.set({{"latex_2.opacity","0"}});
	shared_ptr<LatexScene> latex__10 = make_shared<LatexScene>("(",vec2(0.052,0.151));
	cs.add_scene(latex__10 , "latex__10",vec2(0.861,0.193));
	cs.manager.set({{"latex__10.opacity","0"}});
	shared_ptr<LatexScene> latex_i = make_shared<LatexScene>("i",vec2(0.053,0.19));
	cs.add_scene(latex_i , "latex_i",vec2(0.88,0.193));
	cs.manager.set({{"latex_i.opacity","0"}});
	shared_ptr<LatexScene> latex_j = make_shared<LatexScene>("j",vec2(0.053,0.19));
	cs.add_scene(latex_j , "latex_j",vec2(0.897,0.198));
	cs.manager.set({{"latex_j.opacity","0"}});
	shared_ptr<LatexScene> latex__11 = make_shared<LatexScene>(")",vec2(0.052,0.151));
	cs.add_scene(latex__11 , "latex__11",vec2(0.916,0.193));
	cs.manager.set({{"latex__11.opacity","0"}});

	
	cs.slide_subscene(MICRO,"latex_2",vec2(-0.206,-0.001));
	cs.manager.transition(MICRO,{{"latex_2.opacity","1"}});
	cs.slide_subscene(MICRO,"latex__10",vec2(-0.206,-0.001));
	cs.manager.transition(MICRO,{{"latex__10.opacity","1"}});
	cs.slide_subscene(MICRO,"latex_i",vec2(-0.206,-0.001));
	cs.manager.transition(MICRO,{{"latex_i.opacity","1"}});
	cs.slide_subscene(MICRO,"latex_j",vec2(-0.206,-0.001));
	cs.manager.transition(MICRO,{{"latex_j.opacity","1"}});
	cs.slide_subscene(MICRO,"latex__11",vec2(-0.206,-0.001));
	cs.manager.transition(MICRO,{{"latex__11.opacity","1"}});

    cs.render_microblock();
    cs.render_microblock();



    stage_macroblock(SilenceBlock(4));

	shared_ptr<LatexScene> latex_i2 = make_shared<LatexScene>("i",vec2(0.053,0.19));
	cs.add_scene(latex_i2 , "latex_i2",vec2(0.675,0.192));
	shared_ptr<LatexScene> latex_j2 = make_shared<LatexScene>("j",vec2(0.053,0.19));
	cs.add_scene(latex_j2 , "latex_j2",vec2(0.691,0.198));
	shared_ptr<LatexScene> latex__12 = make_shared<LatexScene>("=",vec2(0.098,0.151));
	cs.add_scene(latex__12 , "latex__12",vec2(0.626,0.196));
	cs.manager.set({{"latex__12.opacity","0"}});
	shared_ptr<LatexScene> latex_i3 = make_shared<LatexScene>("i",vec2(0.053,0.19));
	cs.add_scene(latex_i3 , "latex_i3",vec2(0.675,0.192));
	shared_ptr<LatexScene> latex_j3 = make_shared<LatexScene>("j",vec2(0.053,0.19));
	cs.add_scene(latex_j3 , "latex_j3",vec2(0.691,0.198));

	cs.slide_subscene(MICRO,"latex__12",vec2(-0.007,0.152));
	cs.manager.transition(MICRO,{{"latex__12.opacity","1"}});
	cs.slide_subscene(MICRO,"latex_i2",vec2(-0.016,0.152));
	cs.slide_subscene(MICRO,"latex_j2",vec2(-0.01,0.15));
	cs.slide_subscene(MICRO,"latex_i3",vec2(0.036,0.152));
	cs.slide_subscene(MICRO,"latex_j3",vec2(0.042,0.15));

    cs.render_microblock();


	shared_ptr<LatexScene> latex__14 = make_shared<LatexScene>("=",vec2(0.098,0.151));
	cs.add_scene(latex__14 , "latex__14",vec2(0.619,0.348));
	shared_ptr<LatexScene> latex_i4 = make_shared<LatexScene>("i",vec2(0.053,0.19));
	cs.add_scene(latex_i4 , "latex_i4",vec2(0.659,0.344));
	shared_ptr<LatexScene> latex_j4 = make_shared<LatexScene>("j",vec2(0.053,0.19));
	cs.add_scene(latex_j4 , "latex_j4",vec2(0.681,0.348));
	shared_ptr<LatexScene> latex_i5 = make_shared<LatexScene>("i",vec2(0.053,0.19));
	cs.add_scene(latex_i5 , "latex_i5",vec2(0.711,0.344));
	shared_ptr<LatexScene> latex_j5 = make_shared<LatexScene>("j",vec2(0.053,0.19));
	cs.add_scene(latex_j5 , "latex_j5",vec2(0.733,0.348));

	cs.slide_subscene(MICRO,"latex__14",vec2(0,0.138));
	cs.slide_subscene(MICRO,"latex_i2",vec2(0,0.138));
	cs.slide_subscene(MICRO,"latex_j2",vec2(0.027,0.138));
	cs.slide_subscene(MICRO,"latex_i3",vec2(-0.03,0.138));
	cs.slide_subscene(MICRO,"latex_j3",vec2(0,0.138));

    cs.render_microblock();



	shared_ptr<LatexScene> latex__15 = make_shared<LatexScene>("=",vec2(0.098,0.151));
	cs.add_scene(latex__15 , "latex__15",vec2(0.619,0.486));
	shared_ptr<LatexScene> latex_i6 = make_shared<LatexScene>("i",vec2(0.053,0.19));
	cs.add_scene(latex_i6 , "latex_i6",vec2(0.659,0.482));
	shared_ptr<LatexScene> latex_j6 = make_shared<LatexScene>("j",vec2(0.053,0.19));
	cs.add_scene(latex_j6 , "latex_j6",vec2(0.708,0.486));
	shared_ptr<LatexScene> latex_i7 = make_shared<LatexScene>("i",vec2(0.053,0.19));
	cs.add_scene(latex_i7 , "latex_i7",vec2(0.681,0.482));
	shared_ptr<LatexScene> latex_j7 = make_shared<LatexScene>("j",vec2(0.053,0.19));
	cs.add_scene(latex_j7 , "latex_j7",vec2(0.733,0.486));
	shared_ptr<LatexScene> latex_1 = make_shared<LatexScene>("(-1)",vec2(0.16,0.113));
	cs.add_scene(latex_1 , "latex_1",vec2(0.662,0.487));
	cs.manager.set({{"latex_1.opacity","0"}});
	shared_ptr<LatexScene> latex_12 = make_shared<LatexScene>("(-1)",vec2(0.16,0.113));
	cs.add_scene(latex_12 , "latex_12",vec2(0.724,0.483));
	cs.manager.set({{"latex_12.opacity","0"}});


	cs.slide_subscene(MICRO,"latex__15",vec2(-0.019,0.139));
	cs.slide_subscene(MICRO,"latex_i6",vec2(0.013,0.139));
	cs.manager.transition(MICRO,{{"latex_i6.opacity","0"}});
	cs.slide_subscene(MICRO,"latex_j6",vec2(0.007,0.139));
	cs.manager.transition(MICRO,{{"latex_j6.opacity","0"}});
	cs.slide_subscene(MICRO,"latex_i7",vec2(-0.008,0.139));
	cs.manager.transition(MICRO,{{"latex_i7.opacity","0"}});
	cs.slide_subscene(MICRO,"latex_j7",vec2(-0.015,0.139));
	cs.manager.transition(MICRO,{{"latex_j7.opacity","0"}});
	cs.slide_subscene(MICRO,"latex_1",vec2(0.001,0.143));
	cs.manager.transition(MICRO,{{"latex_1.opacity","1"}});
	cs.slide_subscene(MICRO,"latex_12",vec2(0.001,0.146));
	cs.manager.transition(MICRO,{{"latex_12.opacity","1"}});
	

    cs.render_microblock();



	shared_ptr<LatexScene> latex__16 = make_shared<LatexScene>("=",vec2(0.098,0.151));
	cs.add_scene(latex__16 , "latex__16",vec2(0.6,0.625));
	shared_ptr<LatexScene> latex_13 = make_shared<LatexScene>("(-1)",vec2(0.16,0.113));
	cs.add_scene(latex_13 , "latex_13",vec2(0.663,0.63));
	shared_ptr<LatexScene> latex_14 = make_shared<LatexScene>("(-1)",vec2(0.16,0.113));
	cs.add_scene(latex_14 , "latex_14",vec2(0.725,0.63));
	shared_ptr<LatexScene> latex_15 = make_shared<LatexScene>("1",vec2(0.058,0.134));
	cs.add_scene(latex_15 , "latex_15",vec2(0.693,0.632));
	cs.manager.set({{"latex_15.opacity","0"}});

	
	cs.slide_subscene(MICRO,"latex__16",vec2(0.031,0.13));
	cs.slide_subscene(MICRO,"latex_13",vec2(0.025,0.131));
	cs.manager.transition(MICRO,{{"latex_13.opacity","0"}});
	cs.slide_subscene(MICRO,"latex_14",vec2(-0.036,0.131));
	cs.manager.transition(MICRO,{{"latex_14.opacity","0"}});
	cs.slide_subscene(MICRO,"latex_15",vec2(-0.003,0.122));
	cs.manager.transition(MICRO,{{"latex_15.opacity","1"}});
	
	cs.fade_subscene(MICRO,"latex_j7",0);
	cs.fade_subscene(MICRO,"latex_i7",0);
	cs.fade_subscene(MICRO,"latex_j6",0);
	cs.fade_subscene(MICRO,"latex_i6",0);
	

    cs.render_microblock();
	cs.remove_subscene("latex_j7");
	cs.remove_subscene("latex_i7");
	cs.remove_subscene("latex_j6");
	cs.remove_subscene("latex_i6");
    cs.render_microblock();

    
    stage_macroblock(FileBlock("So to prevent flipping, we must prevent this algebra."));
    cs.render_microblock();
    cs.fade_subscene(MICRO,"table",0);
    cs.fade_subscene(MICRO,"flippy_png",0);
	cs.slide_subscene(MICRO,"table",vec2(-0.15,0));
	cs.slide_subscene(MICRO,"flippy_png",vec2(-0.15,0));
    vec2 leftshift = vec2(-0.2,0);
	cs.slide_subscene(MICRO,"latex_2",leftshift);
	cs.slide_subscene(MICRO,"latex__10",leftshift);
	cs.slide_subscene(MICRO,"latex_i",leftshift);
	cs.slide_subscene(MICRO,"latex_j",leftshift);
	cs.slide_subscene(MICRO,"latex__11",leftshift);
	cs.slide_subscene(MICRO,"latex__12",leftshift);
	cs.slide_subscene(MICRO,"latex__14",leftshift);
	cs.slide_subscene(MICRO,"latex_i4",leftshift);
	cs.slide_subscene(MICRO,"latex_j4",leftshift);
	cs.slide_subscene(MICRO,"latex_i5",leftshift);
	cs.slide_subscene(MICRO,"latex_j5",leftshift);
	cs.slide_subscene(MICRO,"latex_i2",leftshift);
	cs.slide_subscene(MICRO,"latex_j2",leftshift);
	cs.slide_subscene(MICRO,"latex_i3",leftshift);
	cs.slide_subscene(MICRO,"latex_j3",leftshift);
	cs.slide_subscene(MICRO,"latex__15",leftshift);
	cs.slide_subscene(MICRO,"latex_1",leftshift);
	cs.slide_subscene(MICRO,"latex_12",leftshift);
	cs.slide_subscene(MICRO,"latex__16",leftshift);
	cs.slide_subscene(MICRO,"latex_15",leftshift);
    cs.render_microblock();


    stage_macroblock(FileBlock("How about this step, where i and j swap sides?"));


	shared_ptr<LatexScene> latex_a = make_shared<LatexScene>("\\longrightarrow",vec2(0.157,0.076));
	cs.add_scene(latex_a , "latex_a",vec2(0.498,0.42));
	cs.manager.set({{"latex_a.opacity","0"},{"latex_a.angle","1.5708"}});
	
	
	cs.slide_subscene(MICRO,"latex_2",vec2(0,-0.033));
	cs.manager.transition(MICRO,{{"latex_2.opacity","0.2"}});
	cs.slide_subscene(MICRO,"latex__10",vec2(0,-0.033));
	cs.manager.transition(MICRO,{{"latex__10.opacity","0.2"}});
	cs.slide_subscene(MICRO,"latex_i",vec2(0,-0.033));
	cs.manager.transition(MICRO,{{"latex_i.opacity","0.2"}});
	cs.slide_subscene(MICRO,"latex_j",vec2(0,-0.033));
	cs.manager.transition(MICRO,{{"latex_j.opacity","0.2"}});
	cs.slide_subscene(MICRO,"latex__11",vec2(0,-0.033));
	cs.manager.transition(MICRO,{{"latex__11.opacity","0.2"}});
	cs.slide_subscene(MICRO,"latex__12",vec2(0,-0.033));
	cs.slide_subscene(MICRO,"latex__14",vec2(0,0.033));
	cs.slide_subscene(MICRO,"latex_i4",vec2(0,-0.033));
	cs.slide_subscene(MICRO,"latex_j4",vec2(0,-0.033));
	cs.slide_subscene(MICRO,"latex_i5",vec2(0,-0.033));
	cs.slide_subscene(MICRO,"latex_j5",vec2(0,-0.033));
	cs.slide_subscene(MICRO,"latex_i2",vec2(0,0.033));
	cs.slide_subscene(MICRO,"latex_j2",vec2(0,0.033));
	cs.slide_subscene(MICRO,"latex_i3",vec2(0,0.033));
	cs.slide_subscene(MICRO,"latex_j3",vec2(0,0.033));
	cs.slide_subscene(MICRO,"latex__15",vec2(0,0.033));
	cs.manager.transition(MICRO,{{"latex__15.opacity","0.2"}});
	cs.slide_subscene(MICRO,"latex_1",vec2(0,0.033));
	cs.manager.transition(MICRO,{{"latex_1.opacity","0.2"}});
	cs.slide_subscene(MICRO,"latex_12",vec2(0,0.033));
	cs.manager.transition(MICRO,{{"latex_12.opacity","0.2"}});
	cs.slide_subscene(MICRO,"latex__16",vec2(0,0.033));
	cs.manager.transition(MICRO,{{"latex__16.opacity","0.2"}});
	cs.slide_subscene(MICRO,"latex_15",vec2(0,0.033));
	cs.manager.transition(MICRO,{{"latex_15.opacity","0.2"}});
	latex_a->manager.transition(MICRO,{{"w","0.216"},{"h","0.093"}});
	latex_a->begin_latex_transition(MICRO, "\\longrightarrow");
	cs.manager.transition(MICRO,{{"latex_a.opacity","1"}});
	cs.fade_subscene(MICRO,"latex_14",0);
	cs.fade_subscene(MICRO,"latex_13",0);
	cs.fade_subscene(MICRO,"latex__14",0.2);
	cs.fade_subscene(MICRO,"latex__12",0.2);


    cs.render_microblock();
	cs.remove_subscene("latex_14");
	cs.remove_subscene("latex_13");
	

	shared_ptr<LatexScene> latex_a2 = make_shared<LatexScene>("\\longrightarrow",vec2(0.216,0.093));
	cs.add_scene(latex_a2 , "latex_a2",vec2(0.498,0.42));
	cs.manager.set({{"latex_a2.angle","1.5708"}});

	cs.manager.transition(MICRO,{{"latex_i4.opacity","0.2"}});
	cs.manager.transition(MICRO,{{"latex_j5.opacity","0.2"}});
	cs.manager.transition(MICRO,{{"latex_i2.opacity","0.2"}});
	cs.manager.transition(MICRO,{{"latex_j3.opacity","0.2"}});

	cs.manager.transition(MICRO,{{"latex_a.angle","1.9708"}});
	cs.manager.transition(MICRO,{{"latex_a2.angle","1.1708"}});
    cs.render_microblock();




    stage_macroblock(FileBlock("We can stop the swap by saying that ij is not ji."));
    cs.render_microblock();



	latex_j4->manager.transition(MICRO,{{"w","0.08"},{"h","0.3"}});
	latex_j4->begin_latex_transition(MICRO, "j");
	cs.slide_subscene(MICRO,"latex_j4",vec2(0.066,0.097));
	latex_i5->manager.transition(MICRO,{{"w","0.08"},{"h","0.3"}});
	latex_i5->begin_latex_transition(MICRO, "i");
	cs.slide_subscene(MICRO,"latex_i5",vec2(0.067,0.094));
	latex_j2->manager.transition(MICRO,{{"w","0.08"},{"h","0.3"}});
	latex_j2->begin_latex_transition(MICRO, "j");
	cs.slide_subscene(MICRO,"latex_j2",vec2(-0.077,-0.107));
	latex_i3->manager.transition(MICRO,{{"w","0.08"},{"h","0.3"}});
	latex_i3->begin_latex_transition(MICRO, "i");
	cs.slide_subscene(MICRO,"latex_i3",vec2(-0.073,-0.11));

	cs.manager.transition(MICRO,{{"latex_a2.angle","1.9708"},{"latex_a.angle","1.9708"}});
	cs.slide_subscene(MICRO,"latex_a2",vec2(-0.01,-0.01));
	cs.slide_subscene(MICRO,"latex_a",vec2(-0.01,-0.01));
	

	shared_ptr<LatexScene> latex__25 = make_shared<LatexScene>("\\neq",vec2(0.125,0.2));
	cs.add_scene_fade_in(MICRO, latex__25 , "latex__25",vec2(0.5,0.419));
	cs.slide_subscene(MICRO,"latex__25",vec2(-0.01,-0.01));
	
	cs.fade_subscene(MICRO,"latex_a2",0);
	cs.fade_subscene(MICRO,"latex_a",0);
	cs.fade_subscene(MICRO,"latex_15",0);
	cs.fade_subscene(MICRO,"latex__24",0);
	cs.fade_subscene(MICRO,"latex_12",0);
	cs.fade_subscene(MICRO,"latex_1",0);
	cs.fade_subscene(MICRO,"latex__23",0);
	cs.fade_subscene(MICRO,"latex_j5",0);
	cs.fade_subscene(MICRO,"latex_i4",0);
	cs.fade_subscene(MICRO,"latex_j3",0);
	cs.fade_subscene(MICRO,"latex_i2",0);
	cs.fade_subscene(MICRO,"latex__19",0);
	cs.fade_subscene(MICRO,"latex_j",0);
	cs.fade_subscene(MICRO,"latex_i",0);
	cs.fade_subscene(MICRO,"latex__18",0);
	cs.fade_subscene(MICRO,"latex_2",0);
	cs.fade_subscene(MICRO,"latex__14",0);
	cs.fade_subscene(MICRO,"latex__12",0);
	cs.fade_subscene(MICRO,"latex__10",0);
	cs.fade_subscene(MICRO,"latex__11",0);
	cs.fade_subscene(MICRO,"latex__15",0);
	cs.fade_subscene(MICRO,"latex__16",0);
	
	
    cs.render_microblock();
	
	cs.remove_subscene("latex_a2");
	cs.remove_subscene("latex_a");
	cs.remove_subscene("latex_15");
	cs.remove_subscene("latex__24");
	cs.remove_subscene("latex_12");
	cs.remove_subscene("latex_1");
	cs.remove_subscene("latex__23");
	cs.remove_subscene("latex_j5");
	cs.remove_subscene("latex_i4");
	cs.remove_subscene("latex_j3");
	cs.remove_subscene("latex_i2");
	cs.remove_subscene("latex__19");
	cs.remove_subscene("latex_j");
	cs.remove_subscene("latex_i");
	cs.remove_subscene("latex__18");
	cs.remove_subscene("latex_2");
	
	
    cs.render_microblock();

    shared_ptr<FourDAlgebraScene> rot2 = make_shared<FourDAlgebraScene>(vec2(1,1.1));
    cs.add_scene_fade_in(MICRO, rot2, "rot2", vec2(0.7,0.55));

    rot2->manager.set({
        {"offset_1", "0.0"},
        {"offset_2", "0.0"},
        {"slider", "1.1"},
        {"rotate", "1"},
        {"scale", "5.0"},
        {"pov_xz", "0.1"},
        {"pov_y", "0.2"},
    });


    shared_ptr<FourDAlgebraScene> rot1 = make_shared<FourDAlgebraScene>(vec2(1,1.1));
    cs.add_scene_fade_in(MICRO, rot1, "rot1", vec2(0.3,0.55));

    rot1->manager.set({
        {"offset_1", "0.0"},
        {"offset_2", "0.0"},
        {"slider", "1.1"},
        {"rotate", "1"},
        {"scale", "5.0"},
        {"pov_xz", "0.1"},
        {"pov_y", "0.2"},
    });


    cs.fade_subscene(MICRO,"latex_i3",0.3);
    cs.fade_subscene(MICRO,"latex_j2",0.3);
    cs.fade_subscene(MICRO,"latex_j4",0.3);
    cs.fade_subscene(MICRO,"latex_i5",0.3);



    stage_macroblock(FileBlock("This seems weird, but it's how 3D rotations behave."));
	latex_i3->manager.transition(MICRO,{{"w","0.1"},{"h","0.28"}});
	latex_i3->begin_latex_transition(MICRO, "i");
	cs.slide_subscene(MICRO,"latex_i3",vec2(-0.135,-0.24));

	latex_j2->manager.transition(MICRO,{{"w","0.1"},{"h","0.28"}});
	latex_j2->begin_latex_transition(MICRO, "j");
	cs.slide_subscene(MICRO,"latex_j2",vec2(-0.1,-0.24));

	latex__25->manager.transition(MICRO,{{"w","0.1"},{"h","0.28"}});
	latex__25->begin_latex_transition(MICRO, "\\neq");
	cs.slide_subscene(MICRO,"latex__25",vec2(0.01,-0.24));

	latex_j4->manager.transition(MICRO,{{"w","0.1"},{"h","0.28"}});
	latex_j4->begin_latex_transition(MICRO, "j");
	cs.slide_subscene(MICRO,"latex_j4",vec2(0.12,-0.24));

	latex_i5->manager.transition(MICRO,{{"w","0.1"},{"h","0.28"}});
	latex_i5->begin_latex_transition(MICRO, "i");
	cs.slide_subscene(MICRO,"latex_i5",vec2(0.15,-0.24));


    cs.render_microblock();
    cs.remove_subscene("flippy_png");

    cs.render_microblock();




    stage_macroblock(FileBlock("Up then left... is not the same as left then up."));

    rot1->manager.transition(MICRO, {
        {"rot_1", "0.7071"},
        {"rot_i", "0.7071"},
    });
	shared_ptr<LatexScene> latex_up1 = make_shared<LatexScene>("\\Uparrow",vec2(0.17,0.17));
	cs.add_scene_fade_in(MICRO, latex_up1 , "latex_up1",vec2(0.275,0.21));
	cs.slide_subscene(MICRO,"latex_up1",vec2(0,-0.04));
    cs.fade_subscene(MICRO,"latex_i3",0);
	cs.slide_subscene(MICRO,"latex_i3",vec2(0,-0.01));
    cs.render_microblock();

    rot1->manager.transition(MICRO, {
        {"rot_1", "0.5"},
        {"rot_i", "0.5"},
        {"rot_j", "0.5"},
        {"rot_k", "-0.5"},
    });
	shared_ptr<LatexScene> latex_left1 = make_shared<LatexScene>("\\Leftarrow",vec2(0.12,0.12));
	cs.add_scene_fade_in(MICRO, latex_left1 , "latex_left1",vec2(0.37,0.17));
	cs.slide_subscene(MICRO,"latex_left1",vec2(-0.03,0));
    cs.fade_subscene(MICRO,"latex_j2",0);
	cs.slide_subscene(MICRO,"latex_j2",vec2(-0.01,0));

    cs.render_microblock();

    cs.render_microblock();
    cs.render_microblock();
    rot2->manager.transition(MICRO, {
        {"rot_1", "0.7071"},
        {"rot_j", "0.7071"},
    });
	shared_ptr<LatexScene> latex_left2 = make_shared<LatexScene>("\\Leftarrow",vec2(0.12,0.12));
	cs.add_scene_fade_in(MICRO, latex_left2 , "latex_left2",vec2(0.69,0.17));
	cs.slide_subscene(MICRO,"latex_left2",vec2(-0.03,0));
    cs.fade_subscene(MICRO,"latex_j4",0);
	cs.slide_subscene(MICRO,"latex_j4",vec2(-0.01,0));
    cs.render_microblock();

    rot2->manager.transition(MICRO, {
        {"rot_1", "0.5"},
        {"rot_i", "0.5"},
        {"rot_j", "0.5"},
        {"rot_k", "0.5"},
    });
	shared_ptr<LatexScene> latex_up2 = make_shared<LatexScene>("\\Uparrow",vec2(0.17,0.17));
	cs.add_scene_fade_in(MICRO, latex_up2 , "latex_up2",vec2(0.73,0.21));
	cs.slide_subscene(MICRO,"latex_up2",vec2(0,-0.04));
    cs.fade_subscene(MICRO,"latex_i5",0);
	cs.slide_subscene(MICRO,"latex_i5",vec2(0,-0.01));
    cs.render_microblock();



    stage_macroblock(SilenceBlock(2));
    cs.render_microblock();
	cs.remove_subscene("latex_i5");
	cs.remove_subscene("latex_j4");
	cs.remove_subscene("latex_i3");
	cs.remove_subscene("latex_j2");



    stage_macroblock(FileBlock("The double spinny algebra can spin the cube along 2 dimensions,"));

	cs.slide_subscene(MICRO,"latex__25",vec2(0,-0.2));
	cs.slide_subscene(MICRO,"latex_left1",vec2(0,-0.2));
	cs.slide_subscene(MICRO,"latex_left2",vec2(0,-0.2));
	cs.slide_subscene(MICRO,"latex_up1",vec2(0,-0.2));
	cs.slide_subscene(MICRO,"latex_up2",vec2(0,-0.2));
    
    rot1->manager.transition(MICRO, {{"rot_1", "1"},{"rot_i", "0"},{"rot_j", "0"},{"rot_k", "0"},});
    rot2->manager.transition(MICRO, {{"rot_1", "1"},{"rot_i", "0"},{"rot_j", "0"},{"rot_k", "0"},});


    shared_ptr<PngScene> bicomplex_png = make_shared<PngScene>("bicomplex",vec2(0.3,0.15));
    cs.add_scene(bicomplex_png, "bicomplex_png",vec2(0.5,-0.06)); 

    cs.slide_subscene(MICRO, "bicomplex_png",vec2(0,0.2)); 
    cs.slide_subscene(MICRO, "rot1",vec2(0.2,0)); 
    cs.slide_subscene(MICRO, "rot2",vec2(-0.2,0)); 
    
    cs.render_microblock();

	cs.remove_subscene("latex__25");
	cs.remove_subscene("latex_left1");
	cs.remove_subscene("latex_left2");
	cs.remove_subscene("latex_up1");
	cs.remove_subscene("latex_up2");
	cs.remove_subscene("rot2");
    rot1->manager.set({{"conjugate", "0"},{"slider", "-0.1"},});

    cs.render_microblock();


    rot1->manager.transition(MICRO, {{"rot_1", "0"},{"rot_i", "1"},{"rot_j", "0"},{"rot_k", "0"},});
    cs.render_microblock();
    rot1->manager.transition(MICRO, {{"rot_1", "-1"},{"rot_i", "0"},{"rot_j", "0"},{"rot_k", "0"},});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(2));

    rot1->manager.transition(MICRO, {{"rot_1", "0"},{"rot_i", "0"},{"rot_j", "1"},{"rot_k", "0"},});
    cs.render_microblock();
    rot1->manager.transition(MICRO, {{"rot_1", "1"},{"rot_i", "0"},{"rot_j", "0"},{"rot_k", "0"},});
    cs.render_microblock();



    stage_macroblock(CompositeBlock(FileBlock("But the third makes weird flips."),SilenceBlock(0.3)));

    rot1->manager.transition(MICRO, {{"rot_1", "0"},{"rot_i", "0"},{"rot_j", "0"},{"rot_k", "1"},});
    cs.render_microblock();
    rot1->manager.transition(MICRO, {{"rot_1", "1"},{"rot_i", "0"},{"rot_j", "0"},{"rot_k", "0"},});
    cs.render_microblock();

    
    cs.slide_subscene(MICRO, "table",vec2(0.35,0)); 


    stage_macroblock(SilenceBlock(3));

    rot1->manager.set({{"conjugate", "1"},});
    rot1->manager.transition(MICRO, {{"rot_1", "0"},{"rot_i", "-0.7071"},{"rot_j", "0"},{"rot_k", "0.7071"},});
    cs.render_microblock();
    rot1->manager.transition(MICRO, {{"rot_1", "0"},{"rot_i", "0.5"},{"rot_j", "0.5"},{"rot_k", "-0.7071"},});
    cs.render_microblock();
    rot1->manager.transition(MICRO, {{"rot_1", "0"},{"rot_i", "-0.5"},{"rot_j", "-0.7071"},{"rot_k", "0.5"},});
    cs.render_microblock();


    auto table_opacities = [](const vector<string>& cells, const string& on, const string& off){
        StateSet ss;
        for (int r = 0; r < 5; r++) for (int c = 0; c < 5; c++) ss["opacity_" + to_string(r) + to_string(c)] = off;
        for (const string& cell : cells) ss["opacity_" + cell] = on;
        return ss;
    };

    stage_macroblock(FileBlock("It would be nice if ij squares to negative 1, making its own spinny subalgebra."));

    table->manager.set(table_opacities({}, "1", "1"));
    cs.fade_subscene(MICRO,"table",1);
    cs.fade_subscene(MICRO,"rot1",0);
    cs.fade_subscene(MICRO,"bicomplex_png",0.3);
    cs.render_microblock();
    cs.remove_subscene("rot1");
    cs.remove_subscene("rot2");

    table->manager.transition(MICRO, table_opacities({"00","04","40","44"}, "1", "0"));
    cs.render_microblock();

    cs.render_microblock();
    table->manager.set({{"commutative","0"},{"triangle","0"}});
    cs.render_microblock();

    table->manager.transition(MICRO, table_opacities({"00","01","04","10","11","14","40","41","44"}, "1", "0.1"));
    cs.render_microblock();



	shared_ptr<LatexScene> latex_ij2 = make_shared<LatexScene>("(ij)^2 = -1",vec2(0.4,0.2));
	cs.add_scene_fade_in(MICRO, latex_ij2 , "latex_ij2",vec2(0.5,-0.06));

    cs.slide_subscene(MICRO, "bicomplex_png",vec2(0,-0.2)); 
    cs.slide_subscene(MICRO, "latex_ij2",vec2(0,0.2)); 

    stage_macroblock(SilenceBlock(0.6));
    cs.render_microblock();
    cs.remove_subscene("bicomplex_png");



    stage_macroblock(FileBlock("We can rearrange to find that ij is negative ji."));

	latex_ij2->begin_latex_transition(MICRO, "ijij = -1");
    cs.render_microblock();
	latex_ij2->begin_latex_transition(MICRO, "iji = j");
    cs.render_microblock();
	latex_ij2->begin_latex_transition(MICRO, "ij = -ji");
    cs.render_microblock();
    cs.render_microblock();



    stage_macroblock(FileBlock("This rule helps us fill in the rest of the table, to get the triple spinny algebra."));
    table->manager.transition(MICRO,{
        {"opacity_00", "1"},{"opacity_01", "1"},{"opacity_02", "1"},{"opacity_03", "1"},{"opacity_04", "1"},
        {"opacity_10", "1"},{"opacity_11", "1"},{"opacity_12", "1"},{"opacity_13", "1"},{"opacity_14", "1"},
        {"opacity_20", "1"},{"opacity_21", "1"},{"opacity_22", "1"},{"opacity_23", "1"},{"opacity_24", "1"},
        {"opacity_30", "1"},{"opacity_31", "1"},{"opacity_32", "1"},{"opacity_33", "1"},{"opacity_34", "1"},
        {"opacity_40", "1"},{"opacity_41", "1"},{"opacity_42", "1"},{"opacity_43", "1"},{"opacity_44", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();


    cs.manager.transition(MICRO,{{"bg_1_r", "0"},{"bg_1_g", "8"},{"bg_1_b", "80"},});
    shared_ptr<PngScene> quat_png = make_shared<PngScene>("quat",vec2(0.3,0.15));
    cs.add_scene(quat_png, "quat_png",vec2(0.5,-0.06)); 

    cs.slide_subscene(MICRO, "quat_png",vec2(0,0.2)); 
    cs.slide_subscene(MICRO, "latex_ij2",vec2(0,-0.2)); 
    cs.render_microblock();
    cs.render_microblock();




    stage_macroblock(FileBlock("It can spin the cube along all 3 dimensions."));

    shared_ptr<FourDAlgebraScene> rotFull = make_shared<FourDAlgebraScene>(vec2(1,1.1));
    cs.add_scene_fade_in(MICRO, rotFull, "rotFull", vec2(0.5,0.55));

    cs.fade_subscene(MICRO,"table",0);

    rotFull->manager.set({
        {"offset_1", "0.0"},
        {"offset_2", "0.0"},
        {"slider", "1.1"},
        {"rotate", "1"},
        {"scale", "5.0"},
        {"pov_xz", "0.1"},
        {"pov_y", "0.2"},
    });
    cs.render_microblock();
    cs.remove_subscene("table");
    cs.render_microblock();

    rotFull->manager.transition(MICRO, {
        {"rot_1", "0"},
        {"rot_k", "1"},
    });
    cs.render_microblock();
    rotFull->manager.transition(MICRO, {
        {"rot_k", "0"},
        {"rot_j", "1"},
    });
    cs.render_microblock();
    rotFull->manager.transition(MICRO, {
        {"rot_1", "-1"},
        {"rot_j", "0"},
    });
    cs.render_microblock();


    stage_macroblock(SilenceBlock(0.6));
    cs.render_microblock();



    stage_macroblock(FileBlock("This algebra models 3D rotation!"));
    rotFull->manager.transition(MICRO, {{"rot_1", "0.66666"},{"rot_i", "0.333333"},{"rot_j", "0"},{"rot_k", "-0.33333"},});
    cs.render_microblock();
    rotFull->manager.transition(MICRO, {{"rot_1", "1"},{"rot_i", "0"},{"rot_j", "0"},{"rot_k", "0"},});
    cs.render_microblock();




    stage_macroblock(FileBlock("TGC: Who cares? We can already do that with matrices.")); 
    cs.render_microblock();
    shared_ptr<FourDAlgebraScene> matrices = make_shared<FourDAlgebraScene>();
    cs.add_scene_fade_in(MICRO, matrices, "matrices", vec2(1,0.55));
    shared_ptr<PngScene> matrix_png = make_shared<PngScene>("matrices",vec2(0.27,0.135));
    cs.add_scene_fade_in(MICRO, matrix_png, "matrix_png",vec2(0.99,0.14)); 

    cs.slide_subscene(MICRO, "rotFull",vec2(-0.2,0)); 
    cs.slide_subscene(MICRO, "quat_png",vec2(-0.2,0)); 
    cs.slide_subscene(MICRO, "matrices",vec2(-0.3,0)); 
    cs.slide_subscene(MICRO, "matrix_png",vec2(-0.3,0)); 

    matrices->manager.set({
        {"offset_1", "0.0"},
        {"offset_2", "0.0"},
        {"slider", "1.1"},
        {"rotate", "1"},
        {"scale", "5.0"},
        {"pov_xz", "0.1"},
        {"pov_y", "0.2"},
        {"matrices", "1"},
    });
    cs.render_microblock();


    stage_macroblock(SilenceBlock(0.6));
    cs.render_microblock();



    stage_macroblock(FileBlock("Yes, matrices spin nicely about the three axes,")); 
    matrices->manager.transition(MICRO, {
        {"rot_i", "1.5707"},
    });
    cs.render_microblock();
    matrices->manager.transition(MICRO, {
        {"rot_j", "1.5707"},
    });
    cs.render_microblock();
    matrices->manager.transition(MICRO, {
        {"rot_k", "1.5707"},
    });
    cs.render_microblock();


    stage_macroblock(FileBlock("but tilted axes can get overcomplicated.")); 
    cs.render_microblock();
    matrices->manager.transition(MICRO, {
        {"rot_i", "-1.25"},
        {"rot_j", "0.95"},
        {"rot_k", "0.4"},
    });
    cs.render_microblock();


    stage_macroblock(SilenceBlock(0.6));
    cs.render_microblock();


    stage_macroblock(FileBlock("Our algebra can do it more naturally.")); 
    cs.render_microblock();
    rotFull->manager.transition(MICRO, {
        {"rot_1", "0.4695"},
        {"rot_i", "-0.4086"},
        {"rot_j", "0.4512"},
        {"rot_k", "-0.6396"},
    });
    cs.render_microblock();


    stage_macroblock(SilenceBlock(0.6));
    cs.render_microblock();


    
    stage_macroblock(SilenceBlock(5));

    matrices->manager.transition(MICRO, {
        {"rot_i", "0.2"},
        {"rot_j", "1.3"},
        {"rot_k", "-0.7"},
    });
    rotFull->manager.transition(MICRO, {
        {"rot_1", "0.3238"},
        {"rot_i", "-0.5677"},
        {"rot_j", "0.7565"},
        {"rot_k", "-0.0225"},
    });
    cs.render_microblock();

    matrices->manager.transition(MICRO, {
        {"rot_i", "0.9"},
        {"rot_j", "-0.3"},
        {"rot_k", "-2.2"},
    });
    rotFull->manager.transition(MICRO, {
        {"rot_1", "-0.5366"},
        {"rot_i", "-0.2919"},
        {"rot_j", "0.7131"},
        {"rot_k", "-0.3439"},
    });
    cs.render_microblock();


    matrices->manager.transition(MICRO, {
        {"rot_i", "1.4"},
        {"rot_j", "0.3"},
        {"rot_k", "-1.5"},
    });
    rotFull->manager.transition(MICRO, {
        {"rot_1", "0.302"},
        {"rot_i", "0.0165"},
        {"rot_j", "0.9426"},
        {"rot_k", "-0.1414"},
    });
    cs.render_microblock();



    matrices->manager.transition(MICRO, {
        {"rot_i", "0"},
        {"rot_j", "0"},
        {"rot_k", "0"},
    });
    rotFull->manager.transition(MICRO, {
        {"rot_1", "1"},
        {"rot_i", "0"},
        {"rot_j", "0"},
        {"rot_k", "0"},
    });
    cs.render_microblock();


    stage_macroblock(SilenceBlock(0.4));
    cs.render_microblock();


    cs.fade_subscene(MICRO,"quat_png",0.1);
    cs.fade_subscene(MICRO,"matrices",0);
    cs.fade_subscene(MICRO,"matrix_png",0);
    stage_macroblock(FileBlock("Any tilted slice makes its own spinny subalgebra.")); 
    rotFull->manager.transition(MICRO, {
        {"plane", "1"},
    });
    cs.render_microblock();
    cs.remove_subscene("matrices");
    cs.remove_subscene("matrix_png");

    

    shared_ptr<MultiplicationTableScene> slice = make_shared<MultiplicationTableScene>();
    cs.add_scene_fade_in(MICRO, slice, "slice", vec2(1.1,0.52));
    cs.slide_subscene(MICRO, "slice", vec2(-0.4,0));
    slice->manager.set(bg);
    slice->manager.set({
        {"table_size", "2"},
        {"tilted", "1"},
        {"triangle", "0"},
    });

    shared_ptr<PngScene> spinny_png = make_shared<PngScene>("spinny",vec2(0.25,0.12));
    cs.add_scene(spinny_png, "spinny_png",vec2(1.14,0.22)); 
    cs.slide_subscene(MICRO, "spinny_png", vec2(-0.4,0));


    rotFull->manager.transition(MICRO, {
        {"rot_1", "0"},
        {"rot_i", "-0.6666"},
        {"rot_j", "-0.6666"},
        {"rot_k", "0.3333"},
    });
    cs.render_microblock();


    stage_macroblock(SilenceBlock(1.6));
    rotFull->manager.transition(MICRO, {
        {"rot_1", "-1"},
        {"rot_i", "0"},
        {"rot_j", "0"},
        {"rot_k", "0"},
    });
    cs.render_microblock();
    rotFull->manager.transition(MICRO, {
        {"rot_1", "0.2"},
        {"rot_i", "-0.6666"},
        {"rot_j", "-0.6666"},
        {"rot_k", "0.3333"},
    });
    cs.render_microblock();


    stage_macroblock(FileBlock("TGC: So this algebra isn't just triple spinny...")); 
    rotFull->manager.transition(MICRO, {
        {"rot_1", "1"},
        {"rot_i", "0"},
        {"rot_j", "0"},
        {"rot_k", "0"},
    });
    cs.render_microblock();

    rotFull->manager.transition(MICRO, {
        {"plane", "0"},
    });
    cs.fade_subscene(MICRO,"quat_png",1);
    cs.fade_subscene(MICRO,"slice",0);
    cs.fade_subscene(MICRO,"spinny_png",0);
    cs.slide_subscene(MICRO, "quat_png", vec2(0.2,0.03));
    cs.slide_subscene(MICRO, "rotFull", vec2(0.2,0.03));
    cs.slide_subscene(MICRO, "slice", vec2(0.15,0.03));
    cs.slide_subscene(MICRO, "spinny_png", vec2(0.15,0.03));
    cs.render_microblock();
    cs.remove_subscene("slice");
    cs.remove_subscene("spinny_png");


    stage_macroblock(FileBlock("TGC: it's infinny spinny!")); 

    shared_ptr<PngScene> infinny_png = make_shared<PngScene>("infinny",vec2(0.3,0.15));
    cs.add_scene(infinny_png, "infinny_png",vec2(0.5,-0.06)); 

    cs.slide_subscene(MICRO, "infinny_png",vec2(0,0.23)); 
    cs.slide_subscene(MICRO, "quat_png",vec2(0,-0.25)); 
    cs.render_microblock();
    rotFull->manager.transition(MICRO, {{"rot_1", "-0.8"},{"rot_i", "0.2"},{"rot_j", "0.4"},{"rot_k", "0.4"},});
    cs.render_microblock();

    stage_macroblock(SilenceBlock(2.5));
    rotFull->manager.transition(MICRO, {{"rot_1", "0.333333"},{"rot_i", "0.0"},{"rot_j", "0.6666"},{"rot_k", "-0.33333"},});
    cs.render_microblock();
    rotFull->manager.transition(MICRO, {{"rot_1", "1"},{"rot_i", "0"},{"rot_j", "0"},{"rot_k", "0"},});
    cs.render_microblock();
    cs.render_microblock();



}



void quat_graphs(CompositeScene& cs, StateSet& bg){



    shared_ptr<MultiplicationTableScene> table = make_shared<MultiplicationTableScene>(vec2(1,1));
    cs.add_scene_fade_in(MICRO, table, "table",vec2(0.48,1.08));
    cs.slide_subscene(MICRO, "table", vec2(0,-0.6));


    table->manager.set(bg);
    table->manager.set({
        {"triangle", "0"},{"commutative", "1"},{"table_size", "1.5"},{"q_ij", "1"},{"q_jj", "1"},

        {"opacity_00", "0"},{"opacity_01", "0"},{"opacity_02", "0"},{"opacity_03", "0"},{"opacity_04", "0"},
        {"opacity_10", "0"},{"opacity_11", "0"},{"opacity_12", "0"},{"opacity_13", "0"},{"opacity_14", "0"},
        {"opacity_20", "0"},{"opacity_21", "0"},{"opacity_22", "0"},{"opacity_23", "0"},{"opacity_24", "0"},
        {"opacity_30", "0"},{"opacity_31", "0"},{"opacity_32", "0"},{"opacity_33", "0"},{"opacity_34", "0"},
        {"opacity_40", "0"},{"opacity_41", "0"},{"opacity_42", "0"},{"opacity_43", "0"},{"opacity_44", "0"},
    });


    stage_macroblock(FileBlock("To construct it, we began with 1,"));

    cs.fade_subscene(MICRO,"rotFull",0);
    cs.fade_subscene(MICRO,"infinny_png",0);
    cs.slide_subscene(MICRO, "rotFull", vec2(0,-0.6));
    cs.slide_subscene(MICRO, "infinny_png", vec2(0,-0.6));

    cs.render_microblock();

    
    table->manager.transition(MICRO, {
        {"opacity_01", "1"},{"opacity_10", "1"},{"opacity_11", "1"},
    });
    cs.render_microblock();

    stage_macroblock(FileBlock("which makes these neutral multiplications."));
    table->manager.transition(MICRO, {
        {"table_size", "2.6"},
        {"opacity_02", "1"},
        {"opacity_11", "1"},{"opacity_12", "1"},
        {"opacity_20", "1"},{"opacity_21", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();



    stage_macroblock(SilenceBlock(0.6));
    cs.render_microblock();


    stage_macroblock(FileBlock("Then we found that i should square to -1."));
    table->manager.transition(MICRO, {
        {"table_size", "2.8"},
        {"opacity_01", "0.1"},{"opacity_10", "0.1"},
        {"opacity_03", "0"},{"opacity_04", "0"},
        {"opacity_11", "0.1"},{"opacity_12", "0.1"},{"opacity_13", "0"},{"opacity_14", "0"},
        {"opacity_21", "0.1"},
        {"opacity_30", "0"},{"opacity_31", "0"},
        {"opacity_40", "0"},{"opacity_41", "0"},
    });
    cs.render_microblock();
    table->manager.transition(MICRO, {
        {"opacity_22", "1"},
    });
    cs.render_microblock();


    shared_ptr<MultiplicationTableScene> other_choice = make_shared<MultiplicationTableScene>(vec2(1,1));
    cs.add_scene_fade_in(MICRO,other_choice, "other_choice",vec2(0.48,0.48));

    other_choice->manager.set(bg);
    other_choice->manager.set({
        {"triangle", "0"},{"i_flippy", "1"},{"commutative", "0"},
        {"table_size", "2.8"},

        {"opacity_00", "0"},{"opacity_01", "0"},{"opacity_02", "0"},{"opacity_03", "0"},{"opacity_04", "0"},
        {"opacity_10", "0"},{"opacity_11", "0"},{"opacity_12", "0"},{"opacity_13", "0"},{"opacity_14", "0"},
        {"opacity_20", "0"},{"opacity_21", "0"},{"opacity_22", "1"},{"opacity_23", "0"},{"opacity_24", "0"},
        {"opacity_30", "0"},{"opacity_31", "0"},{"opacity_32", "0"},{"opacity_33", "0"},{"opacity_34", "0"},
        {"opacity_40", "0"},{"opacity_41", "0"},{"opacity_42", "0"},{"opacity_43", "0"},{"opacity_44", "0"},
    });


    stage_macroblock(FileBlock("A different choice would make this subalgebra flippy."));
    cs.render_microblock();
    cs.render_microblock();

    shared_ptr<PngScene> flippy_png = make_shared<PngScene>("flippy",vec2(0.28,0.14));
    cs.add_scene(flippy_png, "flippy_png",vec2(0.48,-0.05)); 
    cs.slide_subscene(MICRO, "flippy_png",vec2(0,0.18)); 
    cs.slide_subscene(MICRO,"table",vec2(0,0.03));
    cs.slide_subscene(MICRO,"other_choice",vec2(0,0.03));

    table->manager.transition(MICRO, {
        {"opacity_01", "1"},{"opacity_10", "1"},
        {"opacity_11", "1"},{"opacity_12", "1"},
        {"opacity_21", "1"},
    });
    cs.render_microblock();




    stage_macroblock(CompositeBlock(FileBlock("TGC: And that's bad!"),SilenceBlock(0.4)));
    cs.render_microblock();
    cs.manager.set({{"d","0"},{"bounce","<d> sin 1.09 *"}});
    cs.manager.transition(MICRO,{{"d","1.98"}});
    shared_ptr<PngScene> bad_png = make_shared<PngScene>("bad",vec2(0.1,0.1));
    cs.add_scene_fade_in(MICRO,bad_png, "bad_png",vec2(0.50,0.62)); 
    // cs.slide_subscene(MICRO, "bad_png",vec2(-0.1,0.08)); 
    cs.manager.set({{"bad_png.angle","-0.3"},{"bad_png.x","<bounce> -0.1 * 0.5 +"},{"bad_png.y","<bounce> 0.08 * 0.62 +"}});
    bad_png->manager.set({{"w","[bounce] 0.3 * 0.1 +"},{"h","[bounce] 0.3 * 0.1 +"}});
    cs.render_microblock();
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1.6));
    cs.render_microblock();
    cs.fade_subscene(MICRO,"other_choice",0);
    cs.fade_subscene(MICRO,"bad_png",0);
    cs.fade_subscene(MICRO,"flippy_png",0);
    cs.slide_subscene(MICRO,"flippy_png",vec2(0,-0.18));
    cs.slide_subscene(MICRO,"bad_png",vec2(0.1,0.2));
    cs.render_microblock();
    cs.remove_subscene("other_choice");
    cs.remove_subscene("bad_png");
    cs.remove_subscene("flippy_png");



    stage_macroblock(FileBlock("Similarly, j should square to -1 in its subalgebra."));
    cs.slide_subscene(MICRO,"table",vec2(0,-0.03));
    table->manager.transition(MICRO, {
        {"table_size", "3.2"},
        {"opacity_01", "0"},{"opacity_10", "0"},{"opacity_11", "0"},
        {"opacity_02", "0"},{"opacity_20", "0"},
        {"opacity_12", "0"},{"opacity_21", "0"},{"opacity_22", "0"},
        {"opacity_03", "1"},{"opacity_30", "1"},
        {"opacity_33", "1"},
    });
    cs.render_microblock();
    table->manager.transition(MICRO, {
        {"opacity_01", "1"},{"opacity_10", "1"},{"opacity_11", "1"},
        {"opacity_13", "1"},{"opacity_31", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();


    stage_macroblock(SilenceBlock(1.5));
    cs.render_microblock();


    stage_macroblock(FileBlock("We tried combining i and j to make a 3D algebra, but it was not associative."));
    table->manager.transition(MICRO, {
        {"opacity_02", "1"},{"opacity_20", "1"},
        {"opacity_12", "1"},{"opacity_21", "1"},{"opacity_22", "1"},
        {"opacity_23", "1"},{"opacity_32", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();


    StateSet parentcontrol = {
        {"yz_x", "[yz_x]"}, {"yz_y", "[yz_y]"}, {"yz_z", "[yz_z]"},{"yz_w", "[yz_w]"},
        {"zz_x", "[zz_x]"}, {"zz_y", "[zz_y]"}, {"zz_z", "[zz_z]"},{"zz_w", "[zz_w]"},
        {"d", "[d]"},{"w_slider", "[w_slider]"},
    };
    cs.manager.set({
        {"yz_x", "-0.7"}, {"yz_y", "0.2"}, {"yz_z", "0.4"},
        {"zz_x", "0.7"}, {"zz_y", "0.3"}, {"zz_z", "-0.4"},
        {"w_slider","0"},{"d","30"},
    });
    shared_ptr<ThreeDAlgebraScene> threed1 = make_shared<ThreeDAlgebraScene>(vec2(0.4, 1));
    threed1->manager.set({
        {"associativity", "1"},{"a_x", "0"},{"a_y", "1"},{"b_x", "0"},{"b_z", "1"},{"reach", "3"},
    });
    threed1->manager.set(parentcontrol);

    shared_ptr<ThreeDAlgebraScene> threed2 = make_shared<ThreeDAlgebraScene>(vec2(0.4, 1));
    threed2->manager.set({
        {"associativity", "0"},{"a_x", "0"},{"a_y", "1"},{"b_x", "0"},{"b_z", "1"},{"reach", "3"},
    });
    threed2->manager.set(parentcontrol);

    shared_ptr<LatexScene> latex_ij = make_shared<LatexScene>("\\times ij",vec2(0.25,0.25));
    shared_ptr<LatexScene> latex_i_j = make_shared<LatexScene>("\\times i \\times j",vec2(0.25,0.25));

    cs.manager.transition(MICRO,{
        {"yz_x", "-0.1"}, {"yz_y", "0.6"}, {"yz_z", "0.3"},
        {"zz_x", "0.1"}, {"zz_y", "-0.7"}, {"zz_z", "-0.5"},
    });
    cs.add_scene(threed1, "threed1",vec2(1.2,0.4)); 
    cs.add_scene(threed2, "threed2",vec2(-0.2,0.4));
    cs.add_scene(latex_ij, "latex_ij", vec2(1.2,0.8)); 
    cs.add_scene(latex_i_j, "latex_i_j", vec2(-0.2,0.8)); 
    cs.slide_subscene(MICRO,"threed1",vec2(-0.37,0));
    cs.slide_subscene(MICRO,"threed2",vec2(0.37,0));
    cs.slide_subscene(MICRO,"latex_ij",vec2(-0.37,0));
    cs.slide_subscene(MICRO,"latex_i_j",vec2(0.37,0));
    table->manager.transition(MICRO,{{"w","0.75"},{"h","0.75"}});
    cs.render_microblock();



    stage_macroblock(SilenceBlock(5));

    cs.manager.transition(MICRO,{
        {"yz_x", "0.2"}, {"yz_y", "0.5"}, {"yz_z", "-0.6"},
        {"zz_x", "0.9"}, {"zz_y", "0.2"}, {"zz_z", "-0.6"},
    });
    cs.render_microblock();
    cs.manager.transition(MICRO,{
        {"yz_x", "-1"}, {"yz_y", "0.2"}, {"yz_z", "0"},
        {"zz_x", "0"}, {"zz_y", "1"}, {"zz_z", "-0.2"},
    });
    cs.render_microblock();
    cs.slide_subscene(MICRO,"threed1",vec2(0.4,0));
    cs.slide_subscene(MICRO,"threed2",vec2(-0.4,0));
    cs.slide_subscene(MICRO,"latex_ij",vec2(0.4,0));
    cs.slide_subscene(MICRO,"latex_i_j",vec2(-0.4,0));
    table->manager.transition(MICRO,{{"w","1"},{"h","1"}});
    cs.render_microblock();
    cs.remove_subscene("threed1");
    cs.remove_subscene("threed2");
    cs.remove_subscene("latex_ij");
    cs.remove_subscene("latex_i_j");


    stage_macroblock(FileBlock("i times j must be in a 4th dimension."));
    table->manager.transition(MICRO, {
        {"q_ij", "0"},
        {"opacity_01", "0.1"},{"opacity_10", "0.1"},{"opacity_11", "0.1"},
        {"opacity_02", "0.1"},{"opacity_20", "1"},
        {"opacity_12", "0.1"},{"opacity_21", "0.1"},{"opacity_22", "0.1"},
        {"opacity_03", "1"},{"opacity_30", "0.1"},
        {"opacity_13", "0.1"},{"opacity_31", "0.1"},{"opacity_33", "0.1"},
        {"opacity_32", "0"},
    });
    cs.render_microblock();
    cs.render_microblock();
    table->manager.set({{"q_jj", "0"},{"commutative", "0"}});
    table->manager.transition(MICRO, {
        {"table_size", "4"},
        {"opacity_04", "1"},{"opacity_40", "1"},
    });
    cs.render_microblock();


    stage_macroblock(SilenceBlock(0.6));
    cs.render_microblock();


    stage_macroblock(FileBlock("And here's where the algebra splits."));
    table->manager.transition(MICRO, {
        {"opacity_20", "0.1"},{"opacity_03", "0.1"},{"opacity_23", "0.1"},
    });
    cs.render_microblock();

    shared_ptr<MultiplicationTableScene> bitable = make_shared<MultiplicationTableScene>(vec2(1,1));
    cs.add_scene(bitable, "bitable",vec2(0.48,0.48));
    cs.manager.set({{"slider","0.5"},{"bg_0_r","0"},{"bg_0_g","8"},{"bg_0_b","80"}});

    bitable->manager.set({
        {"triangle", "0"},
        {"commutative", "1"},

        {"opacity_00", "0.1"},{"opacity_01", "0.1"},{"opacity_02", "0.1"},{"opacity_03", "0.1"},{"opacity_04", "1"},
        {"opacity_10", "0.1"},{"opacity_11", "0.1"},{"opacity_12", "0.1"},{"opacity_13", "0.1"},{"opacity_14", "0"},
        {"opacity_20", "0.1"},{"opacity_21", "0.1"},{"opacity_22", "0.1"},{"opacity_23", "0.1"},{"opacity_24", "0"},
        {"opacity_30", "0.1"},{"opacity_31", "0.1"},{"opacity_32", "0"},{"opacity_33", "0.1"},{"opacity_34", "0"},
        {"opacity_40", "1"},{"opacity_41", "0"},{"opacity_42", "0"},{"opacity_43", "0"},{"opacity_44", "0"},
        {"bg_1_r","0"},{"bg_1_g","8"},{"bg_1_b","80"},
    });
    cs.slide_subscene(MICRO, "table", vec2(-0.25,0));
    cs.slide_subscene(MICRO, "bitable", vec2(0.25,0));
    cs.manager.transition(MICRO, {
        {"bg_0_r","48"},{"bg_0_g","0"},{"bg_0_b","60"},
    });
    table->manager.transition(MICRO, {
        {"w", "0.7"},{"h", "0.7"},
    });
    bitable->manager.transition(MICRO, {
        {"w", "0.7"},{"h", "0.7"},
        {"bg_1_r","48"},{"bg_1_g","0"},{"bg_1_b","60"},
    });

    cs.render_microblock();



    stage_macroblock(FileBlock("We can fill the rest of the table with some straightforward calculations,"));
    cs.slide_subscene(MICRO, "table", vec2(-0.05,0));
    cs.slide_subscene(MICRO, "bitable", vec2(-0.05,0));
    cs.manager.transition(MICRO,{{"slider","0.42"}});
    table->manager.transition(MICRO, {{"w", "0.65"},{"h", "0.65"},});
    bitable->manager.transition(MICRO, {{"w", "0.75"},{"h", "0.75"},});

    table->manager.transition(MICRO, {{"opacity_40", "0.2"},{"opacity_04", "0.2"},});

    bitable->manager.transition(MICRO, {
        {"opacity_00", "1"},{"opacity_01", "1"},{"opacity_02", "1"},{"opacity_03", "1"},{"opacity_04", "1"},
        {"opacity_10", "1"},{"opacity_11", "1"},{"opacity_12", "1"},{"opacity_13", "1"},{"opacity_14", "1"},
        {"opacity_20", "1"},{"opacity_21", "1"},{"opacity_22", "1"},{"opacity_23", "1"},{"opacity_24", "1"},
        {"opacity_30", "1"},{"opacity_31", "1"},{"opacity_32", "1"},{"opacity_33", "1"},{"opacity_34", "1"},
        {"opacity_40", "1"},{"opacity_41", "1"},{"opacity_42", "1"},{"opacity_43", "1"},{"opacity_44", "1"},
    });
    cs.render_microblock();

	shared_ptr<LatexScene> latex_bi = make_shared<LatexScene>("ij=ji",vec2(0.3,0.15));
	cs.add_scene(latex_bi , "latex_bi",vec2(0.71,-0.06));
    cs.slide_subscene(MICRO, "latex_bi", vec2(0,0.2));
    cs.slide_subscene(MICRO, "bitable", vec2(0,0.08));
    cs.render_microblock();
    cs.render_microblock();


    stage_macroblock(CompositeBlock(FileBlock("but that makes it flippy."),SilenceBlock(0.4)));
    bitable->manager.transition(MICRO, {
        {"opacity_00", "0"},{"opacity_01", "1"},{"opacity_02", "0"},{"opacity_03", "0"},{"opacity_04", "1"},
        {"opacity_10", "1"},{"opacity_11", "1"},{"opacity_12", "0"},{"opacity_13", "0"},{"opacity_14", "1"},
        {"opacity_20", "0"},{"opacity_21", "0"},{"opacity_22", "0"},{"opacity_23", "0"},{"opacity_24", "0"},
        {"opacity_30", "0"},{"opacity_31", "0"},{"opacity_32", "0"},{"opacity_33", "0"},{"opacity_34", "0"},
        {"opacity_40", "1"},{"opacity_41", "1"},{"opacity_42", "0"},{"opacity_43", "0"},{"opacity_44", "1"},
    });
    cs.render_microblock();
    cs.render_microblock();


    stage_macroblock(SilenceBlock(0.6));
    cs.render_microblock();



    stage_macroblock(FileBlock("Alternatively, we can force the fourth dimension to be spinny,"));
    cs.slide_subscene(MICRO, "table", vec2(0.1,0));
    cs.slide_subscene(MICRO, "bitable", vec2(0.1,0));
    cs.slide_subscene(MICRO, "latex_bi", vec2(0.1,0));
    cs.manager.transition(MICRO,{{"slider","0.58"}});
    table->manager.transition(MICRO, {{"w", "0.75"},{"h", "0.75"},});
    bitable->manager.transition(MICRO, {{"w", "0.65"},{"h", "0.65"},});
    cs.fade_subscene(MICRO,"latex_bi",0.1);

    table->manager.transition(MICRO,{
        {"opacity_00", "0.1"},{"opacity_01", "1"},{"opacity_02", "0.1"},{"opacity_03", "0.1"},{"opacity_04", "1"},
        {"opacity_10", "1"},{"opacity_11", "1"},{"opacity_12", "0.1"},{"opacity_13", "0.1"},{"opacity_14", "1"},
        {"opacity_20", "0.1"},{"opacity_21", "0.1"},{"opacity_22", "0.1"},{"opacity_23", "0.1"},{"opacity_24", "0"},
        {"opacity_30", "0.1"},{"opacity_31", "0.1"},{"opacity_32", "0"},{"opacity_33", "0.1"},{"opacity_34", "0"},
        {"opacity_40", "1"},{"opacity_41", "1"},{"opacity_42", "0"},{"opacity_43", "0"},{"opacity_44", "0"},
    });    
    bitable->manager.transition(MICRO, {
        {"opacity_00", "0.1"},{"opacity_01", "0.2"},{"opacity_04", "0.2"},
        {"opacity_10", "0.2"},{"opacity_11", "0.2"},{"opacity_14", "0.2"},
        {"opacity_40", "0.2"},{"opacity_41", "0.2"},{"opacity_44", "0.2"},
    });

    cs.render_microblock();
    table->manager.transition(MICRO,{
        {"opacity_44", "1"},
    });
    cs.render_microblock();


    stage_macroblock(FileBlock("if we accept the tricky rule that completes the table."));
	shared_ptr<LatexScene> latex_qu = make_shared<LatexScene>("ij=-ji",vec2(0.3,0.15));
	cs.add_scene(latex_qu , "latex_qu",vec2(0.3,-0.06));
    cs.slide_subscene(MICRO, "latex_qu", vec2(0,0.2));
    cs.slide_subscene(MICRO, "table", vec2(0,0.08));
    cs.render_microblock();

    table->manager.transition(MICRO,{
        {"opacity_00", "1"},{"opacity_01", "1"},{"opacity_02", "1"},{"opacity_03", "1"},{"opacity_04", "1"},
        {"opacity_10", "1"},{"opacity_11", "1"},{"opacity_12", "1"},{"opacity_13", "1"},{"opacity_14", "1"},
        {"opacity_20", "1"},{"opacity_21", "1"},{"opacity_22", "1"},{"opacity_23", "1"},{"opacity_24", "1"},
        {"opacity_30", "1"},{"opacity_31", "1"},{"opacity_32", "1"},{"opacity_33", "1"},{"opacity_34", "1"},
        {"opacity_40", "1"},{"opacity_41", "1"},{"opacity_42", "1"},{"opacity_43", "1"},{"opacity_44", "1"},
    });
    cs.render_microblock();


    stage_macroblock(SilenceBlock(0.6));
    cs.render_microblock();



    shared_ptr<FourDAlgebraScene> fd = make_shared<FourDAlgebraScene>(vec2(1.5,1));
    cs.add_scene_fade_in(MICRO, fd, "fd",vec2(0.625,0.5), 1, false, 1);

    fd->manager.set({
        {"offset_1", "-0.13"},
        {"offset_2", "0.093"},
        {"scale", "16.0"},
        {"brightness", "100.0"},
        {"equation", "100"},
        {"pov_xz", "0.5"},
        {"pov_y", "0"},
        {"rotation_jk", "0.0"},
        {"rotation_ik", "-0.28"},
        {"rotation_1k", "-0.121"},
        {"slider", "0.4166666"},
    });
    // fd.slider = (0.75 - fd.x + cs.slider)*2/3

    



    stage_macroblock(FileBlock("This quaternion algebra makes circular graphs."));
    cs.slide_subscene(MICRO, "bitable", vec2(0.4,0));
    cs.slide_subscene(MICRO, "latex_bi", vec2(0.4,0));
    cs.slide_subscene(MICRO, "table", vec2(-0.1,0));
    table->manager.transition(MICRO, {{"w", "0.65"},{"h", "0.65"},});
    cs.slide_subscene(MICRO, "latex_qu", vec2(-0.1,0.05));
    cs.manager.transition(MICRO,{{"slider","1.1"}});
    fd->manager.transition(MICRO,{{"slider","0.816666"}});


    cs.render_microblock();
    
    fd->manager.transition(MICRO, {
        {"rotation_1k", "0.78"},
    });
    cs.render_microblock();




    stage_macroblock(SilenceBlock(1.8));
    fd->manager.transition(MICRO, {
        {"rotation_ik", "1.54"},
    });
    cs.render_microblock();
    
    fd->manager.transition(MICRO, {
        {"rotation_jk", "-0.73"},
        {"rotation_1k", "0.3"},
    });
    cs.render_microblock();




    stage_macroblock(FileBlock("It consists of symmetrically spinny systems."));

    table->manager.transition(MICRO,{
        {"opacity_00", "0.1"},{"opacity_01", "0.1"},{"opacity_02", "1"},{"opacity_03", "1"},{"opacity_04", "1"},
        {"opacity_10", "0.1"},{"opacity_11", "0.1"},{"opacity_12", "0.1"},{"opacity_13", "0.1"},{"opacity_14", "0.1"},
        {"opacity_20", "1"},{"opacity_21", "0.1"},{"opacity_22", "1"},{"opacity_23", "0.1"},{"opacity_24", "0.1"},
        {"opacity_30", "1"},{"opacity_31", "0.1"},{"opacity_32", "0.1"},{"opacity_33", "1"},{"opacity_34", "0.1"},
        {"opacity_40", "1"},{"opacity_41", "0.1"},{"opacity_42", "0.1"},{"opacity_43", "0.1"},{"opacity_44", "1"},
    });
    fd->manager.transition(MICRO, {
        {"rotation_1k", "0.25"},
    });
    cs.render_microblock();


    cs.slide_subscene(MICRO, "latex_bi", vec2(0,0.05));
    fd->manager.transition(MICRO, {
        {"rotation_1k", "-0.6"},
    });
    cs.render_microblock();




    stage_macroblock(SilenceBlock(0.6));
    cs.render_microblock();



    stage_macroblock(FileBlock("Its evil twin looks linear."));

    cs.slide_subscene(MICRO, "table", vec2(-0.4,0));
    cs.slide_subscene(MICRO, "latex_qu", vec2(-0.4,0));
    cs.slide_subscene(MICRO, "bitable", vec2(-0.4,0));
    cs.slide_subscene(MICRO, "latex_bi", vec2(-0.4,0.0));
    cs.fade_subscene(MICRO,"latex_bi",1);

    cs.slide_subscene(MICRO, "fd", vec2(-0.25,0));
    cs.manager.transition(MICRO,{{"slider","-0.1"}});
    fd->manager.transition(MICRO,{{"slider","0.1833333"}});
    
    bitable->manager.transition(MICRO,{
        {"opacity_00", "1"},{"opacity_01", "1"},{"opacity_02", "1"},{"opacity_03", "1"},{"opacity_04", "1"},
        {"opacity_10", "1"},{"opacity_11", "1"},{"opacity_12", "1"},{"opacity_13", "1"},{"opacity_14", "1"},
        {"opacity_20", "1"},{"opacity_21", "1"},{"opacity_22", "1"},{"opacity_23", "1"},{"opacity_24", "1"},
        {"opacity_30", "1"},{"opacity_31", "1"},{"opacity_32", "1"},{"opacity_33", "1"},{"opacity_34", "1"},
        {"opacity_40", "1"},{"opacity_41", "1"},{"opacity_42", "1"},{"opacity_43", "1"},{"opacity_44", "1"},
    });
    fd->manager.transition(MICRO, {
        {"scale", "12.0"},
    });

    cs.render_microblock();

    fd->manager.transition(MICRO, {
        {"rotation_jk", "0"},
        {"pov_xz", "-0.3"},
    });
    cs.render_microblock();


    stage_macroblock(SilenceBlock(0.6));
    fd->manager.transition(MICRO, {
        {"rotation_ik", "0.6"},
    });
    cs.render_microblock();



    stage_macroblock(FileBlock("The small spinny subspaces don't interact."));

    bitable->manager.transition(MICRO,{
        {"opacity_00", "0.1"},{"opacity_01", "0.1"},{"opacity_02", "1"},{"opacity_03", "1"},{"opacity_04", "0.1"},
        {"opacity_10", "0.1"},{"opacity_11", "0.1"},{"opacity_12", "0.1"},{"opacity_13", "0.1"},{"opacity_14", "0.1"},
        {"opacity_20", "1"},{"opacity_21", "0.1"},{"opacity_22", "1"},{"opacity_23", "0.1"},{"opacity_24", "0.1"},
        {"opacity_30", "1"},{"opacity_31", "0.1"},{"opacity_32", "0.1"},{"opacity_33", "1"},{"opacity_34", "0.1"},
        {"opacity_40", "0.1"},{"opacity_41", "0.1"},{"opacity_42", "0.1"},{"opacity_43", "0.1"},{"opacity_44", "0.1"},
    });
    fd->manager.transition(MICRO, {
        {"pov_y", "0.1"},
        {"rotation_1k", "-0.5"},
    });
    cs.render_microblock();

    fd->manager.transition(MICRO, {
        {"pov_y", "1"},
        {"rotation_1k", "0.55"},
    });
    cs.render_microblock();



    stage_macroblock(SilenceBlock(0.4));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(2.4));
    fd->manager.transition(MICRO, {
        {"equation", "101"},
        {"brightness", "10000.0"},
        {"scale", "12.0"},
    });
    cs.render_microblock();
    fd->manager.transition(MICRO, {
        {"brightness", "5000.0"},
        {"rotation_jk", "0.78"},
        {"pov_xz", "-1.1"},
        {"pov_y", "1.4"},  
        {"offset_1", "0.01"},
    });
    cs.render_microblock();



    stage_macroblock(FileBlock("Sticking to familiar ideas can have seemingly nice results."));

    cs.slide_subscene(MICRO, "bitable", vec2(-0.02,0.1));
    cs.fade_subscene(MICRO, "bitable",0);
    latex_bi->manager.transition(MICRO, {{"w", "0.6"},{"h", "0.3"},});
	latex_bi->begin_latex_transition(MICRO, "ij=ji");
    cs.slide_subscene(MICRO, "latex_bi", vec2(-0.05,0.3));
    cs.slide_subscene(MICRO, "latex_qu", vec2(0,0.3));

    fd->manager.transition(MICRO, {
        {"rotation_1k", "0.45"},
        {"pov_xz", "-1.2"},
    });
    cs.render_microblock();

    fd->manager.transition(MICRO, {
        {"rotation_ik", "1.3"},
        {"pov_xz", "-1.57"},
    });
    cs.render_microblock();

    fd->manager.transition(MICRO, {
        {"rotation_1k", "0"},
        {"pov_y", "1.57"},
    });
    cs.render_microblock();

    cs.remove_subscene("bitable");
    cs.remove_subscene("table");



    stage_macroblock(FileBlock("But with open minds and critical curiosity"));

    latex_qu->manager.transition(MICRO, {{"w", "0.65"},{"h", "0.3"},});
	latex_qu->begin_latex_transition(MICRO, "ij=-ji");
    cs.slide_subscene(MICRO, "latex_qu", vec2(0.45,0));
    cs.slide_subscene(MICRO, "latex_bi", vec2(0.45,0));

    cs.slide_subscene(MICRO, "fd", vec2(0.4,0));
    cs.manager.transition(MICRO,{{"slider","1.1"}});
    fd->manager.transition(MICRO,{{"slider","0.716666"}});

    fd->manager.transition(MICRO, {
        {"pov_xz", "-1.82"},
    });
    cs.render_microblock();

    fd->manager.transition(MICRO, {
        {"rotation_jk", "-0.23"},
        {"pov_xz", "-2.48"},
        {"pov_y", "1.53"},
        {"scale", "8.0"},
        {"brightness", "200.0"},
    });
    cs.render_microblock();




    stage_macroblock(FileBlock("we discover unexpectedly beautiful possibilities."));


    fd->manager.transition(MICRO, {
        {"equation", "102"},
        {"brightness", "1000.0"},
    });
    cs.render_microblock();

    fd->manager.transition(MICRO, {
        {"brightness", "10000.0"},
        {"rotation_ik", "0.2"},
        {"pov_y", "1.57"},
    });
    cs.render_microblock();
    
    
    stage_macroblock(SilenceBlock(3));
    fd->manager.transition(MICRO, {
        {"rotation_jk", "-0.83"},
        {"pov_xz", "-2"},
    });
    cs.fade_subscene(MICRO,"latex_qu",0);
    cs.render_microblock();

    fd->manager.transition(MICRO, {
        {"rotation_1k", "0.6"},
        {"pov_y", "0.745"},
    });
    cs.render_microblock();

    stage_macroblock(SilenceBlock(1.2));
    fd->manager.transition(MICRO, {
        {"pov_y", "0.5"},
    });
    cs.fade_subscene(MICRO,"fd",0);
    cs.render_microblock();
}





void render_video() {
    CompositeScene cs;

    StateSet bg_control = {
        {"bg_0_r", "[bg_0_r]"}, {"bg_0_g", "[bg_0_g]"}, {"bg_0_b", "[bg_0_b]"},
        {"bg_1_r", "[bg_1_r]"}, {"bg_1_g", "[bg_1_g]"}, {"bg_1_b", "[bg_1_b]"},
        {"slider", "[slider]"},
    };

    cs.manager.set({
        {"bg_0_r", "48"}, {"bg_0_g", "0"}, {"bg_0_b", "60"},
        {"bg_1_r", "0"}, {"bg_1_g", "8"}, {"bg_1_b", "80"},
        {"slider", "-0.1"},
    });
    set_global_state("xx_pos_x", 0);
    set_global_state("xx_pos_y", 0);
    set_global_state("xy_pos_x", 0);
    set_global_state("xy_pos_y", 0);
    set_global_state("yy_pos_x", 0);
    set_global_state("yy_pos_y", 0);
    set_global_state("drag_x", 0);
    set_global_state("drag_y", 0);

    set_global_draggers(0,0);
    set_global_number_line(1);

    shared_ptr<BackgroundScene> bg = make_shared<BackgroundScene>();
    cs.add_scene(bg, "bg");
    bg->manager.set(bg_control);


    shared_ptr<CursorScene> cursor = make_shared<CursorScene>();
    cs.add_scene(cursor, "cursor"); 

    StateSet cursor_control = {
        {"stroke_r", "[bg_1_r]"}, {"stroke_g", "[bg_1_g]"}, {"stroke_b", "[bg_1_b]"},
    };
    cursor->manager.set({{"cursor_x", "0.65"}, {"cursor_y", "1.1"}, {"cursor_size", "0.08"},});

    cursor->manager.set(cursor_control);

    set_for_real(false);
    

    four_d_intro(cs, bg_control);
    one_dim_algebra(cs);
    two_dim_algebra(cs,bg_control);
    two_dim_smallness(cs,bg_control);
    two_dim_wrapup(cs,bg_control);

    three_dim(cs,bg_control);
    four_d_cosine(cs,bg_control);
    non_commutative(cs,bg_control);
    quat_graphs(cs,bg_control);


    
    // TODO color design

    // TODO for final render set 4d sin and cos back to 60 iterations and drop raymarch step size in ghost and cube scenes

    // TODO resize latex without transitioning
    // TODO gpu flicker on circles
}


// ./go.sh FourDAlgebra 600 360 15
// ./record_audios.py FourDAlgebra
// ffmpeg -i 'Video.mkv' -c copy 'Video.mp4'
