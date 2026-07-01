#include "../Scenes/Math/FourDAlgebraScene.h"
#include "../Scenes/Math/TwoDAlgebraScene.h"
// #include "../Scenes/Math/RealFunctionScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Common/CompositeScene.h"

/*
EQUATION MAP:
    0 - exponential
    1 - sine
    2 - cosine
    3 - 10th necklace polynomial
*/

void render_video() {
    CompositeScene cs;
    shared_ptr<LatexScene> latex = make_shared<LatexScene>(" ", 0.15);
    shared_ptr<FourDAlgebraScene> fd = make_shared<FourDAlgebraScene>();
    shared_ptr<TwoDAlgebraScene> td = make_shared<TwoDAlgebraScene>();
    // shared_ptr<RealFunctionScene> td = make_shared<RealFunctionScene>();


    cs.add_scene(fd, "fd");
    cs.add_scene(latex, "latex");



    cs.add_scene(td, "td");
    td->manager.set("x_equation", "(a) ");
    td->manager.set("y_equation", "(b) ");

    stage_macroblock(FileBlock("2D Algebra Test"), 1);
  
    td->manager.transition(MICRO, {
        {"x_equation", "(a) 2.5 *"},
        {"x_adjustment", "2.5"},
        {"y_equation", "(b) 2 *"},
        {"y_adjustment", "2"},
        {"dragger_x", "1"}
    });
    cs.render_microblock();


//     fd->manager.set("equation", "0");
//     fd->manager.set("scale", "9.4");
//     fd->manager.set("brightness", "2.0");
//     fd->manager.set("slider", "-0.1");
//     fd->manager.set("pov_xz", "0.3");
//     fd->manager.set("offset_1", "0.53");
//     stage_macroblock(FileBlock("This graph shows how exponents change as we move through the 4th dimension."), 3);

//    fd->manager.transition(MICRO, {
//         {"rotation_jk", "-1.36"},
//         {"scale", "8.4"},
//         {"brightness", "3.0"}
//     });
//     cs.render_microblock();
//    fd->manager.transition(MICRO, {
//         {"rotation_ik", "1.29"},
//         {"scale", "12.4"},
//         {"brightness", "1.3"}
//     });
//     cs.render_microblock();
//    fd->manager.transition(MICRO, {
//         {"rotation_1k", "0.73"},
//         {"brightness", "1.3"}
//     });
//     cs.render_microblock();




//     stage_macroblock(FileBlock("and this graph shows how exponents change as we move through the 4th dimension."), 4);

//    fd->manager.transition(MICRO, {
//         {"slider", "1.1"},
//         {"brightness", "1.2"}
//     });
//     cs.render_microblock();
//    fd->manager.transition(MICRO, {
//         {"rotation_1k", "0.1"},
//         {"brightness", "1.4"}
//     });
//     cs.render_microblock();
//    fd->manager.transition(MICRO, {
//         {"rotation_ik", "0.03"},
//         {"brightness", "3.7"}
//     });
//     cs.render_microblock();
//    fd->manager.transition(MICRO, {
//         {"rotation_jk", "-0.19"},
//         {"brightness", "2.7"}
//     });
//     cs.render_microblock();




//     stage_macroblock(FileBlock("Both sides show the same thing."), 1);

//     fd->manager.transition(MICRO, {
//         {"scale", "9.4"},
//         {"slider", "0.5"},
//         {"brightness", "2.0"}
//         // {"brightness", "0.14"}
//     });
//     cs.render_microblock();



//     stage_macroblock(FileBlock("It's the same function on the same inputs,"), 1);

//     fd->manager.transition(MICRO, {
//         {"pov_xz", "1.8"},
//         {"brightness", "2.5"}
//     });
//     cs.render_microblock();;



//     stage_macroblock(FileBlock("but the rules are different."), 1);

//     fd->manager.transition(MICRO, {
//         {"rotation_jk", "1.29"},
//         {"rotation_ik", "-0.43"},
//         {"rotation_1k", "0.57"},
//         {"brightness", "2.9"}
//     });
//     cs.render_microblock();


    // stage_macroblock(FileBlock("Check out the sine function."), 1);

    // fd->manager.set("offset_1", "0.43");
    // fd->manager.set("offset_2", "0.27");
    // fd->manager.set("scale", "10.4");
    // fd->manager.set("brightness", "6.06");
    // fd->manager.set("equation", "1");
    // fd->manager.set("slider", "0.5");
    // fd->manager.set("pov_xz", "0.1");
    // fd->manager.set("rotation_jk", "0");
    // fd->manager.set("rotation_ik", "0");
    // fd->manager.set("rotation_1k", "0");


    // fd->manager.transition(MICRO, {
    //     {"rotation_ik", "-0.83"}
    // });
    // cs.render_microblock();

    // stage_macroblock(FileBlock("The rules on the left make a nice, simple pattern."), 2);

    // fd->manager.transition(MICRO, {
    //     {"slider", "1.1"},
    //     {"pov_xz", "0.2"}
    // });
    // cs.render_microblock();
    // fd->manager.transition(MICRO, {
    //     {"rotation_1k", "0.3"}
    // });
    // cs.render_microblock();

    // stage_macroblock(FileBlock("And the rules on the right produce a messier, yet also more interesting, result."), 3);
    // fd->manager.transition(MICRO, {
    //     {"slider", "-0.1"},
    //     {"pov_xz", "-0.2"}
    // });
    // cs.render_microblock();
    
    // fd->manager.transition(MICRO, {
    //     {"rotation_ik", "1.5"},
    //     {"rotation_jk", "0.33"}
    // });
    // cs.render_microblock();
    
    // fd->manager.transition(MICRO, {
    //     {"rotation_1k", "-0.4"}
    // });
    // cs.render_microblock();


    // stage_macroblock(FileBlock("Each side is using a 4D algebra - a framework for doing math in 4D space."), 2);
    // fd->manager.transition(MICRO, {
    //     {"slider", "0.5"},
    //     {"rotation_ik", "1.3"}
    // });
    // cs.render_microblock();
    
    // fd->manager.transition(MICRO, {
    //     {"pov_y", "0.78"},
    //     {"pov_xz", "0"}
    // });
    // cs.render_microblock();







    // stage_macroblock(FileBlock("Say we want to apply this function."), 2);
    // fd->manager.transition(MICRO, {
    //     {"brightness", "0"}
    // });
    // cs.render_microblock();
    

    // latex->begin_latex_transition(MICRO, "x^{10} - x^5 - x^2 + x");
    // fd->manager.transition(MICRO, {
    //     {"pov_y", "0.78"},
    //     {"pov_xz", "0"}
    // });
    // cs.render_microblock();


    // fd->manager.set("equation", "3");
    // fd->manager.set("offset_1", "0.24");
    // fd->manager.set("offset_2", "0.098");
    // fd->manager.set("scale", "1.8");
    // fd->manager.set("brightness", "0");
    // fd->manager.set("slider", "0.5");
    // fd->manager.set("pov_xz", "0.6");
    // fd->manager.set("pov_y", "0");
    // fd->manager.set("rotation_jk", "0");
    // fd->manager.set("rotation_ik", "0");
    // fd->manager.set("rotation_1k", "0");

    // stage_macroblock(FileBlock("An algebra gives us instructions for how to evaluate it at each point."), 2);
  
    // cs.render_microblock();
    
    // latex->begin_latex_transition(MICRO, " ");
    // fd->manager.transition(MICRO, {
    //     {"brightness", "0.0006"}
    // });
    // cs.render_microblock();



    // stage_macroblock(FileBlock("Different instructions will produce different results,"), 1);

    
    // fd->manager.transition(MICRO, {
    //     {"pov_y", "0.8"}
    // });
    // cs.render_microblock();


    // stage_macroblock(FileBlock("which is why the two graphs don't match."), 1);
  
    // fd->manager.transition(MICRO, {
    //     {"rotation_1k", "0.93"}
    // });
    // cs.render_microblock();


    // stage_macroblock(FileBlock("So, which algebra should we use?"), 1);
  
    // fd->manager.transition(MICRO, {
    //     {"rotation_1k", "1.57"},
    //     {"pov_y", "0.76"},
    //     {"brightness", "0.0003"}
    // });
    // cs.render_microblock();



    // stage_macroblock(FileBlock("Should we prefer the simplicity of the left,"), 1);
  
    // fd->manager.transition(MICRO, {
    //     {"pov_xz", "0.8"},
    //     {"slider", "1.1"}
    // });
    // cs.render_microblock();


    // stage_macroblock(FileBlock("or the complexity of the right?"), 1);
  
    // fd->manager.transition(MICRO, {
    //     {"pov_xz", "0.5"},
    //     {"slider", "-0.1"}
    // });
    // cs.render_microblock();




    // scene.manager.transition(MICRO, "function", "(a)");



    //###################################################
    //###################################################
    // CODE BELOW IS TEST CODE, NOT PART OF WRITTEN VIDEO
    //###################################################
    //###################################################


//     fd->manager.set("scale", "10.4");
//     // fd->manager.set("brightness", "0.3");
//     fd->manager.set("brightness", "0.003");
//     // fd->manager.set("rotation_1", "0.5");
//     // fd->manager.set("rotation_2", "-0.4");
//     // fd->manager.set("rotation_3", "1.2");
//     stage_macroblock(SilenceBlock(2), 1);

//    fd->manager.transition(MICRO, {
//         {"pov_xz", "1.2"}
//     });
//     cs.render_microblock();



    // fd->manager.set("scale", "12.4");
    // fd->manager.set("equation", "0");
    // fd->manager.set("brightness", "0.08");
    // fd->manager.set("offset", "0.65");


    // fd->manager.set("scale", "20.4");
    // fd->manager.set("equation", "3");
    // fd->manager.set("brightness", "20.0");
    // fd->manager.set("offset", "0.65");

    // fd->manager.set("scale", "2.9");
    // fd->manager.set("equation", "3");
    // fd->manager.set("brightness", "600.0");
    // fd->manager.set("offset", "0.35");




    // fd->manager.set("rotation_1", "0.5");
    // fd->manager.set("rotation_2", "-0.4");
    // fd->manager.set("rotation_3", "1.2");

    // stage_macroblock(SilenceBlock(15), 9);

//    fd->manager.transition(MICRO, {
//         {"pov_xz", "0.4"},
//     });
//     cs.render_microblock();

//    fd->manager.transition(MICRO, {
//         {"pov_xz", "0.8"},
//         {"slider", "1.1"}
//     });
//     cs.render_microblock();

//    fd->manager.transition(MICRO, {
//         // {"pov_xz", "1.2"},
//         {"slider", "-0.1"}
//     });
//     cs.render_microblock();
//    fd->manager.transition(MICRO, {
//         // {"pov_xz", "1.2"},
//         {"slider", "0.5"}
//     });
//     cs.render_microblock();

//    fd->manager.transition(MICRO, {
//         {"pov_xz", "1.52"}
//     });
//     cs.render_microblock();
//    fd->manager.transition(MICRO, {
//         {"pov_y", "1.52"}
//     });
//     cs.render_microblock();

//    fd->manager.transition(MICRO, {
//         {"rotation_1", "3.1415"}
//     });
//     cs.render_microblock();
//    fd->manager.transition(MICRO, {
//         {"rotation_3", "3.1415"}
//     });
//     cs.render_microblock();
//    fd->manager.transition(MICRO, {
//         {"rotation_2", "3.1415"}
//     });
//     cs.render_microblock();
//    fd->manager.transition(MICRO, {
//         {"offset", "0.6"}
//     });
//     cs.render_microblock();
//    fd->manager.transition(MICRO, {
//         {"pov_xz", "1.6"}
//     });
//     cs.render_microblock();
}

// ./go.sh FourDAlgebraDemo 240 150 10
// ./record_audios.py FourDAlgebraDemo
/*
TODO
matrix mult?
set final sin and cos limits
*/