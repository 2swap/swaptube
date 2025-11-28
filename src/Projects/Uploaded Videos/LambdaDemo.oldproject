using namespace std;
#include <string>
const string project_name = "LambdaDemo";
#include "../io/PathManager.cpp"
const int width_base = 640;
const int height_base = 360;
const float mult = 6; // 4K!!!

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;

#include "../io/writer.cpp"
#include "../misc/Timer.cpp"

#include "../Scenes/Common/ThreeDimensionScene.cpp"
#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Common/ConvolutionScene.cpp"
#include "../Scenes/Math/LambdaScene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Media/BiographyScene.cpp"
#include "../Scenes/Common/ExposedPixelsScene.cpp"
#include "../Scenes/Math/RealFunctionScene.cpp"
#include "../Scenes/Math/LambdaGraphScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"
#include "../Scenes/Connect4/Connect4Scene.cpp"

void intro() {
    ThreeDimensionScene tds;

    string factorial = "(\\n. (\\f. (((n (\\f. (\\n. (n (f (\\f. (\\x. ((n f) (f x))))))))) (\\x. f)) (\\x. x))))";
    shared_ptr<LambdaExpression> le_factorial = parse_lambda_from_string(factorial);
    le_factorial->set_color_recursive(0xff00ffff); // Cyan

    string church_3 = "(\\a. (\\b. (a (a (a b)))))";
    shared_ptr<LambdaExpression> le_church_3 = parse_lambda_from_string(church_3);
    le_church_3->set_color_recursive(0xffffff00); // Yellow

    shared_ptr<LambdaExpression> term = apply(le_factorial, le_church_3, OPAQUE_WHITE);
    term->set_color_recursive(OPAQUE_WHITE);
    term->flush_uid_recursive();

    LatexScene algebra("y = x^2 + 5x + 3", 1, VIDEO_WIDTH/2, VIDEO_HEIGHT);
    PngScene boolean("BooleanAlgebra", VIDEO_WIDTH/2, VIDEO_HEIGHT);

    shared_ptr<LambdaScene> ls = make_shared<LambdaScene>(term, 800, 800);
    tds.add_surface(Surface(glm::dvec3(0,0,0), glm::dvec3(1,0,0), glm::dvec3(0,1,0), ls));
    tds.state.set(unordered_map<string, string>{
        {"surfaces_opacity", "1"},
        {"lines_opacity", "0"},
        {"points_opacity", "1"},
        {"x", "0"},
        {"y", "0"},
        {"z", "-1"},
        {"d", "2"},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "[qj]"},
        {"qk", "0"},
    });

    int num_reductions = term->count_reductions() + 5;
    vector<string> blurbs = {"What you're watching right now is pure computation.",
                             "Not quite algebraic manipulation,",
                             "Not quite boolean logic either.",
                             "Just... pure... computation.",
                             "Specifically, it's evaluating 3 factorial, and sure enough, it found the result, 6.",
                            };
    CompositeScene cs;
    cs.add_scene(&tds, "tds", 0, 0);
    cs.add_scene(&algebra, "alg", 0, 0);
    cs.add_scene(&boolean, "boo", 0.5, 0.05); // Boolean on the right, algebra on the left
    cs.state.add_equation("alg.opacity", "0");
    cs.state.add_equation("boo.opacity", "0");
    cs.state.add_equation("qj", "0");
    for(int i = 0; i < 5; i++){
        float alg_o = i==1;
        float boo_o = i==2;
        cs.state.add_subscene_transition("alg.opacity", to_string(alg_o));
        cs.state.add_subscene_transition("boo.opacity", to_string(boo_o));
        float qj = 0;
        if(i==1) qj=.19;
        if(i==2) qj=-.19;
        cs.state.add_subscene_transition("qj", to_string(qj));
        cs.stage_macroblock(AudioSegment(blurbs[i]), num_reductions / 5);
        for(int j = 0; j < num_reductions/5; j++) {
            ls->reduce();
            cs.render_microblock();
        }
    }

    tds.stage_macroblock_and_render(AudioSegment(0.5));
    tds.stage_macroblock_and_render(AudioSegment("What are all these weird lines though?"));

    ls->set_expression(term);
    tds.stage_macroblock_and_render(AudioSegment("Going back to the original setup,"));

    dynamic_pointer_cast<LambdaApplication>(term)->get_first()->set_color_recursive(0xffff00ff);
    ls->set_expression(term);
    tds.stage_macroblock(AudioSegment("This pink chunk represents the factorial function."), 3);
    tds.render_microblock();
    tds.render_microblock();
    tds.render_microblock();
    dynamic_pointer_cast<LambdaApplication>(term)->get_second()->set_color_recursive(0xffffff00);
    ls->set_expression(term);
    tds.stage_macroblock(AudioSegment("This yellow chunk represents the number 3."), 3);
    tds.render_microblock();
    tds.render_microblock();
    tds.render_microblock();
    term->set_color(0xff0088ff);
    ls->set_expression(term);
    tds.stage_macroblock(AudioSegment("And this blue chunk represents the application of the function to the number."), 3);
    tds.render_microblock();
    tds.render_microblock();
    tds.render_microblock();

    tds.stage_macroblock_and_render(AudioSegment("We can make all sorts of other values."));

    tds.state.subscene_transition(unordered_map<string, string>{
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"}
    });

    tds.state.subscene_transition(unordered_map<string, string>{
        {"z", "-10"},
    });

    shared_ptr<LambdaExpression> church1 = parse_lambda_from_string("(\\f. (\\x. (f x)))");
    church1->set_color_recursive(0xffff0000);
    church1->flush_uid_recursive();
    shared_ptr<LambdaScene> church1scene = make_shared<LambdaScene>(church1, 800, 800);
    tds.add_surface(Surface(glm::dvec3(-2,-2,-1), glm::dvec3(1,0,0), glm::dvec3(0,1,0), church1scene));

    shared_ptr<LambdaExpression> church2 = parse_lambda_from_string("(\\f. (\\x. (f (f x))))");
    church2->set_color_recursive(0xff00ff00);
    church2->flush_uid_recursive();
    shared_ptr<LambdaScene> church2scene = make_shared<LambdaScene>(church2, 800, 800);
    tds.add_surface(Surface(glm::dvec3(2,2,-3), glm::dvec3(1,0,0), glm::dvec3(0,1,0), church2scene));

    shared_ptr<LambdaExpression> church3 = parse_lambda_from_string("(\\f. (\\x. (f (f (f x)))))");
    church3->set_color_recursive(0xff0088ff);
    church3->flush_uid_recursive();
    shared_ptr<LambdaScene> church3scene = make_shared<LambdaScene>(church3, 800, 800);
    tds.add_surface(Surface(glm::dvec3(2,-2,-5), glm::dvec3(1,0,0), glm::dvec3(0,1,0), church3scene));
    tds.stage_macroblock_and_render(AudioSegment("We've got one, two, three..."));
    
    tds.state.subscene_transition(unordered_map<string, string>{
        {"q1", "1"},
        {"qi", "0"},
        {"qj", ".4"},
        {"qk", "0"},
        {"d", "0"},
        {"x", "0"},
        {"y", "0"},
        {"z", "-10"},
    });
    shared_ptr<LambdaExpression> churchplus = parse_lambda_from_string("(\\m. (\\n. (\\f. (\\x. ((m f) ((n f) x))))))");
    churchplus->set_color_recursive(0xffff8800);
    churchplus->flush_uid_recursive();
    shared_ptr<LambdaScene> churchplusscene = make_shared<LambdaScene>(churchplus, 800, 800);
    tds.add_surface(Surface(glm::dvec3(-5,1,-9), glm::dvec3(0,0,1), glm::dvec3(0,1,0), churchplusscene));

    shared_ptr<LambdaExpression> churchtimes = parse_lambda_from_string("(\\m. (\\n. (\\a. (m (n a)))))");
    churchtimes->set_color_recursive(0xff0088ff);
    churchtimes->flush_uid_recursive();
    shared_ptr<LambdaScene> churchtimesscene = make_shared<LambdaScene>(churchtimes, 800, 800);
    tds.add_surface(Surface(glm::dvec3(-5,-1,-11), glm::dvec3(0,0,1), glm::dvec3(0,1,0), churchtimesscene));
    tds.stage_macroblock_and_render(AudioSegment("as well as plus and times."));
    tds.remove_surface(church1scene);
    tds.remove_surface(church2scene);
    tds.remove_surface(church3scene);
    tds.remove_surface(ls);




    le_factorial->flush_uid_recursive();
    le_church_3->flush_uid_recursive();
    le_factorial->set_color_recursive(0xffff00ff);
    le_church_3->set_color_recursive(0xffffff00);
    shared_ptr<LambdaExpression> term1 = apply(le_factorial, le_church_3, 0xff222222);
    shared_ptr<LambdaScene> ls1 = make_shared<LambdaScene>(term1, 600, 600);
    tds.add_surface(Surface(glm::dvec3(0,0,-8), glm::dvec3(1,0,0), glm::dvec3(0,1,0), ls1));

    tds.stage_macroblock(AudioSegment("We can express any computational procedure, such as the factorial function."), 2);
    tds.render_microblock();
    tds.state.subscene_transition(unordered_map<string, string>{
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"d", "2"},
    });
    tds.render_microblock();
    tds.remove_surface(churchtimesscene);
    tds.remove_surface(churchplusscene);
    tds.stage_macroblock(AudioSegment("But... the magic is that it's not immediately obvious whether a certain expression is a number,"), 4);
    dynamic_pointer_cast<LambdaApplication>(term1)->get_first()->set_color_recursive(0xff222222);
    dynamic_pointer_cast<LambdaApplication>(term1)->get_second()->set_color_recursive(0xff222222);
    ls1->set_expression(term1);
    tds.render_microblock();
    tds.render_microblock();
    dynamic_pointer_cast<LambdaApplication>(term1)->get_first()->set_color_recursive(0xff222222);
    dynamic_pointer_cast<LambdaApplication>(term1)->get_second()->set_color_recursive(0xffffffff);
    ls1->set_expression(term1);
    tds.render_microblock();
    tds.render_microblock();
    dynamic_pointer_cast<LambdaApplication>(term1)->get_first()->set_color_recursive(0xffffffff);
    dynamic_pointer_cast<LambdaApplication>(term1)->get_second()->set_color_recursive(0xff222222);
    ls1->set_expression(term1);
    tds.stage_macroblock(AudioSegment("a function that operates on numbers,"), 2);
    tds.render_microblock();
    tds.render_microblock();
    dynamic_pointer_cast<LambdaApplication>(term1)->get_first()->set_color_recursive(0xffff00ff);
    dynamic_pointer_cast<LambdaApplication>(term1)->get_second()->set_color_recursive(0xffffff00);
    term1->set_color(0xff00ffff);
    ls1->set_expression(term1);
    tds.stage_macroblock(AudioSegment("or something else entirely."), 2);
    tds.render_microblock();
    tds.render_microblock();

    tds.stage_macroblock_and_render(AudioSegment("And that's because, in this language, there _is no difference_."));
    shared_ptr<LambdaExpression> term2 = apply(le_factorial, le_church_3, 0xff00ffff);
    ls1->set_expression(term2);
    tds.stage_macroblock_and_render(AudioSegment("Just like we applied factorial to 3 with function application,"));
    shared_ptr<LambdaExpression> term3 = apply(le_church_3, le_factorial, 0xff00ffff);
    ls1->set_expression(term3);
    tds.stage_macroblock_and_render(AudioSegment("we can apply 3 to factorial in the exact same way, as though 3 was a function and factorial was a value."));
    term3->set_color_recursive(0xff404040);
    ls1->set_expression(term3);

    tds.stage_macroblock_and_render(AudioSegment("OK, but you can't actually evaluate that, right?"));
    num_reductions = 3;
    tds.stage_macroblock(AudioSegment("Well, you can... but it's certainly not the case that when evaluating it, the answer would make any sense... right?"), num_reductions);
    for(int j = 0; j < num_reductions; j++) {
        ls1->reduce();
        tds.render_microblock();
    }

    tds.stage_macroblock(AudioSegment("We're gonna have to totally unlearn the concepts of functions, programs, values, and datatypes."), num_reductions);
    for(int j = 0; j < num_reductions; j++) {
        ls1->reduce();
        tds.render_microblock();
    }

    // Create text which says Lambda Calculus behind where the camera currently is
    shared_ptr<LatexScene> title = make_shared<LatexScene>(latex_text("The \\lambda -Calculus"), 1, 1000, 1000);
    tds.add_surface(Surface(glm::dvec3(0,0,-14), glm::dvec3(1,0,0), glm::dvec3(0,1,0), title));

    // Also add a bunch of grey lambda diagrams parallel to the title with z=12
    vector<shared_ptr<LambdaScene>> lots_of_lambdas;

    // Define the number of lambda scenes and their scattering parameters
    int num_lambdas = 10;
    float scatter_range_x = 0.3f;  // Range for random scattering
    float scatter_range_y = 0.6f;  // Range for random scattering

    // List of unique lambda expressions that require a larger number of reductions
    vector<string> complex_lambdas = {
        "(((\\x. (\\y. (y ((x x) y)))) (\\x. (\\y. (y ((x x) y))))) (\\z. z))",
        "(((\\f. (\\x. (f (f (f (f x)))))) (\\y. (y y))) (\\z. z))",
        "(((\\m. (\\n. (m (n m)))) (\\a. (a a))) (\\c. ((c c) (c c))))",
        "(((\\f. (\\x. (f (f (f x))))) (\\y. (y y))) (\\z. z))",
        "(((\\m. (\\n. (m (n ((m m) n))))) (\\x. x)) (\\y. (y y)))",
        "(((\\x. (\\y. (y ((x x) y)))) (\\x. (\\y. (y ((x x) y))))) (\\x. (\\y. (y ((x x) y)))))",
        "(((\\f. (\\x. (f (f (f (f x)))))) (\\y. (y y))) (\\x. (\\y. (y ((x x) y)))))",
        "(((\\m. (\\n. (m (n m)))) (\\x. (\\y. (y ((x x) y))))) (\\c. ((c c) (c c))))",
        "(((\\f. (\\x. (f (f (f x))))) (\\y. (y y))) (\\x. (\\y. (y ((x x) y)))))",
        "(((\\m. (\\n. (m (n ((m m) n))))) (\\x. (\\y. (y ((x x) y))))) (\\y. (y y)))",
    };

    rand();
    for (int i = 0; i < num_lambdas * 2; ++i) {
        // Parse and color each lambda expression
        shared_ptr<LambdaExpression> le_complex_lambda = parse_lambda_from_string(complex_lambdas[i%num_lambdas]);
        le_complex_lambda->set_color_recursive(0xff404040);  // Set color to grey

        shared_ptr<LambdaScene> lambda_scene = make_shared<LambdaScene>(le_complex_lambda, 400, 400);
        
        // Randomize position and orientation
        float x_position = ((i % 5) + ((rand() % 1000) / 1000.0f * scatter_range_x - scatter_range_x / 2) - 2) * 2;
        float y_position = ((i / 5) + ((rand() % 1000) / 1000.0f * scatter_range_y - scatter_range_y / 2) - 1.5) * 1.5 + i%2- .5;
        float z_position = -13.5 + 2.5 * (rand()%1000)/1000.0;
        float theta = ((rand()%1000)/1000.0f-0.5) * 0.2;
        glm::dvec3 random_tilt_x(cos(theta), sin(theta), 0);
        glm::dvec3 random_tilt_y(-sin(theta), cos(theta), 0);

        tds.add_surface(Surface(glm::dvec3(x_position, y_position, z_position), random_tilt_x * 0.8f, random_tilt_y * 0.8f, lambda_scene));
        
        // Store the lambda scene in the vector for later reduction
        lots_of_lambdas.push_back(lambda_scene);
    }

    // Transition back to be able to see it
    tds.state.superscene_transition(unordered_map<string, string>{
        {"z", "-16"},
    });

    tds.stage_macroblock(AudioSegment("'Cause today, we're learning the lambda calculus."), 8);

    // Reduce all the lambdas in the background in a loop
    for(int i = 0; i < 8; i++) {
        ls1->reduce();
        if(i > 1)
            for (auto& lambda_scene : lots_of_lambdas) {
                lambda_scene->reduce();  // Reduce the lambda expression
            }
        tds.render_microblock();  // Render the scene after each reduction
    }

    tds.state.superscene_transition(unordered_map<string, string>{
        {"z", "20"},
        {"qk", "12"},
    });

    tds.stage_macroblock(AudioSegment(5), 5);
    // Reduce all the lambdas in the background in a loop
    for(int i = 0; i < 5; i++) {
        ls1->reduce();
        for (auto& lambda_scene : lots_of_lambdas) {
            lambda_scene->reduce(); // Reduce the lambda expression
        }
        tds.render_microblock(); // Render the scene after each reduction
    }
}

void history() {
    CompositeScene cs;
    cs.stage_macroblock_and_render(AudioSegment("But what even is computation?"));

    // Create Hilbert's BiographyScene on the left, with a quote on the right
    BiographyScene hilbert("hilbert", "David Hilbert", {}, VIDEO_WIDTH/2, VIDEO_HEIGHT);
    cs.add_scene_fade_in(&hilbert, "hilbert", 0, 0, true);
    LatexScene hilbert_quote(latex_text("Is there a program which can tell\\\\\\\\if a theorem is true or false?"), 0.9, VIDEO_WIDTH/2, VIDEO_HEIGHT/2);
    cs.stage_macroblock(AudioSegment("David Hilbert, one of the greatest mathematicians of the 1900s, wanted to know whether there was some procedure, some algorithm, which can determine whether any given mathematical statement is true or false."), 4);
    cs.render_microblock();
    cs.render_microblock();
    cs.add_scene_fade_in(&hilbert_quote, "hilbert_quote", .5, .25, true);
    cs.render_microblock();
    cs.render_microblock();

    // Move Hilbert to the top half to make room for other mathematicians
    hilbert.state.set(unordered_map<string, string>{
        {"h", "[hilbert.h]"},
    });
    cs.state.set(unordered_map<string, string>{
        {"hilbert.h", to_string(VIDEO_HEIGHT)},
    });
    cs.state.superscene_transition(unordered_map<string, string>{
        {"hilbert.h", to_string(.5 * VIDEO_HEIGHT)},
    });

    // Introduce Church, Turing, and Gödel, moving them from the bottom of the screen, breaking the blurb into parts
    BiographyScene church("church", "Alonzo Church", {}, VIDEO_WIDTH/3, VIDEO_HEIGHT/2);
    BiographyScene turing("turing", "Alan Turing", {}, VIDEO_WIDTH/3, VIDEO_HEIGHT/2);
    BiographyScene godel("godel", "Kurt Gödel", {}, VIDEO_WIDTH/3, VIDEO_HEIGHT/2);

    cs.add_scene(&church, "church", 0, 1);
    cs.add_scene(&turing, "turing", 0.33333, 1);
    cs.add_scene(&godel, "godel", 0.66666, 1);
    cs.state.superscene_transition(unordered_map<string, string>{
        {"turing.y", "0.5"},
        {"godel.y", "0.5"},
        {"church.y", "0.5"},
        {"hilbert_quote.y", "0"},
    });

    // Break up the audio and append to the relevant biographies
    cs.stage_macroblock_and_render(AudioSegment("Three men independently answered this question in different ways."));
    church.append_bio_text("Invented Lambda Calculus");
    turing.append_bio_text("Invented the Turing Machine");
    godel.append_bio_text("Invented General-Recursive Functions");
    cs.stage_macroblock_and_render(AudioSegment("The ideas they encountered along the way were so groundbreaking that they proved Hilbert's task impossible,"));
    church.append_bio_text("Created functional paradigm");
    turing.append_bio_text("Gave way to imperative paradigm");
    cs.stage_macroblock_and_render(AudioSegment("spawned two of the paradigms underlying modern programming languages,"));
    godel.append_bio_text("Gödel's Incompleteness Theorems");
    cs.stage_macroblock_and_render(AudioSegment("showed that mathematics is essentially incomplete,"));
    turing.append_bio_text("Named Father of Computer Science");
    cs.stage_macroblock_and_render(AudioSegment("and spawned the entire field of computer science."));
    hilbert_quote.begin_latex_transition(latex_text("Is there a " + latex_color(0xffff7777, "program") + " which can tell\\\\\\\\if a theorem is true or false?"));
    cs.state.subscene_transition(unordered_map<string, string>{
        {"turing.opacity", "0.15"},
        {"godel.opacity", "0.15"},
        {"church.opacity", "0.15"},
        {"hilbert.opacity", "0.15"},
        {"hilbert_quote.x", "0.25"},
        {"hilbert_quote.y", "0.25"},
    });

    cs.stage_macroblock(AudioSegment("But Hilbert's question is a question about procedures. About ways of doing computation."), 2);
    cs.render_microblock();
    cs.render_microblock();
    hilbert_quote.begin_latex_transition(latex_text("Is there a program which can tell\\\\\\\\if a theorem is true or false?"));
    cs.state.superscene_transition(unordered_map<string, string>{
        {"hilbert_quote.x", "0.5"},
        {"hilbert_quote.y", "0"},
        {"turing.opacity", "1"},
        {"godel.opacity", "1"},
        {"church.opacity", "1"},
        {"hilbert.opacity", "1"},
    });
    cs.stage_macroblock_and_render(AudioSegment("We won't answer his question explicitly here."));
    cs.stage_macroblock_and_render(AudioSegment("My goal, instead, is to give you a more visceral understanding of how computation itself can be formalized."));

    // Slide Turing and Gödel out the right side, and introduce a LatexScene title "The \\lambda-Calculus" on the right side
    cs.state.superscene_transition(unordered_map<string, string>{
        {"turing.y", "1.5"},
        {"godel.y", "1.5"},
        {"hilbert.y", "-1"},
        {"hilbert_quote.y", "-1"},
        {"church.y", ".35"},
    });

    LatexScene lambda_title(latex_text("The \\lambda-Calculus"), 1, VIDEO_WIDTH, VIDEO_HEIGHT*0.25);
    cs.add_scene(&lambda_title, "lambda_title", 0, -0.25);
    cs.state.superscene_transition(unordered_map<string, string>{
        {"lambda_title.y", "0"},
    });
    cs.stage_macroblock_and_render(AudioSegment("This video is the story of Alonzo Church's answer."));
    cs.remove_scene(&hilbert);
    cs.remove_scene(&turing);
    cs.remove_scene(&godel);
    LatexScene calc("\\frac{d}{dx}", 1, VIDEO_WIDTH*.5, VIDEO_HEIGHT*.5);
    cs.add_scene_fade_in(&calc, "calc", 0.4, 0.25, true);
    cs.stage_macroblock(AudioSegment("His 'Lambda Calculus', to be clear, had nothing to do with derivatives or integrals or what you learned in Calculus class in High School."), 2);
    cs.render_microblock();
    calc.begin_latex_transition("\\int_a^b");
    cs.render_microblock();
    cs.state.superscene_transition(unordered_map<string, string>{
        {"calc.opacity", "0"},
    });
    cs.stage_macroblock_and_render(AudioSegment("The term calculus, up until recently, was used to describe all sorts of logical systems."));

    // Add LatexScenes showing Lambda expressions
    LatexScene lambda_examples1("(\\lambda x. x)",                   1, VIDEO_WIDTH*.333, VIDEO_HEIGHT*.333);
    LatexScene lambda_examples2("y",                                 1, VIDEO_WIDTH*.333, VIDEO_HEIGHT*.333);
    LatexScene lambda_examples3("(\\lambda z. (z (\\lambda w. w)))", 1, VIDEO_WIDTH*.333, VIDEO_HEIGHT*.333);
    LatexScene lambda_examples4("(a (b c))"                        , 1, VIDEO_WIDTH*.333, VIDEO_HEIGHT*.333);
    cs.add_scene_fade_in(&lambda_examples1, "lambda_examples1", 0.333, 0.266, true);
    cs.add_scene_fade_in(&lambda_examples2, "lambda_examples2", 0.333, 0.6  , true);
    cs.add_scene_fade_in(&lambda_examples3, "lambda_examples3", 0.666, 0.266, true);
    cs.add_scene_fade_in(&lambda_examples4, "lambda_examples4", 0.666, 0.6  , true);
    cs.stage_macroblock_and_render(AudioSegment("It might as well be called 'The Way of the Lambda'."));
    LatexScene algebra("x+4=2x+1", .7, VIDEO_WIDTH*.666, VIDEO_HEIGHT*.666);
    cs.add_scene_fade_in(&algebra, "algebra", 0.333, 0.266, true);
    cs.state.subscene_transition(unordered_map<string, string>{
        {"lambda_examples1.opacity", "0.1"},
        {"lambda_examples2.opacity", "0.1"},
        {"lambda_examples3.opacity", "0.1"},
        {"lambda_examples4.opacity", "0.1"},
    });
    cs.stage_macroblock_and_render(AudioSegment("It's fundamentally a system for manipulating strings of symbols,"));
    cs.stage_macroblock(AudioSegment("just like Algebra teaches us to manipulate numbers and pluses and equals signs."), 4);
    algebra.begin_latex_transition("x+4-1=2x+1-1");
    cs.render_microblock();
    algebra.begin_latex_transition("x+3=2x");
    cs.render_microblock();
    algebra.begin_latex_transition("x-x+3=2x-x");
    cs.render_microblock();
    algebra.begin_latex_transition("3=x");
    cs.render_microblock();
    lambda_examples1.begin_latex_transition(latex_color(0xff00ff00, "(\\lambda x. x)"                  ));
    lambda_examples2.begin_latex_transition(latex_color(0xff00ff00, "y"                                ));
    lambda_examples3.begin_latex_transition(latex_color(0xff00ff00, "(\\lambda z. (z (\\lambda w. w)))"));
    lambda_examples4.begin_latex_transition(latex_color(0xff00ff00, "(a (b c))"                        ));
    cs.state.subscene_transition(unordered_map<string, string>{
        {"lambda_examples1.opacity", "1"},
        {"lambda_examples2.opacity", "1"},
        {"lambda_examples3.opacity", "1"},
        {"lambda_examples4.opacity", "1"},
        {"algebra.opacity", "0"},
    });
    cs.stage_macroblock(AudioSegment("In the lambda calculus, the strings look something like this."), 2);
    cs.render_microblock();
    lambda_examples1.begin_latex_transition("(\\lambda x. x)"                  );
    lambda_examples2.begin_latex_transition("y"                                );
    lambda_examples3.begin_latex_transition("(\\lambda z. (z (\\lambda w. w)))");
    lambda_examples4.begin_latex_transition("(a (b c))"                        );
    cs.render_microblock();
    lambda_examples1.begin_latex_transition(latex_color(0xffff0000, "(") + latex_color(0xff444444, "\\lambda x. x") + latex_color(0xffff0000, ")"));
    lambda_examples2.begin_latex_transition(latex_color(0xff444444, "y"));
    lambda_examples3.begin_latex_transition(latex_color(0xffff0000, "(") + latex_color(0xff444444, "\\lambda z. ") + latex_color(0xffff0000, "(") + latex_color(0xff444444, "z ") + latex_color(0xffff0000, "(") + latex_color(0xff444444, "\\lambda w. w") + latex_color(0xffff0000, ")))"));
    lambda_examples4.begin_latex_transition(latex_color(0xffff0000, "(") + latex_color(0xff444444, "a ") + latex_color(0xffff0000, "(") + latex_color(0xff444444, "b c") + latex_color(0xffff0000, "))"));
    cs.stage_macroblock(AudioSegment("They're composed of parentheses,"), 2);
    cs.render_microblock();
    cs.render_microblock();
    cs.stage_macroblock_and_render(AudioSegment(.5));
    lambda_examples1.begin_latex_transition(latex_color(0xff444444, "(\\lambda") + latex_color(0xff00ff00, "x") + latex_color(0xff444444, ".") + latex_color(0xff00ff00, "x") + latex_color(0xff444444, ")"));
    lambda_examples2.begin_latex_transition(latex_color(0xff00ff00, "y"));
    lambda_examples3.begin_latex_transition(latex_color(0xff444444, "(\\lambda ") + latex_color(0xff00ff00, "z") + latex_color(0xff444444, ". (") + latex_color(0xff00ff00, "z ") + latex_color(0xff444444, "(\\lambda") + latex_color(0xff00ff00, "w") + latex_color(0xff444444, ".") + latex_color(0xff00ff00, "w") + latex_color(0xff444444, ")))"));
    lambda_examples4.begin_latex_transition(latex_color(0xff444444, "(") + latex_color(0xff00ff00, "a ") + latex_color(0xff444444, "(") + latex_color(0xff00ff00, "b c") + latex_color(0xff444444, "))"));
    cs.stage_macroblock(AudioSegment("letters of the alphabet,"), 2);
    cs.render_microblock();
    cs.render_microblock();
    cs.stage_macroblock_and_render(AudioSegment(.5));
    lambda_examples1.begin_latex_transition(latex_color(0xff444444, "(") + latex_color(0xff0044ff, "\\lambda") + latex_color(0xff444444, "x") + latex_color(0xff0044ff, ".") + latex_color(0xff444444, "x)"));
    lambda_examples2.begin_latex_transition(latex_color(0xff444444, "y"));
    lambda_examples3.begin_latex_transition(latex_color(0xff444444, "(") + latex_color(0xff0044ff, "\\lambda") + latex_color(0xff444444, "z ") + latex_color(0xff0044ff, ".") + latex_color(0xff444444, "(z (") + latex_color(0xff0044ff, "\\lambda") + latex_color(0xff444444, "w ") + latex_color(0xff0044ff, ".") + latex_color(0xff444444, "w)))"));
    lambda_examples4.begin_latex_transition(latex_color(0xff444444, "(a (b c))"));
    cs.stage_macroblock(AudioSegment("and this notation involving a lambda and a dot. Those two always come together in a pair."), 2);
    cs.render_microblock();
    cs.render_microblock();
    cs.stage_macroblock_and_render(AudioSegment(.5));

    // On the left, add the production rules of the lambda calculus
    LatexScene lambda_rule_var(latex_color(0xffffff00, "a"                ), 0.8, VIDEO_WIDTH*.5, VIDEO_HEIGHT*.25);
    LatexScene lambda_rule_abs(latex_color(0xffff00ff, "(\\lambda a. \\_)"), 0.8, VIDEO_WIDTH*.5, VIDEO_HEIGHT*.25);
    LatexScene lambda_rule_app(latex_color(0xff00ffff, "(\\_ \\_)"        ), 0.8, VIDEO_WIDTH*.5, VIDEO_HEIGHT*.25);
    cs.add_scene(&lambda_rule_var, "lambda_rule_var", -0.5, 0.25);
    cs.add_scene(&lambda_rule_abs, "lambda_rule_abs", -0.5, 0.45);
    cs.add_scene(&lambda_rule_app, "lambda_rule_app", -0.5, 0.65);

    // Slide Church out to the left
    cs.state.superscene_transition(unordered_map<string, string>{
        {"church.x", "-1"},
        {"lambda_examples1.x", "1.5"},
        {"lambda_examples2.x", "1.5"},
        {"lambda_examples3.x", "1.5"},
        {"lambda_examples4.x", "1.5"},
        {"lambda_rule_var.x"    , "0"},
        {"lambda_rule_abs.x"    , "0"},
        {"lambda_rule_app.x"    , "0"},
    });
    cs.stage_macroblock_and_render(AudioSegment("We can build these strings ourselves, following 3 templates."));
    cs.remove_scene(&lambda_examples1);
    cs.remove_scene(&lambda_examples2);
    cs.remove_scene(&lambda_examples3);
    cs.remove_scene(&lambda_examples4);


    lambda_rule_var.begin_latex_transition(latex_color(0xffffff00, "a"                ));
    lambda_rule_abs.begin_latex_transition(latex_color(0xffff00ff, "(\\lambda a. "+latex_color(0xffffffff, "\\_") + ")"));
    lambda_rule_app.begin_latex_transition(latex_color(0xff00ffff, "("+latex_color(0xffffffff, "\\_ \\_") + ")"        ));
    cs.stage_macroblock(AudioSegment("The blanks are places where you can put any other template."), 3);
    cs.render_microblock();
    cs.render_microblock();
    lambda_rule_var.begin_latex_transition(latex_color(0xffffff00, "a"));
    lambda_rule_abs.begin_latex_transition(latex_color(0xffff00ff, "(\\lambda a. \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xff00ffff, "(\\_ \\_)"));
    cs.render_microblock();
    lambda_rule_var.begin_latex_transition(latex_color(0xffffffff, "a"));
    lambda_rule_abs.begin_latex_transition(latex_color(0xffff00ff, "(\\lambda "+latex_color(0xffffffff, "a") + ". \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xff00ffff, "(\\_ \\_)"));
    cs.stage_macroblock(AudioSegment("and the 'a's can be substituted for any letter in the alphabet."), 3);
    cs.render_microblock();
    cs.render_microblock();
    lambda_rule_var.begin_latex_transition(latex_color(0xffffff00, "a"));
    lambda_rule_abs.begin_latex_transition(latex_color(0xffff00ff, "(\\lambda a. \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xff00ffff, "(\\_ \\_)"));
    cs.render_microblock();
    LatexScene lambda_construct(latex_color(0xff00ffff, "(\\_ \\_)"), 0.8, VIDEO_WIDTH*.5, VIDEO_HEIGHT*.25);
    cs.add_scene(&lambda_construct, "lambda_construct", 0, 0.65);

    cs.state.superscene_transition(unordered_map<string, string>{
        {"lambda_construct.x", ".5"},
        {"lambda_construct.y", ".25"},
    });
    cs.stage_macroblock_and_render(AudioSegment("As an example, we can grab the third template,"));
    lambda_construct.begin_latex_transition(latex_color(0xffffff00, "a") + "\\\\\\\\" + latex_color(0xff00ffff, "(\\_ \\_)"));
    cs.stage_macroblock_and_render(AudioSegment("take a variable from the first template,"));
    lambda_construct.begin_latex_transition(latex_color(0xff00ffff, "("+latex_color(0xffffff00, "a")+ "\\_)"));
    cs.stage_macroblock_and_render(AudioSegment("slap it in one of the blanks,"));
    lambda_construct.begin_latex_transition(latex_color(0xffff00ff, "(\\lambda a. \\_)") + "\\\\\\\\" + latex_color(0xff00ffff, "("+latex_color(0xffffff00, "a")+ "\\_)"));
    cs.stage_macroblock_and_render(AudioSegment("take template 2,"));
    lambda_construct.begin_latex_transition(latex_color(0xff00ffff, "("+latex_color(0xffffff00, "a") + latex_color(0xffff00ff, "\\ (\\lambda a. \\_)") + ")"));
    cs.stage_macroblock_and_render(AudioSegment("and slap it in the other."));
    lambda_construct.begin_latex_transition(latex_color(0xff00ffff, "("+latex_color(0xffffff00, "a") + latex_color(0xffff00ff, "\\ (\\lambda a. "+latex_color(0xffffff00, "a")+")") + ")"));
    cs.stage_macroblock_and_render(AudioSegment("There's still a blank left, so we can put another variable in."));
    cs.stage_macroblock_and_render(AudioSegment(.5));
    lambda_construct.begin_latex_transition("(a\\ (\\lambda a. a))");
    cs.stage_macroblock_and_render(AudioSegment("And just like that, we've made a lambda expression."));
    lambda_construct.begin_latex_transition("(x\\ (\\lambda y. z))");
    cs.stage_macroblock_and_render(AudioSegment("Remember, these a's can be changed to any letter we want."));
    LatexScene lc2("(\\_ \\_)", 0.7, VIDEO_WIDTH*.5, VIDEO_HEIGHT*.25);
    cs.add_scene_fade_in(&lc2, "lc2", 0.5, 0.4, true);
    cs.stage_macroblock(AudioSegment("Any combination of these templates forms a valid expression."), 12);
    lc2.begin_latex_transition("(\\_ (\\_ \\_))");
    cs.render_microblock();
    lc2.begin_latex_transition("(\\_ (\\_ (\\_ \\_)))");
    cs.render_microblock();
    lc2.begin_latex_transition("(\\_ (\\_ (\\_ (\\_ \\_))))");
    cs.render_microblock();
    lc2.begin_latex_transition("(\\_ (\\_ (\\_ ((\\_ \\_) \\_))))");
    cs.render_microblock();
    lc2.begin_latex_transition("(\\_ (\\_ (\\_ (((\\_ \\_) \\_) \\_))))");
    cs.render_microblock();
    lc2.begin_latex_transition("(t (\\_ (\\_ (((\\_ \\_) \\_) \\_))))");
    cs.render_microblock();
    lc2.begin_latex_transition("(t (w (\\_ (((\\_ \\_) \\_) \\_))))");
    cs.render_microblock();
    lc2.begin_latex_transition("(t (w (o (((\\_ \\_) \\_) \\_))))");
    cs.render_microblock();
    lc2.begin_latex_transition("(t (w (o (((s \\_) \\_) \\_))))");
    cs.render_microblock();
    lc2.begin_latex_transition("(t (w (o (((s w) \\_) \\_))))");
    cs.render_microblock();
    lc2.begin_latex_transition("(t (w (o (((s w) a) \\_))))");
    cs.render_microblock();
    lc2.begin_latex_transition("(t (w (o (((s w) a) p))))");
    cs.render_microblock();
    LatexScene lc3("(\\lambda a. (\\lambda b. (a b)))", 0.8, VIDEO_WIDTH*.5, VIDEO_HEIGHT*.25);
    cs.add_scene_fade_in(&lc3, "lc3", 0.5, 0.55, true);
    cs.stage_macroblock_and_render(AudioSegment("Using them, we can create all sorts of different lambda terms."));
    cs.state.superscene_transition(unordered_map<string, string>{
        {"lambda_rule_var.x", ".5"},
        {"lambda_rule_var.y", ".7"},
    });
    lambda_rule_var.begin_latex_transition("a");
    cs.stage_macroblock_and_render(AudioSegment("Even a variable itself is a valid expression."));
    cs.stage_macroblock_and_render(AudioSegment(0.5));
    cs.state.subscene_transition(unordered_map<string, string>{
        {"lambda_rule_var.opacity", "0"},
        {"lambda_construct.opacity", "0"},
        {"lc2.opacity", "0"},
        {"lc3.opacity", "0"},
        {"lambda_rule_app.x", ".5"},
        {"lambda_rule_app.y", ".45"},
    });
    cs.stage_macroblock(AudioSegment("Now, one thing you should know- these two templates have special interpretations."), 2);
    cs.render_microblock();
    cs.render_microblock();
    /*
    cs.stage_macroblock_and_render(AudioSegment("The first rule says that any letter is a valid lambda expression."));

    // Highlight the first rule for variables
    cs.stage_macroblock(AudioSegment("a, b, c, you name it."), 4);

    lambda_rule_var.begin_latex_transition("b");
    cs.render_microblock();

    lambda_rule_var.begin_latex_transition("c");
    cs.render_microblock();

    lambda_rule_var.begin_latex_transition("a");
    cs.render_microblock();

    // Fade out the first rule by transitioning its opacity to 0.
    cs.state.subscene_transition(unordered_map<string, string>{
        {"lambda_rule_var.x", "0"},
        {"lambda_rule_var.y", ".25"},
    });
    cs.render_microblock();

    // Move out the second rule.
    cs.state.superscene_transition(unordered_map<string, string>{
        {"lambda_rule_abs.x", ".5"},
        {"lambda_rule_abs.y", ".5"},
    });
    cs.stage_macroblock_and_render(AudioSegment("The second rule says that we can make a valid expression of this form."));

    // Highlight the abstraction rule, keeping the variable "a" constant
    lambda_rule_abs.begin_latex_transition("(\\lambda a. \\_)");
    cs.stage_macroblock(AudioSegment("Once again, the 'a' represents any letter."), 3);

    // Shuffle through letters, transitioning back to 'a' like before
    lambda_rule_abs.begin_latex_transition("(\\lambda b. \\_)"); cs.render_microblock();
    lambda_rule_abs.begin_latex_transition("(\\lambda c. \\_)"); cs.render_microblock();
    lambda_rule_abs.begin_latex_transition("(\\lambda a. \\_)"); cs.render_microblock();

    // Highlight the blank in white, make the rest of the term gray.
    lambda_rule_abs.begin_latex_transition(latex_color(0xff444444, "(\\lambda a. ") + latex_color(0xffffffff, "\\_") + latex_color(0xff444444, ")"));
    cs.stage_macroblock_and_render(AudioSegment("The blank, in this case, is a placeholder for any other valid lambda expression."));

    // Transition the latex to have an 'a' where the blank was.
    lambda_rule_abs.begin_latex_transition("(\\lambda a. a)");
    cs.stage_macroblock_and_render(AudioSegment("That could, for example, be a lone variable, such as the ones which we made in expression 1."));

    // Show the completed valid lambda expression.
    cs.stage_macroblock_and_render(AudioSegment("So, this is a valid lambda expression which matches the proper form."));

    // Slide out another modifiable copy of the abstraction rule and place the last expression inside of it.
    cs.stage_macroblock(AudioSegment("And therefore, it can also be placed inside the blank of the same rule."), 2);
    lambda_rule_abs.begin_latex_transition("(\\lambda x. \\_)\\\\\\\\(\\lambda a. a)");
    cs.render_microblock();
    lambda_rule_abs.begin_latex_transition("(\\lambda x. (\\lambda a. a))");
    cs.render_microblock();
    */

    cs.state.superscene_transition(unordered_map<string, string>{
        {"lambda_rule_app.opacity", "0.1"},
    });
    // Transition back to the identity function.
    lambda_rule_abs.begin_latex_transition("(\\lambda a. a)");
    // Fade-in a Python identity function which models this lambda expression identity function.
    PngScene python_identity("python_identity_color", VIDEO_WIDTH/2, VIDEO_HEIGHT/4);
    cs.add_scene(&python_identity, "python_identity", -.5, 0.25);
    cs.stage_macroblock_and_render(AudioSegment("The template with a lambda and a dot represents a function definition."));
    cs.state.superscene_transition(unordered_map<string, string>{
        {"python_identity.x", "0"},
    });

    lambda_rule_abs.begin_latex_transition("(\\lambda " + latex_color(0xff0088ff, "a") + ".a)");
    cs.stage_macroblock_and_render(AudioSegment("The letter that we put in is the name of the input,"));
    lambda_rule_abs.begin_latex_transition("(\\lambda " + latex_color(0xff0088ff, "a") + "." + latex_color(0xffff0000, "a") + ")");
    cs.stage_macroblock_and_render(AudioSegment("and the blank represents the return-statement of the function."));
    cs.state.superscene_transition(unordered_map<string, string>{
        {"python_identity.x", "-.5"},
        {"lambda_rule_app.opacity", "1"},
        {"lambda_rule_abs.opacity", "0.1"},
    });

    // Repeat by making an example with the third function, and then explain its role as function application.
    lambda_rule_app.begin_latex_transition("(\\_\\_)");
    cs.stage_macroblock_and_render(AudioSegment(0.5));
    cs.stage_macroblock_and_render(AudioSegment("The third template involves _applying_ such functions."));

    // Show an example where a function is applied to a variable.
    lambda_rule_app.begin_latex_transition("(" + latex_color(0xff0088ff, "a") + latex_color(0xffff0000, "b") + ")");
    cs.stage_macroblock(AudioSegment("In this case, we're suggesting the thing on the left, 'a', is gonna be used as a function which takes in 'b'."), 2);
    cs.render_microblock();
    cs.render_microblock();

    // Show an example where a function is applied to a variable.
    cs.state.superscene_transition(unordered_map<string, string>{
        {"lambda_rule_app.y", "1.5"},
        {"lambda_rule_abs.y", "1.5"},
        {"lambda_title.y"   , "-1"},
    });
    cs.fade_out_all_scenes();

    cs.stage_macroblock_and_render(AudioSegment("That's all we need to generate lambda expressions."));
}

void visualize(){
    CompositeScene cs;
    LatexScene rep_classic("((\\lambda x. (\\lambda y. (y ((x x) y)))) (\\lambda x. (\\lambda y. (y ((x x) y))))))", 1, VIDEO_WIDTH, VIDEO_HEIGHT);
    LatexScene rep_dbi("(" + latex_color(0xff0088ff, "\\lambda") + latex_color(0xffff0000, "\\lambda") + " (" + latex_color(0xffff0000, "1") + " ((" + latex_color(0xff0088ff, "2") + " " + latex_color(0xff0088ff, "2") + ") " + latex_color(0xffff0000, "1") + ")))" +
                       "(" + latex_color(0xff88ff00, "\\lambda") + latex_color(0xff8800ff, "\\lambda") + " (" + latex_color(0xff8800ff, "1") + " ((" + latex_color(0xff88ff00, "2") + " " + latex_color(0xff88ff00, "2") + ") " + latex_color(0xff8800ff, "1") + ")))", 1, VIDEO_WIDTH/3, VIDEO_HEIGHT/2);
    PngScene rep_vex("vex", VIDEO_WIDTH/3, VIDEO_HEIGHT/2);
    PngScene rep_graph("graph", VIDEO_WIDTH/3, VIDEO_HEIGHT/2);
    PngScene rep_viktor("viktor", VIDEO_WIDTH/3, VIDEO_HEIGHT/2);
    PngScene rep_keenan("keenan", VIDEO_WIDTH/3, VIDEO_HEIGHT/2);
    shared_ptr<LambdaExpression> term = parse_lambda_from_string("((\\x. (\\y. (y ((x x) y)))) (\\x. (\\y. (y ((x x) y)))))");
    term->flush_uid_recursive();
    LambdaScene rep_tromp(term, VIDEO_WIDTH/3, VIDEO_HEIGHT/2);
    cs.add_scene(&rep_classic, "rep_classic", 0    , -1);
    cs.add_scene(&rep_keenan , "rep_keenan" , 0    , -1.5);
    cs.add_scene(&rep_graph  , "rep_graph"  , 0.333, -2);
    cs.add_scene(&rep_dbi    , "rep_dbi"    , 0.666, -2.5);
    cs.add_scene(&rep_vex    , "rep_vex"    , 0    , -3);
    cs.add_scene(&rep_tromp  , "rep_tromp"  , 0.333, -3.5);
    cs.add_scene(&rep_viktor , "rep_viktor" , 0.666, -4);
    cs.state.superscene_transition(unordered_map<string, string>{
        {"rep_classic.y", "0"},
        {"rep_graph.y", "0"},
        {"rep_dbi.y", "0"},
        {"rep_keenan.y", "0"},
        {"rep_vex.y", "0.5"},
        {"rep_tromp.y", "0.5"},
        {"rep_viktor.y", "0.5"},
    });
    cs.stage_macroblock_and_render(AudioSegment("We'll try evaluating these expressions in a sec, but first let's visualize them."));
    cs.stage_macroblock(AudioSegment("There's a ton of styles, but the one I chose is John Tromp's Lambda Diagrams."), 2);
    cs.render_microblock();
    cs.state.subscene_transition(unordered_map<string, string>{
        {"rep_classic.y", "-1.5"},
        {"rep_keenan.y", "-1.5"},
        {"rep_graph.y", "-1.5"},
        {"rep_dbi.y", "-1.5"},
        {"rep_vex.y", "-1"},
        {"rep_tromp.y", "0"},
        {"rep_tromp.x", "0"},
        {"rep_viktor.y", "-1"},
    });
    rep_tromp.state.subscene_transition(unordered_map<string, string>{
        {"w", to_string(VIDEO_WIDTH)},
        {"h", to_string(VIDEO_HEIGHT)},
    });
    cs.render_microblock();
    C4Scene c4s("43334444343", VIDEO_WIDTH*1.2, VIDEO_HEIGHT*1.2);
    cs.add_scene(&c4s, "c4s", -.1, -.1);
    cs.state.set(unordered_map<string, string>{
        {"ntf", "<subscene_transition_fraction> .5 -"},
        {"c4s.opacity", "1 <ntf> 2 * 2 ^ -"},
    });
    c4s.stage_transition("433344443437731155222211");
    cs.stage_macroblock_and_render(AudioSegment("After all, I'm biased- he was the first person to strongly solve Connect 4."));
    cs.remove_scene(&c4s);



    LatexScene lambda_rule_var("a"                , 0.8, VIDEO_WIDTH*.25, VIDEO_HEIGHT*.25);
    LatexScene lambda_rule_abs("(\\lambda a. \\_)", 0.8, VIDEO_WIDTH*.25, VIDEO_HEIGHT*.25);
    LatexScene lambda_rule_app("(\\_ \\_)"        , 0.8, VIDEO_WIDTH*.25, VIDEO_HEIGHT*.25);
    cs.add_scene(&lambda_rule_var, "lambda_rule_var", -0.5, 0.125);
    cs.add_scene(&lambda_rule_abs, "lambda_rule_abs", -0.5, 0.375);
    cs.add_scene(&lambda_rule_app, "lambda_rule_app", -0.5, 0.625);
    cs.state.subscene_transition(unordered_map<string, string>{
        {"lambda_rule_var.x"    , "0"},
        {"lambda_rule_abs.x"    , "0"},
        {"lambda_rule_app.x"    , "0"},
        {"rep_tromp.x", ".25"},
    });
    rep_tromp.state.subscene_transition(unordered_map<string, string>{
        {"w", to_string(VIDEO_WIDTH*3/4)},
        {"latex_opacity", "1"},
    });
    cs.remove_scene(&rep_classic);
    cs.remove_scene(&rep_keenan);
    cs.remove_scene(&rep_graph);
    cs.remove_scene(&rep_dbi);
    cs.remove_scene(&rep_vex);
    cs.remove_scene(&rep_viktor);
    cs.stage_macroblock_and_render(AudioSegment("Each of our three templates is part of a different shape in a lambda diagram."));
    LambdaExpression::Iterator it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(current->get_type() == "Variable" ? OPAQUE_WHITE : 0xff222222);
    }
    rep_tromp.set_expression(term);
    lambda_rule_var.begin_latex_transition(latex_color(0xffffffff, "a"                ));
    lambda_rule_abs.begin_latex_transition(latex_color(0xff222222, "(\\lambda a. \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xff222222, "(\\_ \\_)"        ));
    cs.stage_macroblock_and_render(AudioSegment(.5));
    cs.stage_macroblock_and_render(AudioSegment("Variables from the first template are these vertical lines."));
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(current->get_type() == "Abstraction" ? OPAQUE_WHITE : 0xff222222);
    }
    rep_tromp.set_expression(term);
    lambda_rule_var.begin_latex_transition(latex_color(0xff222222, "a"                ));
    lambda_rule_abs.begin_latex_transition(latex_color(0xffffffff, "(\\lambda a. \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xff222222, "(\\_ \\_)"        ));
    cs.stage_macroblock_and_render(AudioSegment(.5));
    cs.stage_macroblock(AudioSegment("Lambda Abstractions, or template 2, are the horizontal bars at the top."), 2);
    cs.render_microblock();
    cs.render_microblock();
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(current->get_type() == "Application" ? OPAQUE_WHITE : 0xff222222);
    }
    rep_tromp.set_expression(term);
    lambda_rule_var.begin_latex_transition(latex_color(0xff222222, "a"                ));
    lambda_rule_abs.begin_latex_transition(latex_color(0xff222222, "(\\lambda a. \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xffffffff, "(\\_ \\_)"        ));
    cs.stage_macroblock_and_render(AudioSegment(.5));
    cs.stage_macroblock_and_render(AudioSegment("This branching structure at the bottom is template 3, representing function application."));

    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(current->get_type() == "Variable" ? OPAQUE_WHITE : 0xff222222);
    }
    rep_tromp.set_expression(term);
    lambda_rule_var.begin_latex_transition(latex_color(0xffffffff, "a"                ));
    lambda_rule_abs.begin_latex_transition(latex_color(0xff222222, "(\\lambda a. \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xff222222, "(\\_ \\_)"        ));
    cs.stage_macroblock_and_render(AudioSegment(1));
    cs.stage_macroblock_and_render(AudioSegment("Notice that there's eight vertical lines here, just how the expression itself has eight variables."));
    vector<LatexScene> lsv;
    for(int i = 0; i < 8; i++){
        string ch = "y";
        if(((i+3)/2)%2 == 0) ch = "x";
        LatexScene varls(ch, 1, VIDEO_WIDTH/10, VIDEO_WIDTH/10);
        lsv.push_back(varls);
    }
    for(int i = 0; i < 8; i++){
        cs.add_scene(&(lsv[i]), "lamvar" + to_string(i), .27+i/12., .07);
    }
    cs.stage_macroblock_and_render(AudioSegment("These correspond one-to-one, left to right."));
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(current->get_type() == "Abstraction" ? OPAQUE_WHITE : 0xff222222);
    }
    rep_tromp.set_expression(term);
    lambda_rule_var.begin_latex_transition(latex_color(0xff222222, "a"                ));
    lambda_rule_abs.begin_latex_transition(latex_color(0xffffffff, "(\\lambda a. \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xff222222, "(\\_ \\_)"        ));
    cs.stage_macroblock_and_render(AudioSegment("Note how they collide into different horizontal abstraction bars."));
    vector<LatexScene> lsv_abstr;
    for(int i = 0; i < 4; i++){
        string ch = "y";
        if(i%2 == 0) ch = "x";
        LatexScene absls("\\lambda " + ch + ".", 1, VIDEO_WIDTH/16, VIDEO_WIDTH/16);
        lsv_abstr.push_back(absls);
    }
    for(int i = 0; i < 4; i++){
        cs.add_scene_fade_in(&(lsv_abstr[i]), "lamabs" + to_string(i), .27+(i/2)*.335, .18 + .08 * (i%2), true);
    }
    cs.stage_macroblock_and_render(AudioSegment("Those horizontal bars, corresponding to template 2,"));
    cs.stage_macroblock_and_render(AudioSegment("are associated each with one of the lambda-dot pairs."));
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        int color = 0xff222222;
        if(current->get_type() == "Abstraction" && current->get_string()[2] == 'x') color = 0xff660000;
        if(current->get_type() == "Variable" && current->get_string() == "x") color = 0xff660000;
        if(current->get_type() == "Abstraction" && current->get_string()[2] == 'y') color = 0xff002266;
        if(current->get_type() == "Variable" && current->get_string() == "y") color = 0xff002266;
        current->set_color(color);
    }
    cs.stage_macroblock(AudioSegment("Variables touch the bars which bind them."), 3);
    rep_tromp.set_expression(term);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    for(int i = 0; i < 8; i++){
        string ch = "y";
        if(((i+3)/2)%2 == 0) ch = latex_color(0xffff0000, "x");
        lsv[i].begin_latex_transition(ch);
    }
    cs.stage_macroblock(AudioSegment("The Xs connect to the lambda abstraction which binds the variable X"), 3);
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        int color = 0xff222222;
        if(current->get_type() == "Abstraction" && current->get_string()[2] == 'x') color = 0xffff0000;
        if(current->get_type() == "Variable" && current->get_string() == "x") color = 0xffff0000;
        if(current->get_type() == "Abstraction" && current->get_string()[2] == 'y') color = color;
        if(current->get_type() == "Variable" && current->get_string() == "y") color = color;
        current->set_color(color);
    }
    rep_tromp.set_expression(term);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    for(int i = 0; i < 8; i++){
        string ch = latex_color(0xff0044ff, "y");
        if(((i+3)/2)%2 == 0) ch = "x";
        lsv[i].begin_latex_transition(ch);
    }
    cs.stage_macroblock(AudioSegment("and the Ys touch the bar which binds Y."), 3);
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        int color = 0xff222222;
        if(current->get_type() == "Abstraction" && current->get_string()[2] == 'x') color = color;
        if(current->get_type() == "Variable" && current->get_string() == "x") color = color;
        if(current->get_type() == "Abstraction" && current->get_string()[2] == 'y') color = 0xff0044ff;
        if(current->get_type() == "Variable" && current->get_string() == "y") color = 0xff0044ff;
        current->set_color(color);
    }
    rep_tromp.set_expression(term);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    lambda_rule_var.begin_latex_transition(latex_color(0xff222222, "a"                ));
    lambda_rule_abs.begin_latex_transition(latex_color(0xff222222, "(\\lambda a. \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xffffffff, "(\\_ \\_)"        ));
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(current->get_type() == "Application" ? OPAQUE_WHITE : 0xff222222);
    }
    rep_tromp.set_expression(term);
    cs.state.subscene_transition(unordered_map<string, string>{
        {"lamabs0.opacity", "0"},
        {"lamabs1.opacity", "0"},
        {"lamabs2.opacity", "0"},
        {"lamabs3.opacity", "0"},
        {"lamvar0.opacity", "0"},
        {"lamvar1.opacity", "0"},
        {"lamvar2.opacity", "0"},
        {"lamvar3.opacity", "0"},
        {"lamvar4.opacity", "0"},
        {"lamvar5.opacity", "0"},
        {"lamvar6.opacity", "0"},
        {"lamvar7.opacity", "0"},
    });
    cs.stage_macroblock_and_render(AudioSegment(1));
    cs.stage_macroblock(AudioSegment("Template 3, for function application, has two subexpressions corresponding to the two blanks."), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    dynamic_pointer_cast<LambdaApplication>(term)->get_first()->set_color_recursive(0xff006600);
    dynamic_pointer_cast<LambdaApplication>(term)->get_second()->set_color_recursive(0xff660066);
    rep_tromp.set_expression(term);
    cs.stage_macroblock(AudioSegment("This lambda expression, as a whole, has an application surrounding everything."), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    dynamic_pointer_cast<LambdaApplication>(term)->get_first()->set_color_recursive(0xff003300);
    dynamic_pointer_cast<LambdaApplication>(term)->get_second()->set_color_recursive(0xff330033);
    rep_tromp.set_expression(term);
    cs.stage_macroblock_and_render(AudioSegment("It's shown in white."));
    dynamic_pointer_cast<LambdaApplication>(term)->get_first()->set_color_recursive(0xff006600);
    dynamic_pointer_cast<LambdaApplication>(term)->get_second()->set_color_recursive(0xff660066);
    rep_tromp.set_expression(term);
    cs.stage_macroblock_and_render(AudioSegment("The lambda diagram for each of the subexpressions"));
    cs.stage_macroblock(AudioSegment("is drawn on the left and right branch of the application."), 4);
    dynamic_pointer_cast<LambdaApplication>(term)->get_first()->set_color_recursive(0xff00ff00);
    dynamic_pointer_cast<LambdaApplication>(term)->get_second()->set_color_recursive(0xff330033);
    rep_tromp.set_expression(term);
    cs.render_microblock();
    cs.render_microblock();
    dynamic_pointer_cast<LambdaApplication>(term)->get_first()->set_color_recursive(0xff003300);
    dynamic_pointer_cast<LambdaApplication>(term)->get_second()->set_color_recursive(0xffff00ff);
    rep_tromp.set_expression(term);
    cs.render_microblock();
    cs.render_microblock();
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(OPAQUE_WHITE);
    }
    rep_tromp.set_expression(term);
    cs.stage_macroblock_and_render(AudioSegment(1));
    cs.stage_macroblock_and_render(AudioSegment("Don't worry too much about trying to read these, it's the kind of skill you have to practice for a while."));
    cs.state.superscene_transition(unordered_map<string, string>{
        {"lambda_rule_var.opacity", "0"},
        {"lambda_rule_abs.opacity", "0"},
        {"lambda_rule_app.opacity", "0"},
        {"rep_tromp.opacity", "0"},
        {"lamabs0.opacity", "0"},
        {"lamabs1.opacity", "0"},
        {"lamabs2.opacity", "0"},
        {"lamabs3.opacity", "0"},
        {"lamvar0.opacity", "0"},
        {"lamvar1.opacity", "0"},
        {"lamvar2.opacity", "0"},
        {"lamvar3.opacity", "0"},
        {"lamvar4.opacity", "0"},
        {"lamvar5.opacity", "0"},
        {"lamvar6.opacity", "0"},
        {"lamvar7.opacity", "0"},
    });
    cs.stage_macroblock_and_render(AudioSegment("I'll color in subcomponents so you can tell what's going on!"));
}

void beta_reduction(){
    CompositeScene cs;
    LatexScene beta_title(latex_text("\\beta-Reduction"), 1, VIDEO_WIDTH*0.5, VIDEO_HEIGHT*0.25);
    cs.add_scene(&beta_title, "beta_title", 0.25, -.25);
    cs.state.superscene_transition(unordered_map<string, string>{
        {"beta_title.y", "0"},
    });
    cs.stage_macroblock_and_render(AudioSegment("I've mentioned that these expressions can be 'evaluated', but what does that even mean?"));
    LatexScene latexbeta("(\\lambda x. \\_)", 0.6, VIDEO_WIDTH*.75, VIDEO_HEIGHT*.5);
    cs.add_scene(&latexbeta, "latex_beta", -.5, .25);
    cs.state.subscene_transition(unordered_map<string, string>{
        {"latex_beta.x", ".125"},
    });
    cs.stage_macroblock(AudioSegment("Remember that template 2 can be interpreted as a function."), 2);
    cs.render_microblock(); cs.render_microblock();
    latexbeta.begin_latex_transition("(\\lambda x. \\_)\\\\\\\\ \\big(\\_ \\_\\big)");
    cs.stage_macroblock(AudioSegment("And template 3 can be interpreted as applying a value to a function."), 3);
    cs.render_microblock(); cs.render_microblock(); cs.render_microblock();
    cs.stage_macroblock_and_render(AudioSegment("We wanna create a template for evaluation that combines these interpretations."));

    // Scene showing lambda abstraction
    latexbeta.begin_latex_transition("(\\lambda x. (x\\ b))\\\\\\\\ \\big(\\_ \\_\\big)");
    cs.stage_macroblock_and_render(AudioSegment("First, let's make a function by using template 2."));
    cs.stage_macroblock_and_render(AudioSegment("This function is pretty simple."));
    latexbeta.begin_latex_transition("(\\lambda " + latex_color(0xff00ff00, "x") + ". (x\\ b))\\\\\\\\ \\big(\\_ \\_\\big)");
    cs.stage_macroblock_and_render(AudioSegment("It just takes a value x in,"));
    latexbeta.begin_latex_transition("(\\lambda " + latex_color(0xff00ff00, "x") + ". " + latex_color(0xffff0000, "(x\\ b)") + ")\\\\\\\\ \\big(\\_ \\_\\big)");
    cs.stage_macroblock_and_render(AudioSegment("and applies x to b."));

    // Scene showing some random expression to be applied to the function
    latexbeta.begin_latex_transition("\\big((\\lambda x. (x\\ b))\\ \\_\\big)");
    cs.stage_macroblock_and_render(AudioSegment("We'll place that function in the spot where the function goes in template 3."));

    // Scene showing some random expression to be applied to the function
    latexbeta.begin_latex_transition("\\big((\\lambda x. (x\\ b)) E\\big)");
    cs.stage_macroblock_and_render(AudioSegment("We'll make our 'value' just be some random expression."));
    cs.stage_macroblock_and_render(AudioSegment("Now, we're gonna perform what's called 'beta reduction'."));

    // Show substitution
    latexbeta.begin_latex_transition("\\big((" + latex_color(0xff8888ff, "\\lambda " + latex_color(0xff00ff00, "x") + ".") + "(x\\ b)) E\\big)");
    cs.stage_macroblock_and_render(AudioSegment("We first note that 'x' is the variable bound by the lambda."));
    latexbeta.begin_latex_transition("\\big((\\lambda x. (" + latex_color(0xffff0000, "x") + "\\ b)) E\\big)");
    cs.stage_macroblock_and_render(AudioSegment("Since the function binds 'x', we first find all 'x's in the body of the function."));
    cs.stage_macroblock_and_render(AudioSegment("In our case there's just this one."));

    // Substitution result
    latexbeta.begin_latex_transition("(" + latex_color(0xffff0000, "x") + "\\ b)\\quad E");
    cs.stage_macroblock_and_render(AudioSegment("Now, we discard everything but the function body itself and the value we're plugging in."));
    latexbeta.begin_latex_transition("(" + latex_color(0xffff0000, "x") + "\\ b)\\quad " + latex_color(0xff0088ff, "E"));
    cs.stage_macroblock_and_render(AudioSegment("Now, we take our value,"));
    latexbeta.begin_latex_transition("(" + latex_color(0xff0088ff, "E") + "\\ b)");
    cs.stage_macroblock_and_render(AudioSegment("and replace every instance of 'x' with that value."));

    // Show final result
    latexbeta.begin_latex_transition("(E\\ b)");
    cs.stage_macroblock_and_render(AudioSegment("And we're left with the reduced expression."));
    // Ok, let's try a slightly trickier one.
    latexbeta.begin_latex_transition("\\Big(\\big(\\lambda x. (x (y x))\\big) (f f)\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("Ok, let's try a slightly trickier one."));
    cs.stage_macroblock(AudioSegment("To make it clear, this is the function, and this is the value."), 3);
    cs.render_microblock();
    latexbeta.begin_latex_transition("\\Big("+latex_color(0xffff7777, "\\big(\\lambda x. (x (y x))\\big)") + "(f f)\\Big)");
    cs.render_microblock();
    latexbeta.begin_latex_transition("\\Big("+latex_color(0xffff7777, "\\big(\\lambda x. (x (y x))\\big) ")+latex_color(0xff0088ff, "(f f)")+"\\Big)");
    cs.render_microblock();

    // Show substitution
    latexbeta.begin_latex_transition("\\Big("+latex_color(0xffff7777, "\\big(\\lambda " + latex_color(0xff00ff00, "x") + ". (x (y x))\\big) ")+latex_color(0xff0088ff, "(f f)")+"\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("Same as before, X is the variable bound by the lambda."));
    latexbeta.begin_latex_transition("\\Big("+latex_color(0xffff7777, "\\big(\\lambda " + latex_color(0xff00ff00, "x") + ". (" + latex_color(0xffff0000, "x") + " (y " + latex_color(0xffff0000, "x") + "))\\big) ")+latex_color(0xff0088ff, "(f f)")+"\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("So we highlight all the instances of 'x' in the body of the function."));
    cs.stage_macroblock_and_render(AudioSegment("There's 2 this time."));

    // Substitution result
    latexbeta.begin_latex_transition(latex_color(0xffff7777, "(" + latex_color(0xffff0000, "x") + " (y " + latex_color(0xffff0000, "x") + "))\\quad ")+latex_color(0xff0088ff, "(f f)"));
    cs.stage_macroblock_and_render(AudioSegment("Discard all the scaffolding,"));
    latexbeta.begin_latex_transition(latex_color(0xffff7777, "(" + latex_color(0xff0088ff, "(f f)") + " (y " + latex_color(0xff0088ff, "(f f)") + "))"));
    cs.stage_macroblock_and_render(AudioSegment("and perform the replacement."));

    // Show final result

    latexbeta.begin_latex_transition(latex_color(0xff010101, "((f f) (y (f f)))"));
    cs.stage_macroblock_and_render(AudioSegment("Still looks like abstract nonsense?"));
    cs.remove_scene(&latexbeta);
    cs.stage_macroblock_and_render(AudioSegment("This actually isn't as unfamiliar to you as it might seem."));

    // Transition to algebra
    LatexScene quadratic_equation("f(x) = x^2 + x + 3", .6, VIDEO_WIDTH*.8, VIDEO_HEIGHT*.5);
    cs.add_scene_fade_in(&quadratic_equation, "quadratic_equation", 0.1, 0.25, true);
    cs.stage_macroblock_and_render(AudioSegment("You already know how to do this in plain old algebra."));
    cs.stage_macroblock_and_render(AudioSegment("Here's an algebraic function."));
    cs.stage_macroblock_and_render(AudioSegment("We wanna evaluate it at x=5."));
    quadratic_equation.begin_latex_transition(latex_color(0xffff8888, "(f(x) = x^2 + x + 3)") + "\\\\\\\\" + latex_color(0xff88ff88, "f(5)"));

    // Show evaluation
    cs.stage_macroblock_and_render(AudioSegment("We can first establish the function, and then on a separate line, apply it to 5."));
    quadratic_equation.begin_latex_transition(latex_color(0xffff8888, "(f(x) = x^2 + x + 3)") + "\\ " + latex_color(0xff88ff88, "f(5)"));
    cs.stage_macroblock_and_render(AudioSegment("But let's stick em on the same line."));
    quadratic_equation.begin_latex_transition("\\Big(" + latex_color(0xffff8888, "(f(x) = x^2 + x + 3)\\ ") + latex_color(0xff88ff88, "f(5)") + "\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("We'll wrap these two things in parentheses to imply that the first is a function and the second is the value we wanna put in."));

    quadratic_equation.begin_latex_transition("\\Big((" + latex_color(0xffff0000, "f") + "(x) = x^2 + x + 3)\\ " + latex_color(0xffff0000, "f") + "(5)\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("Now, since we tied the function to its value, there's no point in naming it 'f'."));
    quadratic_equation.begin_latex_transition("\\Big((x \\rightarrow x^2 + x + 3)\\quad 5\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("Our inline notation is unambiguous about which function we wanna apply to what value."));
    quadratic_equation.begin_latex_transition("\\Big((" + latex_color(0xff00ff00, "x") + " \\rightarrow x^2 + x + 3)\\quad 5\\Big)");

    // Transform into lambda calculus
    cs.stage_macroblock_and_render(AudioSegment("We do still need to express that x is the thing in the body which 5'll be replacing,"));
    quadratic_equation.begin_latex_transition("\\Big((\\lambda x.\\ x^2 + x + 3)\\quad 5\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("and the Lambda-Calculus way to express that is by putting a lambda-dot pair around it."));
    cs.stage_macroblock_and_render(AudioSegment("Now how do we evaluate it?"));
    cs.stage_macroblock_and_render(AudioSegment("Well, exactly the same procedure!"));
    quadratic_equation.begin_latex_transition("\\Big((\\lambda x.\\ " + latex_color(0xffff0000, "x") + "^2 + " + latex_color(0xffff0000, "x") + " + 3)\\quad 5\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("Identify all the variables in the body which match the one bound by the lambda, in this case all the 'x's."));
    quadratic_equation.begin_latex_transition(latex_color(0xffff0000, "x") + "^2 + " + latex_color(0xffff0000, "x") + " + 3\\quad 5");
    cs.stage_macroblock_and_render(AudioSegment("Drop all the slack,"));
    quadratic_equation.begin_latex_transition(latex_color(0xffff0000, "x") + "^2 + " + latex_color(0xffff0000, "x") + " + 3\\quad " + latex_color(0xff0088ff, "5"));
    cs.stage_macroblock_and_render(AudioSegment("grab our value,"));
    quadratic_equation.begin_latex_transition(latex_color(0xff0088ff, "5") + "^2 + " + latex_color(0xff0088ff, "5") + " + 3");
    cs.stage_macroblock_and_render(AudioSegment("and shove it everywhere we had an x."));
    cs.stage_macroblock(AudioSegment("We're left with an algebraic expression which represents the answer."), 3);
    quadratic_equation.begin_latex_transition("25 + 5 + 3");
    cs.render_microblock();
    quadratic_equation.begin_latex_transition("30 + 3");
    cs.render_microblock();
    quadratic_equation.begin_latex_transition("33");
    cs.render_microblock();
    cs.stage_macroblock_and_render(AudioSegment("Not too alien now, is it?"));

    /*
    quadratic_equation.begin_latex_transition("((\\lambda x. x) (\\lambda a. (\\lambda f. (a f))))");
    cs.stage_macroblock_and_render(AudioSegment("Alright, one more."));
    quadratic_equation.begin_latex_transition("(" + latex_color(0xffffaaaa, "(\\lambda x. x)") + "(\\lambda a. (\\lambda f. (a f))))");
    cs.stage_macroblock_and_render(AudioSegment("In this case, we're applying the identity function..."));
    quadratic_equation.begin_latex_transition("(" + latex_color(0xffffaaaa, "(\\lambda x. x)") + latex_color(0xffaaffaa, "(\\lambda a. (\\lambda f. (a f)))") + ")");
    cs.stage_macroblock_and_render(AudioSegment("...to another random term which I made up."));
    cs.stage_macroblock_and_render(AudioSegment("Let's follow the procedure once again."));
    quadratic_equation.begin_latex_transition("((\\lambda x. " + latex_color(0xffff0000, "x") + ") (\\lambda a. (\\lambda f. (a f))))");
    cs.stage_macroblock_and_render(AudioSegment("First, we identify all instances of the bound variable in the function."));
    quadratic_equation.begin_latex_transition("(\\lambda x. " + latex_color(0xffff0000, "x") + ") (\\lambda a. (\\lambda f. (a f)))");
    cs.stage_macroblock_and_render(AudioSegment("Drop the application parentheses,"));
    quadratic_equation.begin_latex_transition("(\\lambda x. " + latex_color(0xffff0000, "x") + ") " + latex_color(0xff0088ff, "(\\lambda a. (\\lambda f. (a f)))"));
    cs.stage_macroblock_and_render(AudioSegment("Grab the value,"));
    quadratic_equation.begin_latex_transition("(\\lambda x. (\\lambda a. (\\lambda f. (a f))))");
    cs.stage_macroblock_and_render(AudioSegment("and replace the bound variable with the argument being passed into the function."));
    quadratic_equation.begin_latex_transition("(\\lambda a. (\\lambda f. (a f)))");
    cs.stage_macroblock_and_render(AudioSegment("Scrap the function definition,"));
    cs.stage_macroblock_and_render(AudioSegment("and we get my function back out!"));
    */

    quadratic_equation.begin_latex_transition("((\\lambda x. (\\lambda w. (x w))) (\\lambda a. (\\lambda f. f)))");
    cs.state.subscene_transition(unordered_map<string, string>{
        {"quadratic_equation.y", "0.58"},
    });
    shared_ptr<LambdaExpression> term = parse_lambda_from_string("((\\x. (\\w. (x w))) (\\a. (\\f. f)))");
    term->flush_uid_recursive();
    LambdaScene betadiagram(term, 0.4*VIDEO_WIDTH, 0.4*VIDEO_HEIGHT);
    cs.add_scene_fade_in(&betadiagram, "betadiagram", 0.3, 0.3, true);
    cs.stage_macroblock_and_render(AudioSegment("Let's try one more, but alongside the diagram this time."));
    quadratic_equation.begin_latex_transition("((\\lambda x. (\\lambda w. (x w))) " + latex_color(0xff0088ff, "(\\lambda a. (\\lambda f. f))") + ")");
    dynamic_pointer_cast<LambdaApplication>(term)->get_second()->set_color_recursive(0xff0088ff);
    betadiagram.set_expression(term);
    cs.stage_macroblock_and_render(AudioSegment("Let's color the value in blue,"));
    quadratic_equation.begin_latex_transition("((\\lambda x. (\\lambda w. (" + latex_color(0xffff0000, "x") + " w))) " + latex_color(0xff0088ff, "(\\lambda a. (\\lambda f. f))") + ")");
    dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaApplication>(term)->get_first())->get_body())->get_body())->get_first()->set_color_recursive(0xffff0000);
    shared_ptr<LambdaExpression> termclone = term->clone();
    betadiagram.set_expression(term);
    cs.stage_macroblock_and_render(AudioSegment("and the variable which it's about to replace as red."));
    cs.stage_macroblock_and_render(AudioSegment("Now let's perform the beta reduction."));
    quadratic_equation.begin_latex_transition("(\\lambda w. (" + latex_color(0xff7f00ff, "(\\lambda a. (\\lambda f. f))") + "w))");
    betadiagram.reduce();
    cs.stage_macroblock_and_render(AudioSegment("See how the value takes the place of the variable, and then the application scaffolding drops out?"));
    quadratic_equation.begin_latex_transition("((\\lambda x. (\\lambda w. (" + latex_color(0xffff0000, "x") + " w))) " + latex_color(0xff0088ff, "(\\lambda a. (\\lambda f. f))") + ")");
    betadiagram.set_expression(termclone);
    cs.stage_macroblock_and_render(AudioSegment("Let's watch that a few times over."));
    quadratic_equation.begin_latex_transition("(\\lambda w. (" + latex_color(0xff7f00ff, "(\\lambda a. (\\lambda f. f))") + "w))");
    betadiagram.reduce();
    cs.stage_macroblock_and_render(AudioSegment(1));
    quadratic_equation.begin_latex_transition("((\\lambda x. (\\lambda w. (" + latex_color(0xffff0000, "x") + " w))) " + latex_color(0xff0088ff, "(\\lambda a. (\\lambda f. f))") + ")");
    betadiagram.set_expression(termclone);
    cs.stage_macroblock_and_render(AudioSegment(1));
    quadratic_equation.begin_latex_transition("(\\lambda w. (" + latex_color(0xff7f00ff, "(\\lambda a. (\\lambda f. f))") + "w))");
    betadiagram.reduce();
    cs.stage_macroblock_and_render(AudioSegment(1));
    cs.state.subscene_transition(unordered_map<string, string>{
        {"quadratic_equation.opacity", "0"},
    });

    cs.stage_macroblock_and_render(AudioSegment("This doesn't have to happen at the top level of the expression."));
    shared_ptr<LambdaExpression> someterm = parse_lambda_from_string("((a (\\n. (\\m. (n ((m m) n))))) (\\x. (\\y. (y ((x x) y)))))");
    shared_ptr<LambdaExpression> newterm = abstract('a', apply(someterm, term, OPAQUE_WHITE), OPAQUE_WHITE);
    newterm->flush_uid_recursive();
    betadiagram.set_expression(newterm);
    cs.stage_macroblock_and_render(AudioSegment("Let's make a big expression which contains this somewhere inside."));
    betadiagram.reduce();
    cs.stage_macroblock(AudioSegment("We can still beta-reduce the entire expression by reducing this subcomponent."), 2);
    cs.render_microblock();
    cs.render_microblock();
    cs.stage_macroblock_and_render(AudioSegment(0.5));
    cs.state.subscene_transition(unordered_map<string, string>{
        {"betadiagram.opacity", "0"},
        {"beta_title.opacity", "0"},
    });
    cs.stage_macroblock_and_render(AudioSegment("Alright, time to write some code!"));
}

void currying() {
    CompositeScene cs;
    LatexScene currying_title(latex_text("Currying"), 1, VIDEO_WIDTH * 0.5, VIDEO_HEIGHT * 0.25);
    cs.add_scene(&currying_title, "currying_title", 0.25, -0.25);
    cs.state.superscene_transition(unordered_map<string, string>{
        {"currying_title.y", "0"},
    });
    LatexScene multi_argument("\\big(\\lambda x y .\\ 5 * y + x\\big)", 0.5, VIDEO_WIDTH, VIDEO_HEIGHT * 0.4);
    cs.add_scene_fade_in(&multi_argument, "multi_argument", 0, 0.3, true);
    cs.stage_macroblock_and_render(AudioSegment("First, we've gotta make a function that takes two variables."));
    cs.stage_macroblock_and_render(AudioSegment("Pseudocode like this isn't permitted by our strict templates for making expressions."));
    multi_argument.begin_latex_transition("\\big(\\lambda x y .\\ "+latex_color(0xff00ff00, "5\\: *")+" y "+latex_color(0xff00ff00, "+")+" x\\big)");
    cs.stage_macroblock_and_render(AudioSegment("Of course, we have yet to define numbers and times and plus,"));
    cs.stage_macroblock_and_render(AudioSegment("but let's assume we know how those work for now."));
    multi_argument.begin_latex_transition("\\big(\\lambda " + latex_color(0xffff0000, "x y") + ".\\ 5 * y + x\\big)");
    cs.stage_macroblock_and_render(AudioSegment("The real problem is that we're trying to make a function that takes in two variables."));
    cs.stage_macroblock_and_render(AudioSegment("Only one letter is allowed to go between the lambda and the dot."));
    cs.stage_macroblock_and_render(AudioSegment("Luckily, there's a workaround, called 'currying'."));

    multi_argument.begin_latex_transition("\\big(\\lambda x.\\ (\\lambda y.\\ 5 * y + x)\\big)");
    cs.stage_macroblock_and_render(AudioSegment("The trick here is to make a function wrap another function."));
    multi_argument.begin_latex_transition("\\bigg(\\big(\\lambda x.\\ (\\lambda y.\\ 5 * y + x)\\big)\\ 5\\ 3\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("Let's see what happens when we stick two arguments after this function and beta reduce it."));
    multi_argument.begin_latex_transition("\\bigg(\\Big(\\big(\\lambda x.\\ (\\lambda y.\\ 5 * y + x)\\big)\\ 5\\Big)\\ 3\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("It's best to be clear what the application order is, so we don't misinterpret 5 as being a function which takes in 3."));
    cs.stage_macroblock_and_render(AudioSegment("Now let's beta reduce the innermost function application."));
    multi_argument.begin_latex_transition("\\bigg(\\qquad \\Big(" + latex_color(0xffff8888, "\\big(\\lambda x.\\ (\\lambda y.\\ 5 * y + x)\\big)\\ ") + latex_color(0xff0088ff, "5") + "\\Big)\\qquad 3\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("Here's the function, and the value we're passing in."));
    multi_argument.begin_latex_transition("\\bigg(\\qquad \\Big(" + latex_color(0xffff8888, "\\big(\\lambda "+latex_color(0xff00ff00,"x")+".\\ (\\lambda y.\\ 5 * y + x)\\big)\\ ") + latex_color(0xff0088ff, "5") + "\\Big)\\qquad 3\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("Notice the bound variable is 'x'."));
    multi_argument.begin_latex_transition("\\bigg(\\qquad \\Big(" + latex_color(0xffff8888, "\\big(\\lambda "+latex_color(0xff00ff00,"x")+".\\ (\\lambda y.\\ 5 * y + "+latex_color(0xffff0000,"x")+")\\big)\\ ") + latex_color(0xff0088ff, "5") + "\\Big)\\qquad 3\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("Find all the x's in the body,"));
    multi_argument.begin_latex_transition("\\bigg(\\qquad " + latex_color(0xffff8888, "(\\lambda y.\\ 5 * y + "+latex_color(0xffff0000,"x")+")") + "\\ 5\\qquad 3\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("rip off the scaffolding,"));
    multi_argument.begin_latex_transition("\\bigg(\\qquad " + latex_color(0xffff8888, "(\\lambda y.\\ 5 * y + 5)") + "\\qquad 3\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("and replace."));
    multi_argument.begin_latex_transition("\\bigg(\\qquad (\\lambda y.\\ 5 * y + 5)\\qquad 3\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment(1));
    multi_argument.begin_latex_transition("\\bigg((\\lambda y.\\ 5 * y + 5)\\ 3\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("Getting the hang of beta reduction?"));
    multi_argument.begin_latex_transition("5 * 3 + 5");
    cs.stage_macroblock_and_render(AudioSegment("Reduce one more time, and we have the answer."));
    multi_argument.begin_latex_transition("15 + 5");
    cs.stage_macroblock_and_render(AudioSegment(1));
    multi_argument.begin_latex_transition("20");
    cs.stage_macroblock_and_render(AudioSegment(2));
    multi_argument.begin_latex_transition("\\big(\\lambda x.\\ (\\lambda y.\\ 5 * y + x)\\big)");
    cs.stage_macroblock_and_render(AudioSegment("Let that sink in a little bit-"));
    cs.stage_macroblock_and_render(AudioSegment("Since a function can only _really_ take one argument,"));
    multi_argument.begin_latex_transition("\\big(\\lambda x.\\ "+latex_color(0xffff7777, "(\\lambda y.\\ 5 * y + x)")+"\\big)");
    cs.stage_macroblock_and_render(AudioSegment("A two-argument function is the same thing as a function which spits out a one-argument function."));
    multi_argument.begin_latex_transition("\\big(\\lambda x.\\ "+latex_color(0xffff7777, "(\\lambda y.\\ "+latex_color(0xffff77ff, "(\\lambda z.\\ "+latex_text("foo")+")")+")")+"\\big)");
    cs.stage_macroblock_and_render(AudioSegment(.5));
    cs.stage_macroblock_and_render(AudioSegment("A three-argument function would spit out a two-argument function."));
    cs.stage_macroblock_and_render(AudioSegment("And so on!"));
    multi_argument.begin_latex_transition("\\big(\\lambda x"+latex_color(0xffff7777, "y"+latex_color(0xffff77ff, "z.\\ "+latex_text("foo")))+"\\big)");

    cs.stage_macroblock_and_render(AudioSegment("So from now on, we'll permit two variables in the lambda,"));
    multi_argument.begin_latex_transition("\\big(\\lambda " + latex_color(0xffff0000, "x y") + ".\\ 5 * y + x\\big)");
    cs.stage_macroblock_and_render(AudioSegment("since we can think of it as shorthand for a curried function."));
    multi_argument.begin_latex_transition("\\bigg(\\Big(\\big(\\lambda x y.\\ 5 * y + x\\big)\\ 5\\Big)\\ 3\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("Before, we applied the function like this..."));
    multi_argument.begin_latex_transition("\\bigg(\\big(\\lambda x y.\\ 5 * y + x\\big)\\ 5\\ 3\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("but now we can just drop the extra parentheses as shorthand too."));
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment("Alright, it's programming time. We're gonna start with the values TRUE and FALSE."));
}

void booleans() {
    CompositeScene cs;

    // Create Lambda scenes for TRUE and FALSE
    shared_ptr<LambdaExpression> lambda_true = parse_lambda_from_string("(\\a. (\\b. a))");
    lambda_true->flush_uid_recursive();
    LambdaScene true_scene(lambda_true, VIDEO_WIDTH/4, VIDEO_HEIGHT/4);
    true_scene.set_title("TRUE");

    shared_ptr<LambdaExpression> lambda_false = parse_lambda_from_string("(\\a. (\\b. b))");
    lambda_false->flush_uid_recursive();
    LambdaScene false_scene(lambda_false, VIDEO_WIDTH/4, VIDEO_HEIGHT/4);
    false_scene.set_title("FALSE");

    // Position both lambda scenes side by side in the center
    LatexScene true_latex( "(\\lambda a. (\\lambda b. a))", 0.7, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 4);
    LatexScene false_latex("(\\lambda a. (\\lambda b. b))", 0.7, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 4);
    cs.add_scene(&true_latex, "true_latex", 0, 1);
    cs.add_scene(&false_latex, "false_latex", 0.5, 1);
    cs.add_scene(&true_scene, "true_scene", 0.125, 1);
    cs.add_scene(&false_scene, "false_scene", 0.625, 1);
    cs.state.set(unordered_map<string, string>{
        {"true_scene.title_opacity", "1"},
        {"false_scene.title_opacity", "1"},
    });
    true_scene.state.set(unordered_map<string, string>{
        {"title_opacity", "[true_scene.title_opacity]"},
    });
    false_scene.state.set(unordered_map<string, string>{
        {"title_opacity", "[false_scene.title_opacity]"},
    });
    cs.state.superscene_transition(unordered_map<string, string>{
        {"true_latex.y", "0.5"},
        {"false_latex.y", "0.5"},
        {"true_scene.y", "0.2"},
        {"false_scene.y", "0.2"},
    });
    cs.stage_macroblock_and_render(AudioSegment("Conventionally, these are the two expressions which are used."));

    // Highlight the functions in red and green
    true_scene.set_expression(lambda_true);
    false_scene.set_expression(lambda_false);
    cs.stage_macroblock_and_render(AudioSegment("Note that they're curried functions!"));
    true_latex.begin_latex_transition( "(\\lambda a. " +latex_color(0xffff0000, "(\\lambda b. a)")+")");
    false_latex.begin_latex_transition("(\\lambda a. " +latex_color(0xffff0000, "(\\lambda b. b)")+")");
    cs.stage_macroblock_and_render(AudioSegment("One function,"));
    true_latex.begin_latex_transition( latex_color(0xff0088ff, "(\\lambda a. ")+latex_color(0xffff0000, "(\\lambda b. a)") + latex_color(0xff0088ff, ")"));
    false_latex.begin_latex_transition(latex_color(0xff0088ff, "(\\lambda a. ")+latex_color(0xffff0000, "(\\lambda b. b)") + latex_color(0xff0088ff, ")"));
    cs.stage_macroblock_and_render(AudioSegment("directly inside the next."));
    true_latex.begin_latex_transition( "(\\lambda ab. a)");
    false_latex.begin_latex_transition("(\\lambda ab. b)");
    cs.stage_macroblock_and_render(AudioSegment("Let's use our shorthand."));

    cs.stage_macroblock_and_render(AudioSegment("To get a sense of what they do, let's pass in two values."));

    shared_ptr<LambdaExpression> arg1 = parse_lambda_from_string("(\\x. x)");
    shared_ptr<LambdaExpression> arg2 = parse_lambda_from_string("(\\y. y)");
    arg1->flush_uid_recursive();
    arg2->flush_uid_recursive();
    arg1->set_color_recursive(0xffff0000);
    arg2->set_color_recursive(0xff0088ff);

    // Pass in V_1 and V_2 to TRUE and perform the reduction
    true_latex.begin_latex_transition("((\\lambda ab. a)\\ " + latex_color(0xffff0000, "V_1") + "\\ " + latex_color(0xff0088ff, "V_2") + ")");
    shared_ptr<LambdaExpression> true_with_args = apply(apply(lambda_true, arg1, OPAQUE_WHITE), arg2, OPAQUE_WHITE);
    true_scene.set_expression(true_with_args);
    cs.stage_macroblock_and_render(AudioSegment("We'll start with TRUE."));
    true_scene.reduce();
    true_latex.begin_latex_transition("((\\lambda b. " + latex_color(0xffff0000, "V_1") + ")\\ " + latex_color(0xff0088ff, "V_2") + ")");
    cs.stage_macroblock_and_render(AudioSegment("Reduce once for the first argument,"));
    true_scene.reduce();
    true_latex.begin_latex_transition(latex_color(0xffff0000, "V_1"));
    cs.stage_macroblock_and_render(AudioSegment("Reduce again for the second."));
    cs.stage_macroblock_and_render(AudioSegment("We passed in V1 and V2, and we got out just V1."));

    false_latex.begin_latex_transition("((\\lambda ab. b)\\ " + latex_color(0xffff0000, "V_1") + "\\ " + latex_color(0xff0088ff, "V_2") + ")");
    shared_ptr<LambdaExpression> false_with_args = apply(apply(lambda_false, arg1, OPAQUE_WHITE), arg2, OPAQUE_WHITE);
    false_scene.set_expression(false_with_args);
    cs.stage_macroblock_and_render(AudioSegment("Let's try again with FALSE."));
    false_scene.reduce();
    false_latex.begin_latex_transition("((\\lambda b. b)\\ "+ latex_color(0xff0088ff, "V_2") + ")");
    cs.stage_macroblock_and_render(AudioSegment("Reduce once,"));
    false_scene.reduce();
    false_latex.begin_latex_transition(latex_color(0xff0088ff, "V_2"));
    cs.stage_macroblock_and_render(AudioSegment("Reduce again."));

    // Pass in V_1 and V_2 to FALSE and perform the reduction
    cs.stage_macroblock_and_render(AudioSegment("This time, we got out V2 instead of V1."));

    true_scene.set_expression(lambda_true);
    false_scene.set_expression(lambda_false);
    true_latex .begin_latex_transition("(\\lambda ab. a)");
    false_latex.begin_latex_transition("(\\lambda ab. b)");
    cs.stage_macroblock_and_render(AudioSegment("This means, these functions are fundamentally selectors."));
    cs.stage_macroblock_and_render(AudioSegment("TRUE picks out the first of 2 arguments, and FALSE picks the second."));
    cs.stage_macroblock_and_render(AudioSegment(.5));

    // Transition to a conditional expression
    LatexScene conditional("\\big(X\\ t\\ f\\big)", 0.5, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2);
    cs.add_scene_fade_in(&conditional, "conditional", 0.25, 0.3, true);
    cs.stage_macroblock_and_render(AudioSegment("We can make a term like this,"));
    cs.stage_macroblock_and_render(AudioSegment("where X represents either true or false."));

    // Highlight t and f
    conditional.begin_latex_transition("\\big("+latex_color(0xff00ff00, "(\\lambda ab. a)") + "\\ "+latex_color(0xff00ff00, "t")+"\\ f\\big)");
    cs.stage_macroblock_and_render(AudioSegment("If X is true, then the output'll be t."));
    conditional.begin_latex_transition("\\big("+latex_color(0xffff0000, "(\\lambda ab. b)") + "\\ t"+latex_color(0xffff0000, "\\ f") + "\\big)");
    cs.stage_macroblock_and_render(AudioSegment("If X is false, the output'll be f."));

    // Move conditional out and prepare for logic gates
    cs.state.superscene_transition(unordered_map<string, string>{
        {"true_scene.opacity", "0"},
        {"false_scene.opacity", "0"},
        {"true_latex.opacity", "0"},
        {"false_latex.opacity", "0"},
        {"conditional.opacity", "0"},
    });

    cs.stage_macroblock_and_render(AudioSegment("So how do we make common logic gates using these values?"));
    cs.remove_scene(&true_latex);
    cs.remove_scene(&false_latex);
    cs.remove_scene(&true_scene);
    cs.remove_scene(&false_scene);

    // Create NOT logic gate
    LatexScene not_gate(latex_text("NOT = "), 0.6, VIDEO_WIDTH, VIDEO_HEIGHT / 4);
    cs.add_scene_fade_in(&not_gate, "not_gate", 0, 0.375, true);
    cs.stage_macroblock_and_render(AudioSegment("Let's start with NOT."));

    LatexScene not_help(latex_text("NOT(TRUE) = FALSE") + "\\\\\\\\" + latex_text("NOT(FALSE) = TRUE"), 1, VIDEO_WIDTH*.4, VIDEO_HEIGHT * .3);
    cs.add_scene_fade_in(&not_help, "not_help", 0.3, 0, true);
    cs.stage_macroblock_and_render(AudioSegment("In other words, we want a function that maps TRUE onto FALSE, and vice versa."));

    // Show the abstraction for NOT
    not_gate.begin_latex_transition(latex_text("NOT = ") + "\\Big(\\lambda x.\\ \\_\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("'NOT' only takes one variable in, so our function should too."));

    not_gate.begin_latex_transition(latex_text("NOT = ") + "\\Big(\\lambda x. \\big(x\\ \\_\\ \\_\\big)\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("The trick is to use that argument as a selector."));
    not_gate.begin_latex_transition(latex_text("NOT = ") + "\\Big(\\lambda x. \\big(x\\ "+latex_color(0xffff0000, "\\_")+"\\ \\_\\big)\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("If the input is TRUE, then the first argument is the one which'll be picked."));
    not_gate.begin_latex_transition(latex_text("NOT = ") + "\\Big(\\lambda x. \\big(x\\ "+latex_text("FALSE")+"\\ \\_\\big)\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("Since NOT TRUE is FALSE, FALSE should be the thing in this blank."));
    not_gate.begin_latex_transition(latex_text("NOT = ") + "\\Big(\\lambda x. \\big(x\\ "+latex_text("FALSE")+"\\ "+latex_text("TRUE")+"\\big)\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("If the input is FALSE, we want the selected value to be TRUE."));
    cs.state.subscene_transition(unordered_map<string, string>{
        {"not_help.opacity", "0"},
    });
    cs.stage_macroblock_and_render(AudioSegment("So, this is NOT!"));

    // Substitute TRUE and FALSE expressions
    not_gate.begin_latex_transition(latex_text("NOT = ") + "\\Big(\\lambda x. \\big(x\\ (\\lambda ab. b)\\ (\\lambda ab. a)\\big)\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("If we wanna write it in full, we can substitute in the TRUEs and FALSEs."));
    cs.stage_macroblock(AudioSegment("But for now, let's leave it like this."), 2);
    not_gate.begin_latex_transition(latex_text("NOT = ") + "\\Big(\\lambda x. \\big(x\\ "+latex_text("FALSE")+"\\ "+latex_text("TRUE")+"\\big)\\Big)");
    cs.render_microblock();
    not_gate.begin_latex_transition("\\Big(\\lambda x. \\big(x\\ "+latex_text("FALSE")+"\\ "+latex_text("TRUE")+"\\big)\\Big)");
    cs.render_microblock();
    not_gate.begin_latex_transition("\\bigg(\\Big(\\lambda x. \\big(x\\ "+latex_text("FALSE")+"\\ "+latex_text("TRUE")+"\\big)\\Big)\\ " + latex_text("FALSE") + "\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("Let's see what happens, by passing in FALSE."));
    cs.stage_macroblock_and_render(AudioSegment("We're expecting TRUE out, since NOT FALSE is TRUE."));
    not_gate.begin_latex_transition("\\big(" + latex_text("FALSE FALSE")+"\\ "+latex_text("TRUE")+"\\big)");
    cs.stage_macroblock_and_render(AudioSegment("Beta reduce once,"));
    not_gate.begin_latex_transition("\\big((\\lambda ab. b)\\ "+latex_text("FALSE")+"\\ "+latex_text("TRUE")+"\\big)");
    cs.stage_macroblock_and_render(AudioSegment("Substitute in the known value of FALSE,"));
    not_gate.begin_latex_transition("\\big((\\lambda b. b)\\ "+latex_text("TRUE")+"\\big)");
    cs.stage_macroblock_and_render(AudioSegment("Beta reduce twice more,"));
    not_gate.begin_latex_transition(latex_text("TRUE"));
    cs.stage_macroblock_and_render(AudioSegment(1));
    cs.stage_macroblock_and_render(AudioSegment("And sure enough, it's TRUE!"));

    shared_ptr<LambdaExpression> lambda_not = parse_lambda_from_string("(\\x. ((x (\\a. (\\b. b))) (\\a. (\\b. a))))");
    lambda_not->flush_uid_recursive();
    LambdaScene not_scene(lambda_not, VIDEO_WIDTH*.4, VIDEO_HEIGHT*.4);
    cs.add_scene(&not_scene, "not_scene", 0.3, 0.3);
    cs.state.set(unordered_map<string, string>{
        {"not_scene.opacity", "0"},
    });
    cs.state.superscene_transition(unordered_map<string, string>{
        {"not_gate.opacity", "0"},
        {"not_scene.opacity", "1"},
    });
    cs.stage_macroblock_and_render(AudioSegment("Here's what NOT looks like as a diagram!"));
    cs.remove_scene(&not_gate);

    shared_ptr<LambdaApplication> not_body = dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaAbstraction>(lambda_not)->get_body());
    not_body->get_second()->set_color_recursive(0xff00ff00);
    dynamic_pointer_cast<LambdaApplication>(not_body->get_first())->get_second()->set_color_recursive(0xffff0000);
    not_scene.set_expression(lambda_not);
    cs.stage_macroblock_and_render(AudioSegment("I'll highlight the TRUE subcomponent as green, and FALSE as red."));

    shared_ptr<LambdaExpression> lambda_not_true = apply(lambda_not, parse_lambda_from_string("(\\m. (\\n. m))"), OPAQUE_WHITE);
    shared_ptr<LambdaExpression> lambda_not_false = apply(lambda_not, parse_lambda_from_string("(\\m. (\\n. n))"), OPAQUE_WHITE);
    not_scene.set_expression(lambda_not_true);
    cs.stage_macroblock_and_render(AudioSegment("Let's plug in TRUE this time, and check what we get."));
    not_scene.reduce();
    cs.stage_macroblock_and_render(AudioSegment(2));
    not_scene.reduce();
    cs.stage_macroblock_and_render(AudioSegment(2));
    not_scene.reduce();
    cs.stage_macroblock_and_render(AudioSegment(2));
    cs.stage_macroblock_and_render(AudioSegment("It's FALSE, as desired!"));
    LatexScene and_gate(latex_text("AND = "), 0.6, VIDEO_WIDTH, VIDEO_HEIGHT / 4);
    cs.add_scene(&and_gate, "and_gate", 0, 0.375);
    cs.state.set(unordered_map<string, string>{
        {"and_gate.opacity", "0"},
    });
    cs.state.superscene_transition(unordered_map<string, string>{
        {"and_gate.opacity", "1"},
        {"not_scene.opacity", "0"},
    });

    cs.stage_macroblock_and_render(AudioSegment(.5));
    cs.stage_macroblock_and_render(AudioSegment("How about AND?"));
    cs.remove_scene(&not_scene);

    and_gate.begin_latex_transition(latex_text("AND = ") + "\\Big(\\lambda xy.\\ (\\_)\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("'AND' takes two variables in, so our function should too."));

    and_gate.begin_latex_transition(latex_text("AND = ") + "\\Big(\\lambda xy.\\ (x\\ \\_\\ \\_)\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("We're gonna use the same trick as with not, using x as a selector."));
    and_gate.begin_latex_transition(latex_text("AND = ") + "\\Big(\\lambda xy.\\ (x\\ \\_\\ "+latex_text("FALSE")+")\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("If X is FALSE, then the answer is FALSE too."));
    cs.stage_macroblock_and_render(AudioSegment("Remember that TRUE AND something is just that something."));
    and_gate.begin_latex_transition(latex_text("AND = ") + "\\Big(\\lambda xy.\\ (x\\ y\\ "+latex_text("FALSE")+")\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("So, if X is TRUE, then the answer is whatever Y is."));
    and_gate.begin_latex_transition("\\Big(\\lambda xy.\\ (x\\ y\\ "+latex_text("FALSE")+")\\Big)");
    cs.stage_macroblock_and_render(AudioSegment("So, this is AND!"));
    cs.stage_macroblock_and_render(AudioSegment(.2));
    //TODO plug in and true false
    cs.state.superscene_transition(unordered_map<string, string>{
        {"and_gate.opacity", "0"},
    });
    cs.stage_macroblock_and_render(AudioSegment("I'll leave it as a challenge to you to find OR!"));
}

void numerals() {
    CompositeScene cs;
    cs.stage_macroblock(AudioSegment("Ok, now how about numbers?"), 3);
    LatexScene zero("0", 1, VIDEO_WIDTH/7, VIDEO_HEIGHT/7);
    LatexScene one ("1", 1, VIDEO_WIDTH/7, VIDEO_HEIGHT/7);
    LatexScene two ("2", 1, VIDEO_WIDTH/7, VIDEO_HEIGHT/7);
    cs.add_scene_fade_in(&zero, "zero", 0.15, 0.1, true);
    cs.render_microblock();
    cs.add_scene_fade_in(&one , "one" , 0.15, 0.3, true);
    cs.render_microblock();
    cs.add_scene_fade_in(&two , "two" , 0.15, 0.5, true);
    cs.render_microblock();
    cs.stage_macroblock(AudioSegment("The underlying inspiration here is to represent 1 as f(x), 2 as f(f(x)), and so on."), 3);
    LatexScene f_zero("x"      , 0.8, VIDEO_WIDTH/3, VIDEO_HEIGHT/7);
    LatexScene f_one ("(f\\ x)"   , 1, VIDEO_WIDTH/3, VIDEO_HEIGHT/7);
    LatexScene f_two ("(f\\ (f\\ x))", 1, VIDEO_WIDTH/3, VIDEO_HEIGHT/7);
    shared_ptr<LambdaExpression> numeral_0 = parse_lambda_from_string("(\\f. (\\x. x))");
    numeral_0->flush_uid_recursive();
    LambdaScene zero_scene(numeral_0, VIDEO_WIDTH / 4, VIDEO_HEIGHT / 5);

    shared_ptr<LambdaExpression> numeral_1 = parse_lambda_from_string("(\\f. (\\x. (f x)))");
    numeral_1->flush_uid_recursive();
    LambdaScene one_scene(numeral_1, VIDEO_WIDTH / 4, VIDEO_HEIGHT / 5);

    shared_ptr<LambdaExpression> numeral_2 = parse_lambda_from_string("(\\f. (\\x. (f (f x))))");
    numeral_2->flush_uid_recursive();
    LambdaScene two_scene(numeral_2, VIDEO_WIDTH / 4, VIDEO_HEIGHT / 5);

    cs.add_scene_fade_in(&f_zero, "f_zero", 0.333, 0.1, true);
    cs.add_scene_fade_in(&zero_scene, "zero_scene", 0.6, .07, true);
    cs.render_microblock();
    cs.add_scene_fade_in(&f_one , "f_one" , 0.333, 0.3, true);
    cs.add_scene_fade_in(&one_scene, "one_scene", 0.6, .27, true);
    cs.render_microblock();
    cs.add_scene_fade_in(&f_two , "f_two" , 0.333, 0.5, true);
    cs.add_scene_fade_in(&two_scene, "two_scene", 0.6, .47, true);
    cs.render_microblock();

    LatexScene f_n("(f^n\\ x)", 1, VIDEO_WIDTH/3, VIDEO_HEIGHT/7);
    cs.add_scene_fade_in(&f_n, "f_n", 0.333, 0.7, true);
    LatexScene n   ("n", 1, VIDEO_WIDTH/7, VIDEO_HEIGHT/7);
    cs.add_scene_fade_in(&n   , "n"   , 0.15, 0.7, true);
    cs.stage_macroblock_and_render(AudioSegment("In other words, the number n is represented by applying some function f to some value x, n times over."));
    cs.stage_macroblock_and_render(AudioSegment(0.2));


    f_zero.begin_latex_transition("(\\lambda fx.\\ x)");
    f_one .begin_latex_transition("(\\lambda fx.\\ (f\\ x))");
    f_two .begin_latex_transition("(\\lambda fx.\\ (f\\ (f\\ x)))");
    f_n   .begin_latex_transition("(\\lambda fx.\\ (f^n\\ x))");
    cs.stage_macroblock_and_render(AudioSegment("Well, almost- the numbers are really two-argument functions which take in f and x, and then apply f to x n times."));
    cs.stage_macroblock_and_render(AudioSegment("It's important to understand the difference there."));
    cs.stage_macroblock(AudioSegment("These numbers ARE FUNCTIONS that TAKE IN A FUNCTION AND A VARIABLE, and then iterate that function on that variable."), 4);
    cs.render_microblock();
    f_zero.begin_latex_transition("(\\lambda "+latex_color(0xffff0000, "f")+"x.\\ x)");
    f_one .begin_latex_transition("(\\lambda "+latex_color(0xffff0000, "f")+"x.\\ (f\\ x))");
    f_two .begin_latex_transition("(\\lambda "+latex_color(0xffff0000, "f")+"x.\\ (f\\ (f\\ x)))");
    f_n   .begin_latex_transition("(\\lambda "+latex_color(0xffff0000, "f")+"x.\\ (f^n\\ x))");
    cs.render_microblock();
    f_zero.begin_latex_transition("(\\lambda "+latex_color(0xffff0000, "f")+latex_color(0xff0088ff, "x") + ".\\ x)");
    f_one .begin_latex_transition("(\\lambda "+latex_color(0xffff0000, "f")+latex_color(0xff0088ff, "x") + ".\\ (f\\ x))");
    f_two .begin_latex_transition("(\\lambda "+latex_color(0xffff0000, "f")+latex_color(0xff0088ff, "x") + ".\\ (f\\ (f\\ x)))");
    f_n   .begin_latex_transition("(\\lambda "+latex_color(0xffff0000, "f")+latex_color(0xff0088ff, "x") + ".\\ (f^n\\ x))");
    cs.render_microblock();
    cs.render_microblock();
    f_zero.begin_latex_transition("(\\lambda fx.\\ x)");
    f_one .begin_latex_transition("(\\lambda fx.\\ (f\\ x))");
    f_two .begin_latex_transition("(\\lambda fx.\\ (f\\ (f\\ x)))");
    f_n   .begin_latex_transition("(\\lambda fx.\\ (f^n\\ x))");
    // Create Lambda scenes for Church numerals
    cs.fade_out_all_scenes();

    cs.stage_macroblock(AudioSegment("So, we can give 2 the function sin and the value 5,"), 2);
    cs.render_microblock();
    LatexScene two_sin_5("2\\ sin\\ 5", 0.8, VIDEO_WIDTH, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&two_sin_5, "2s5", 0, .375, true);
    cs.render_microblock();
    two_sin_5.begin_latex_transition("(sin\\ (sin\\ 5))");
    cs.stage_macroblock_and_render(AudioSegment("and 2 will apply sin to 5 twice."));


    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment("So how do we do math on these?"));
    cs.remove_all_scenes();

    LatexScene succ_gate(latex_text("SUCC"), 0.6, VIDEO_WIDTH, VIDEO_HEIGHT / 4);
    cs.add_scene_fade_in(&succ_gate, "succ_gate", 0, 0.375, true);
    cs.stage_macroblock_and_render(AudioSegment("Let's start with succession."));
    succ_gate.begin_latex_transition(latex_text("SUCC(500)=501"));
    cs.stage_macroblock_and_render(AudioSegment("This is the function that, when given n, returns n+1."));

    succ_gate.begin_latex_transition(latex_text("SUCC = ") + "\\bigg(\\lambda n.\\ \\_\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("It takes one number as an input."));

    succ_gate.begin_latex_transition(latex_text("SUCC = ") + "\\bigg(\\lambda n.\\ \\Big(\\lambda fx.\\ \\_\\Big)\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("The thing that it spits out is also a number, so we're gonna make the body look like a number."));
    cs.stage_macroblock_and_render(AudioSegment(.5));

    cs.stage_macroblock_and_render(AudioSegment("We wanna apply f to x n+1 times."));

    succ_gate.begin_latex_transition(latex_text("SUCC = ") + "\\bigg(\\lambda n.\\ \\Big(\\lambda fx.\\ \\big(n f x\\big)\\Big)\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("We can start by applying f to x n times."));
    cs.stage_macroblock(AudioSegment("We do this by using the numeral as a function and passing f and x into it."), 3);
    succ_gate.begin_latex_transition(latex_text("SUCC = ") + "\\bigg(\\lambda n.\\ \\Big(\\lambda fx.\\ \\big("+latex_color(0xffff0000, "n") + " f x\\big)\\Big)\\bigg)");
    cs.render_microblock();
    succ_gate.begin_latex_transition(latex_text("SUCC = ") + "\\bigg(\\lambda n.\\ \\Big(\\lambda fx.\\ \\big("+latex_color(0xffff0000, "n")+latex_color(0xff00ff00, "f") + " x\\big)\\Big)\\bigg)");
    cs.render_microblock();
    succ_gate.begin_latex_transition(latex_text("SUCC = ") + "\\bigg(\\lambda n.\\ \\Big(\\lambda fx.\\ \\big("+latex_color(0xffff0000, "n")+latex_color(0xff00ff00, "f") + latex_color(0xff0088ff, "x") + "\\big)\\Big)\\bigg)");
    cs.render_microblock();

    succ_gate.begin_latex_transition(latex_text("SUCC = ") + "\\bigg(\\lambda n.\\ \\Big(\\lambda fx.\\ \\big(f (n f x)\\big)\\Big)\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("All that's left is to apply f just one more time!"));

    // Create LambdaScene for SUCC(5)
    shared_ptr<LambdaExpression> numeral_5 = parse_lambda_from_string("(\\f. (\\x. (f (f (f (f (f x)))))))"); // Church numeral for 5
    numeral_5->set_color_recursive(0xff0000ff);
    
    shared_ptr<LambdaExpression> lambda_succ = parse_lambda_from_string("(\\n. (\\f. (\\x. (f ((n f) x)))))");
    dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaAbstraction>(lambda_succ)->get_body())->get_body())->get_body())->get_first()->set_color_recursive(0xffff0000);
    shared_ptr<LambdaExpression> succ_5 = apply(lambda_succ, numeral_5, OPAQUE_WHITE); // SUCC(5)
    succ_5->flush_uid_recursive();
    LambdaScene succ_5_scene(succ_5, VIDEO_WIDTH/2, VIDEO_HEIGHT/2);

    cs.add_scene(&succ_5_scene, "succ_5_scene", 0.25, 0.15);
    cs.state.set(unordered_map<string, string>{
        {"succ_5_scene.opacity", "0"},
    });

    cs.state.subscene_transition(unordered_map<string, string>{
        {"succ_5_scene.opacity", "1"},
        {"succ_gate.y", ".65"},
    });
    cs.stage_macroblock_and_render(AudioSegment("Let's evaluate the successor of 5."));
    succ_gate.begin_latex_transition(latex_text("SUCC = ") + "\\bigg(\\lambda n.\\ \\Big(\\lambda fx.\\ \\big("+latex_color(0xffff0000, "f") + " (n f x)\\big)\\Big)\\bigg)");
    cs.stage_macroblock_and_render(AudioSegment("I've colored the extra f red. Keep an eye on it during reduction!"));
    succ_5_scene.reduce();
    cs.stage_macroblock_and_render(AudioSegment(1));
    succ_5_scene.reduce();
    cs.stage_macroblock_and_render(AudioSegment(1));
    succ_5_scene.reduce();
    cs.stage_macroblock_and_render(AudioSegment(1));
    cs.stage_macroblock_and_render(AudioSegment("Check it out- that's 6!"));
    cs.state.superscene_transition(unordered_map<string, string>{
        {"succ_gate.opacity", "0"},
        {"succ_5_scene.opacity", "0"},
    });
    cs.stage_macroblock_and_render(AudioSegment(1));
    cs.remove_scene(&succ_gate);
    cs.remove_scene(&succ_5_scene);

    // Create the ADD function
    LatexScene add_gate(latex_text("+"), 0.6, VIDEO_WIDTH, VIDEO_HEIGHT / 4);
    cs.add_scene(&add_gate, "add_gate", 0, 0.375);
    cs.state.set(unordered_map<string, string>{
        {"add_gate.opacity", "0"},
    });
    cs.state.subscene_transition(unordered_map<string, string>{
        {"add_gate.opacity", "1"},
    });
    cs.stage_macroblock_and_render(AudioSegment("How 'bout addition then?"));

    add_gate.begin_latex_transition(latex_text("+ = ") + "\\big(\\lambda mn.\\ \\_\\big)");
    cs.stage_macroblock_and_render(AudioSegment("Addition is a function that takes in two numbers."));
    cs.stage_macroblock_and_render(AudioSegment("Since we already defined the successor function, we can use it here!"));
    LatexScene addition_help("n + 1 = SUCC\\big(n\\big)", 0.6, VIDEO_WIDTH/2, VIDEO_HEIGHT/4);
    cs.add_scene(&addition_help, "addition_help", 0.25, 0.65);
    cs.stage_macroblock_and_render(AudioSegment("Calling successor adds one to a number!"));
    addition_help.begin_latex_transition("n + m = SUCC\\big(SUCC(...\\ n)\\big)");
    cs.stage_macroblock_and_render(AudioSegment("To get n+m, we just have to call successor on n, m times over."));
    cs.stage_macroblock_and_render(AudioSegment("Remember, the number m itself _is a function_,"));
    addition_help.begin_latex_transition("n + m = \\big(m\\ SUCC\\ n\\big)");
    cs.stage_macroblock_and_render(AudioSegment("which when applied to the successor function, will iterate that function m times!"));
    cs.stage_macroblock_and_render(AudioSegment("So, m SUCC n is the same as adding one to n, m times over."));
    add_gate.begin_latex_transition(latex_text("+ = ") + "\\big(\\lambda mn. (m\\ SUCC\\ n)\\big)");
    cs.state.subscene_transition(unordered_map<string, string>{
        {"addition_help.opacity", "0"},
    });
    cs.stage_macroblock_and_render(AudioSegment("Plugging this into our function body, we get this."));

    // Create LambdaScene for ADD(5, 3)
    shared_ptr<LambdaExpression> numeral_3 = parse_lambda_from_string("(\\f. (\\x. (f (f (f x)))))"); // Church numeral for 3
    numeral_3->set_color_recursive(0xff00ff00);

    shared_ptr<LambdaExpression> add_function = parse_lambda_from_string("(\\m. (\\n. ((m (\\n. (\\f. (\\x. (f ((n f) x)))))) n)))"); // ADD function
    add_function->flush_uid_recursive();
    numeral_3->flush_uid_recursive();
    numeral_5->flush_uid_recursive();
    shared_ptr<LambdaExpression> add_5_3 = apply(apply(add_function, numeral_5, OPAQUE_WHITE), numeral_3, OPAQUE_WHITE); // ADD(5, 3)

    LambdaScene add_5_3_scene(add_function, VIDEO_WIDTH/2, VIDEO_HEIGHT/2);

    cs.add_scene(&add_5_3_scene, "add_5_3_scene", 0.25, 0.2);
    cs.state.set(unordered_map<string, string>{
        {"add_5_3_scene.opacity", "0"},
    });

    cs.state.subscene_transition(unordered_map<string, string>{
        {"add_5_3_scene.opacity", "1"},
        {"add_gate.y", ".65"},
    });

    cs.stage_macroblock_and_render(AudioSegment("That's an addition function!"));
    cs.remove_scene(&addition_help);
    add_5_3_scene.set_expression(add_5_3);
    cs.stage_macroblock_and_render(AudioSegment("Let's add 5 and 3 together."));

    int num_reductions = add_5_3->count_reductions();
    cs.stage_macroblock(AudioSegment(6), num_reductions);
    for(int i = 0; i < num_reductions; i++) {
        add_5_3_scene.reduce();
        cs.render_microblock();
    }
    cs.stage_macroblock_and_render(AudioSegment("It's eight!"));

    // Return to the plain addition function
    cs.stage_macroblock_and_render(AudioSegment("This addition function takes a lot of beta reductions to complete."));
    shared_ptr<LambdaExpression> add_fast = parse_lambda_from_string("(\\m. (\\n. (\\f. (\\x. ((m f) ((n f) x))))))");
    add_fast->set_color_recursive(0xffff0000);
    LambdaScene add_fast_scene(add_fast, VIDEO_WIDTH/2, VIDEO_HEIGHT/2);
    cs.add_scene(&add_fast_scene, "add_fast_scene", 0.25, 0.35);
    add_fast_scene.state.set(unordered_map<string, string>{
        {"w", "[add_fast_scene.w]"},
        {"h", "[add_fast_scene.h]"},
    });
    cs.state.set(unordered_map<string, string>{
        {"add_fast_scene.opacity", "0"},
        {"add_fast_scene.w", to_string(VIDEO_WIDTH*.5)},
        {"add_fast_scene.h", to_string(VIDEO_HEIGHT*.5)},
    });
    cs.state.subscene_transition(unordered_map<string, string>{
        {"add_5_3_scene.opacity", "0"},
        {"add_fast_scene.opacity", "1"},
        {"add_fast_scene.y", ".25"},
        {"add_5_3_scene.y", ".15"},
        {"add_gate.y", "1.75"},
    });
    cs.stage_macroblock_and_render(AudioSegment("I'll spoil a faster one here."));
    cs.remove_scene(&add_gate);
    cs.state.subscene_transition(unordered_map<string, string>{
        {"add_fast_scene.x", ".05"},
        {"add_fast_scene.y", ".05"},
        {"add_fast_scene.w", to_string(VIDEO_WIDTH*.3)},
        {"add_fast_scene.h", to_string(VIDEO_HEIGHT*.3)},
    });
    cs.stage_macroblock_and_render(AudioSegment(1));
    cs.stage_macroblock_and_render(AudioSegment("Multiplication can be defined in terms of repeated addition,"));
    shared_ptr<LambdaExpression> mult_function = parse_lambda_from_string("(\\m. (\\n. (\\s. (n (m s)))))");
    mult_function->set_color_recursive(0xff00ff00);
    LambdaScene mult_scene(mult_function, VIDEO_WIDTH/2, VIDEO_HEIGHT/2);
    cs.add_scene(&mult_scene, "mult_scene", 0.25, 0.35);
    mult_scene.state.set(unordered_map<string, string>{
        {"w", "[mult_scene.w]"},
        {"h", "[mult_scene.h]"},
    });
    cs.state.set(unordered_map<string, string>{
        {"mult_scene.opacity", "0"},
        {"mult_scene.w", to_string(VIDEO_WIDTH*.5)},
        {"mult_scene.h", to_string(VIDEO_HEIGHT*.5)},
    });
    cs.state.subscene_transition(unordered_map<string, string>{
        {"mult_scene.opacity", "1"},
        {"mult_scene.y", ".25"},
    });
    cs.stage_macroblock_and_render(AudioSegment("but there's also a quicker way."));
    cs.state.subscene_transition(unordered_map<string, string>{
        {"mult_scene.x", ".35"},
        {"mult_scene.y", ".05"},
        {"mult_scene.w", to_string(VIDEO_WIDTH*.3)},
        {"mult_scene.h", to_string(VIDEO_HEIGHT*.3)},
    });
    cs.stage_macroblock_and_render(AudioSegment(1));
    shared_ptr<LambdaExpression> exp_function = parse_lambda_from_string("(\\m. (\\n. (n m)))");
    exp_function->set_color_recursive(0xff0000ff);
    LambdaScene exp_scene(exp_function, VIDEO_WIDTH/2, VIDEO_HEIGHT/2);
    cs.add_scene(&exp_scene, "exp_scene", 0.25, 0.35);
    exp_scene.state.set(unordered_map<string, string>{
        {"w", "[exp_scene.w]"},
        {"h", "[exp_scene.h]"},
    });
    cs.state.set(unordered_map<string, string>{
        {"exp_scene.opacity", "0"},
        {"exp_scene.w", to_string(VIDEO_WIDTH*.5)},
        {"exp_scene.h", to_string(VIDEO_HEIGHT*.5)},
    });
    cs.state.subscene_transition(unordered_map<string, string>{
        {"exp_scene.opacity", "1"},
        {"exp_scene.y", ".25"},
    });
    cs.stage_macroblock_and_render(AudioSegment("Here's exponentiation too."));
    cs.state.subscene_transition(unordered_map<string, string>{
        {"exp_scene.x", ".65"},
        {"exp_scene.y", ".05"},
        {"exp_scene.w", to_string(VIDEO_WIDTH*.3)},
        {"exp_scene.h", to_string(VIDEO_HEIGHT*.3)},
    });
    cs.stage_macroblock_and_render(AudioSegment(1));
    LatexScene ptp("+ \\times +", 0.8, VIDEO_WIDTH, VIDEO_HEIGHT/4);
    cs.add_scene(&ptp, "ptp", 0, 0.375);
    cs.state.subscene_transition(unordered_map<string, string>{
        {"add_fast_scene.opacity", "0.2"},
        {"mult_scene.opacity", "0.2"},
        {"exp_scene.opacity", "0.2"},
    });
    cs.stage_macroblock(AudioSegment("Now I promised you in the thumbnail that we'd find the answer to 'plus times plus',"), 2);
    cs.state.set(unordered_map<string, string>{
        {"ptp.opacity", "0"},
    });
    cs.state.subscene_transition(unordered_map<string, string>{
        {"ptp.opacity", "1"},
    });
    cs.render_microblock();
    ptp.begin_latex_transition("((\\times +) +)");
    cs.render_microblock();
    cs.stage_macroblock_and_render(AudioSegment("so here goes."));
    ptp.begin_latex_transition("(((\\lambda nms. (n(ms))) +) +)");
    cs.stage_macroblock_and_render(AudioSegment("Let's plug in the function for times."));
    cs.stage_macroblock(AudioSegment("Reducing, we get this."), 2);
    ptp.begin_latex_transition("((\\lambda ms. (+(ms))) +)");
    cs.render_microblock();
    ptp.begin_latex_transition("(\\lambda s. (+(+s)))");
    cs.render_microblock();
    cs.stage_macroblock_and_render(AudioSegment("Ok, that's kinda weird,"));
    ptp.begin_latex_transition("("+latex_color(0xffff4444, "\\lambda s.") + "(+(+s)))");
    cs.stage_macroblock_and_render(AudioSegment("we get another function."));
    cs.stage_macroblock_and_render(AudioSegment("But let's just roll with it."));
    ptp.begin_latex_transition("((\\lambda s. (+(+s)))a)");
    cs.stage_macroblock_and_render(AudioSegment("What happens when we stick a number in?"));
    cs.stage_macroblock_and_render(AudioSegment("Let's call it 'a'."));
    ptp.begin_latex_transition("(+(+a))");
    cs.stage_macroblock_and_render(AudioSegment("Reduce..."));
    ptp.begin_latex_transition("((\\lambda mnfx.((mf)((nf)x)))(+a))");
    cs.stage_macroblock_and_render(AudioSegment("Substitute the definition of plus..."));
    ptp.begin_latex_transition("(\\lambda nfx.(((+a)f)((nf)x)))");
    cs.stage_macroblock_and_render(AudioSegment(2));
    cs.stage_macroblock_and_render(AudioSegment("This is still a function...!"));
    ptp.begin_latex_transition("((\\lambda nfx.(((+a)f)((nf)x)))b)");
    cs.stage_macroblock_and_render(AudioSegment("Let's keep sticking numbers in until we get a number out."));
    ptp.begin_latex_transition("(\\lambda fx.(((+a)f)((bf)x)))");
    cs.stage_macroblock_and_render(AudioSegment(1));
    ptp.begin_latex_transition("((\\lambda fx.(((+a)f)((bf)x)))c)");
    cs.stage_macroblock_and_render(AudioSegment(1));
    ptp.begin_latex_transition("(\\lambda x.(((+a)c)((bc)x)))");
    cs.stage_macroblock_and_render(AudioSegment(1));
    ptp.begin_latex_transition("((\\lambda x.(((+a)c)((bc)x)))d)");
    cs.stage_macroblock_and_render(AudioSegment(1));
    ptp.begin_latex_transition("(((+a)c)((bc)d))");
    cs.stage_macroblock_and_render(AudioSegment("Ok, after inserting 4 arguments we got a number out!"));
    cs.stage_macroblock_and_render(AudioSegment("Let's write it in a more familiar notation..."));
    ptp.begin_latex_transition("((a+c)((bc)d))");
    cs.stage_macroblock_and_render(AudioSegment("This term is just a+c."));
    ptp.begin_latex_transition("((a+c)((c^b)d))");
    cs.stage_macroblock_and_render(AudioSegment("and there's a few examples of exponentiation going on here."));
    ptp.begin_latex_transition("((a+c)(d^{(c^b)}))");
    cs.stage_macroblock_and_render(AudioSegment(1));
    ptp.begin_latex_transition("(d^{(c^b)})^{a+c}");
    cs.stage_macroblock_and_render(AudioSegment(1));
    ptp.begin_latex_transition("+\\times+(a,b,c,d) = (d^{(c^b)})^{a+c}");
    cs.stage_macroblock(AudioSegment("Ok, so what we learned here is that 'plus times plus' is a function of four arguments which spits out this power tower."), 2);
    cs.render_microblock();
    cs.render_microblock();
    cs.stage_macroblock_and_render(AudioSegment("Obviously this is completely ridiculous."));
    cs.stage_macroblock_and_render(AudioSegment("But it serves to prove a point..."));
    cs.state.subscene_transition(unordered_map<string, string>{
        {"add_fast_scene.x", "0.133"},
        {"add_fast_scene.y", "0.133"},
        {"add_fast_scene.opacity", "0.15"},
        {"mult_scene.x", "0.333"},
        {"mult_scene.y", "0.333"},
        {"mult_scene.opacity", "0.15"},
        {"exp_scene.x", "0.533"},
        {"exp_scene.y", "0.533"},
        {"exp_scene.opacity", "0.15"},
    });
    cs.stage_macroblock_and_render(AudioSegment("In a world where everything is a function,"));
    ptp.begin_latex_transition(latex_color(0xffff0000, "+\\times+") + "(a,b,c,d) = (d^{(c^b)})^{a+c}");
    cs.stage_macroblock_and_render(AudioSegment("you can do things that normal math just doesn't permit,"));
    ptp.begin_latex_transition(latex_color(0xffff0000, "+\\times+") + latex_color(0xff0088ff, "(a,b,c,d) = (d^{(c^b)})^{a+c}"));
    cs.stage_macroblock_and_render(AudioSegment("and strange emergent behavior is the norm."));
    cs.state.subscene_transition(unordered_map<string, string>{
        {"ptp.opacity", "0"},
        {"add_fast_scene.opacity", "0"},
        {"mult_scene.opacity", "0"},
        {"exp_scene.opacity", "0"},
    });
    cs.stage_macroblock_and_render(AudioSegment("But it's gonna get plenty weirder yet."));
}

shared_ptr<LambdaExpression> factorial(){
    CompositeScene cs;
    /*
    LatexScene title(latex_text("Recursion"), 1, VIDEO_WIDTH*0.5, VIDEO_HEIGHT*0.25);
    cs.add_scene(&title, "title", 0.25, -.25);
    cs.state.superscene_transition(unordered_map<string, string>{
        {"title.y", "0"},
    });
    */
    cs.stage_macroblock_and_render(AudioSegment("We're gonna need some voodoo magic to make the factorial function."));
    ConvolutionScene py_fac(png_to_pix_bounding_box("factorial_first", VIDEO_WIDTH*.6, VIDEO_HEIGHT*.6), VIDEO_WIDTH*.6, VIDEO_HEIGHT*.6);
    cs.add_scene_fade_in(&py_fac, "py_fac", 0.2, 0.2, true);
    cs.stage_macroblock_and_render(AudioSegment("Here's a typical implementation in python."));
    py_fac.begin_transition(png_to_pix_bounding_box("factorial_first_colored", VIDEO_WIDTH*.6, VIDEO_HEIGHT*.6));
    cs.stage_macroblock_and_render(AudioSegment("The core idea is to return n times the factorial of the previous number."));
    py_fac.begin_transition(png_to_pix_bounding_box("factorial", VIDEO_WIDTH*.6, VIDEO_HEIGHT*.6));
    cs.stage_macroblock_and_render(AudioSegment("However, we don't wanna go into the negatives, so we add a base case."));
    cs.state.superscene_transition(unordered_map<string, string>{
        {"py_fac.y", "0"},
    });
    LatexScene fac(latex_text("fac = "), 0.6, VIDEO_WIDTH, VIDEO_HEIGHT/6);
    cs.add_scene_fade_in(&fac, "fac", 0, 0.5, true);
    cs.stage_macroblock_and_render(AudioSegment("Alright, let's try it in the lambda calculus!"));
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ \\_)");
    cs.stage_macroblock_and_render(AudioSegment("Factorial just takes in one variable."));
    cs.stage_macroblock_and_render(AudioSegment("Let's say you have an IS\\_0 function available, which checks if a number is 0 and gives back TRUE or FALSE."));
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ "+latex_text("IS\\_0")+"\\big)");
    cs.stage_macroblock_and_render(AudioSegment("We could make it, but let's not get distracted."));
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ ("+latex_text("IS\\_0")+"\\ n)\\big)");
    cs.stage_macroblock_and_render(AudioSegment("IS\\_0(n) yields a boolean. And as we know, booleans work as selectors!"));
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ "+latex_text("then_case")+"\\ "+latex_text("else_case")+")\\big)");
    cs.stage_macroblock_and_render(AudioSegment("Thus, we already know how to make this if-then-else block."));
    cs.stage_macroblock_and_render(AudioSegment("Therefore, factorial takes this form:"));
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ ("+latex_color(0xff0088ff,"("+latex_text("IS\\_0")+"\\ n)\\ ")+latex_text("then_case")+"\\ "+latex_text("else_case")+")\\big)");
    cs.stage_macroblock_and_render(AudioSegment("The boolean,"));
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ "+latex_color(0xff0088ff,latex_text("then_case"))+"\\ "+latex_text("else_case")+")\\big)");
    cs.stage_macroblock_and_render(AudioSegment("what we wanna return if it's true,"));
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ "+latex_text("then_case")+"\\ "+latex_color(0xff0088ff,latex_text("else_case"))+")\\big)");
    cs.stage_macroblock_and_render(AudioSegment("and what we wanna return if it's false."));
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ "+latex_text("then_case")+"\\ "+latex_text("else_case")+")\\big)");
    cs.stage_macroblock_and_render(AudioSegment("If it's true, just like our python function,"));
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ "+latex_text("else_case")+")\\big)");
    cs.stage_macroblock_and_render(AudioSegment("we simply return 1."));
    cs.stage_macroblock_and_render(AudioSegment(.3));
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ \\_)\\big)");
    cs.stage_macroblock_and_render(AudioSegment("If it's false, we return the recursive call."));
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ \\_))\\big)");
    cs.stage_macroblock_and_render(AudioSegment("We multiply n,"));
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ ("+latex_text("fac")+"(-\\ n\\ 1))))\\big)");
    cs.stage_macroblock_and_render(AudioSegment("by the factorial of n-1."));
    cs.stage_macroblock_and_render(AudioSegment("Now wait a minute."));
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ ("+latex_color(0xffff0000,latex_text("fac"))+"(-\\ n\\ 1))))\\big)");
    cs.stage_macroblock_and_render(AudioSegment("Our definition of the factorial function contains itself inside."));
    py_fac.begin_transition(png_to_pix_bounding_box("factorial_nested", VIDEO_WIDTH*.6, VIDEO_HEIGHT*.6));
    cs.stage_macroblock_and_render(AudioSegment("In python this is fine..."));
    cs.stage_macroblock_and_render(AudioSegment("we can instantiate the function anywhere, even within itself."));
    py_fac.begin_transition(png_to_pix_bounding_box("factorial", VIDEO_WIDTH*.6, VIDEO_HEIGHT*.6));
    cs.stage_macroblock_and_render(AudioSegment("But a lambda calculus term is self-contained."));
    cs.stage_macroblock(AudioSegment("How would I draw a diagram of this term, if I don't know what this component is?"), 7);
    cs.render_microblock();
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ ("+latex_color(0xffffffff,latex_text("fac"))+"(-\\ n\\ 1))))\\big)");
    cs.render_microblock();
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ ("+latex_color(0xffff0000,latex_text("fac"))+"(-\\ n\\ 1))))\\big)");
    cs.render_microblock();
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ ("+latex_color(0xffffffff,latex_text("fac"))+"(-\\ n\\ 1))))\\big)");
    cs.render_microblock();
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ ("+latex_color(0xffff0000,latex_text("fac"))+"(-\\ n\\ 1))))\\big)");
    cs.render_microblock();
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ ("+latex_color(0xffffffff,latex_text("fac"))+"(-\\ n\\ 1))))\\big)");
    cs.render_microblock();
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ ("+latex_color(0xffff0000,latex_text("fac"))+"(-\\ n\\ 1))))\\big)");
    cs.render_microblock();
    fac.begin_latex_transition(latex_text("fac = ") + "\\big(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ (\\scriptsize(\\lambda n. (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n(...(-\\ n\\ 1)))))\\normalsize(-\\ n\\ 1))))\\big)");
    cs.stage_macroblock_and_render(AudioSegment("You could argue, as is, this definition's infinitely long."));
    fac.begin_latex_transition(latex_text("fac = ") + "\\scriptsize(\\lambda n. (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n(\\tiny(\\lambda n. (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n(...(-\\ n\\ 1)))))(-\\ n\\ 1)))))\\scriptsize (-\\ n\\ 1)))))");
    cs.stage_macroblock_and_render(AudioSegment("How on earth are we gonna define recursion?"));
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment("Time for a mind-blowing detour."));
    cs.remove_all_scenes();
    RealFunctionScene rfs;
    cs.add_scene_fade_in(&rfs, "rfs", 0, 0, true);
    rfs.add_function("? sin", 0xffff0000);
    rfs.begin_transition(0, "? sin ? 2 * cos +");
    cs.stage_macroblock_and_render(AudioSegment("Let's talk about normal functions on the real numbers."));
    cs.stage_macroblock_and_render(AudioSegment("A fixed point is some number that you can plug into a function, and you get the same number back."));
    LatexScene fp_math("x = f(x)", 0.5, VIDEO_WIDTH/2, VIDEO_HEIGHT/3);
    cs.add_scene_fade_in(&fp_math, "fp_math", 0, 0.06, true);
    cs.stage_macroblock(AudioSegment("In other words, x is a fixed point if f(x) matches x."), 2);
    cs.render_microblock();
    cs.render_microblock();
    rfs.add_function("?", 0xff0088ff);
    cs.stage_macroblock_and_render(AudioSegment("Graphing this, it's where the line y=x intersects with your function."));
    rfs.begin_transition(0, "? sin");
    fp_math.begin_latex_transition("x = sin(x)");
    cs.stage_macroblock_and_render(AudioSegment("Sine evidently has exactly one fixed point,"));
    cs.stage_macroblock_and_render(AudioSegment("exactly at zero."));
    rfs.begin_transition(0, "? ? *");
    fp_math.begin_latex_transition("x = x^2");
    cs.stage_macroblock_and_render(AudioSegment("y=x^2 has 2 fixed points, 0 and 1."));
    cs.stage_macroblock_and_render(AudioSegment("Those are the only reals which are their own squares."));
    rfs.begin_transition(0, "? ? * 2 +");
    fp_math.begin_latex_transition("x = x^2 + 2");
    cs.stage_macroblock_and_render(AudioSegment("But what about y=x^2+2?"));
    cs.stage_macroblock_and_render(AudioSegment("It doesn't even have a fixed point!"));
    cs.stage_macroblock_and_render(AudioSegment(0.5));
    rfs.begin_transition(0, "? sin ? 2.1 * cos ? 1.5 sin * 2.6 ? cos * + + +");
    fp_math.begin_latex_transition("x = \\sum_{n=1}^{\\infty}\\frac{1}{n^x} + x");
    cs.stage_macroblock_and_render(AudioSegment("What about arbitrary functions?"));
    rfs.begin_transition(0, "? sin ? 2.1 * cos ? 1.5 sin * 2.6 ? cos * + + + ? cos *");
    cs.stage_macroblock_and_render(AudioSegment("If you can find all fixed points of the Riemann Zeta Function plus its input, you would solve the Riemann Hypothesis, and win a million dollars!"));
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment("Ok. Now hold that thought."));
    cs.remove_all_scenes();
    LatexScene ls("U = (\\lambda xy. (y ((x x) y)))", 0.8, VIDEO_WIDTH*.8, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&ls, "ls", 0, 0.25, true);
    cs.stage_macroblock_and_render(AudioSegment("Check out this lambda term."));
    LatexScene ls2("UU", 0.8, VIDEO_WIDTH/2, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&ls2, "ls2", 0.25, 0.5, true);
    cs.stage_macroblock_and_render(AudioSegment("If we apply it to itself, we get this term,"));
    ls2.begin_latex_transition("\\Theta = (UU)");
    cs.stage_macroblock_and_render(AudioSegment("called the Turing Fixed Point Combinator!"));
    cs.state.superscene_transition(unordered_map<string, string>{
        {"ls.x", "-.1"},
        {"ls2.x", "0.55"},
        {"ls.y",  "0.02"},
        {"ls2.y", "0.02"},
    });
    cs.stage_macroblock_and_render(AudioSegment("It does something so cool it's unreal."));
    LatexScene ls3("F", 0.8, VIDEO_WIDTH, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&ls3, "ls3", 0, 0.4, true);
    cs.stage_macroblock_and_render(AudioSegment("Let's just imagine you have some function F."));
    ls3.begin_latex_transition("(\\Theta F)");
    cs.stage_macroblock_and_render(AudioSegment("Apply the fixed point combinator to F, and see what happens!"));
    ls3.begin_latex_transition("((UU)F)");
    cs.stage_macroblock_and_render(AudioSegment("We know Theta is UU, so let's apply that identity..."));
    ls3.begin_latex_transition("(((\\lambda xy. (y ((x x) y)))U)F)");
    cs.stage_macroblock_and_render(AudioSegment("Expand the first U to its definition,"));
    ls3.begin_latex_transition("((\\lambda y. (y ((U U) y)))F)");
    cs.stage_macroblock_and_render(AudioSegment("Beta reduce the first argument U,"));
    ls3.begin_latex_transition("(F ((U U) F))");
    cs.stage_macroblock_and_render(AudioSegment("Beta reduce the second argument F,"));
    ls3.begin_latex_transition("(F (\\Theta F))");
    cs.stage_macroblock_and_render(AudioSegment("and substitute theta for UU."));
    cs.stage_macroblock_and_render(AudioSegment("Do you see it?"));
    ls3.begin_latex_transition("(\\Theta F) \\twoheadrightarrow_\\beta (F (\\Theta F))");
    cs.stage_macroblock_and_render(AudioSegment("We reduced theta F to F(theta F)."));
    ls3.begin_latex_transition("(\\Theta F) =_\\beta (F(\\Theta F))");
    cs.stage_macroblock_and_render(AudioSegment("In other words, up to beta reduction, theta F = F(theta F)..."));
    cs.stage_macroblock_and_render(AudioSegment("So theta F is a fixed point of F!"));
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment("Just think about that for a second. Unlike real valued functions, not only does every function in the Lambda Calculus have a fixed point, but there is a trivial way to find them too."));
    cs.remove_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment(1));
    cs.stage_macroblock_and_render(AudioSegment("But, wait... if that's the case,"));
    cs.add_scene_fade_in(&rfs, "rfs", 0, 0, true);
    rfs.begin_transition(0, "? 6 * sin ? 1.1 * cos ? 2.5 sin * 1.6 ? cos * + + +");
    cs.stage_macroblock_and_render(AudioSegment("why am I making this video instead of solving the Riemann Hypothesis?"));
    cs.stage_macroblock_and_render(AudioSegment("It's kinda like trying to solve for the square root of negative 1."));
    rfs.begin_transition(0, "? 6 * sin ? 1.1 * cos ? 2.5 sin * 1.6 ? cos * + + + ? 4 * sin *");
    cs.stage_macroblock_and_render(AudioSegment("There's always a fixed point out there, but it's not among the real numbers."));
    cs.add_scene_fade_in(&ls3, "ls3", 0, 0.375, true);
    cs.state.superscene_transition(unordered_map<string, string>{
        {"rfs.opacity", ".2"},
    });
    cs.stage_macroblock_and_render(AudioSegment("In our case, Theta F isn't necessarily a real number."));
    ls3.begin_latex_transition("(\\Theta F)=" + latex_text("potentially nonsense"));
    cs.stage_macroblock_and_render(AudioSegment("It can be an arbitrary, non-numerical lambda expression."));
    shared_ptr<LambdaExpression> xsqp2 = parse_lambda_from_string("(\\x. (((\\m. (\\n. (\\f. (\\x. ((m f) ((n f) x)))))) (((\\m. (\\n. (\\s. (n (m s))))) x) x)) (\\f. (\\x. (f (f x))))))");
    xsqp2->set_color_recursive(0xffff3333);
    xsqp2->flush_uid_recursive();
    shared_ptr<LambdaExpression> U = parse_lambda_from_string("(\\x. (\\y. (y ((x x) y))))");
    shared_ptr<LambdaExpression> Theta = apply(U, U, OPAQUE_WHITE);
    Theta->flush_uid_recursive();
    Theta->set_color_recursive(0xff0088ff);
    shared_ptr<LambdaExpression> txsqp2 = apply(Theta, xsqp2, OPAQUE_WHITE);
    LambdaScene xsqp2_scene(xsqp2, VIDEO_WIDTH/2, VIDEO_HEIGHT/2);
    cs.fade_out_all_scenes();
    cs.add_scene_fade_in(&xsqp2_scene, "xsqp2", 0.25, 0.25, true);
    cs.stage_macroblock_and_render(AudioSegment("So we can go ahead and define x^2+2,"));
    xsqp2_scene.set_expression(txsqp2);
    cs.stage_macroblock_and_render(AudioSegment("slap theta in front of it,"));
    xsqp2_scene.reduce();
    cs.stage_macroblock_and_render(AudioSegment("and sure enough we find a fixed point."));
    xsqp2_scene.reduce();
    cs.stage_macroblock_and_render(AudioSegment("But this fixed point isn't even a number."));
    xsqp2_scene.reduce();
    cs.stage_macroblock_and_render(AudioSegment("It's a fixed point of a lambda calculus construct which emulates the function x^2+2."));
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment(.5));
    cs.stage_macroblock_and_render(AudioSegment("Ok, so maybe Turing's Fixed Point Combinator can't do the impossible,"));
    cs.remove_all_scenes();
    fac.begin_latex_transition(latex_text("fac = ") + "(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ ("+latex_text("fac")+"(-\\ n\\ 1)))))");
    fac.end_transition();
    cs.add_scene_fade_in(&fac, "fac", 0, 0.25, true);
    cs.stage_macroblock_and_render(AudioSegment("but it _is_ the magic bullet to help us make our factorial."));
    cs.stage_macroblock_and_render(AudioSegment("This is where the voodoo magic comes in."));
    cs.stage_macroblock_and_render(AudioSegment("Hang with me."));
    fac.begin_latex_transition(latex_text("fac = ") + "(\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ (f(-\\ n\\ 1)))))");
    cs.stage_macroblock_and_render(AudioSegment("I'm gonna remove the problematic recursive call, and replace it with some f,"));
    fac.begin_latex_transition(latex_text("fac = ") + "(\\lambda fn.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ (f(-\\ n\\ 1)))))");
    cs.stage_macroblock_and_render(AudioSegment("which we'll add as an argument for this function."));
    fac.begin_latex_transition("F = (\\lambda fn.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ (f(-\\ n\\ 1)))))");
    cs.stage_macroblock_and_render(AudioSegment("I'm gonna call this new function big F."));
    LatexScene fac_solution("(\\Theta F)", 0.6, VIDEO_WIDTH, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&fac_solution, "fac_solution", 0, 0.4, true);
    cs.stage_macroblock_and_render(AudioSegment("Now, let's consider what a fixed point of big F would be like."));
    fac_solution.begin_latex_transition("(\\Theta F) =_\\beta (F (\\Theta F))");
    cs.stage_macroblock(AudioSegment("We already know that this is true of any fixed point by definition, using the magic of Turing's Fixed Point Combinator."), 2);
    cs.render_microblock();
    cs.render_microblock();
    fac_solution.begin_latex_transition("(\\Theta F) =_\\beta ((\\lambda fn.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ (f(-\\ n\\ 1))))) (\\Theta F))");
    cs.stage_macroblock_and_render(AudioSegment("Plugging in the first F,"));
    fac_solution.begin_latex_transition("(\\Theta F) =_\\beta (\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ (\\Theta F(-\\ n\\ 1)))))");
    cs.stage_macroblock_and_render(AudioSegment("and beta reducing once,"));
    cs.stage_macroblock_and_render(AudioSegment("we get this expression...!"));
    cs.stage_macroblock_and_render(AudioSegment("Now, remember that there's no recursive nesting this time."));
    fac_solution.begin_latex_transition(latex_color(0xff0088ff, "(\\Theta F)") + " =_\\beta (\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ (" + latex_color(0xff0088ff, "\\Theta F") + "(-\\ n\\ 1)))))");
    cs.stage_macroblock_and_render(AudioSegment("This component is finite and closed-form."));
    cs.stage_macroblock_and_render(AudioSegment("but this time, through beta reduction,"));
    LatexScene oldfac(latex_color(0xffff7777,latex_text("fac")) + "= (\\lambda n.\\ (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ ("+latex_color(0xffff7777,latex_text("fac"))+"(-\\ n\\ 1)))))", 0.85, VIDEO_WIDTH, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&oldfac, "oldfac", 0.023, 0.55, true);
    cs.stage_macroblock_and_render(AudioSegment("we _derived_ the recursive equivalence relation that we wanted to begin with!"));
    cs.stage_macroblock_and_render(AudioSegment("This term, Theta F, satisfies the recursive equivalence characteristic of the factorial function."));
    cs.state.superscene_transition(unordered_map<string, string>{
        {"fac.y", "-.5"},
        {"fac_solution.y", "0.0"},
        {"oldfac.y", "0.1"},
    });
    shared_ptr<LambdaExpression> IS0m = parse_lambda_from_string("((\\n. (\\x. (\\y. ((n (\\z. y)) x)))) m)");
    shared_ptr<LambdaExpression> one = parse_lambda_from_string("(\\f. (\\x. (f x)))");
    shared_ptr<LambdaExpression> three = parse_lambda_from_string("(\\f. (\\x. (f (f (f x)))))");
    three->flush_uid_recursive();
    shared_ptr<LambdaExpression> pred = parse_lambda_from_string("(\\n. (\\f. (\\x. (((n (\\g. (\\h. (h (g f))))) (\\u. x)) (\\u. u)))))");
    shared_ptr<LambdaExpression> mult_function = parse_lambda_from_string("(\\m. (\\n. (\\s. (n (m s)))))");
    shared_ptr<LambdaExpression> f_pred_m = apply(parse_lambda_from_string("c"), apply(pred, parse_lambda_from_string("m"), OPAQUE_WHITE), OPAQUE_WHITE);
    shared_ptr<LambdaExpression> n_times_fpm = apply(apply(mult_function, parse_lambda_from_string("m"), OPAQUE_WHITE), f_pred_m, OPAQUE_WHITE);
    shared_ptr<LambdaExpression> F = abstract('c', abstract('m', apply(apply(IS0m, one, OPAQUE_WHITE), n_times_fpm, OPAQUE_WHITE), OPAQUE_WHITE), OPAQUE_WHITE);
    shared_ptr<LambdaExpression> theta = parse_lambda_from_string("((\\x. (\\y. (y ((x x) y)))) (\\x. (\\y. (y ((x x) y)))))");
    shared_ptr<LambdaExpression> TF = apply(theta, F, OPAQUE_WHITE);
    TF->flush_uid_recursive();
    LambdaScene fac_lambda(TF, VIDEO_WIDTH, VIDEO_HEIGHT*.6);
    cs.add_scene_fade_in(&fac_lambda, "fac_lambda", 0, 0.3, true);
    cs.stage_macroblock_and_render(AudioSegment("And that means that Theta F IS the factorial function."));
    cs.stage_macroblock_and_render(AudioSegment("Lemme color it in so it makes some more sense."));
    dynamic_pointer_cast<LambdaApplication>(TF)->get_first()->set_color_recursive(0xff0088ff);
    fac_lambda.set_expression(TF);
    cs.stage_macroblock_and_render(AudioSegment("This is the fixed point operator in blue."));
    dynamic_pointer_cast<LambdaApplication>(TF)->get_second()->set_color_recursive(0xffff9999);
    fac_lambda.set_expression(TF);
    cs.stage_macroblock_and_render(AudioSegment("Here's F."));
    dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaApplication>(TF)->get_second())->get_body())->get_body())->get_first())->get_first()->set_color_recursive(0xffaaaa00);
    fac_lambda.set_expression(TF);
    cs.stage_macroblock_and_render(AudioSegment("Inside of it, here's IS_ZERO,"));
    dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaApplication>(TF)->get_second())->get_body())->get_body())->get_first())->get_second()->set_color_recursive(0xffaa00aa);
    fac_lambda.set_expression(TF);
    cs.stage_macroblock_and_render(AudioSegment("the number 1 in the base case,"));
    dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaApplication>(TF)->get_second())->get_body())->get_body())->get_second())->get_first())->get_first()->set_color_recursive(0xff00ff77);
    fac_lambda.set_expression(TF);
    cs.stage_macroblock_and_render(AudioSegment("the multiplication function,"));
    dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaApplication>(TF)->get_second())->get_body())->get_body())->get_second())->get_second())->get_second())->get_first()->set_color_recursive(0xff6600ff);
    fac_lambda.set_expression(TF);
    cs.stage_macroblock_and_render(AudioSegment("the function which subtracts 1,"));
    dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaApplication>(TF)->get_second())->get_body())->get_body())->get_second())->get_second())->get_first()->set_color_recursive(0xff0066bb);
    fac_lambda.set_expression(TF);
    cs.stage_macroblock_and_render(AudioSegment("and the recursive function call."));
    shared_ptr<LambdaExpression> TF3 = apply(TF, three, OPAQUE_WHITE);
    fac_lambda.set_expression(TF3);
    cs.stage_macroblock_and_render(AudioSegment("Now, let's apply this factorial function to the number three."));
    cs.stage_macroblock_and_render(AudioSegment("Hold on tight...!"));
    int num_reductions = TF3->count_reductions();
    shared_ptr<LambdaExpression> TF3_copy = TF3->clone();
    cs.stage_macroblock(AudioSegment(20), num_reductions);
    for(int i = 0; i < num_reductions; i++) {
        TF3 = TF3->reduce();
        fac_lambda.reduce();
        cs.render_microblock();
    }
    cout << TF3->get_string() << endl;
    cs.stage_macroblock_and_render(AudioSegment("It's six!!"));
    cs.fade_out_all_scenes();
    /*
    LatexScene fac_three("3! = fac(3) = 6", 1, VIDEO_WIDTH, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&fac_three, "fac_three", 0, 0.2, true);
    cs.stage_macroblock_and_render(AudioSegment("Now, we know factorial(3) is 6."));
    LatexScene three_fac("3(fac) = ???", 1, VIDEO_WIDTH, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&three_fac, "three_fac", 0, 0.5, true);
    cs.stage_macroblock_and_render(AudioSegment("But what about 3(factorial)?"));
    three_fac.begin_latex_transition("(3(fac))(x) = fac(fac(fac(x)))");
    cs.stage_macroblock_and_render(AudioSegment("Given our knowledge that the numeral 3 iterates a function 3 times,"));
    cs.stage_macroblock(AudioSegment("you can probably guess what 3(factorial) is now."), 3);
    three_fac.begin_latex_transition("(3(fac))(x) = fac(fac(x!))");
    cs.render_microblock();
    three_fac.begin_latex_transition("(3(fac))(x) = fac(x!!)");
    cs.render_microblock();
    three_fac.begin_latex_transition("(3(fac))(x) = x!!!");
    cs.render_microblock();
    cs.stage_macroblock_and_render(AudioSegment("It's the triple factorial function!"));
    cs.fade_out_all_scenes();
    */
    cs.stage_macroblock_and_render(AudioSegment(2));
    return TF3_copy;
}

void reduction_graph(shared_ptr<LambdaExpression> TF3){
    CompositeScene cs;
    std::unordered_map<std::string, std::string> closequat{
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "{t} -20 / sin 10 /"},
        {"qk", "0"},
        {"d", "3"},
        {"x", "0"},
        {"y", "2"},
        {"z", "0"},
        {"surfaces_opacity", ".5"},
        {"lines_opacity", "1"},
        {"points_opacity", "0"},
    };
    Graph<HashableString> g;
    g.dimensions = 2;
    string iter1 = "(((\\m. (\\n. (\\f. (\\x. ((m f) ((n f) x)))))) (\\f. (\\x. (f x)))) (\\f. (\\x. (f x))))";
    LambdaGraphScene lgs(&g, iter1, VIDEO_WIDTH*.5, VIDEO_HEIGHT);
    unordered_map<string, string> use_parent = {
        {"q1", "[q1]"},
        {"qi", "[qi]"},
        {"qj", "[qj]"},
        {"qk", "[qk]"},
        {"d", "[d]"},
        {"x", "[x]"},
        {"y", "[y]"},
        {"z", "[z]"},
        {"surfaces_opacity", "[surfaces_opacity]"},
        {"lines_opacity", "[lines_opacity]"},
        {"points_opacity", "[points_opacity]"},
    };
    lgs.state.superscene_transition(use_parent);
    cs.add_scene_fade_in(&lgs, "lgs", .25, 0, true);
    cs.state.set(closequat);
    cs.stage_macroblock_and_render(AudioSegment("Now, I knowingly hid some complexity about beta reduction from you, but now I think you're ready."));
    cs.stage_macroblock_and_render(AudioSegment("Consider this term for 'one plus one'."));
    cs.state.superscene_transition(unordered_map<string, string>{
        {"d", "9"},
        {"y", "8"},
    });
    unordered_set<HashableString*> children;
    auto nodescopy = g.nodes;
    double lastid = 0;
    nodescopy = g.nodes;
    for(auto& p : nodescopy){
        Node<HashableString>& n = p.second;
        unordered_set<HashableString*> children = n.data->get_children();
        for(HashableString* child : children){
            lastid = g.add_node(child);
        }
    }
    cs.stage_macroblock_and_render(AudioSegment("As we've done a million times, we can beta reduce it to two,"));
    children = g.nodes.find(lastid)->second.data->get_children();
    cs.stage_macroblock(AudioSegment("but this time, let's keep track of the intermediate steps."), 5);
    for(HashableString* child : children){
        lastid = g.add_node(child);
        break;
    }
    cs.render_microblock();
    children = g.nodes.find(lastid)->second.data->get_children();
    for(HashableString* child : children){
        lastid = g.add_node(child);
        break;
    }
    cs.render_microblock();
    children = g.nodes.find(lastid)->second.data->get_children();
    for(HashableString* child : children){
        lastid = g.add_node(child);
        break;
    }
    cs.render_microblock();
    children = g.nodes.find(lastid)->second.data->get_children();
    for(HashableString* child : children){
        lastid = g.add_node(child);
        break;
    }
    cs.render_microblock();
    children = g.nodes.find(lastid)->second.data->get_children();
    for(HashableString* child : children){
        lastid = g.add_node(child);
        break;
    }
    cs.render_microblock();
    for(auto& p : g.nodes){
        Node<HashableString>& n = p.second;
        if(n.data->representation.size() == 21)
            n.color = 0xff0088ff;
    }
    cs.stage_macroblock_and_render(AudioSegment(0.5));
    cs.stage_macroblock_and_render(AudioSegment("Ok, we've arrived at 2."));
    cs.stage_macroblock_and_render(AudioSegment("I've colored it blue, because it's special- it can't be reduced further."));
    cs.stage_macroblock(AudioSegment("The fancy word for this is 'Beta Normal Form'."), 3);
    LatexScene bnf(latex_text("Beta Normal Form")+"\\rightarrow", 0.9, VIDEO_WIDTH/2, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&bnf, "bnf", 0, 0.69, true);
    cs.render_microblock();
    cs.render_microblock();
    cs.state.subscene_transition(unordered_map<string, string>{
        {"bnf.opacity", "0"},
    });
    cs.render_microblock();
    cs.stage_macroblock(AudioSegment("But I wanna take a closer look at the second node."), 2);
    cs.render_microblock();

    string iter2 = "((\\n. (\\f. (\\x. (((\\f. (\\x. (f x))) f) ((n f) x))))) (\\f. (\\x. (f x))))";
    glm::dvec4 iter2pos;
    for(auto& p : g.nodes){
        Node<HashableString>& n = p.second;
        lastid = p.first;
        if(n.data->representation == iter2){
            n.color = 0xffff3333;
            iter2pos = n.position;
        }
    }
    g.clear();
    g.add_node(new HashableString(iter1));
    g.add_node(new HashableString(iter2));
    for(auto& p : g.nodes){
        Node<HashableString>& n = p.second;
        lastid = p.first;
        if(n.data->representation == iter2){
            n.color = 0xffff3333;
            n.position = iter2pos;
        }
    }
    g.iterate_physics(100);
    cs.render_microblock();

    cs.state.superscene_transition(unordered_map<string, string>{
        {"lgs.x", "0"},
    });
    shared_ptr<LambdaExpression> lam_iter2 = parse_lambda_from_string(iter2);
    lam_iter2->flush_uid_recursive();
    LambdaScene lam(lam_iter2, VIDEO_WIDTH/2, VIDEO_HEIGHT);
    cs.add_scene_fade_in(&lam, "lam", .5, 0, true);
    cs.stage_macroblock_and_render(AudioSegment("If I beta-reduce it,"));
    lam.reduce();
    cs.stage_macroblock_and_render(AudioSegment("it looks like this."));
    lam.set_expression(lam_iter2);
    cs.stage_macroblock(AudioSegment("But... stepping back,"), 2);
    cs.render_microblock();
    cs.render_microblock();
    cs.stage_macroblock_and_render(AudioSegment("Lemme highlight the function-value pair which just got reduced."));
    dynamic_pointer_cast<LambdaApplication>(lam_iter2)->get_first()->set_color_recursive(0xffff7700);
    lam.set_expression(lam_iter2);
    cs.stage_macroblock_and_render(AudioSegment("Here's the function,"));
    dynamic_pointer_cast<LambdaApplication>(lam_iter2)->get_second()->set_color_recursive(0xff77ff00);
    lam.set_expression(lam_iter2);
    cs.stage_macroblock_and_render(AudioSegment("and here's the value."));
    lam.reduce();
    cs.stage_macroblock_and_render(AudioSegment("And here it is getting reduced again."));
    lam_iter2->set_color_recursive(OPAQUE_WHITE);
    lam.set_expression(lam_iter2);
    cs.stage_macroblock_and_render(AudioSegment("But..."));
    cs.stage_macroblock_and_render(AudioSegment("there's actually a different reduction available here!"));
    dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaApplication>(lam_iter2)->get_first())->get_body())->get_body())->get_body())->get_first())->get_first()->set_color_recursive(0xff00ff00);
    dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaApplication>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaApplication>(lam_iter2)->get_first())->get_body())->get_body())->get_body())->get_first())->get_second()->set_color_recursive(0xff6600ff);
    lam.set_expression(lam_iter2);
    cs.stage_macroblock_and_render(AudioSegment("Here it is."));
    lam_iter2->specific_reduction(1);
    lam.set_expression(lam_iter2);
    cs.stage_macroblock_and_render(AudioSegment("And here it is actually taking place."));

    cs.state.superscene_transition(unordered_map<string, string>{
        {"lgs.x", ".25"},
        {"lam.opacity", "0"},
    });
    cs.stage_macroblock_and_render(AudioSegment("Going back to our tree of reductions,"));

    nodescopy = g.nodes;
    for(auto& p : nodescopy){
        Node<HashableString>& n = p.second;
        unordered_set<HashableString*> children = n.data->get_children();
        for(HashableString* child : children){
            g.add_node(child);
        }
    }
    cs.stage_macroblock_and_render(AudioSegment("since there's two options at this node,"));
    cs.stage_macroblock_and_render(AudioSegment("let's follow both paths at the same time."));
    while(g.size() != 13){
        nodescopy = g.nodes;
        for(auto& p : nodescopy){
            Node<HashableString>& n = p.second;
            unordered_set<HashableString*> children = n.data->get_children();
            for(HashableString* child : children){
                g.add_node(child);
            }
        }
        for(auto& p : g.nodes){
            Node<HashableString>& n = p.second;
            if(n.data->representation.size() == 21)
                n.color = 0xff0088ff;
            else
                n.color = 0xffffffff;
        }
        cs.stage_macroblock_and_render(AudioSegment(1));
    }
    cs.stage_macroblock_and_render(AudioSegment("Looks like all paths lead to 2!"));
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment("Of course, that's what we would expect."));
    cs.remove_all_scenes();
    Graph<HashableString> h;
    string onetimesone = "(((\\m. (\\n. (\\s. (n (m s))))) (\\f. (\\x. (f (f x))))) (\\f. (\\x. (f (f x)))))";
    LambdaGraphScene lg2(&h, onetimesone, VIDEO_WIDTH, VIDEO_HEIGHT);
    cs.add_scene_fade_in(&lg2, "lg2", 0, 0, true);
    h.dimensions = 3;
    lg2.state.superscene_transition(use_parent);
    cs.state.set(unordered_map<string, string>{
        {"d", "7"},
        {"y", "3"},
        {"qj", "{t} 6 / sin"},
        {"q1", "{t} 6 / cos"},
    });
    cs.stage_macroblock_and_render(AudioSegment("The answer to the problem shouldn't depend on the order that you do the steps. Right?"));
    cs.state.superscene_transition(unordered_map<string, string>{
        {"d", "20"},
        {"y", "9"},
    });
    cs.stage_macroblock_and_render(AudioSegment("Let's try two times two."));
    int gs = -1;
    while(gs != h.size()){
        gs = h.size();
        nodescopy = h.nodes;
        for(auto& p : nodescopy){
            Node<HashableString>& n = p.second;
            unordered_set<HashableString*> children = n.data->get_children();
            for(HashableString* child : children){
                h.add_node(child);
            }
        }
        for(auto& p : h.nodes){
            Node<HashableString>& n = p.second;
            if(!parse_lambda_from_string(n.data->representation)->is_reducible())
                n.color = 0xff0088ff;
            else
                n.color = 0xffffffff;
        }
        cs.stage_macroblock_and_render(AudioSegment(1));
    }
    cs.stage_macroblock_and_render(AudioSegment("Here's the reduction graph!"));
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment("Looks like in this case, we can only get to four also."));
    cs.stage_macroblock_and_render(AudioSegment(0.5));
    cs.remove_all_scenes();
    g.clear();
    cs.add_scene(&lgs, "lgs", 0, 0);
    cs.state.superscene_transition(unordered_map<string, string>{
        {"y", "14"},
    });
    string o3_str = "((\\x. ((x x) x)) (\\x. ((x x) x)))";
    LambdaScene omega3(parse_lambda_from_string(o3_str), VIDEO_WIDTH*0.5, VIDEO_HEIGHT);
    cs.add_scene_fade_in(&omega3, "omega3", 0.5, 0, true);
    g.add_node(new HashableString(o3_str));
    cs.stage_macroblock_and_render(AudioSegment("Alright, I'll stop leading you on. This is a lambda term called Omega 3."));
    gs = -1;
    while(gs != g.size() && gs < 10){
        gs = g.size();
        omega3.reduce();
        nodescopy = g.nodes;
        for(auto& p : nodescopy){
            Node<HashableString>& n = p.second;
            unordered_set<HashableString*> children = n.data->get_children();
            for(HashableString* child : children){
                g.add_node(child);
            }
        }
        for(auto& p : g.nodes){
            Node<HashableString>& n = p.second;
            if(!parse_lambda_from_string(n.data->representation)->is_reducible())
                n.color = 0xff0088ff;
            else
                n.color = 0xffffffff;
        }
        cs.stage_macroblock_and_render(AudioSegment(1));
    }
    cs.stage_macroblock_and_render(AudioSegment("See where this is going?"));
    cs.stage_macroblock_and_render(AudioSegment("When we beta reduce this term, it actually just gets bigger."));
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment("There's no branching, but this is never gonna get to Beta Normal Form."));
    cs.remove_all_scenes();
    LambdaScene omega(parse_lambda_from_string("((\\x. (x x)) (\\x. (x x)))"), VIDEO_WIDTH/2, VIDEO_HEIGHT/2);
    cs.add_scene_fade_in(&omega, "omega", .25, .25, true);
    cs.stage_macroblock_and_render(AudioSegment("Here's an even weirder term."));
    cs.stage_macroblock(AudioSegment("It is reducible, but it doesn't reduce to a _different_ lambda term..."), 2);
    for(int i = 0; i < 2; i++){
        omega.reduce();
        cs.render_microblock();
    }
    omega.reduce();
    PngScene self_arrow("self_arrow", VIDEO_WIDTH/3, VIDEO_HEIGHT/3);
    cs.add_scene_fade_in(&self_arrow, "self_arrow", .6, 0.0666, true);
    cs.stage_macroblock(AudioSegment("There's not much of a graph to draw, because it reduces to itself!"), 2);
    for(int i = 0; i < 2; i++){
        omega.reduce();
        cs.render_microblock();
    }
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment(1));
    cs.remove_all_scenes();
    g.clear();
    g.dimensions = 3;
    LambdaGraphScene lg3(&g, TF3->get_string(), VIDEO_WIDTH, VIDEO_HEIGHT);
    lg3.state.set(use_parent);
    cs.add_scene_fade_in(&lg3, "lg3", 0, 0, true);
    cs.state.superscene_transition(unordered_map<string, string>{
        {"d", "80"},
        {"y", "16"},
        {"points_opacity", "1"},
        {"surfaces_opacity", "0"},
    });
    cs.stage_macroblock_and_render(AudioSegment("Here's our old friend, factorial of 3."));
    auto TF3_clone = TF3->clone();
    int reductions = TF3->count_reductions();

    auto TF3_infinite_depth = TF3->clone();
    int num_bfs = 1000;
    cs.state.superscene_transition(unordered_map<string, string>{
        {"d", "200"},
        {"y", "36"},
    });
    cs.stage_macroblock(AudioSegment("The search tree on this is suuuper deep, so I'll just show you the tip of the iceberg."), num_bfs);
    if(FOR_REAL) for(int i = 0; i < num_bfs; i++){
        g.expand_graph_once();
        cs.render_microblock();
    }

    cs.stage_macroblock_and_render(AudioSegment("That's the first thousand nodes."));
    cs.state.superscene_transition(unordered_map<string, string>{
        {"d", "1200"},
        {"y", "160"},
    });

    cs.stage_macroblock(AudioSegment("Here's the path straight to the answer of 6."), reductions);
    if(FOR_REAL) for(int i = 0; i < reductions; i++){
        TF3_clone = TF3_clone->reduce();
        HashableString* hs = new HashableString(TF3_clone->get_string());
        g.add_node(hs);
        for (auto& p : g.nodes){
            auto& n = p.second;
            n.opacity = 0;
            n.color = 0xffffffff;
            if(n.data->representation.size() == 37){
                n.color = 0xff0088ff;
                n.opacity = 1;
            }
        }
        cs.render_microblock();
    }
    cs.state.superscene_transition(unordered_map<string, string>{
        {"d", "1800"},
        {"y", "200"},
    });
    reductions = 200;
    cs.stage_macroblock_and_render(AudioSegment("But, lemme show you something else..."));
    TF3_clone = TF3->clone();
    cs.stage_macroblock(AudioSegment("This is another path that seemingly goes in a totally different direction."), reductions);
    if(FOR_REAL) for(int i = 0; i < reductions; i++){
        TF3_clone = TF3_clone->specific_reduction(TF3_clone->count_parallel_reductions()-1);
        HashableString* hs = new HashableString(TF3_clone->get_string());
        g.add_node(hs);
        for (auto& p : g.nodes){
            auto& n = p.second;
            n.opacity = 0;
            n.color = 0xffffffff;
            if(n.data->representation.size() == 37){
                n.color = 0xff0088ff;
                n.opacity = 1;
            }
        }
        cs.render_microblock();
    }
    cs.stage_macroblock_and_render(AudioSegment("In fact, this alternate reduction path goes on indefinitely, sort of like Omega."));
    LatexScene unroll("(\\lambda n. (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ ((\\Theta F)(-\\ n\\ 1)))))", 0.5, VIDEO_WIDTH, VIDEO_HEIGHT*.25);
    cs.add_scene_fade_in(&unroll, "unroll", 0, .375, true);
    cs.state.superscene_transition(unordered_map<string, string>{
        {"lg3.opacity", "0.3"},
    });
    cs.stage_macroblock_and_render(AudioSegment("Up to the order that you perform your reductions, you may or may not ever reach the answer."));

    unroll.begin_latex_transition("(\\lambda n. (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ (\\tiny(\\lambda n. (("+latex_text("IS\\_0")+"\\ n)\\ 1\\ (\\times\\ n\\ ((\\Theta F)(-\\ n\\ 1)))))\\normalsize(-\\ n\\ 1)))))");
    cs.stage_macroblock_and_render(AudioSegment("It turns out, in this path, it's 'unrolling' the recursive definition of factorial, so-to-speak,"));
    cs.stage_macroblock_and_render(AudioSegment("before the base case is ever taken advantage of."));
    cs.stage_macroblock_and_render(AudioSegment(0.5));
    cs.state.superscene_transition(unordered_map<string, string>{
        {"unroll.opacity", "0"},
    });
    LatexScene church_rosser(latex_text("Church-Rosser Theorem"), 1, VIDEO_WIDTH, VIDEO_HEIGHT*.25);
    cs.add_scene_fade_in(&church_rosser, "church_rosser", 0, 0, true);
    cs.stage_macroblock_and_render(AudioSegment("The proof is unfortunately too long for this video,"));
    LatexScene cr_statement(latex_text("No two different expressions in beta-normal form can be reached by reducing the same expression."), 1, VIDEO_WIDTH, VIDEO_HEIGHT*.25);
    cs.add_scene_fade_in(&cr_statement, "cr_statement", 0, 0.375, true);
    cs.stage_macroblock_and_render(AudioSegment("but the Church-Rosser theorem shows that if there's _one_ of these blue irreducible answers in our tree, there isn't another."));
    cs.stage_macroblock_and_render(AudioSegment("Sometimes there isn't one at all, and sometimes you can go down a beta-reduction path in the wrong direction,"));
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment("but there's never more than one node in this special 'beta normal form'."));
}

void extra(){
    CompositeScene cs;
    LatexScene lc(latex_text("The \\lambda-Calculus"), 0.4, VIDEO_WIDTH, VIDEO_HEIGHT*.25);
    cs.add_scene_fade_in(&lc, "lc", 0, 0.375, true);
    cs.stage_macroblock_and_render(AudioSegment("Lambda calculus is a deep subject, and we've only scratched the surface."));
    cs.stage_macroblock_and_render(AudioSegment("Just to give you a taste of what else is out there:"));
    cs.stage_macroblock(AudioSegment("The Church-Turing Thesis showed that the Lambda Calculus can do everything that any traditional computer can."), 2);
    LatexScene ctt(latex_text("Church-Turing Thesis"), 1, VIDEO_WIDTH, VIDEO_HEIGHT*.25);
    cs.add_scene_fade_in(&ctt, "ctt", 0, 0.125, true);
    lc.begin_latex_transition(latex_text("The \\lambda-Calculus is as powerful as a Turing Machine"));
    cs.render_microblock();
    cs.render_microblock();
    ctt.begin_latex_transition(latex_text("Typed \\lambda-Calculus"));
    lc.begin_latex_transition(latex_text("Variables in the \\lambda-Calculus can take datatypes"));
    cs.stage_macroblock(AudioSegment("There's also the entire field of typed lambda calculus, where we assign datatypes to variables."), 2);
    cs.render_microblock();
    cs.render_microblock();
    cs.stage_macroblock_and_render(AudioSegment("It even turns out that any given typed program in the Lambda Calculus corresponds to some proof."));
    cs.stage_macroblock_and_render(AudioSegment("_Merely writing a program in the typed Lambda Calculus proves something._"));
    PngScene coq("coq", VIDEO_WIDTH/2, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&coq, "coq", 0.25, 0.57, true);
    cs.stage_macroblock_and_render(AudioSegment("People have even made programming languages which convert a mathematical proof _into a runnable program_."));
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment("Which brings us to the whole topic of how this is actually used in programming."));
    cs.remove_all_scenes();
    cs.stage_macroblock(AudioSegment("Lisp, Haskell, and other programming languages are intrinsically based in the lambda calculus, unlike, say, C or Rust."), 9);
    LatexScene functional(latex_text("Functional Languages"), 0.8, VIDEO_WIDTH/2, VIDEO_HEIGHT*.25);
    cs.add_scene_fade_in(&functional, "functional", 0, 0, true);
    LatexScene imperative(latex_text("Imperative Languages"), 0.8, VIDEO_WIDTH/2, VIDEO_HEIGHT*.25);
    cs.add_scene_fade_in(&imperative, "imperative", 0.5, 0, true);
    cs.render_microblock();
    PngScene lisp("lisp", VIDEO_WIDTH/4, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&lisp, "lisp", 2/16., 4/16., true);
    cs.render_microblock();
    PngScene haskell("haskell", VIDEO_WIDTH/4, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&haskell, "haskell", 2/16., 9/16., true);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    PngScene c_lang("c_lang", VIDEO_WIDTH/4, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&c_lang, "c_lang", 10/16., 4/16., true);
    cs.render_microblock();
    PngScene rust("rust", VIDEO_WIDTH/4, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&rust, "rust", 10/16., 9/16., true);
    cs.render_microblock();
    cs.fade_out_all_scenes();
    PngScene python("python", VIDEO_WIDTH/4, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&python, "python", .375, .375, true);
    cs.stage_macroblock_and_render(AudioSegment("Even Python has adopted the ability to make small anonymous functions,"));
    PngScene py_lambda("py_lambda", VIDEO_WIDTH/2, VIDEO_HEIGHT/4);
    cs.add_scene_fade_in(&py_lambda, "py_lambda", .25, .625, true);
    cs.stage_macroblock_and_render(AudioSegment("called Lambdas, which emulate the properties of the Lambda Calculus."));
    cs.stage_macroblock_and_render(AudioSegment("But I don't like the Lambda Calculus for its utility."));
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment("The philosophical implications of these alternative models of computing run deep. They give us a fresh perspective on the structure of information and the nature of computation itself."));
    cs.stage_macroblock_and_render(AudioSegment(1));
}

void credits(){
    CompositeScene cs;
    LatexScene links(latex_text("Links in the Description!"), .8, VIDEO_WIDTH, VIDEO_HEIGHT/5);
    cs.add_scene_fade_in(&links, "links", 0, .6, true);
    PngScene seef("seef", VIDEO_WIDTH/3, VIDEO_HEIGHT/2);
    PngScene tromp("tromp_y", VIDEO_WIDTH/3, VIDEO_HEIGHT/2);
    PngScene selinger("selinger", VIDEO_WIDTH/3, VIDEO_HEIGHT/2);
    LatexScene seef_text(latex_text("6884"), .8, VIDEO_WIDTH/3, VIDEO_HEIGHT/6);
    LatexScene tromp_text(latex_text("John Tromp"), .8, VIDEO_WIDTH/3, VIDEO_HEIGHT/6);
    LatexScene selinger_text(latex_text("Peter Selinger"), .8, VIDEO_WIDTH/3, VIDEO_HEIGHT/6);
    cs.add_scene_fade_in(&seef, "seef", 0, .1, true);
    cs.add_scene_fade_in(&seef_text, "seef_text", 0, .533, true);
    cs.stage_macroblock(AudioSegment("I wanna give an enormous thank-you to 6884, who made all the music which made this video come to life."), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.add_scene_fade_in(&tromp, "tromp", .333, .1, true);
    cs.add_scene_fade_in(&tromp_text, "tromp_text", .333, .533, true);
    cs.stage_macroblock(AudioSegment("This also couldn't have been possible if not for John Tromp, the inventor of this sick diagrammatic notation for lambda expressions."), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();
    cs.add_scene_fade_in(&selinger, "selinger", .666, .1, true);
    cs.add_scene_fade_in(&selinger_text, "selinger_text", .666, .533, true);
    cs.stage_macroblock(AudioSegment("Or Peter Selinger, who wrote this book, Lecture Notes on the Lambda Calculus."), 3);
    cs.render_microblock();
    cs.render_microblock();
    cs.render_microblock();

    shared_ptr<LambdaExpression> add = parse_lambda_from_string("(\\m. (\\n. (\\f. (\\x. ((m f) ((n f) x))))))");
    shared_ptr<LambdaExpression> mult = parse_lambda_from_string("(\\m. (\\n. (\\s. (n (m s)))))");
    shared_ptr<LambdaExpression> ptp_skeleton = parse_lambda_from_string("(\\t. (\\p. ((t p) p)))");
    shared_ptr<LambdaExpression> ptp = apply(apply(ptp_skeleton, mult, OPAQUE_WHITE), add, OPAQUE_WHITE);
    LambdaScene ptp_scene(ptp, VIDEO_WIDTH*.6, VIDEO_HEIGHT*.6);
    cs.fade_out_all_scenes();
    cs.add_scene_fade_in(&ptp_scene, "ptp_scene", .2, .2, true);
    cs.stage_macroblock(AudioSegment("His clear formalizations of a lot of the things I _thought_ I understood helped me get through plenty of snags in rendering these diagrams."), 4);
    cs.render_microblock();
    ptp_scene.reduce();
    cs.render_microblock();
    ptp_scene.reduce();
    cs.render_microblock();
    ptp_scene.reduce();
    cs.render_microblock();
    ptp_scene.reduce();
    cs.stage_macroblock(AudioSegment("This video, to the best of my knowledge, is the first which shows beta-reduction animations of _any_ visualization method for the Lambda Calculus."), 4);
    cs.render_microblock();
    ptp_scene.reduce();
    cs.render_microblock();
    ptp_scene.reduce();
    cs.render_microblock();
    ptp_scene.reduce();
    cs.render_microblock();
    ptp_scene.reduce();
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment("And geez, did it take a while to make."));
    cs.remove_all_scenes();

    PngScene discord("discord", VIDEO_WIDTH/3, VIDEO_HEIGHT/3);
    cs.add_scene_fade_in(&discord, "discord", .333, .333, true);
    cs.stage_macroblock(AudioSegment("If you like this content, then there's nothing I would appreciate more than you joining our discord server!"), 2);
    cs.render_microblock();
    cs.add_scene_fade_in(&links, "links", 0, .666, true);
    cs.render_microblock();
    cs.stage_macroblock_and_render(AudioSegment("We talk about math, puzzles, game theory, and so on!"));
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment("I can't wait to see you there."));
    cs.remove_all_scenes();

    TwoswapScene twoswap;
    cs.add_scene_fade_in(&twoswap, "twoswap", 0, 0, true);
    cs.stage_macroblock_and_render(AudioSegment("But until then, this has been 2swap."));
    cs.fade_out_all_scenes();
    cs.stage_macroblock_and_render(AudioSegment(1.5));
    cs.stage_macroblock_and_render(AudioSegment(.5));
}

void chapter_number(int number, string subtitle){
    cout << "Rendering chapter "<< number << endl;
    CompositeScene cs;
    LatexScene ls(latex_text(subtitle), 1, 1800, 500);
    LatexScene ls2(latex_text("\\textbf{Chapter " + to_string(number) + "}"), 1, 1800, 500);
    int w = OPAQUE_WHITE;
    shared_ptr<LambdaExpression> lex = parse_lambda_from_string("x");
    for(int i = 0; i < number; i++){
        lex = apply(parse_lambda_from_string("f"), lex, w);
    }
    lex = abstract('f', abstract('x', lex, w), w);
    lex->set_color_recursive(0xff005599);
    LambdaScene lam(lex, 800, 800);
    ThreeDimensionScene tds1;
    ThreeDimensionScene tds2;
    tds1.add_surface(Surface(glm::dvec3(0, .25,0),glm::dvec3(1,0,0),glm::dvec3(0, 1        , 0),make_shared<LambdaScene>(lam)));
    tds2.add_surface(Surface(glm::dvec3(0, .1,0),glm::dvec3(1,0,0),glm::dvec3(0, 500/1800., 0),make_shared<LatexScene>(ls)));
    tds2.add_surface(Surface(glm::dvec3(0,-.25,0),glm::dvec3(1,0,0),glm::dvec3(0, 500/1800., 0),make_shared<LatexScene>(ls2)));

    tds1.state.set(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "1 <subscene_transition_fraction> 4 ^ -"},
        {"lines_opacity", "1"},
        {"points_opacity", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "2"},
        {"ntf", "<subscene_transition_fraction> 2 / .5 -"},
        {"x", "3 <ntf> * 3 ^ -1 *"},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "1 sin 9 / .0 +"},
    });
    tds2.state.set(std::unordered_map<std::string, std::string>{
        {"ntf", "<subscene_transition_fraction> .5 -"},
        {"surfaces_opacity", "1 <ntf> 2 * 4 ^ -"},
        {"lines_opacity", "1"},
        {"points_opacity", "0"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "2"},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
    });
    cs.add_scene(&tds1, "tds1", 0, 0);
    cs.add_scene(&tds2, "tds2", 0, 0);
    cs.stage_macroblock_and_render(AudioSegment(3));
}

void render_thumbnail(){
}

int main() {
    Timer timer;
    FOR_REAL = true;
    PRINT_TO_TERMINAL = true;
    shared_ptr<LambdaExpression> TF3 = parse_lambda_from_string("(\\x. x)");
    intro();
    chapter_number(1, "The Way of the Lambda");
    history();
    chapter_number(2, "Tromp's Diagrams");
    visualize();
    chapter_number(3, "Beta Reduction");
    beta_reduction();
    chapter_number(4, "Currying");
    currying();
    chapter_number(5, "Boolean Arithmetic");
    booleans();
    chapter_number(6, "Church Numerals");
    numerals();
    chapter_number(7, "Recursion");
    TF3 = factorial();
    chapter_number(8, "Reduction Graphs");
    reduction_graph(TF3);
    chapter_number(9, "Beyond the Basics");
    extra();
    credits();
    return 0;
}

