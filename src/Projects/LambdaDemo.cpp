using namespace std;
#include <string>
const string project_name = "LambdaDemo";
#include "../io/PathManager.cpp"
const int width_base = 640;
const int height_base = 360;
const float mult = 1;

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
    tds.add_surface(Surface(glm::vec3(0,0,0), glm::vec3(1,0,0), glm::vec3(0,1,0), ls));
    tds.state_manager.set(unordered_map<string, string>{
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
    cs.state_manager.add_equation("alg.opacity", "0");
    cs.state_manager.add_equation("boo.opacity", "0");
    cs.state_manager.add_equation("qj", "0");
    for(int i = 0; i < 5; i++){
        float alg_o = i==1;
        float boo_o = i==2;
        cs.state_manager.add_subscene_transition("alg.opacity", to_string(alg_o));
        cs.state_manager.add_subscene_transition("boo.opacity", to_string(boo_o));
        float qj = 0;
        if(i==1) qj=.19;
        if(i==2) qj=-.19;
        cs.state_manager.add_subscene_transition("qj", to_string(qj));
        cs.inject_audio(AudioSegment(blurbs[i]), num_reductions / 5);
        for(int j = 0; j < num_reductions/5; j++) {
            ls->reduce();
            cs.render();
        }
    }

    tds.inject_audio_and_render(AudioSegment("What are all these weird lines though?"));

    ls->set_expression(term);
    tds.inject_audio_and_render(AudioSegment("Going back to the original setup,"));

    dynamic_pointer_cast<LambdaApplication>(term)->get_first()->set_color_recursive(0xffff00ff);
    ls->set_expression(term);
    tds.inject_audio_and_render(AudioSegment("This pink chunk represents the factorial function."));
    dynamic_pointer_cast<LambdaApplication>(term)->get_second()->set_color_recursive(0xffffff00);
    ls->set_expression(term);
    tds.inject_audio_and_render(AudioSegment("This yellow chunk represents the number 3."));
    term->set_color(0xff00ffff);
    ls->set_expression(term);
    tds.inject_audio_and_render(AudioSegment("And this blue chunk represents the application of the function to the number."));

    tds.inject_audio_and_render(AudioSegment("We can make all sorts of other values."));

    tds.state_manager.subscene_transition(unordered_map<string, string>{
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"}
    });

    tds.state_manager.subscene_transition(unordered_map<string, string>{
        {"z", "-10"},
    });

    shared_ptr<LambdaExpression> church1 = parse_lambda_from_string("(\\f. (\\x. (f x)))");
    church1->set_color_recursive(0xffff0000);
    church1->flush_uid_recursive();
    shared_ptr<LambdaScene> church1scene = make_shared<LambdaScene>(church1, 800, 800);
    tds.add_surface(Surface(glm::vec3(-2,-2,-1), glm::vec3(1,0,0), glm::vec3(0,1,0), church1scene));

    shared_ptr<LambdaExpression> church2 = parse_lambda_from_string("(\\f. (\\x. (f (f x))))");
    church2->set_color_recursive(0xff00ff00);
    church2->flush_uid_recursive();
    shared_ptr<LambdaScene> church2scene = make_shared<LambdaScene>(church2, 800, 800);
    tds.add_surface(Surface(glm::vec3(2,2,-3), glm::vec3(1,0,0), glm::vec3(0,1,0), church2scene));

    shared_ptr<LambdaExpression> church3 = parse_lambda_from_string("(\\f. (\\x. (f (f (f x)))))");
    church3->set_color_recursive(0xff7777ff);
    church3->flush_uid_recursive();
    shared_ptr<LambdaScene> church3scene = make_shared<LambdaScene>(church3, 800, 800);
    tds.add_surface(Surface(glm::vec3(2,-2,-5), glm::vec3(1,0,0), glm::vec3(0,1,0), church3scene));
    tds.inject_audio_and_render(AudioSegment("We've got one, two, three..."));
    
    tds.state_manager.subscene_transition(unordered_map<string, string>{
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
    tds.add_surface(Surface(glm::vec3(-5,1,-9), glm::vec3(0,0,1), glm::vec3(0,1,0), churchplusscene));

    shared_ptr<LambdaExpression> churchtimes = parse_lambda_from_string("(\\m. (\\n. (\\a. (m (n a)))))");
    churchtimes->set_color_recursive(0xff0088ff);
    churchtimes->flush_uid_recursive();
    shared_ptr<LambdaScene> churchtimesscene = make_shared<LambdaScene>(churchtimes, 800, 800);
    tds.add_surface(Surface(glm::vec3(-5,-1,-11), glm::vec3(0,0,1), glm::vec3(0,1,0), churchtimesscene));
    tds.inject_audio_and_render(AudioSegment("and plus and times too."));
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
    tds.add_surface(Surface(glm::vec3(0,0,-8), glm::vec3(1,0,0), glm::vec3(0,1,0), ls1));

    tds.inject_audio(AudioSegment("We can express any computational procedure, such as the factorial function."), 2);
    tds.render();
    tds.state_manager.subscene_transition(unordered_map<string, string>{
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"d", "2"},
    });
    tds.render();
    tds.remove_surface(churchtimesscene);
    tds.remove_surface(churchplusscene);
    tds.inject_audio(AudioSegment("But... the magic is that it's not immediately obvious whether a certain expression is a number,"), 4);
    dynamic_pointer_cast<LambdaApplication>(term1)->get_first()->set_color_recursive(0xff222222);
    dynamic_pointer_cast<LambdaApplication>(term1)->get_second()->set_color_recursive(0xff222222);
    ls1->set_expression(term1);
    tds.render();
    tds.render();
    dynamic_pointer_cast<LambdaApplication>(term1)->get_first()->set_color_recursive(0xff222222);
    dynamic_pointer_cast<LambdaApplication>(term1)->get_second()->set_color_recursive(0xffffffff);
    ls1->set_expression(term1);
    tds.render();
    tds.render();
    dynamic_pointer_cast<LambdaApplication>(term1)->get_first()->set_color_recursive(0xffffffff);
    dynamic_pointer_cast<LambdaApplication>(term1)->get_second()->set_color_recursive(0xff222222);
    ls1->set_expression(term1);
    tds.inject_audio(AudioSegment("a function that operates on numbers,"), 2);
    tds.render();
    tds.render();
    dynamic_pointer_cast<LambdaApplication>(term1)->get_first()->set_color_recursive(0xffff00ff);
    dynamic_pointer_cast<LambdaApplication>(term1)->get_second()->set_color_recursive(0xffffff00);
    term1->set_color(0xff00ffff);
    ls1->set_expression(term1);
    tds.inject_audio(AudioSegment("or what."), 2);
    tds.render();
    tds.render();

    tds.inject_audio_and_render(AudioSegment("And that's because, in this language, there _is no difference_."));
    shared_ptr<LambdaExpression> term2 = apply(le_factorial, le_church_3, 0xff00ffff);
    ls1->set_expression(term2);
    tds.inject_audio_and_render(AudioSegment("Just like we applied factorial to 3 with function application,"));
    shared_ptr<LambdaExpression> term3 = apply(le_church_3, le_factorial, 0xff00ffff);
    ls1->set_expression(term3);
    tds.inject_audio_and_render(AudioSegment("we can apply 3 to factorial in the exact same way, as though 3 was a function and factorial was a value."));
    term3->set_color_recursive(0xff404040);
    ls1->set_expression(term3);

    tds.inject_audio_and_render(AudioSegment("OK, but you can't actually evaluate that, right?"));
    num_reductions = 3;
    tds.inject_audio(AudioSegment("Well, you can... but it's certainly not the case that when evaluating it, the answer would make any sense... right?"), num_reductions);
    for(int j = 0; j < num_reductions; j++) {
        ls1->reduce();
        tds.render();
    }

    tds.inject_audio(AudioSegment("We're gonna have to totally unlearn the concepts of functions, programs, values, and datatypes."), num_reductions);
    for(int j = 0; j < num_reductions; j++) {
        ls1->reduce();
        tds.render();
    }

    // Create text which says Lambda Calculus behind where the camera currently is
    shared_ptr<LatexScene> title = make_shared<LatexScene>(latex_text("The \\lambda -Calculus"), 1, 1000, 1000);
    tds.add_surface(Surface(glm::vec3(0,0,-14), glm::vec3(1,0,0), glm::vec3(0,1,0), title));

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
        glm::vec3 random_tilt_x(cos(theta), sin(theta), 0);
        glm::vec3 random_tilt_y(-sin(theta), cos(theta), 0);

        tds.add_surface(Surface(glm::vec3(x_position, y_position, z_position), random_tilt_x * 0.8f, random_tilt_y * 0.8f, lambda_scene));
        
        // Store the lambda scene in the vector for later reduction
        lots_of_lambdas.push_back(lambda_scene);
    }

    // Transition back to be able to see it
    tds.state_manager.superscene_transition(unordered_map<string, string>{
        {"z", "-16"},
    });

    tds.inject_audio(AudioSegment("'Cause today, we're learning the lambda calculus."), 8);

    // Reduce all the lambdas in the background in a loop
    for(int i = 0; i < 8; i++) {
        ls1->reduce();
        if(i > 1)
            for (auto& lambda_scene : lots_of_lambdas) {
                lambda_scene->reduce();  // Reduce the lambda expression
            }
        tds.render();  // Render the scene after each reduction
    }

    tds.state_manager.superscene_transition(unordered_map<string, string>{
        {"z", "20"},
        {"qk", "12"},
    });

    tds.inject_audio(AudioSegment(6), 5);
    // Reduce all the lambdas in the background in a loop
    for(int i = 0; i < 5; i++) {
        ls1->reduce();
        for (auto& lambda_scene : lots_of_lambdas) {
            lambda_scene->reduce(); // Reduce the lambda expression
        }
        tds.render(); // Render the scene after each reduction
    }
}

void history() {
    CompositeScene cs;
    cs.inject_audio_and_render(AudioSegment("But what even is computation?"));

    // Create Hilbert's BiographyScene on the left, with a quote on the right
    BiographyScene hilbert("hilbert", {"David Hilbert", "One of the greatest mathematicians of the 1900s."}, VIDEO_WIDTH/2, VIDEO_HEIGHT);
    cs.add_scene(&hilbert, "hilbert", 0, 0);
    cs.inject_audio_and_render(AudioSegment("David Hilbert, one of the greatest mathematicians of the 1900s, wanted to know whether there was some procedure, some algorithm, which, can determine whether any given mathematical statement is true or false."));

    // Move Hilbert to the top half to make room for other mathematicians
    hilbert.state_manager.set(unordered_map<string, string>{
        {"h", "[hilbert.h]"},
    });
    cs.state_manager.set(unordered_map<string, string>{
        {"hilbert.h", to_string(VIDEO_HEIGHT)},
    });
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"hilbert.h", to_string(.5 * VIDEO_HEIGHT)},
    });

    // Introduce Church, Turing, and Gödel, moving them from the bottom of the screen, breaking the blurb into parts
    BiographyScene church("church", {"Alonzo Church", "Invented the Lambda Calculus"}, VIDEO_WIDTH/3, VIDEO_HEIGHT/2);
    BiographyScene turing("turing", {"Alan Turing", "Father of modern computer science"}, VIDEO_WIDTH/3, VIDEO_HEIGHT/2);
    BiographyScene godel("godel", {"Kurt Gödel", "Proved mathematics is incomplete"}, VIDEO_WIDTH/3, VIDEO_HEIGHT/2);

    cs.add_scene(&church, "church", 0, 1);
    cs.add_scene(&turing, "turing", 0.33333, 1);
    cs.add_scene(&godel, "godel", 0.66666, 1);
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"turing.y", "0.5"},
        {"godel.y", "0.5"},
        {"church.y", "0.5"},
    });

    // Break up the audio and append to the relevant biographies
    cs.inject_audio_and_render(AudioSegment("Three men independently answered this question in different ways."));
    church.append_bio_text("First to disprove Hilbert's proposed algorithm");
    cs.inject_audio_and_render(AudioSegment("The ideas they encountered along the way were so groundbreaking that they proved Hilbert's task impossible,"));
    church.append_bio_text("Invented functional programming");
    turing.append_bio_text("Invented imperative programming");
    cs.inject_audio_and_render(AudioSegment("spawned two of the paradigms underlying modern programming languages,"));
    godel.append_bio_text("Gödel's Incompleteness Theorems");
    cs.inject_audio_and_render(AudioSegment("showed that mathematics is essentially incomplete,"));
    turing.append_bio_text("Formalized the Turing Machine");
    cs.inject_audio_and_render(AudioSegment("and spawned the entire field of computer science."));

    cs.inject_audio_and_render(AudioSegment("However, to answer this question in any rigorous sense, we need some sort of understanding of what a 'procedure' is in the first place."));

    // Slide Turing and Gödel out the right side, and introduce a LatexScene title "The \\lambda-Calculus" on the right side
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"turing.x", "2"},
        {"godel.x", "2"},
        {"hilbert.x", "2"},
    });

    LatexScene lambda_title(latex_text("The \\lambda-Calculus"), 1, VIDEO_WIDTH, VIDEO_HEIGHT*0.25);
    cs.add_scene(&lambda_title, "lambda_title", 0, -0.25);
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"lambda_title.y", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("Alonzo Church was the first to answer this question by inventing the Lambda Calculus."));
    cs.remove_scene(&hilbert);
    cs.remove_scene(&turing);
    cs.remove_scene(&godel);

    // Add LatexScenes showing Lambda expressions
    LatexScene lambda_examples1("(\\lambda x. x)",                   1, VIDEO_WIDTH*.333, VIDEO_HEIGHT*.333);
    LatexScene lambda_examples2("y",                                 1, VIDEO_WIDTH*.333, VIDEO_HEIGHT*.333);
    LatexScene lambda_examples3("(\\lambda z. (z (\\lambda w. w)))", 1, VIDEO_WIDTH*.333, VIDEO_HEIGHT*.333);
    LatexScene lambda_examples4("(a (b c))"                        , 1, VIDEO_WIDTH*.333, VIDEO_HEIGHT*.333);
    cs.add_scene(&lambda_examples1, "lambda_examples1", 0.333, 0.333);
    cs.add_scene(&lambda_examples2, "lambda_examples2", 0.333, 0.666);
    cs.add_scene(&lambda_examples3, "lambda_examples3", 0.666, 0.333);
    cs.add_scene(&lambda_examples4, "lambda_examples4", 0.666, 0.666);
    cs.inject_audio_and_render(AudioSegment("In its original formulation, the Lambda Calculus was defined in terms of strings that look like this."));
    lambda_examples1.begin_latex_transition(latex_color(0xffff0000, "(") + latex_color(0xff444444, "\\lambda x. x") + latex_color(0xffff0000, ")"));
    lambda_examples2.begin_latex_transition(latex_color(0xff444444, "y"));
    lambda_examples3.begin_latex_transition(latex_color(0xffff0000, "(") + latex_color(0xff444444, "\\lambda z. ") + latex_color(0xffff0000, "(") + latex_color(0xff444444, "z ") + latex_color(0xffff0000, "(") + latex_color(0xff444444, "\\lambda w. w") + latex_color(0xffff0000, ")))"));
    lambda_examples4.begin_latex_transition(latex_color(0xffff0000, "(") + latex_color(0xff444444, "a ") + latex_color(0xffff0000, "(") + latex_color(0xff444444, "b c") + latex_color(0xffff0000, "))"));
    cs.inject_audio(AudioSegment("They are composed of parentheses,"), 2);
    cs.render();
    cs.render();
    cs.inject_audio_and_render(AudioSegment(1));
    lambda_examples1.begin_latex_transition(latex_color(0xff444444, "(") + latex_color(0xff00ff00, "\\lambda x. x") + latex_color(0xff444444, ")"));
    lambda_examples2.begin_latex_transition(latex_color(0xff00ff00, "y"));
    lambda_examples3.begin_latex_transition(latex_color(0xff444444, "(\\lambda ") + latex_color(0xff00ff00, "z") + latex_color(0xff444444, ". (") + latex_color(0xff00ff00, "z ") + latex_color(0xff444444, "(\\lambda") + latex_color(0xff00ff00, "w") + latex_color(0xff444444, ".") + latex_color(0xff00ff00, "w") + latex_color(0xff444444, ")))"));
    lambda_examples4.begin_latex_transition(latex_color(0xff444444, "(") + latex_color(0xff00ff00, "a ") + latex_color(0xff444444, "(") + latex_color(0xff00ff00, "b c") + latex_color(0xff444444, "))"));
    cs.inject_audio(AudioSegment("letters of the alphabet,"), 2);
    cs.render();
    cs.render();
    cs.inject_audio_and_render(AudioSegment(1));
    lambda_examples1.begin_latex_transition(latex_color(0xff444444, "(") + latex_color(0xff0044ff, "\\lambda") + latex_color(0xff444444, "x") + latex_color(0xff0044ff, ".") + latex_color(0xff444444, "x)"));
    lambda_examples2.begin_latex_transition(latex_color(0xff444444, "y"));
    lambda_examples3.begin_latex_transition(latex_color(0xff444444, "(") + latex_color(0xff0044ff, "\\lambda") + latex_color(0xff444444, "z ") + latex_color(0xff0044ff, ".") + latex_color(0xff444444, "(z (") + latex_color(0xff0044ff, "\\lambda") + latex_color(0xff444444, "w ") + latex_color(0xff0044ff, ".") + latex_color(0xff444444, "w)))"));
    lambda_examples4.begin_latex_transition(latex_color(0xff444444, "(a (b c))"));
    cs.inject_audio(AudioSegment("and this notation involving a lambda and a dot. Those two always come together in a pair."), 2);
    cs.render();
    cs.render();
    cs.inject_audio_and_render(AudioSegment(1));

    // On the left, add the production rules of the lambda calculus
    LatexScene lambda_rule_var("a"                , 0.8, VIDEO_WIDTH*.5, VIDEO_HEIGHT*.25);
    LatexScene lambda_rule_abs("(\\lambda a. \\_)", 0.8, VIDEO_WIDTH*.5, VIDEO_HEIGHT*.25);
    LatexScene lambda_rule_app("(\\_ \\_)"        , 0.8, VIDEO_WIDTH*.5, VIDEO_HEIGHT*.25);
    cs.add_scene(&lambda_rule_var, "lambda_rule_var", -0.5, 0.25);
    cs.add_scene(&lambda_rule_abs, "lambda_rule_abs", -0.5,  0.5);
    cs.add_scene(&lambda_rule_app, "lambda_rule_app", -0.5, 0.75);

    // Slide Church out to the left
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"church.x", "-1"},
        {"lambda_examples1.x", "1.5"},
        {"lambda_examples2.x", "1.5"},
        {"lambda_examples3.x", "1.5"},
        {"lambda_examples4.x", "1.5"},
        {"lambda_rule_var.x"    , "0"},
        {"lambda_rule_abs.x"    , "0"},
        {"lambda_rule_app.x"    , "0"},
    });
    cs.inject_audio_and_render(AudioSegment("We can build these strings ourselves, following 3 rules."));
    cs.remove_scene(&lambda_examples1);
    cs.remove_scene(&lambda_examples2);
    cs.remove_scene(&lambda_examples3);
    cs.remove_scene(&lambda_examples4);

    // Start Animation Help!

    // Dim rules 2 and 3 to gray, and slide the variable "mod" rule out to the right.
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"lambda_rule_var.x", ".5"},
        {"lambda_rule_var.y", ".5"},
    });
    cs.inject_audio_and_render(AudioSegment("The first rule says that any letter is a valid lambda expression."));

    // Highlight the first rule for variables
    cs.inject_audio(AudioSegment("a, b, c, you name it. Simple as that."), 4);

    lambda_rule_var.begin_latex_transition("b");
    cs.render();

    lambda_rule_var.begin_latex_transition("c");
    cs.render();

    lambda_rule_var.begin_latex_transition("a");
    cs.render();

    // Fade out the first rule by transitioning its opacity to 0.
    cs.state_manager.subscene_transition(unordered_map<string, string>{
        {"lambda_rule_var.x", "0"},
        {"lambda_rule_var.y", ".25"},
    });
    cs.render();

    // Move out the second rule.
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"lambda_rule_abs.x", ".5"},
        {"lambda_rule_abs.y", ".5"},
    });
    cs.inject_audio_and_render(AudioSegment("The second rule says that we can make a valid expression of this form."));

    // Highlight the abstraction rule, keeping the variable "a" constant
    lambda_rule_abs.begin_latex_transition("(\\lambda a. \\_)");
    cs.inject_audio(AudioSegment("Once again, the 'a' represents any letter."), 3);

    // Shuffle through letters, transitioning back to 'a' like before
    lambda_rule_abs.begin_latex_transition("(\\lambda b. \\_)"); cs.render();
    lambda_rule_abs.begin_latex_transition("(\\lambda c. \\_)"); cs.render();
    lambda_rule_abs.begin_latex_transition("(\\lambda a. \\_)"); cs.render();

    // Highlight the blank in white, make the rest of the term gray.
    lambda_rule_abs.begin_latex_transition(latex_color(0xff444444, "(\\lambda a. ") + latex_color(0xffffffff, "\\_") + latex_color(0xff444444, ")"));
    cs.inject_audio_and_render(AudioSegment("The blank, in this case, is a placeholder for any other valid lambda expression."));

    // Transition the latex to have an 'a' where the blank was.
    lambda_rule_abs.begin_latex_transition("(\\lambda a. a)");
    cs.inject_audio_and_render(AudioSegment("That could, for example, be a lone variable, such as the ones which we made in expression 1."));

    // Show the completed valid lambda expression.
    lambda_rule_abs.begin_latex_transition("(\\lambda x. \\_)\\\\\\\\(\\lambda a. a)");
    cs.inject_audio_and_render(AudioSegment("So, this is a valid lambda expression which matches the proper form."));

    // Slide out another modifiable copy of the abstraction rule and place the last expression inside of it.
    lambda_rule_abs.begin_latex_transition("(\\lambda x. (\\lambda a. a))");
    cs.inject_audio_and_render(AudioSegment("And therefore, it can also be placed inside the blank of the same rule."));

    // Transition back to the identity function.
    lambda_rule_abs.begin_latex_transition("(\\lambda a. a)");
    // Fade-in a Python identity function which models this lambda expression identity function.
    PngScene python_identity("python_identity", VIDEO_WIDTH/2, VIDEO_HEIGHT/4);
    cs.add_scene(&python_identity, "python_identity", 0.5, 0.75);
    cs.state_manager.set(unordered_map<string, string>{
        {"python_identity.y", "1"},
    });
    cs.inject_audio_and_render(AudioSegment("This rule can kind of be thought of as a function definition."));
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"python_identity.y", ".75"},
    });

    cs.inject_audio_and_render(AudioSegment("The variable is the name of the input, and the blank represents the return-statement of the function."));
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"python_identity.y", "1"},
    });

    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"lambda_rule_abs.x", "0"},
        {"lambda_rule_abs.y", ".5"},
        {"lambda_rule_app.x", ".5"},
        {"lambda_rule_app.y", ".5"},
    });

    // Repeat by making an example with the third function, and then explain its role as function application.
    lambda_rule_app.begin_latex_transition("(\\_\\_)");
    cs.inject_audio_and_render(AudioSegment("If the second rule represents a function, the third rule involves _applying_ such functions."));

    // Show an example where a function is applied to a variable.
    lambda_rule_app.begin_latex_transition("(" + latex_color(0xff0088ff, "a") + latex_color(0xffff0000, "b") + ")");
    cs.inject_audio(AudioSegment("In this case, we are sort of implying 'a' is going to be used as a function which takes in 'b'."), 2);
    cs.render();
    cs.render();

    // Show an example where a function is applied to a variable.
    lambda_rule_app.begin_latex_transition("(" + latex_color(0xff0088ff, "(\\lambda x. (\\lambda a. a))") + latex_color(0xffff0000, "(b c)") + ")");
    cs.inject_audio(AudioSegment("And of course, just like before, any other valid expressions are fair game to put in here."), 2);
    cs.render();
    cs.render();

    lambda_rule_app.begin_latex_transition("(\\_\\_)");
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"lambda_rule_app.x", "-1"},
        {"lambda_rule_abs.x", "-1"},
        {"lambda_rule_var.x", "-1"},
        {"lambda_title.y"   , "-1"},
    });

    cs.inject_audio_and_render(AudioSegment("That's all we need to generate lambda expressions."));
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
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"rep_classic.y", "0"},
        {"rep_graph.y", "0"},
        {"rep_dbi.y", "0"},
        {"rep_keenan.y", "0"},
        {"rep_vex.y", "0.5"},
        {"rep_tromp.y", "0.5"},
        {"rep_viktor.y", "0.5"},
    });
    cs.inject_audio_and_render(AudioSegment("We'll try evaluating them in a sec, but first let's visualize them."));
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"rep_classic.y", "-1.5"},
        {"rep_keenan.y", "-1.5"},
        {"rep_graph.y", "-1.5"},
        {"rep_dbi.y", "-1.5"},
        {"rep_vex.y", "-1"},
        {"rep_tromp.y", "0"},
        {"rep_tromp.x", "0"},
        {"rep_viktor.y", "-1"},
    });
    rep_tromp.state_manager.superscene_transition(unordered_map<string, string>{
        {"w", to_string(VIDEO_WIDTH)},
        {"h", to_string(VIDEO_HEIGHT)},
    });
    cs.inject_audio_and_render(AudioSegment("There are a ton of styles, but the one I chose is John Tromp's Lambda Diagrams."));
    cs.inject_audio_and_render(AudioSegment("After all, I'm biased- he was the first to strongly solve 7x6 Connect 4."));



    LatexScene lambda_rule_var("a"                , 0.8, VIDEO_WIDTH*.25, VIDEO_HEIGHT*.25);
    LatexScene lambda_rule_abs("(\\lambda a. \\_)", 0.8, VIDEO_WIDTH*.25, VIDEO_HEIGHT*.25);
    LatexScene lambda_rule_app("(\\_ \\_)"        , 0.8, VIDEO_WIDTH*.25, VIDEO_HEIGHT*.25);
    cs.add_scene(&lambda_rule_var, "lambda_rule_var", -0.5, 0.125);
    cs.add_scene(&lambda_rule_abs, "lambda_rule_abs", -0.5, 0.375);
    cs.add_scene(&lambda_rule_app, "lambda_rule_app", -0.5, 0.625);
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"lambda_rule_var.x"    , "0"},
        {"lambda_rule_abs.x"    , "0"},
        {"lambda_rule_app.x"    , "0"},
        {"rep_tromp.x", ".25"},
    });
    rep_tromp.state_manager.superscene_transition(unordered_map<string, string>{
        {"w", to_string(VIDEO_WIDTH*3/4)},
        {"latex_opacity", "1"},
    });
    cs.remove_scene(&rep_classic);
    cs.remove_scene(&rep_keenan);
    cs.remove_scene(&rep_graph);
    cs.remove_scene(&rep_dbi);
    cs.remove_scene(&rep_vex);
    cs.remove_scene(&rep_viktor);
    cs.inject_audio_and_render(AudioSegment("Each of our three rules is part of a different shape in a lambda diagram."));
    LambdaExpression::Iterator it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(current->get_type() == "Variable" ? OPAQUE_WHITE : 0xff222222);
    }
    rep_tromp.set_expression(term);
    lambda_rule_var.begin_latex_transition(latex_color(0xffffffff, "a"                ));
    lambda_rule_abs.begin_latex_transition(latex_color(0xff222222, "(\\lambda a. \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xff222222, "(\\_ \\_)"        ));
    cs.inject_audio_and_render(AudioSegment(1));
    cs.inject_audio_and_render(AudioSegment("Variables - that is, the first rule, are these vertical lines."));
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(current->get_type() == "Abstraction" ? OPAQUE_WHITE : 0xff222222);
    }
    rep_tromp.set_expression(term);
    lambda_rule_var.begin_latex_transition(latex_color(0xff222222, "a"                ));
    lambda_rule_abs.begin_latex_transition(latex_color(0xffffffff, "(\\lambda a. \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xff222222, "(\\_ \\_)"        ));
    cs.inject_audio_and_render(AudioSegment(1));
    cs.inject_audio_and_render(AudioSegment("Lambda Abstractions, or rule 2, correspond to the vertical bars at the top."));
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(current->get_type() == "Application" ? OPAQUE_WHITE : 0xff222222);
    }
    rep_tromp.set_expression(term);
    lambda_rule_var.begin_latex_transition(latex_color(0xff222222, "a"                ));
    lambda_rule_abs.begin_latex_transition(latex_color(0xff222222, "(\\lambda a. \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xffffffff, "(\\_ \\_)"        ));
    cs.inject_audio_and_render(AudioSegment(1));
    cs.inject_audio_and_render(AudioSegment("This branching structure at the bottom is rule 3, for function application."));

    cs.inject_audio_and_render(AudioSegment("Let's take a close look at each of the three rules."));
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(current->get_type() == "Variable" ? OPAQUE_WHITE : 0xff222222);
    }
    rep_tromp.set_expression(term);
    lambda_rule_var.begin_latex_transition(latex_color(0xffffffff, "a"                ));
    lambda_rule_abs.begin_latex_transition(latex_color(0xff222222, "(\\lambda a. \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xff222222, "(\\_ \\_)"        ));
    cs.inject_audio_and_render(AudioSegment(1));
    cs.inject_audio_and_render(AudioSegment("Notice that there are eight vertical lines here, just how the expression itself has eight variables."));
    vector<LatexScene> lsv;
    for(int i = 0; i < 8; i++){
        string ch = "y";
        if(((i+3)/2)%2 == 0) ch = "x";
        LatexScene varls(ch, 1, VIDEO_WIDTH/10, VIDEO_WIDTH/10);
        lsv.push_back(varls);
    }
    for(int i = 0; i < 8; i++){
        cs.add_scene(&(lsv[i]), "lamvar" + to_string(i), .22+i/10.2, .07);
    }
    cs.inject_audio_and_render(AudioSegment("These correspond one-to-one, left to right."));
    cs.inject_audio_and_render(AudioSegment("Note how they collide into different vertical bars."));
    lambda_rule_var.begin_latex_transition(latex_color(0xff222222, "a"                ));
    lambda_rule_abs.begin_latex_transition(latex_color(0xffffffff, "(\\lambda a. \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xff222222, "(\\_ \\_)"        ));
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(current->get_type() == "Abstraction" ? OPAQUE_WHITE : 0xff222222);
    }
    rep_tromp.set_expression(term);
    vector<LatexScene> lsv_abstr;
    for(int i = 0; i < 4; i++){
        string ch = "y";
        if(i%2 == 0) ch = "x";
        LatexScene absls("\\lambda " + ch + ".", 1, VIDEO_WIDTH/16, VIDEO_WIDTH/16);
        lsv_abstr.push_back(absls);
    }
    for(int i = 0; i < 4; i++){
        cs.add_scene(&(lsv_abstr[i]), "lamabs" + to_string(i), .23+(i/2)*.375, .15 + .08 * (i%2));
    }
    cs.inject_audio_and_render(AudioSegment("Those vertical bars, corresponding to rule 2, are associated each with one of the lambda abstraction rules."));
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
    rep_tromp.set_expression(term);
    cs.inject_audio_and_render(AudioSegment("A variable makes contact with it if that variable name corresponds to the variable name which is in between the lambda and the dot."));
    cs.inject_audio(AudioSegment("The Xs connect to the lambda abstraction which binds the variable X, and the Ys touch the bar which binds Y."), 4);
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        int color = 0xff222222;
        if(current->get_type() == "Abstraction" && current->get_string()[2] == 'x') color = 0xffff0000;
        if(current->get_type() == "Variable" && current->get_string() == "x") color = 0xffff0000;
        if(current->get_type() == "Abstraction" && current->get_string()[2] == 'y') color = 0xff002266;
        if(current->get_type() == "Variable" && current->get_string() == "y") color = 0xff002266;
        current->set_color(color);
    }
    rep_tromp.set_expression(term);
    cs.render();
    cs.render();
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        int color = 0xff222222;
        if(current->get_type() == "Abstraction" && current->get_string()[2] == 'x') color = 0xffff0000;
        if(current->get_type() == "Variable" && current->get_string() == "x") color = 0xffff0000;
        if(current->get_type() == "Abstraction" && current->get_string()[2] == 'y') color = 0xff0044ff;
        if(current->get_type() == "Variable" && current->get_string() == "y") color = 0xff0044ff;
        current->set_color(color);
    }
    rep_tromp.set_expression(term);
    cs.render();
    cs.render();
    cs.inject_audio_and_render(AudioSegment(1));
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(OPAQUE_WHITE);
    }
    rep_tromp.set_expression(term);
    lambda_rule_var.begin_latex_transition(latex_color(0xff222222, "a"                ));
    lambda_rule_abs.begin_latex_transition(latex_color(0xff222222, "(\\lambda a. \\_)"));
    lambda_rule_app.begin_latex_transition(latex_color(0xffffffff, "(\\_ \\_)"        ));
    cs.inject_audio_and_render(AudioSegment(1));
    it = LambdaExpression::Iterator(term);
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(current->get_type() == "Application" ? OPAQUE_WHITE : 0xff222222);
    }
    rep_tromp.set_expression(term);
    cs.inject_audio_and_render(AudioSegment("Rule 3, for function application, has two subexpressions corresponding to the two blanks."));
    dynamic_pointer_cast<LambdaApplication>(term)->get_first()->set_color_recursive(0xff006600);
    dynamic_pointer_cast<LambdaApplication>(term)->get_second()->set_color_recursive(0xff660066);
    rep_tromp.set_expression(term);
    cs.inject_audio_and_render(AudioSegment("This lambda expression has an application surrounding everything. It's shown in white."));
    cs.inject_audio(AudioSegment("The lambda diagram for each of the subexpressions is drawn on the left and right branch of the application."), 2);
    dynamic_pointer_cast<LambdaApplication>(term)->get_first()->set_color_recursive(0xff00ff00);
    dynamic_pointer_cast<LambdaApplication>(term)->get_second()->set_color_recursive(0xff660066);
    rep_tromp.set_expression(term);
    cs.render();
    dynamic_pointer_cast<LambdaApplication>(term)->get_first()->set_color_recursive(0xff006600);
    dynamic_pointer_cast<LambdaApplication>(term)->get_second()->set_color_recursive(0xffff00ff);
    rep_tromp.set_expression(term);
    cs.render();
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        current->set_color(OPAQUE_WHITE);
    }
    rep_tromp.set_expression(term);
    cs.inject_audio_and_render(AudioSegment(1));
    cs.inject_audio_and_render(AudioSegment("Don't worry too much about how exactly these are read, it's the kind of skill you have to practice for a little while."));
    cs.state_manager.superscene_transition(unordered_map<string, string>{
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
    cs.inject_audio_and_render(AudioSegment("I'll color in subcomponents so that you can tell what's going on!"));
}

void beta_reduction(){
    CompositeScene cs;
    LatexScene beta_title(latex_text("\\beta-Reduction"), 1, VIDEO_WIDTH*0.5, VIDEO_HEIGHT*0.25);
    cs.add_scene(&beta_title, "beta_title", 0.25, 0);
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"beta_title.y", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("I've mentioned that these expressions can be 'evaluated'- what exactly does that mean?"));
    LatexScene latexbeta("(\\lambda x. \\_)", 0.6, VIDEO_WIDTH*.5, VIDEO_HEIGHT*.5);
    cs.add_scene(&latexbeta, "latex_beta", -1, .25);
    cs.state_manager.subscene_transition(unordered_map<string, string>{
        {"latex_beta.x", ".25"},
    });
    cs.inject_audio(AudioSegment("Remember that rule 2 can be interpreted as a function."), 3);
    cs.render(); cs.render(); cs.render();
    latexbeta.begin_latex_transition("(\\lambda x. \\_)\\\\\\\\(\\_ \\_)");
    cs.inject_audio(AudioSegment("And rule 3 can be interpreted as applying a value to a function."), 3);
    cs.render(); cs.render(); cs.render();
    cs.inject_audio_and_render(AudioSegment("We want to create a rule for evaluation that combines these interpretations."));

    // Scene showing lambda abstraction
    latexbeta.begin_latex_transition("(\\lambda x. x)\\\\\\\\(\\_ \\_)");
    cs.inject_audio_and_render(AudioSegment("First, let's make a function by using rule 2."));
    cs.inject_audio_and_render(AudioSegment("This function is pretty simple."));
    latexbeta.begin_latex_transition("(\\lambda " + latex_color(0xff00ff00, "x") + ". x)\\\\\\\\(\\_ \\_)");
    cs.inject_audio_and_render(AudioSegment("It just takes a value in,"));
    latexbeta.begin_latex_transition("(\\lambda " + latex_color(0xff00ff00, "x") + ". " + latex_color(0xffff0000, "x") + ")\\\\\\\\(\\_ \\_)");
    cs.inject_audio_and_render(AudioSegment("and spits that same value right back out."));

    // Scene showing some random expression to be applied to the function
    latexbeta.begin_latex_transition("((\\lambda x. x) \\_)");
    cs.inject_audio_and_render(AudioSegment("We will place that function in the spot where the function goes in rule 3."));

    // Scene showing some random expression to be applied to the function
    latexbeta.begin_latex_transition("((\\lambda x. x) E)");
    cs.inject_audio_and_render(AudioSegment("We'll make our 'value' just be some random expression."));
    cs.inject_audio_and_render(AudioSegment("Now, we are going to perform what is called 'beta reduction' to evaluate this function application."));

    // Show substitution
    latexbeta.begin_latex_transition("((" + latex_color(0xff8888ff, "\\lambda " + latex_color(0xff00ff00, "x") + ".") + "x) E)");
    cs.inject_audio_and_render(AudioSegment("We first note that 'x' is the variable bound by the lambda."));
    latexbeta.begin_latex_transition("((\\lambda x. " + latex_color(0xffff0000, "x") + ") E)");
    cs.inject_audio_and_render(AudioSegment("Since the function binds 'x', we first find all the instances of 'x' in the body of the function."));
    cs.inject_audio_and_render(AudioSegment("In this case there's just this one."));

    // Substitution result
    latexbeta.begin_latex_transition("(\\lambda x. " + latex_color(0xffff0000, "x") + ") E");
    cs.inject_audio_and_render(AudioSegment("Drop the application parentheses,"));
    latexbeta.begin_latex_transition("(\\lambda x. " + latex_color(0xffff0000, "x") + ") " + latex_color(0xff7777ff, "E"));
    cs.inject_audio_and_render(AudioSegment("Now, we take our value,"));
    latexbeta.begin_latex_transition("(\\lambda x. E)");
    cs.inject_audio_and_render(AudioSegment("and replace that instance of 'x' with that value."));

    // Drop unnecessary parts
    cs.inject_audio_and_render(AudioSegment("Now that we have placed the value in the function body,"));
    latexbeta.begin_latex_transition("E");
    cs.inject_audio_and_render(AudioSegment("we drop the binding lambda and its parentheses."));

    // Show final result
    cs.inject_audio_and_render(AudioSegment("And we are left with precisely the random value we picked."));
    cs.inject_audio_and_render(AudioSegment("This is exactly the behavior we should expect from a function that spits out exactly what we feed it."));

    latexbeta.begin_latex_transition(latex_color(0xff010101, "E"));
    cs.inject_audio_and_render(AudioSegment("Still looks like abstract nonsense?"));
    cs.remove_scene(&latexbeta);
    cs.inject_audio_and_render(AudioSegment("This actually isn't as unfamiliar to you as it might seem."));

    // Transition to algebra
    cs.inject_audio_and_render(AudioSegment("We can use the same notation with plain old algebra instead of lambda calculus."));
    LatexScene quadratic_equation("f(x) = x^2 + x + 3", .6, VIDEO_WIDTH*.8, VIDEO_HEIGHT*.5);
    cs.add_scene(&quadratic_equation, "quadratic_equation", 0.1, 0.25);
    cs.inject_audio_and_render(AudioSegment("Here's a quadratic function."));
    quadratic_equation.begin_latex_transition(latex_color(0xffff8888, "f(x) = x^2 + x + 3") + "\\\\\\\\" + latex_color(0xff88ff88, "f(5)"));
    cs.inject_audio_and_render(AudioSegment("We want to evaluate it at x=5."));

    // Show evaluation
    cs.inject_audio_and_render(AudioSegment("We can first establish the function, and then on a separate line, apply it to 5."));
    quadratic_equation.begin_latex_transition(latex_color(0xffff8888, "f(x) = x^2 + x + 3") + "; " + latex_color(0xff88ff88, "f(5)"));
    cs.inject_audio_and_render(AudioSegment("But let's stick em on the same line."));
    quadratic_equation.begin_latex_transition("(" + latex_color(0xffff8888, "f(x) = x^2 + x + 3") + "; " + latex_color(0xff88ff88, "f(5)") + ")");
    cs.inject_audio_and_render(AudioSegment("We'll wrap these two things in parentheses to imply that the first is a function and the second is the value we want to put in."));

    quadratic_equation.begin_latex_transition("(" + latex_color(0xffff0000, "f") + "(x) = x^2 + x + 3; " + latex_color(0xffff0000, "f") + "(5))");
    cs.inject_audio_and_render(AudioSegment("Now, since we have tied the function to its value, there is no point in naming it 'f'."));
    quadratic_equation.begin_latex_transition("((x) = x^2 + x + 3; 5)");
    cs.inject_audio_and_render(AudioSegment("Our inline notation is unambiguous about which function we want to apply to what value."));
    quadratic_equation.begin_latex_transition("((" + latex_color(0xff00ff00, "x") + ") = x^2 + x + 3; 5)");

    // Transform into lambda calculus
    cs.inject_audio_and_render(AudioSegment("We do still need to express that x is the thing in the body which the 5 will be replacing,"));
    quadratic_equation.begin_latex_transition("(\\lambda x. x^2 + x + 3; 5)");
    cs.inject_audio_and_render(AudioSegment("so let's slap a 'lambda' in front of it."));
    cs.inject_audio_and_render(AudioSegment("Now how do we evaluate it?"));
    cs.inject_audio_and_render(AudioSegment("Exactly the same procedure!"));
    quadratic_equation.begin_latex_transition("(\\lambda x. " + latex_color(0xffff0000, "x") + "^2 + " + latex_color(0xffff0000, "x") + " + 3; 5)");
    cs.inject_audio_and_render(AudioSegment("Identify all of the variables in the body which match the one bound by the lambda, in this case all of the 'x's."));
    quadratic_equation.begin_latex_transition("\\lambda x. " + latex_color(0xffff0000, "x") + "^2 + " + latex_color(0xffff0000, "x") + " + 3; 5");
    cs.inject_audio_and_render(AudioSegment("Drop the application parentheses,"));
    quadratic_equation.begin_latex_transition("\\lambda x. " + latex_color(0xffff0000, "x") + "^2 + " + latex_color(0xffff0000, "x") + " + 3;" + latex_color(0xff7777ff, "5"));
    cs.inject_audio_and_render(AudioSegment("grab our value,"));
    quadratic_equation.begin_latex_transition("\\lambda x. 5^2 + 5 + 3");
    cs.inject_audio_and_render(AudioSegment("and shove it everywhere we had an x."));
    quadratic_equation.begin_latex_transition("5^2 + 5 + 3");
    cs.inject_audio_and_render(AudioSegment("Drop the function definition,"));
    cs.inject_audio(AudioSegment("and we are left with an algebraic expression which represents the answer."), 3);
    quadratic_equation.begin_latex_transition("25 + 5 + 3");
    cs.render();
    quadratic_equation.begin_latex_transition("30 + 3");
    cs.render();
    quadratic_equation.begin_latex_transition("33");
    cs.render();
    cs.inject_audio_and_render(AudioSegment("Not too alien now, is it?"));

    quadratic_equation.begin_latex_transition("((\\lambda x. x) (\\lambda a. (\\lambda f. (a f))))");
    cs.inject_audio_and_render(AudioSegment("Alright, one more."));
    quadratic_equation.begin_latex_transition("(" + latex_color(0xffffaaaa, "(\\lambda x. x)") + "(\\lambda a. (\\lambda f. (a f))))");
    cs.inject_audio_and_render(AudioSegment("In this case, we are applying the identity function..."));
    quadratic_equation.begin_latex_transition("(" + latex_color(0xffffaaaa, "(\\lambda x. x)") + latex_color(0xffaaffaa, "(\\lambda a. (\\lambda f. (a f)))") + ")");
    cs.inject_audio_and_render(AudioSegment("...to another random term which I made up."));
    cs.inject_audio_and_render(AudioSegment("Let's follow the procedure once again."));
    quadratic_equation.begin_latex_transition("((\\lambda x. " + latex_color(0xffff0000, "x") + ") (\\lambda a. (\\lambda f. (a f))))");
    cs.inject_audio_and_render(AudioSegment("First, we identify all instances of the bound variable in the function."));
    quadratic_equation.begin_latex_transition("(\\lambda x. " + latex_color(0xffff0000, "x") + ") (\\lambda a. (\\lambda f. (a f)))");
    cs.inject_audio_and_render(AudioSegment("Drop the application parentheses,"));
    quadratic_equation.begin_latex_transition("(\\lambda x. " + latex_color(0xffff0000, "x") + ") " + latex_color(0xff7777ff, "(\\lambda a. (\\lambda f. (a f)))"));
    cs.inject_audio_and_render(AudioSegment("Grab the value,"));
    quadratic_equation.begin_latex_transition("(\\lambda x. (\\lambda a. (\\lambda f. (a f))))");
    cs.inject_audio_and_render(AudioSegment("and replace the bound variable with the argument being passed into the function."));
    quadratic_equation.begin_latex_transition("(\\lambda a. (\\lambda f. (a f)))");
    cs.inject_audio_and_render(AudioSegment("Scrap the function definition,"));
    quadratic_equation.begin_latex_transition("((\\lambda x. x) (\\lambda a. (\\lambda f. (a f))))");
    cs.inject_audio_and_render(AudioSegment("and we get my function back out!"));

    cs.state_manager.subscene_transition(unordered_map<string, string>{
        {"quadratic_equation.y", "0.58"},
    });
    cs.inject_audio_and_render(AudioSegment("Let's watch that again, but alongside the diagram this time."));
    shared_ptr<LambdaExpression> term = parse_lambda_from_string("((\\x. x) (\\a. (\\f. (a f))))");
    term->flush_uid_recursive();
    LambdaScene betadiagram(term, 0.4*VIDEO_WIDTH, 0.4*VIDEO_HEIGHT);
    cs.add_scene(&betadiagram, "betadiagram", 0.3, 0.3);
    quadratic_equation.begin_latex_transition("((\\lambda x. x) " + latex_color(0xff7777ff, "(\\lambda a. (\\lambda f. (a f)))") + ")");
    dynamic_pointer_cast<LambdaApplication>(term)->get_second()->set_color_recursive(0xff7777ff);
    betadiagram.set_expression(term);
    cs.inject_audio_and_render(AudioSegment("Let's color the value in blue,"));
    quadratic_equation.begin_latex_transition("((\\lambda x. " + latex_color(0xffff0000, "x") + ") " + latex_color(0xff7777ff, "(\\lambda a. (\\lambda f. (a f)))") + ")");
    dynamic_pointer_cast<LambdaAbstraction>(dynamic_pointer_cast<LambdaApplication>(term)->get_first())->get_body()->set_color_recursive(0xffff0000);
    shared_ptr<LambdaExpression> termclone = term->clone();
    betadiagram.set_expression(term);
    cs.inject_audio_and_render(AudioSegment("and the variable which it is about to replace as red."));
    cs.inject_audio_and_render(AudioSegment("Now let's perform the beta reduction."));
    quadratic_equation.begin_latex_transition(latex_color(0xff7f00ff, "(\\lambda a. (\\lambda f. (a f)))"));
    betadiagram.reduce();
    cs.inject_audio_and_render(AudioSegment("See how the value takes the place of the variable, and then the application scaffolding drops out?"));
    quadratic_equation.begin_latex_transition("((\\lambda x. " + latex_color(0xffff0000, "x") + ") " + latex_color(0xff7777ff, "(\\lambda a. a)") + ")");
    betadiagram.set_expression(termclone);
    cs.inject_audio_and_render(AudioSegment("Let's watch that a few times over."));
    quadratic_equation.begin_latex_transition(latex_color(0xff7f00ff, "(\\lambda a. (\\lambda f. (a f)))"));
    betadiagram.reduce();
    cs.inject_audio_and_render(AudioSegment(1));
    quadratic_equation.begin_latex_transition("((\\lambda x. " + latex_color(0xffff0000, "x") + ") " + latex_color(0xff7777ff, "(\\lambda a. (\\lambda f. (a f)))") + ")");
    betadiagram.set_expression(termclone);
    cs.inject_audio_and_render(AudioSegment(1));
    quadratic_equation.begin_latex_transition(latex_color(0xff7f00ff, "(\\lambda a. (\\lambda f. (a f)))"));
    betadiagram.reduce();
    cs.inject_audio_and_render(AudioSegment(1));
    cs.state_manager.subscene_transition(unordered_map<string, string>{
        {"quadratic_equation.opacity", "0"},
    });

    cs.inject_audio_and_render(AudioSegment("This doesn't have to happen at the top level of the expression."));
    shared_ptr<LambdaExpression> someterm = parse_lambda_from_string("((\\m. (\\n. (m (n ((m m) n))))) (\\x. (\\y. (y ((x x) y)))))");
    term = apply(someterm, term, OPAQUE_WHITE);
    term->flush_uid_recursive();
    betadiagram.set_expression(term);
    cs.inject_audio_and_render(AudioSegment("Let's make a big expression which contains this somewhere inside."));
    betadiagram.reduce();
    cs.inject_audio_and_render(AudioSegment("We can still beta-reduce the entire expression by reducing this subcomponent."));
    cs.state_manager.subscene_transition(unordered_map<string, string>{
        {"betadiagram.opacity", "0"},
        {"beta_title.opacity", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("Alright, time to write some code!"));
}

void currying() {
    CompositeScene cs;
    LatexScene currying_title(latex_text("Currying"), 1, VIDEO_WIDTH * 0.5, VIDEO_HEIGHT * 0.25);
    cs.add_scene(&currying_title, "currying_title", 0.25, -0.25);
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"currying_title.y", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("First, we've gotta make a function that takes two variables."));
    LatexScene multi_argument("(\\lambda x y . (5 * y + x))", 0.6, VIDEO_WIDTH, VIDEO_HEIGHT * 0.25);
    cs.add_scene(&multi_argument, "multi_argument", 0, 0.375);
    cs.inject_audio_and_render(AudioSegment("Pseudocode like this isn't permitted by our rules for making expressions."));
    multi_argument.begin_latex_transition("(\\lambda " + latex_color(0xffff0000, "x y") + ". (5 * y + x))");
    cs.inject_audio_and_render(AudioSegment("Only one letter is allowed to go between the lambda and the dot."));
    cs.inject_audio_and_render(AudioSegment("Luckily, the rules as-written are expressive enough for us to work around this, using a technique called 'currying'."));

    multi_argument.begin_latex_transition("(\\lambda x. (\\lambda y. (5 * y + x)))");
    cs.inject_audio_and_render(AudioSegment("The trick here is to make a function wrap another function."));
    cs.inject_audio_and_render(AudioSegment("Of course, we have yet to define numbers and plus and times, but let's assume we know how those work for now."));
    multi_argument.begin_latex_transition("(\\lambda x. (\\lambda y. (5 * y + x))) 5, 3)");
    cs.inject_audio_and_render(AudioSegment("Let's see what happens when we stick two arguments after this function and beta reduce it."));
    multi_argument.begin_latex_transition("(((\\lambda x. (\\lambda y. (5 * y + x))) 5) 3)");
    cs.inject_audio_and_render(AudioSegment("It's best to be clear what the application order is, so we don't misinterpret 5 as being a function which takes in 3."));
    cs.inject_audio_and_render(AudioSegment("Now let's beta reduce the innermost function application."));
    multi_argument.begin_latex_transition("(\\quad (" + latex_color(0xffff8888, "(\\lambda x. (\\lambda y. (5 * y + x)))") + latex_color(0xff7777ff, "5") + ")\\quad 3)");
    cs.inject_audio_and_render(AudioSegment("Here's the function, and the value we are passing in."));
    multi_argument.begin_latex_transition("(\\quad (" + latex_color(0xffff8888, "(\\lambda "+latex_color(0xff00ff00,"x")+". (\\lambda y. (5 * y + x)))") + latex_color(0xff7777ff, "5") + ")\\quad 3)");
    cs.inject_audio_and_render(AudioSegment("Notice the bound variable is 'x'."));
    multi_argument.begin_latex_transition("(\\quad (" + latex_color(0xffff8888, "(\\lambda "+latex_color(0xff00ff00,"x")+". (\\lambda y. (5 * y + "+latex_color(0xffff0000,"x")+")))") + latex_color(0xff7777ff, "5") + ")\\quad 3)");
    cs.inject_audio_and_render(AudioSegment("Find all the x's in the body,"));
    multi_argument.begin_latex_transition("(\\quad (" + latex_color(0xffff8888, "(\\lambda "+latex_color(0xff00ff00,"x")+". (\\lambda y. (5 * y + 5)))") + ")\\quad 3)");
    cs.inject_audio_and_render(AudioSegment("Replace,"));
    multi_argument.begin_latex_transition("(\\quad " + latex_color(0xffff8888, "(\\lambda y. (5 * y + 5))") + "\\quad 3)");
    cs.inject_audio_and_render(AudioSegment("and rip off the scaffolding."));
    multi_argument.begin_latex_transition("(\\quad (\\lambda y. (5 * y + 5))\\quad 3)");
    cs.inject_audio_and_render(AudioSegment(1));
    multi_argument.begin_latex_transition("((\\lambda y. (5 * y + 5)) 3)");
    cs.inject_audio_and_render(AudioSegment("Getting the hang of beta reduction?"));
    multi_argument.begin_latex_transition("(5 * 3 + 5)");
    cs.inject_audio_and_render(AudioSegment("Reduce one more time."));

    cs.state_manager.subscene_transition(unordered_map<string, string>{
        {"currying_title.opacity", "0"},
        {"multi_argument.opacity", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("and sure enough, we evaluated our 2-argument function."));
}

void booleans() {
    CompositeScene cs;
    cs.inject_audio_and_render(AudioSegment("Ok. If we want to write code, we're gonna need a way to represent TRUE and FALSE."));

    // Create Lambda scenes for TRUE and FALSE
    shared_ptr<LambdaExpression> lambda_true = parse_lambda_from_string("(\\a. (\\b. a))");
    LambdaScene true_scene(lambda_true, VIDEO_WIDTH/4, VIDEO_HEIGHT/4);

    shared_ptr<LambdaExpression> lambda_false = parse_lambda_from_string("(\\a. (\\b. b))");
    LambdaScene false_scene(lambda_false, VIDEO_WIDTH/4, VIDEO_HEIGHT/4);

    // Position both lambda scenes side by side in the center
    cs.add_scene(&true_scene, "true_scene", 0.25, 0.375);
    cs.add_scene(&false_scene, "false_scene", 0.5, 0.375);
    cs.inject_audio_and_render(AudioSegment("Conventionally, these are the two expressions which are used."));
    cs.inject_audio_and_render(AudioSegment("Note that they are both functions, and they are not reducible."));

    // Highlight the functions in red and green
    lambda_true->set_color_recursive(0xffff8888); // Light red
    lambda_false->set_color_recursive(0xff88ff88); // Light green
    true_scene.set_expression(lambda_true);
    false_scene.set_expression(lambda_false);
    cs.inject_audio_and_render(AudioSegment("They are also curried functions- one function directly inside the next."));
    // Move TRUE and FALSE to the top left
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"true_scene.x", "0.1"},
        {"true_scene.y", "0.1"},
        {"false_scene.x", "0.1"},
        {"false_scene.y", "0.3"},
    });

    // Create LatexScenes to manipulate the text
    LatexScene true_latex("(\\lambda a. (\\lambda b. a))", 0.5, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 4);
    LatexScene false_latex("(\\lambda a. (\\lambda b. b))", 0.5, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 4);
    cs.add_scene(&true_latex, "true_latex", 0, 0.5);
    cs.add_scene(&false_latex, "false_latex", 0.5, 0.5);

    cs.inject_audio_and_render(AudioSegment("To get a sense of what they do, let's pass in two values. We'll start with TRUE."));

    // Pass in V_1 and V_2 to TRUE and perform the reduction
    true_latex.begin_latex_transition("(\\lambda a. (\\lambda b. a) V_1 V_2)");
    // TODO beta reduce the expression over the course of a few animations.
    cs.inject_audio_and_render(AudioSegment("We passed in V1 and V2, and we got out just V1."));

    cs.inject_audio_and_render(AudioSegment("Let's try again with FALSE."));

    // Pass in V_1 and V_2 to FALSE and perform the reduction
    false_latex.begin_latex_transition("(\\lambda a. (\\lambda b. b) V_1 V_2)");
    cs.inject_audio_and_render(AudioSegment("This time, we got out V2 instead of V1."));

    cs.inject_audio_and_render(AudioSegment("These functions are fundamentally selectors."));

    // Highlight TRUE and FALSE variables
    true_latex.begin_latex_transition(latex_color(0xffff0000, "V1"));
    false_latex.begin_latex_transition(latex_color(0xffff0000, "V2"));
    cs.inject_audio_and_render(AudioSegment("TRUE picks out the first of 2 arguments, and FALSE picks the second."));

    // Transition to a conditional expression
    LatexScene conditional("(A t f)", 0.5, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2);
    cs.add_scene(&conditional, "conditional", 0.5, 0.6);
    cs.inject_audio_and_render(AudioSegment("We can make a term like this."));

    // Highlight t and f
    conditional.begin_latex_transition(latex_color(0xff00ff00, "t"));
    cs.inject_audio_and_render(AudioSegment("If A is true, then the output will be t."));
    conditional.begin_latex_transition(latex_color(0xff0000ff, "f"));
    cs.inject_audio_and_render(AudioSegment("If A is false, the output will be f."));

    // Move conditional out and prepare for logic gates
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"true_latex.opacity", "0"},
        {"false_latex.opacity", "0"},
        {"conditional.opacity", "0"},
    });

    cs.inject_audio_and_render(AudioSegment("Now, how do we make common logic gates using these values?"));

    // Create NOT logic gate
    LatexScene not_gate(latex_text("NOT = "), 0.5, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2);
    cs.add_scene(&not_gate, "not_gate", 0.5, 0.4);
    cs.inject_audio_and_render(AudioSegment("Let's start with NOT."));

    cs.inject_audio_and_render(AudioSegment("In other words, we want a function that maps TRUE onto FALSE, and vice versa."));

    // Show the abstraction for NOT
    not_gate.begin_latex_transition(latex_text("NOT = ") + "(\\lambda x. ???)");
    cs.inject_audio_and_render(AudioSegment("'NOT' only takes one variable in, so our function should too."));

    not_gate.begin_latex_transition(latex_text("NOT = ") + "(\\lambda x. ((x \\_) \\_))");
    cs.inject_audio_and_render(AudioSegment("The trick is to use that argument as a selector."));
    not_gate.begin_latex_transition(latex_text("NOT = ") + "(\\lambda x. ((x "+latex_color(0xffff0000, "\\_")+") \\_))");
    cs.inject_audio_and_render(AudioSegment("If the input is TRUE, then the first argument is the one which will be picked."));
    not_gate.begin_latex_transition(latex_text("NOT = ") + "(\\lambda x. ((x "+latex_text("FALSE")+") \\_))");
    cs.inject_audio_and_render(AudioSegment("Since NOT TRUE is FALSE, FALSE should be the thing in this blank."));
    not_gate.begin_latex_transition(latex_text("NOT = ") + "(\\lambda x. ((x "+latex_text("FALSE")+") "+latex_text("TRUE")+"))");
    cs.inject_audio_and_render(AudioSegment("If the input is FALSE, we similarly want the thing which FALSE selects for to be TRUE."));
    cs.inject_audio_and_render(AudioSegment("So, this is our expression for not."));

    // Substitute TRUE and FALSE expressions
    not_gate.begin_latex_transition("(\\lambda x. ((x (\\lambda a. (\\lambda b. b))) (\\lambda a. (\\lambda b. a))))");
    cs.inject_audio_and_render(AudioSegment("If we want to write it in full, we can substitute in the TRUEs and FALSEs."));
    
    cs.inject_audio_and_render(AudioSegment("But for readability's sake..."));
    // Simplify the expression back
    not_gate.begin_latex_transition("NOT = (\\lambda x. ((x FALSE) TRUE))");
    cs.inject_audio_and_render(AudioSegment("let's just leave it like this."));

    // LambdaScene for the NOT gate
    shared_ptr<LambdaExpression> lambda_not = parse_lambda_from_string("(\\x. ((x (\\a. (\\b. b))) (\\a. (\\b. a))))");
    LambdaScene not_scene(lambda_not, VIDEO_WIDTH*.4, VIDEO_HEIGHT*.4);
    cs.add_scene(&not_scene, "not_scene", 0.3, 0.3);

    cs.inject_audio_and_render(AudioSegment("Let's plug in TRUE and FALSE, and check what we get."));
    // TODO transition the LambdaScene to NOT TRUE.
    // TODO continue, for AND and OR
}

int main() {
    Timer timer;
    FOR_REAL = true;
    PRINT_TO_TERMINAL = true;
    //intro();
    //history();
    //visualize();
    //beta_reduction();
    currying();
    booleans();

    //credits
    // 6884
    // Book by Peter Selinger
    // Tromp Lambda Diagrams
    // Viktor Massalogin's paper / https://github.com/bntre/visual-lambda
    // Vex: https://www.researchgate.net/publication/2726047_Programming_with_Visual_Expressions
    // dkeenan
    return 0;
}

