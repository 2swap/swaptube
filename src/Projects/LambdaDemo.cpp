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
    PngScene boolean("BooleanAlgebra");

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
    cs.add_scene(&tds, "tds", 0, 0, 1, 1);
    cs.add_scene(&algebra, "alg", 0, 0, .5, 1);
    cs.add_scene(&boolean, "boo", 0.5, 0.05, .5, 1); // Boolean on the right, algebra on the left
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
    church3->set_color_recursive(0xff0000ff);
    church3->flush_uid_recursive();
    shared_ptr<LambdaScene> church3scene = make_shared<LambdaScene>(church3, 800, 800);
    tds.add_surface(Surface(glm::vec3(2,-2,-5), glm::vec3(1,0,0), glm::vec3(0,1,0), church3scene));
    tds.inject_audio_and_render(AudioSegment("We've got one, two, three, and so on..."));
    
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
    tds.inject_audio_and_render(AudioSegment("as well as plus and times."));
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

    tds.inject_audio(AudioSegment("We can express any computational procedure, even functions such as factorial."), 2);
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
    tds.inject_audio(AudioSegment("But... the magic of it comes from the fact that it's not immediately obvious whether a certain expression is a number,"), 4);
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

    tds.inject_audio_and_render(AudioSegment("And that's because, in this language, there _fundamentally is no difference_."));
    shared_ptr<LambdaExpression> term2 = apply(le_factorial, le_church_3, 0xff00ffff);
    ls1->set_expression(term2);
    tds.inject_audio_and_render(AudioSegment("Just like we applied the factorial function to 3 with function application,"));
    shared_ptr<LambdaExpression> term3 = apply(le_church_3, le_factorial, 0xff00ffff);
    ls1->set_expression(term3);
    tds.inject_audio_and_render(AudioSegment("we can apply 3 to the factorial function in the exact same way, as though 3 was a function and factorial was a value."));
    term3->set_color_recursive(0xff404040);
    ls1->set_expression(term3);

    tds.inject_audio_and_render(AudioSegment("OK, but you can't actually evaluate that, right?"));
    num_reductions = 3;
    tds.inject_audio(AudioSegment("Well, you can... but it's certainly not the case that when evaluating it, the answer would make any sense... right?"), num_reductions);
    for(int j = 0; j < num_reductions; j++) {
        ls1->reduce();
        tds.render();
    }

    tds.inject_audio(AudioSegment("We're going to have to completely unlearn the concepts of functions, programs, values, and datatypes."), num_reductions);
    for(int j = 0; j < num_reductions; j++) {
        ls1->reduce();
        tds.render();
    }

    // Create text which says Lambda Calculus behind where the camera currently is
    shared_ptr<LatexScene> title = make_shared<LatexScene>(latex_text("The \\lambda -Calculus"), 1);
    tds.add_surface(Surface(glm::vec3(0,0,-14), glm::vec3(1,0,0), glm::vec3(0,static_cast<float>(title->h)/title->w,0), title));

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
        {"z", "-15"},
    });

    tds.inject_audio(AudioSegment("Because today, we're learning the lambda calculus."), 8);

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
    BiographyScene hilbert("hilbert", {"David Hilbert", "One of the greatest mathematicians of the 1900s."});
    cs.add_scene(&hilbert, "hilbert", 0, 0, 0.5, 1);
    cs.inject_audio_and_render(AudioSegment("David Hilbert, one of the greatest mathematicians of the 1900s, wanted to know whether there was some procedure that could be employed to determine, in a finite amount of time, whether some mathematical statement is true or false."));

    // Move Hilbert to the top half to make room for other mathematicians
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"hilbert.h", ".5"},
    });

    // Introduce Church, Turing, and Gödel, moving them from the bottom of the screen, breaking the blurb into parts
    BiographyScene church("church", {"Alonzo Church", "Invented the Lambda Calculus"});
    BiographyScene turing("turing", {"Alan Turing", "Father of modern computer science"});
    BiographyScene godel("godel", {"Kurt Gödel", "Proved mathematics is incomplete"});

    cs.add_scene(&church, "church", 0, 1, 0.5, 0.5);
    cs.add_scene(&turing, "turing", 0.5, 1, 0.5, 0.5);
    cs.add_scene(&godel, "godel", 0.5, 1, 0.5, 0.5);

    // Break up the audio and append to the relevant biographies
    cs.inject_audio_and_render(AudioSegment("Three men independently answered this question in different ways..."));
    church.set_bio_text({"Alonzo Church", "Invented the Lambda Calculus", "Proposed a formal system for computation"});
    cs.inject_audio_and_render(AudioSegment("... and the ideas they encountered along the way were so groundbreaking..."));
    turing.set_bio_text({"Alan Turing", "Father of modern computer science", "Developed the Turing Machine"});
    cs.inject_audio_and_render(AudioSegment("... that they proved Hilbert's task impossible..."));
    godel.set_bio_text({"Kurt Gödel", "Proved mathematics is incomplete", "Gödel's Incompleteness Theorems"});

    cs.inject_audio_and_render(AudioSegment("However, to answer this question in any rigorous sense, we need some sort of understanding of what a 'procedure' is in the first place."));

    // Slide Turing and Gödel out the right side, and introduce a LatexScene title "The \\lambda-Calculus" on the right side
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"turing.x", "1"},
        {"godel.x", "1"},
        {"church.x", "1"},
    });

    LatexScene lambda_calculus_title("The \\lambda-Calculus", 1);
    cs.add_scene(&lambda_calculus_title, "lambda_title", 0.5, 0, 0.5, 0.5);
    cs.inject_audio_and_render(AudioSegment("Alonzo Church was the first to answer this question by inventing the Lambda Calculus."));

    // Slide Church out to the left
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"church.x", "-1"},
    });

    // Add LatexScenes showing Lambda expressions
    LatexScene lambda_examples("\\lambda x. x \\quad \\lambda x. (x x) \\quad \\lambda x. (x (x x))", 1);
    cs.add_scene(&lambda_examples, "lambda_examples", 0.5, 0.5, 0.5, 0.5);
    cs.inject_audio_and_render(AudioSegment("In its original formulation, the Lambda Calculus was defined in terms of strings that look like this."));

    // Slide those examples to the right side of the screen
    cs.state_manager.superscene_transition(unordered_map<string, string>{
        {"lambda_examples.x", "0.5"},
    });

    // On the left, add the production rules of the lambda calculus
    LatexScene lambda_rules("1. Variable\n2. Abstraction\n3. Application", 1);
    cs.add_scene(&lambda_rules, "lambda_rules", 0, 0.5, 0.5, 0.5);
    cs.inject_audio_and_render(AudioSegment("We can make these strings following exactly 3 rules."));

    cs.inject_audio_and_render(AudioSegment(""));
    cs.inject_audio_and_render(AudioSegment(""));
    cs.inject_audio_and_render(AudioSegment(""));

    cs.inject_audio_and_render(AudioSegment(3));
}

int main() {
    Timer timer;
    FOR_REAL = false;
    //intro();
    PRINT_TO_TERMINAL = false;
    FOR_REAL = true;
    history();
    return 0;
}

