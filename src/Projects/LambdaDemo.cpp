using namespace std;
#include <string>
const string project_name = "LambdaDemo";
#include "../io/PathManager.cpp"
const int width_base = 640;
const int height_base = 360;
const float mult = 2;

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
#include "../Scenes/Common/ExposedPixelsScene.cpp"

void intro() {
    ThreeDimensionScene tds;

    string factorial = "(\\n. (\\f. (((n (\\f. (\\n. (n (f (\\f. (\\x. ((n f) (f x))))))))) (\\x. f)) (\\x. x))))";
    shared_ptr<LambdaExpression> le_factorial = parse_lambda_from_string(factorial);
    le_factorial->set_color_recursive(0xffffff00);

    string church_3 = "(\\a. (\\b. (a (a (a b)))))";
    shared_ptr<LambdaExpression> le_church_3 = parse_lambda_from_string(church_3);
    le_church_3->set_color_recursive(0xffff00ff);

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

    PRINT_TO_TERMINAL = false;
    FOR_REAL = false;
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
    cs.add_scene(&boolean, "boo", 0, 0, .5, 1);
    cs.state_manager.add_equation("alg.opacity", "0");
    cs.state_manager.add_equation("boo.opacity", "0");
    cs.state_manager.add_equation("qj", "0");
    for(int i = 0; i < 5; i++){
        float alg_o = i==1;
        float boo_o = i==2;
        cs.state_manager.add_subscene_transition("alg.opacity", to_string(alg_o));
        cs.state_manager.add_subscene_transition("boo.opacity", to_string(boo_o));
        cs.state_manager.add_subscene_transition("qj", to_string(i==1 || i==2 ? .19: 0));
        cs.inject_audio(AudioSegment(blurbs[i]), num_reductions / 5);
        for(int j = 0; j < num_reductions/5; j++) {
            ls->reduce();
            cs.render();
        }
    }

    tds.inject_audio_and_render(AudioSegment("What are all these weird lines though?"));
    tds.inject_audio_and_render(AudioSegment("Well, this here represents the answer, 6."));

    ls->set_expression(term);
    tds.inject_audio_and_render(AudioSegment("Going back to the original setup,"));

    dynamic_pointer_cast<LambdaApplication>(term)->get_first()->set_color_recursive(0xffff0000);
    ls->set_expression(term);
    tds.inject_audio_and_render(AudioSegment("This red chunk represents the factorial function."));
    dynamic_pointer_cast<LambdaApplication>(term)->get_second()->set_color_recursive(0xff00ff00);
    ls->set_expression(term);
    tds.inject_audio_and_render(AudioSegment("This green chunk represents the number 3."));
    term->set_color(0xff0000ff);
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
    church1->set_color_recursive(OPAQUE_WHITE);
    church1->flush_uid_recursive();
    shared_ptr<LambdaScene> church1scene = make_shared<LambdaScene>(church1, 800, 800);
    tds.add_surface(Surface(glm::vec3(-2,-2,-1), glm::vec3(1,0,0), glm::vec3(0,1,0), church1scene));

    shared_ptr<LambdaExpression> church2 = parse_lambda_from_string("(\\f. (\\x. (f (f x))))");
    church2->set_color_recursive(OPAQUE_WHITE);
    church2->flush_uid_recursive();
    shared_ptr<LambdaScene> church2scene = make_shared<LambdaScene>(church2, 800, 800);
    tds.add_surface(Surface(glm::vec3(2,2,-3), glm::vec3(1,0,0), glm::vec3(0,1,0), church2scene));

    shared_ptr<LambdaExpression> church3 = parse_lambda_from_string("(\\f. (\\x. (f (f (f x)))))");
    church3->set_color_recursive(OPAQUE_WHITE);
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
    churchplus->set_color_recursive(OPAQUE_WHITE);
    churchplus->flush_uid_recursive();
    shared_ptr<LambdaScene> churchplusscene = make_shared<LambdaScene>(churchplus, 800, 800);
    tds.add_surface(Surface(glm::vec3(-5,1,-9), glm::vec3(0,0,1), glm::vec3(0,1,0), churchplusscene));

    shared_ptr<LambdaExpression> churchtimes = parse_lambda_from_string("(\\m. (\\n. (\\a. (m (n a)))))");
    churchtimes->set_color_recursive(OPAQUE_WHITE);
    churchtimes->flush_uid_recursive();
    shared_ptr<LambdaScene> churchtimesscene = make_shared<LambdaScene>(churchtimes, 800, 800);
    tds.add_surface(Surface(glm::vec3(-5,-1,-11), glm::vec3(0,0,1), glm::vec3(0,1,0), churchtimesscene));
    tds.inject_audio_and_render(AudioSegment("as well as plus and times."));
    tds.remove_surface(church1scene);
    tds.remove_surface(church2scene);
    tds.remove_surface(church3scene);
    tds.remove_surface(ls);



    tds.state_manager.subscene_transition(unordered_map<string, string>{
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"d", "2"},
    });
    le_factorial->flush_uid_recursive();
    le_church_3->flush_uid_recursive();
    le_factorial->set_color_recursive(0xffff0000);
    le_church_3->set_color_recursive(0xff00ff00);
    shared_ptr<LambdaExpression> term1 = apply(le_factorial, le_church_3, 0x00000000);
    shared_ptr<LambdaScene> ls1 = make_shared<LambdaScene>(term1, 600, 600);
    tds.add_surface(Surface(glm::vec3(0,0,-8), glm::vec3(1,0,0), glm::vec3(0,1,0), ls1));

    tds.inject_audio_and_render(AudioSegment("This mathematical language permits us to express any computational procedure, including functions such as factorial, which can then be applied to numbers like 3."));
    tds.remove_surface(churchtimesscene);
    tds.remove_surface(churchplusscene);
    tds.inject_audio_and_render(AudioSegment("But... the magic of it comes from the fact that it's not immediately obvious whether a certain expression is a number, a function that operates on numbers, or what."));

    tds.inject_audio_and_render(AudioSegment("And that's because, in this language, there _fundamentally is no difference_."));
    shared_ptr<LambdaExpression> term2 = apply(le_factorial, le_church_3, 0xff0000ff);
    ls1->set_expression(term2);
    tds.inject_audio_and_render(AudioSegment("Just like we applied the factorial function to 3 with function application,"));
    shared_ptr<LambdaExpression> term3 = apply(le_church_3, le_factorial, 0xff0000ff);
    ls1->set_expression(term3);
    tds.inject_audio_and_render(AudioSegment("we can apply 3 to the factorial function in the exact same way, as though 3 was a function and factorial was a value."));
    term3->set_color_recursive(0xff404040);
    ls1->set_expression(term3);

    tds.inject_audio_and_render(AudioSegment("OK, but you can't actually evaluate that, right? That's not a real thing. Right?"));
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
    FOR_REAL = true;

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
        "(((\\x. (\\y. (y ((x x) y)))) (\\x. (\\y. (y ((x x) y))))) (\\z. z))",  // Y combinator with extra application
        "(((\\f. (\\x. (f (f (f (f x)))))) (\\y. (y y))) (\\z. z))",  // Deep nested application with self-application
        "(((\\m. (\\n. (m (n m)))) (\\a. (a a))) (\\c. ((c c) (c c))))",  // Complex self-application
        "(((\\f. (\\x. (f (f (f x))))) (\\y. (y y))) (\\z. z))",  // Deep nested application with self-application
        "(((\\m. (\\n. (m (n ((m m) n))))) (\\x. x)) (\\y. (y y)))",  // Complex identity function application
        "(((\\x. (\\y. (y ((x x) y)))) (\\x. (\\y. (y ((x x) y))))) (\\x. (\\y. (y ((x x) y)))))",  // Y combinator with extra application
        "(((\\f. (\\x. (f (f (f (f x)))))) (\\y. (y y))) (\\x. (\\y. (y ((x x) y)))))",  // Deep nested application with self-application
        "(((\\m. (\\n. (m (n m)))) (\\x. (\\y. (y ((x x) y))))) (\\c. ((c c) (c c))))",  // Complex self-application
        "(((\\f. (\\x. (f (f (f x))))) (\\y. (y y))) (\\x. (\\y. (y ((x x) y)))))",  // Deep nested application with self-application
        "(((\\m. (\\n. (m (n ((m m) n))))) (\\x. (\\y. (y ((x x) y))))) (\\y. (y y)))"  // Complex identity function application
    };

    for (int i = 0; i < num_lambdas; ++i) {
        // Parse and color each lambda expression
        shared_ptr<LambdaExpression> le_complex_lambda = parse_lambda_from_string(complex_lambdas[i]);
        le_complex_lambda->set_color_recursive(0xff404040);  // Set color to grey

        shared_ptr<LambdaScene> lambda_scene = make_shared<LambdaScene>(le_complex_lambda, 400, 400);
        
        // Randomize position and orientation
        float x_position = ((i % 5) + ((rand() % 1000) / 1000.0f * scatter_range_x - scatter_range_x / 2) - 2) * 2;
        float y_position = ((i / 5) + ((rand() % 1000) / 1000.0f * scatter_range_y - scatter_range_y / 2) - .5) * 3 + i%2- .5;
        float z_position = -11.5 + (rand()%1000)/1000.0;
        float theta = ((rand()%1000)/1000.0f-0.5) * 0.2;
        glm::vec3 random_tilt_x(cos(theta), sin(theta), 0);
        glm::vec3 random_tilt_y(-sin(theta), cos(theta), 0);

        tds.add_surface(Surface(glm::vec3(x_position, y_position, z_position), random_tilt_x, random_tilt_y, lambda_scene));
        
        // Store the lambda scene in the vector for later reduction
        lots_of_lambdas.push_back(lambda_scene);
    }

    // Transition back to be able to see it
    tds.state_manager.superscene_transition(unordered_map<string, string>{
        {"z", "-15"},
    });

    tds.inject_audio(AudioSegment("Because today, we're learning the lambda calculus."), 10);

    // Reduce all the lambdas in the background in a loop
    for(int i = 0; i < 10; i++) {
        if(i > 2)
            for (auto& lambda_scene : lots_of_lambdas) {
                lambda_scene->reduce();  // Reduce the lambda expression
            }
        tds.render();  // Render the scene after each reduction
    }

    tds.inject_audio(AudioSegment(3), 10);
    // Reduce all the lambdas in the background in a loop
    for(int i = 0; i < 10; i++) {
        for (auto& lambda_scene : lots_of_lambdas) {
            lambda_scene->reduce(); // Reduce the lambda expression
        }
        tds.render(); // Render the scene after each reduction
    }

    tds.state_manager.superscene_transition(unordered_map<string, string>{
        {"z", "0"},
        {"qk", "4"},
    });

    tds.inject_audio(AudioSegment(3), 10);
    // Reduce all the lambdas in the background in a loop
    for(int i = 0; i < 10; i++) {
        for (auto& lambda_scene : lots_of_lambdas) {
            lambda_scene->reduce(); // Reduce the lambda expression
        }
        tds.render(); // Render the scene after each reduction
    }
}

void history(){
    ThreeDimensionScene tds;
    tds.inject_audio_and_render(AudioSegment("But what even is computation?"));
    tds.inject_audio_and_render(AudioSegment("David Hilbert, one of the greatest mathematicians of the 1900s, wanted to know whether there was some procedure that could be employed to determine, in a finite amount of time, whether some mathematical statement is true or false."));
    tds.inject_audio_and_render(AudioSegment("Three men independently answered this question in different ways, and the ideas they encountered along the way were so groundbreaking that they proved Hilbert's task impossible, invented Functional and Imperative programming, showed that mathematics is essentially incomplete, and spawned the entire field of computer science."));
    tds.inject_audio_and_render(AudioSegment("However, to answer this question in any rigorous sense, we need some sort of understanding of what a 'procedure' is in the first place."));
    tds.inject_audio_and_render(AudioSegment("Alonzo Church was the first to answer this question by inventing the Lambda Calculus."));
    tds.inject_audio_and_render(AudioSegment("In its original formulation, the Lambda Calculus was defined in terms of strings that look like this."));
    tds.inject_audio_and_render(AudioSegment("We can make these strings following exactly 3 rules."));
    tds.inject_audio_and_render(AudioSegment(""));
    tds.inject_audio_and_render(AudioSegment(""));
    tds.inject_audio_and_render(AudioSegment(""));

    tds.inject_audio_and_render(AudioSegment(3));
}

int main() {
    Timer timer;
    intro();
    return 0;
}

