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

    shared_ptr<LambdaScene> ls = make_shared<LambdaScene>(term, 800, 800);
    tds.add_surface(Surface(glm::vec3(0,0,0), glm::vec3(1,0,0), glm::vec3(0,1,0), ls));
    tds.state_manager.set(unordered_map<string, string>{
        {"surfaces_opacity", "1"},
        {"lines_opacity", "0"},
        {"points_opacity", "1"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "2"},
        {"q1", "1"},
        {"qi", "<t> 2 / cos 20 /"},
        {"qj", "<t> 2 / sin 20 /"},
        {"qk", "0"}
    });

    PRINT_TO_TERMINAL = false;
    FOR_REAL = false;
    // Show the lambda expression for 
    int num_reductions = term->count_reductions() + 4;
    //tds.inject_audio(AudioSegment(40), num_reductions);
    tds.inject_audio(AudioSegment("What you're looking at right now is a computation taking place."), num_reductions/2);
    tds.render();
    tds.render();
    for(int i = 0; i < num_reductions/2 - 2; i++){
        ls->reduce();
        tds.render();
    }
    tds.inject_audio(AudioSegment("Specifically, it's evaluating 3 factorial, and soon, it will arrive at the result of 6."), num_reductions/2);
    for(int i = 0; i < num_reductions/2; i++){
        ls->reduce();
        tds.render();
    }
    tds.inject_audio_and_render(AudioSegment("Oh look, it's done!"));

    // 
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
    tds.inject_audio_and_render(AudioSegment("And this blue chunk represents the application of the number to the function."));

    tds.inject_audio_and_render(AudioSegment("We can make all sorts of other values."));

    tds.state_manager.transition(unordered_map<string, string>{
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"}
    });

    tds.state_manager.transition(unordered_map<string, string>{
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
    
    tds.state_manager.transition(unordered_map<string, string>{
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



    tds.state_manager.transition(unordered_map<string, string>{
        {"q1", "1"},
        {"qi", ".4"},
        {"qj", "0"},
        {"qk", "0"},
    });
    le_factorial->set_color_recursive(0xffff0000);
    le_factorial->flush_uid_recursive();
    shared_ptr<LambdaScene> factorial_scene = make_shared<LambdaScene>(le_factorial, 800, 800);
    tds.add_surface(Surface(glm::vec3(-.6,10,-10), glm::vec3(1,0,0), glm::vec3(0,0,-1), factorial_scene));
    church3->set_color_recursive(0xff00ff00);
    church3->flush_uid_recursive();
    shared_ptr<LambdaScene> c3s = make_shared<LambdaScene>(church3, 800, 800);
    tds.add_surface(Surface(glm::vec3(.9,10,-9.6), glm::vec3(.4,0,0), glm::vec3(0,0,-.4), c3s));

    tds.inject_audio_and_render(AudioSegment("This mathematical language permits us to express any computational procedure, including functions such as factorial, which can then be applied to numbers like 3."));
    tds.remove_surface(churchtimesscene);
    tds.remove_surface(churchplusscene);
    tds.state_manager.transition(unordered_map<string, string>{
        {"y", "5"},
        {"z", "-11"},
    });
    FOR_REAL = true;
    tds.inject_audio_and_render(AudioSegment("But... the magic of it comes from the fact that it's not immediately obvious whether a certain expression is a number, a function that operates on numbers, or what."));

    le_factorial->flush_uid_recursive();
    le_church_3->flush_uid_recursive();
    le_factorial->set_color_recursive(0xffff0000);
    le_church_3->set_color_recursive(0xff00ff00);
    shared_ptr<LambdaExpression> term1 = apply(le_factorial, le_church_3, 0xff0000ff);
    shared_ptr<LambdaScene> ls1 = make_shared<LambdaScene>(term1, 600, 600);
    shared_ptr<LambdaExpression> term2 = apply(le_church_3, le_factorial, 0xff0000ff);
    tds.inject_audio_and_render(AudioSegment("And that's because, in this language, there _fundamentally is no difference_."));
    tds.add_surface(Surface(glm::vec3(-.2,10,-12), glm::vec3(1.4,0,0), glm::vec3(0,0,-1.4), ls1));
    tds.inject_audio_and_render(AudioSegment("Just like we applied the factorial function to 3 with function application,"));
    ls1->set_expression(term2);
    tds.inject_audio_and_render(AudioSegment("we can apply 3 to the factorial function in the exact same way, as though 3 was a function and factorial was a value."));
    tds.inject_audio_and_render(AudioSegment("OK, you can't actually evaluate that, right? That's not a real thing. Right?"));
    return;
    tds.inject_audio_and_render(AudioSegment("And it's certainly not the case that when evaluating it, the answer would make any sense... right?"));
    tds.inject_audio_and_render(AudioSegment("We're going to have to completely unlearn the concepts of functions, programs, values, and datatypes."));
    tds.inject_audio_and_render(AudioSegment("Because they're a mirage distracting us from the essence of computation itself."));
    tds.inject_audio_and_render(AudioSegment("Because today, we're learning the lambda calculus."));
}

void history(){
    ThreeDimensionScene tds;
    // 
    tds.inject_audio_and_render(AudioSegment("Our story starts in the 1930's. David Hilbert ..."));
    tds.inject_audio_and_render(AudioSegment(""));
    tds.inject_audio_and_render(AudioSegment(""));

    // 
    tds.inject_audio_and_render(AudioSegment("These diagrams and their actions turn out to have some super cool properties."));
    tds.inject_audio_and_render(AudioSegment("You might remember from math class that there is this notion of a function."));
    tds.inject_audio_and_render(AudioSegment("Functions always have a domain and a range. It takes in one type of data, and spits out another."));
    tds.inject_audio_and_render(AudioSegment("For example, the factorial function takes in a positive integer, and spits out another positive integer."));
    tds.inject_audio_and_render(AudioSegment("Addition is a function that takes a pair of real numbers, and gives a single real number back."));
    tds.inject_audio_and_render(AudioSegment("A function is a thing which takes an object from the Domain, and spits out a corresponding object in the Range."));
    tds.inject_audio_and_render(AudioSegment("Now, out of curiosity we can also talk about the set of functions, say, which take two real numbers and give one back."));
    tds.inject_audio_and_render(AudioSegment("Addition, subtraction, multiplication, and division come to mind."));
    tds.inject_audio_and_render(AudioSegment(""));
    tds.inject_audio_and_render(AudioSegment(""));
    tds.inject_audio_and_render(AudioSegment(""));

    // 
    tds.inject_audio_and_render(AudioSegment(""));
    tds.inject_audio_and_render(AudioSegment(""));
    tds.inject_audio_and_render(AudioSegment(""));

    // 
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

