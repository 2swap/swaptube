using namespace std;
#include <string>
const string project_name = "LambdaDemo";
#include "../io/PathManager.cpp"
const int width_base = 640;
const int height_base = 360;
const int mult = 2;

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;

#include "../io/writer.cpp"
#include "../misc/Timer.cpp"

#include "../Scenes/Common/ThreeDimensionScene.cpp"
#include "../Scenes/Math/LambdaScene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Common/ExposedPixelsScene.cpp"

void render_video() {
    ThreeDimensionScene tds;

    string factorial = "(\\n. (\\f. (((n (\\f. (\\n. (n (f (\\f. (\\x. ((n f) (f x))))))))) (\\x. f)) (\\x. x))))";
    shared_ptr<LambdaExpression> le_factorial = parse_lambda_from_string(factorial);
    le_factorial->set_color_recursive(0xffffff00);

    string church_3 = "(\\a. (\\b. (a (a (a b)))))";
    shared_ptr<LambdaExpression> le_church_3 = parse_lambda_from_string(church_3);
    le_church_3->set_color_recursive(0xffff00ff);

    string pred = "(\\n. (\\f. (\\x. (((n (\\g. (\\h. (h (g f))))) (\\u. x)) (\\u. u)))))";
    shared_ptr<LambdaExpression> le_pred = parse_lambda_from_string(pred);
    le_pred->set_color_recursive(0xff00ffff);

    // Boolean True
    string boolean_true = "(\\a. (\\b. a))";
    shared_ptr<LambdaExpression> le_boolean_true = parse_lambda_from_string(boolean_true);
    le_boolean_true->set_color_recursive(0xffff0000); // Red color

    // Boolean False
    string boolean_false = "(\\a. (\\b. b))";
    shared_ptr<LambdaExpression> le_boolean_false = parse_lambda_from_string(boolean_false);
    le_boolean_false->set_color_recursive(0xff00ff00); // Green color

    // Boolean AND
    string boolean_and = "(\\p. (\\q. ((p q) p)))";
    shared_ptr<LambdaExpression> le_boolean_and = parse_lambda_from_string(boolean_and);
    le_boolean_and->set_color_recursive(0xffff8080); // Blue color

    // Boolean OR
    string boolean_or = "(\\p. (\\q. ((p p) q)))";
    shared_ptr<LambdaExpression> le_boolean_or = parse_lambda_from_string(boolean_or);
    le_boolean_or->set_color_recursive(0xffff8000); // Yellow color

    // Boolean NOT
    shared_ptr<LambdaVariable> x = make_shared<LambdaVariable>('x', OPAQUE_WHITE);
    shared_ptr<LambdaExpression> le_boolean_not = abstract('x', apply(apply(x, le_boolean_false, OPAQUE_WHITE), le_boolean_true, OPAQUE_WHITE), OPAQUE_WHITE);

    LambdaScene ls(apply(le_boolean_not, le_boolean_true, OPAQUE_WHITE), 400, 400);
    tds.add_surface(Surface(glm::vec3(0,0,0),glm::vec3(8,0,0),glm::vec3(0,9,0),&ls));

    // Show the lambda expression for 
    tds.inject_audio_and_render(AudioSegment("What you're looking at right now is a computation taking place."));
    tds.inject_audio_and_render(AudioSegment("More specifically, it's evaluating 4 factorial, and soon, it will arrive at the result of 24."));
    tds.inject_audio_and_render(AudioSegment("Oh look, it's done!"));

    // 
    tds.inject_audio_and_render(AudioSegment("What are all these weird lines though?"));
    tds.inject_audio_and_render(AudioSegment("Well, this bundle of pink lines represents the number 3."));
    tds.inject_audio_and_render(AudioSegment("These green lines represent the factorial function."));
    tds.inject_audio_and_render(AudioSegment("We've got one, two, three, and so on..."));
    tds.inject_audio_and_render(AudioSegment("as well as plus and times."));
    tds.inject_audio_and_render(AudioSegment("But, they enable us to ask the question 'what is plus times plus'."));
    tds.inject_audio_and_render(AudioSegment("We saw factorial(3), but we can just as easily compute the function 3(factorial) and evaluate the result!"));
    tds.inject_audio_and_render(AudioSegment("OK, you can't actually do that, right? That's not a real thing. Right?"));
    tds.inject_audio_and_render(AudioSegment("And it's certainly not the case that when evaluating it, the answer would make any sense... right?"));
    tds.inject_audio_and_render(AudioSegment("To learn the answer, we're going to have to completely unlearn the ideas of functions, programs, values, and datatypes."));
    tds.inject_audio_and_render(AudioSegment("Because they're just the stickers we've put on top of math to make it understandable by our tiny monkey brains."));
    tds.inject_audio_and_render(AudioSegment("Because they're a mirage distracting us from the essence of computation itself."));
    tds.inject_audio_and_render(AudioSegment("Because today, we're learning the lambda calculus."));

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

    tds.state_manager.add_equations(unordered_map<string, string>{
        {"surfaces_opacity", "1"},
        {"lines_opacity", "0"},
        {"points_opacity", "1"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "20"},
        {"q1", "1"},
        {"qi", "<t> cos 10 /"},
        {"qj", "<t> sin 10 /"},
        {"qk", "0"}
    });
    tds.inject_audio_and_render(AudioSegment(3));
}

int main() {
    Timer timer;
    render_video();
    return 0;
}

