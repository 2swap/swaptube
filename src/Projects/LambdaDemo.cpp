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

