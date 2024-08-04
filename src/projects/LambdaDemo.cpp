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

#include "../scenes/Common/ThreeDimensionScene.cpp"
#include "../scenes/Math/Lambda/LambdaScene.cpp"
#include "../scenes/Media/PngScene.cpp"
#include "../scenes/Common/ExposedPixelsScene.cpp"

void render_video() {
    ThreeDimensionScene tds;

    LambdaScene ls("((\\n. (\\f. (\\x. (((n (\\g. (\\h. (h (g f))))) (\\u. x)) (\\u. u))))) (\\a. (\\b. (a (a b)))))", 400, 400);

    tds.add_surface(Surface(glm::vec3(0,0,0),glm::vec3(8,0,0),glm::vec3(0,9,0),&ls));

    tds.dag->add_equations(std::unordered_map<std::string, std::string>{
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

