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

    PngScene eps("Mona_Lisa");
    LambdaScene ls("(\\x. (\\y. (\\z. ((x z) (y z)))))");

    tds.add_surface(Surface(glm::vec3(0,0,0),glm::vec3(-8,2,8),glm::vec3(-2,-9,0),&eps));

    tds.dag->add_equations(std::unordered_map<std::string, std::string>{
        {"surfaces_opacity", "1"},
        {"lines_opacity", "0"},
        {"points_opacity", "1"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"d", "40"},
        {"q1", "<t> 4 / cos"},
        {"qi", "0"},
        {"qj", "<t> -4 / sin"},
        {"qk", "0"}
    });
    tds.inject_audio_and_render(AudioSegment(5));
}

int main() {
    Timer timer;
    render_video();
    timer.stop_timer();
    return 0;
}

