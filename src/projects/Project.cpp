using namespace std;
#include <string>
#include "../io/PathManager.cpp"
#include "../io/writer.cpp"
#include "../misc/Timer.cpp"

// I kinda gave up on this abstraction, cause the Scene object needs to know too much about random global stuff...
using VideoDimensions = glm::vec2;
VideoDimensions VS_NHD    (640 , 360 );
VideoDimensions VS_HD     (1280, 720 );
VideoDimensions VS_FULL_HD(1920, 1080);
VideoDimensions VS_4K     (3480, 2160);

class Project {
public:
    Project(string project_name, VideoDimensions vd, float vf) : video_dimensions(vd), video_framerate(vf), timer(), writer(), path_manager(project_name) {}
    VideoDimensions video_dimensions;
    float video_framerate;
private:
    Timer timer;
    MovieWriter writer;
    PathManager path_manager;
};
