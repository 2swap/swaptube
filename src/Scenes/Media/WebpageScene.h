#include "../Scene.h"

class WebpageScene : public Scene {
public:
    std::string url;
    WebpageScene(const std::string& src, const vec2& dimensions = vec2(1,1));

    void draw();

    const StateQuery populate_state_query() const;
};
