#include "../Scene.h"

class WebpageScene : public Scene {
public:
    std::string url;
    WebpageScene(const std::string& src, const vec2& dimensions = vec2(1,1));

    bool check_if_data_changed() const;
    void mark_data_unchanged();
    void change_data();

    void draw();

    const StateQuery populate_state_query() const;
};
