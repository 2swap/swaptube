#include "../Scene.h"
#include <gtk/gtk.h>
#include <gtksourceview/gtksource.h>

class CodeScene : public Scene {
public:
    CodeScene(const std::string& source, const vec2& dimensions = vec2(1,1));

    void draw();

    const StateQuery populate_state_query() const;
private:
    std::string code;

    GtkSource::View* view = nullptr;
    Glib::RefPtr<GtkSource::Buffer> buffer;
};
