#include "../Scene.h"
#include <gtk/gtk.h>
#include <gtksourceview/gtksource.h>

class CodeScene : public Scene {
public:
    CodeScene(const std::string& source, const vec2& dimensions = vec2(1,1));

    bool check_if_data_changed() const;
    void mark_data_unchanged();
    void change_data();

    void draw();

    const StateQuery populate_state_query() const;
private:
    std::string code;

    GtkSource::View* view = nullptr;
    Glib::RefPtr<GtkSource::Buffer> buffer;
};
