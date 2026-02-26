#pragma once

class DataObject {
public:
    // Initialize to updated, since on the first step we wish to redraw the scene.
    void mark_unchanged();
    bool has_been_updated_since_last_scene_query() const;
private:
    bool updated_since_last_scene_query = true;
protected:
    // Called by child classes when the underlying data is changed.
    void mark_updated();
};
