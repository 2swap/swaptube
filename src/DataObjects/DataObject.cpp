class DataObject {
public:
    // Initialize to updated, since on the first step we wish to redraw the scene.
    bool has_been_updated_since_last_scene_query(){
        bool tmp = updated_since_last_scene_query;
        updated_since_last_scene_query = false;
        return tmp;
    }
private:
    bool updated_since_last_scene_query = true;
protected:
    // Called by child classes when the underlying data is changed.
    void mark_updated() {
        updated_since_last_scene_query = true;
    }
};
