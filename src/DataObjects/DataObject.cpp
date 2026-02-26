#include "DataObject.h"

void DataObject::mark_unchanged(){
    updated_since_last_scene_query = false;
}
bool DataObject::has_been_updated_since_last_scene_query() const{
    return updated_since_last_scene_query;
}
void DataObject::mark_updated() {
    updated_since_last_scene_query = true;
}
