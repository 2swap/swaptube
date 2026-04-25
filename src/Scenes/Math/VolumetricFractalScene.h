#pragma once

#include "../Scene.h"
#include "../../Host_Device_Shared/vec.h"
#include <complex>

class VolumetricScene : public Scene {
private:
    float* voxel_data;
public:
    VolumetricScene(const vec2& dimensions = vec2(1,1));
    const StateQuery populate_state_query() const override;
    void load_voxels(const int width, const int height, const int depth);
    void initialize_voxel_grid();
    bool check_if_data_changed() const override;
    void draw() override;
    void change_data() override;
    void mark_data_unchanged() override;
    ~VolumetricScene();
};
