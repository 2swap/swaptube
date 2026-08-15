#pragma once

#include "../Scene.h"
#include "../../Host_Device_Shared/vec.h"
#include <complex>

class VoxelScene : public Scene {
private:
    float* voxel_data;
public:
    VoxelScene(const vec2& dimensions = vec2(1,1));
    void load_voxels(const int width, const int height, const int depth);
    void initialize_voxel_grid();
    void draw() override;
    ~VoxelScene();
};
