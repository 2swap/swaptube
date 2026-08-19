#pragma once

#include "../Common/ThreeDimensionScene.h"

class ThreeDAlgebraScene : public ThreeDimensionScene {
public:
    ThreeDAlgebraScene(const vec2& dimensions = vec2(1, 1));

    void draw() override;

private:
    vec3 multiply(const vec3& p, const vec3& q,
                  const vec3& xx, const vec3& xy, const vec3& xz,
                  const vec3& yy, const vec3& yz, const vec3& zz) const;
};
