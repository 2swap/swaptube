#pragma once

#include "../Common/ThreeDimensionScene.h"

class ThreeDAlgebraScene : public ThreeDimensionScene {
public:
    ThreeDAlgebraScene(const vec2& dimensions = vec2(1, 1));

    void draw() override;

private:
    // vec4 multiply(const vec4& p, const vec4& q,
    //               const vec4& xx, const vec4& xy, const vec4& xz,
    //               const vec4& yy, const vec4& yz, const vec4& zz) const;
    vec4 multiply(const vec4& p, const vec4& q,
                  const vec4& xx, const vec4& xy, const vec4& xz,
                  const vec4& yy, const vec4& yz, const vec4& zz) const;

    vec3 project(const vec4& p) const;
};

