// Definition of data structures in the style of GLM
// Including vec2, vec3, vec4, and quat

#pragma once
#include <cmath>
#include "shared_precompiler_directives.h"

SHARED_FILE_PREFIX

struct vec4 {
    float x, y, z, w;
    HOST_DEVICE vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    HOST_DEVICE vec4(float v) : x(v), y(v), z(v), w(v) {}
    HOST_DEVICE vec4() {}
};
struct vec3 {
    float x, y, z;
    HOST_DEVICE vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    HOST_DEVICE vec3(float v) : x(v), y(v), z(v) {}
    HOST_DEVICE vec3() {}
    HOST_DEVICE vec3(const vec4& v) : x(v.x), y(v.y), z(v.z) {}
};
struct vec2 {
    float x, y;
    HOST_DEVICE vec2(float x, float y) : x(x), y(y) {}
    HOST_DEVICE vec2(float v) : x(v), y(v) {}
    HOST_DEVICE vec2() {}
    HOST_DEVICE vec2(const vec3& v) : x(v.x), y(v.y) {}
    HOST_DEVICE vec2(const vec4& v) : x(v.x), y(v.y) {}
};
struct quat {
    float x, y, z, w;
    HOST_DEVICE quat(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    HOST_DEVICE quat() {}
};

// Common operators
HOST_DEVICE inline vec2 operator+(const vec2& a, const vec2& b) { return vec2{ a.x + b.x, a.y + b.y }; }
HOST_DEVICE inline vec3 operator+(const vec3& a, const vec3& b) { return vec3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
HOST_DEVICE inline vec4 operator+(const vec4& a, const vec4& b) { return vec4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
HOST_DEVICE inline quat operator+(const quat& a, const quat& b) { return quat{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }

HOST_DEVICE inline vec2& operator+=(vec2& a, const vec2& b) { a.x += b.x; a.y += b.y; return a; }
HOST_DEVICE inline vec3& operator+=(vec3& a, const vec3& b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
HOST_DEVICE inline vec4& operator+=(vec4& a, const vec4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
HOST_DEVICE inline quat& operator+=(quat& a, const quat& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }

HOST_DEVICE inline vec2 operator-(const vec2& a, const vec2& b) { return vec2{ a.x - b.x, a.y - b.y }; }
HOST_DEVICE inline vec3 operator-(const vec3& a, const vec3& b) { return vec3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
HOST_DEVICE inline vec4 operator-(const vec4& a, const vec4& b) { return vec4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
HOST_DEVICE inline quat operator-(const quat& a, const quat& b) { return quat{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }

HOST_DEVICE inline vec2& operator-=(vec2& a, const vec2& b) { a.x -= b.x; a.y -= b.y; return a; }
HOST_DEVICE inline vec3& operator-=(vec3& a, const vec3& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
HOST_DEVICE inline vec4& operator-=(vec4& a, const vec4& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
HOST_DEVICE inline quat& operator-=(quat& a, const quat& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }

HOST_DEVICE inline vec2 operator*(const vec2& a, float scalar) { return vec2{ a.x * scalar, a.y * scalar }; }
HOST_DEVICE inline vec3 operator*(const vec3& a, float scalar) { return vec3{ a.x * scalar, a.y * scalar, a.z * scalar }; }
HOST_DEVICE inline vec4 operator*(const vec4& a, float scalar) { return vec4{ a.x * scalar, a.y * scalar, a.z * scalar, a.w * scalar }; }
HOST_DEVICE inline quat operator*(const quat& a, float scalar) { return quat{ a.x * scalar, a.y * scalar, a.z * scalar, a.w * scalar }; }

HOST_DEVICE inline vec2& operator*=(vec2& a, float scalar) { a.x *= scalar; a.y *= scalar; return a; }
HOST_DEVICE inline vec3& operator*=(vec3& a, float scalar) { a.x *= scalar; a.y *= scalar; a.z *= scalar; return a; }
HOST_DEVICE inline vec4& operator*=(vec4& a, float scalar) { a.x *= scalar; a.y *= scalar; a.z *= scalar; a.w *= scalar; return a; }
HOST_DEVICE inline quat& operator*=(quat& a, float scalar) { a.x *= scalar; a.y *= scalar; a.z *= scalar; a.w *= scalar; return a; }

HOST_DEVICE inline vec2 operator*(float scalar, const vec2& a) { return a * scalar; }
HOST_DEVICE inline vec3 operator*(float scalar, const vec3& a) { return a * scalar; }
HOST_DEVICE inline vec4 operator*(float scalar, const vec4& a) { return a * scalar; }
HOST_DEVICE inline quat operator*(float scalar, const quat& a) { return a * scalar; }

HOST_DEVICE inline vec2 operator/(const vec2& a, float scalar) { return a * (1.0f / scalar); }
HOST_DEVICE inline vec3 operator/(const vec3& a, float scalar) { return a * (1.0f / scalar); }
HOST_DEVICE inline vec4 operator/(const vec4& a, float scalar) { return a * (1.0f / scalar); }
HOST_DEVICE inline quat operator/(const quat& a, float scalar) { return a * (1.0f / scalar); }

HOST_DEVICE inline vec2& operator/=(vec2& a, float scalar) { return a *= (1.0f / scalar); }
HOST_DEVICE inline vec3& operator/=(vec3& a, float scalar) { return a *= (1.0f / scalar); }
HOST_DEVICE inline vec4& operator/=(vec4& a, float scalar) { return a *= (1.0f / scalar); }
HOST_DEVICE inline quat& operator/=(quat& a, float scalar) { return a *= (1.0f / scalar); }

HOST_DEVICE inline vec2 operator*(const vec2& a, const vec2& b) { return vec2{ a.x * b.x, a.y * b.y }; }
HOST_DEVICE inline vec3 operator*(const vec3& a, const vec3& b) { return vec3{ a.x * b.x, a.y * b.y, a.z * b.z }; }
HOST_DEVICE inline vec4 operator*(const vec4& a, const vec4& b) { return vec4{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w }; }

HOST_DEVICE inline vec2& operator*=(vec2& a, const vec2& b) { a.x *= b.x; a.y *= b.y; return a; }
HOST_DEVICE inline vec3& operator*=(vec3& a, const vec3& b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
HOST_DEVICE inline vec4& operator*=(vec4& a, const vec4& b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }

HOST_DEVICE inline quat operator*(const quat& a, const quat& b) {
    return quat{
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    };
}

HOST_DEVICE inline vec2 operator/(const vec2& a, const vec2& b) { return vec2{ a.x / b.x, a.y / b.y }; }
HOST_DEVICE inline vec3 operator/(const vec3& a, const vec3& b) { return vec3{ a.x / b.x, a.y / b.y, a.z / b.z }; }
HOST_DEVICE inline vec4 operator/(const vec4& a, const vec4& b) { return vec4{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w }; }

HOST_DEVICE inline vec2& operator/=(vec2& a, const vec2& b) { a.x /= b.x; a.y /= b.y; return a; }
HOST_DEVICE inline vec3& operator/=(vec3& a, const vec3& b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; return a; }
HOST_DEVICE inline vec4& operator/=(vec4& a, const vec4& b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w; return a; }

HOST_DEVICE inline float length(const vec2& v) { return sqrtf(v.x * v.x + v.y * v.y); }
HOST_DEVICE inline float length(const vec3& v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }
HOST_DEVICE inline float length(const vec4& v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w); }
HOST_DEVICE inline float length(const quat& q) { return sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w); }

HOST_DEVICE inline vec2 normalize(const vec2& v) { float len = length(v); return vec2{ v.x / len, v.y / len }; }
HOST_DEVICE inline vec3 normalize(const vec3& v) { float len = length(v); return vec3{ v.x / len, v.y / len, v.z / len }; }
HOST_DEVICE inline vec4 normalize(const vec4& v) { float len = length(v); return vec4{ v.x / len, v.y / len, v.z / len, v.w / len }; }
HOST_DEVICE inline quat normalize(const quat& q) { float len = length(q); return quat{ q.x / len, q.y / len, q.z / len, q.w / len }; }

HOST_DEVICE inline float dot(const vec2& a, const vec2& b) { return a.x * b.y + a.y * b.x; }
HOST_DEVICE inline float dot(const vec3& a, const vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HOST_DEVICE inline float dot(const vec4& a, const vec4& b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
HOST_DEVICE inline float dot(const quat& a, const quat& b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

HOST_DEVICE inline bool hasnan(const vec2& v) { return std::isnan(v.x) || std::isnan(v.y); }
HOST_DEVICE inline bool hasnan(const vec3& v) { return std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z); }
HOST_DEVICE inline bool hasnan(const vec4& v) { return std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z) || std::isnan(v.w); }
HOST_DEVICE inline bool hasnan(const quat& q) { return std::isnan(q.x) || std::isnan(q.y) || std::isnan(q.z) || std::isnan(q.w); }

HOST_DEVICE inline vec3 integerize(const vec3& v) { return vec3{ floorf(v.x), floorf(v.y), floorf(v.z) }; }

HOST_DEVICE inline vec2 clamp(const vec2& v, const vec2& min, const vec2& max) {
    return vec2{
        fmaxf(min.x, fminf(max.x, v.x)),
        fmaxf(min.y, fminf(max.y, v.y))
    };
}
HOST_DEVICE inline vec3 clamp(const vec3& v, const vec3& min, const vec3& max) {
    return vec3{
        fmaxf(min.x, fminf(max.x, v.x)),
        fmaxf(min.y, fminf(max.y, v.y)),
        fmaxf(min.z, fminf(max.z, v.z))
    };
}
HOST_DEVICE inline vec4 clamp(const vec4& v, const vec4& min, const vec4& max) {
    return vec4{
        fmaxf(min.x, fminf(max.x, v.x)),
        fmaxf(min.y, fminf(max.y, v.y)),
        fmaxf(min.z, fminf(max.z, v.z)),
        fmaxf(min.w, fminf(max.w, v.w))
    };
}

HOST_DEVICE inline vec3 cross(const vec3& a, const vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

// Rotation of a vec3 by a quat
HOST_DEVICE inline vec3 operator*(const quat& q, const vec3& v)
{
    // Convert the quaternion to a rotation matrix
    float xx = q.x * q.x;
    float yy = q.y * q.y;
    float zz = q.z * q.z;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yz = q.y * q.z;
    float wx = q.w * q.x;
    float wy = q.w * q.y;
    float wz = q.w * q.z;

    // Rotate the vector
    return {
        (1.0f - 2.0f * (yy + zz)) * v.x + (2.0f * (xy - wz)) * v.y + (2.0f * (xz + wy)) * v.z,
        (2.0f * (xy + wz)) * v.x + (1.0f - 2.0f * (xx + zz)) * v.y + (2.0f * (yz - wx)) * v.z,
        (2.0f * (xz - wy)) * v.x + (2.0f * (yz + wx)) * v.y + (1.0f - 2.0f * (xx + yy)) * v.z
    };
}

SHARED_FILE_SUFFIX
