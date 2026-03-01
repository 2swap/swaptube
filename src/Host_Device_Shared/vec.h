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
    float u, i, j, k;
    HOST_DEVICE quat(float u, float i, float j, float k) : u(u), i(i), j(j), k(k) {}
    HOST_DEVICE quat() {}
};

// Common operators
HOST_DEVICE inline vec2 operator+(const vec2& a, const vec2& b) { return vec2{ a.x + b.x, a.y + b.y }; }
HOST_DEVICE inline vec3 operator+(const vec3& a, const vec3& b) { return vec3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
HOST_DEVICE inline vec4 operator+(const vec4& a, const vec4& b) { return vec4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
HOST_DEVICE inline quat operator+(const quat& a, const quat& b) { return quat{ a.u + b.u, a.i + b.i, a.j + b.j, a.k + b.k }; }

HOST_DEVICE inline vec2& operator+=(vec2& a, const vec2& b) { a.x += b.x; a.y += b.y; return a; }
HOST_DEVICE inline vec3& operator+=(vec3& a, const vec3& b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
HOST_DEVICE inline vec4& operator+=(vec4& a, const vec4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
HOST_DEVICE inline quat& operator+=(quat& a, const quat& b) { a.u += b.u; a.i += b.i; a.j += b.j; a.k += b.k; return a; }

HOST_DEVICE inline vec2 operator-(const vec2& a, const vec2& b) { return vec2{ a.x - b.x, a.y - b.y }; }
HOST_DEVICE inline vec3 operator-(const vec3& a, const vec3& b) { return vec3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
HOST_DEVICE inline vec4 operator-(const vec4& a, const vec4& b) { return vec4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
HOST_DEVICE inline quat operator-(const quat& a, const quat& b) { return quat{ a.u - b.u, a.i - b.i, a.j - b.j, a.k - b.k }; }

HOST_DEVICE inline vec2 operator-(const vec2& v) { return vec2{ -v.x, -v.y }; }
HOST_DEVICE inline vec3 operator-(const vec3& v) { return vec3{ -v.x, -v.y, -v.z }; }
HOST_DEVICE inline vec4 operator-(const vec4& v) { return vec4{ -v.x, -v.y, -v.z, -v.w }; }
HOST_DEVICE inline quat operator-(const quat& q) { return quat{ -q.u, -q.i, -q.j, -q.k }; }

HOST_DEVICE inline vec2& operator-=(vec2& a, const vec2& b) { a.x -= b.x; a.y -= b.y; return a; }
HOST_DEVICE inline vec3& operator-=(vec3& a, const vec3& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
HOST_DEVICE inline vec4& operator-=(vec4& a, const vec4& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
HOST_DEVICE inline quat& operator-=(quat& a, const quat& b) { a.u -= b.u; a.i -= b.i; a.j -= b.j; a.k -= b.k; return a; }

HOST_DEVICE inline vec2 operator*(const vec2& a, float scalar) { return vec2{ a.x * scalar, a.y * scalar }; }
HOST_DEVICE inline vec3 operator*(const vec3& a, float scalar) { return vec3{ a.x * scalar, a.y * scalar, a.z * scalar }; }
HOST_DEVICE inline vec4 operator*(const vec4& a, float scalar) { return vec4{ a.x * scalar, a.y * scalar, a.z * scalar, a.w * scalar }; }
HOST_DEVICE inline quat operator*(const quat& a, float scalar) { return quat{ a.u * scalar, a.i * scalar, a.j * scalar, a.k * scalar }; }

HOST_DEVICE inline vec2& operator*=(vec2& a, float scalar) { a.x *= scalar; a.y *= scalar; return a; }
HOST_DEVICE inline vec3& operator*=(vec3& a, float scalar) { a.x *= scalar; a.y *= scalar; a.z *= scalar; return a; }
HOST_DEVICE inline vec4& operator*=(vec4& a, float scalar) { a.x *= scalar; a.y *= scalar; a.z *= scalar; a.w *= scalar; return a; }
HOST_DEVICE inline quat& operator*=(quat& a, float scalar) { a.u *= scalar; a.i *= scalar; a.j *= scalar; a.k *= scalar; return a; }

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
        a.u * b.u - a.i * b.i - a.j * b.j - a.k * b.k,
        a.u * b.i + a.i * b.u + a.j * b.k - a.k * b.j,
        a.u * b.j - a.i * b.k + a.j * b.u + a.k * b.i,
        a.u * b.k + a.i * b.j - a.j * b.i + a.k * b.u
    };
}

HOST_DEVICE inline quat& operator*=(quat& a, const quat& b) { return a = a * b; }

HOST_DEVICE inline vec2 operator/(const vec2& a, const vec2& b) { return vec2{ a.x / b.x, a.y / b.y }; }
HOST_DEVICE inline vec3 operator/(const vec3& a, const vec3& b) { return vec3{ a.x / b.x, a.y / b.y, a.z / b.z }; }
HOST_DEVICE inline vec4 operator/(const vec4& a, const vec4& b) { return vec4{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w }; }

HOST_DEVICE inline vec2& operator/=(vec2& a, const vec2& b) { a.x /= b.x; a.y /= b.y; return a; }
HOST_DEVICE inline vec3& operator/=(vec3& a, const vec3& b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; return a; }
HOST_DEVICE inline vec4& operator/=(vec4& a, const vec4& b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w; return a; }

HOST_DEVICE inline float length(const vec2& v) { return sqrtf(v.x * v.x + v.y * v.y); }
HOST_DEVICE inline float length(const vec3& v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }
HOST_DEVICE inline float length(const vec4& v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w); }
HOST_DEVICE inline float length(const quat& q) { return sqrtf(q.u * q.u + q.i * q.i + q.j * q.j + q.k * q.k); }

HOST_DEVICE inline vec2 normalize(const vec2& v) { float len = length(v); return vec2{ v.x / len, v.y / len }; }
HOST_DEVICE inline vec3 normalize(const vec3& v) { float len = length(v); return vec3{ v.x / len, v.y / len, v.z / len }; }
HOST_DEVICE inline vec4 normalize(const vec4& v) { float len = length(v); return vec4{ v.x / len, v.y / len, v.z / len, v.w / len }; }
HOST_DEVICE inline quat normalize(const quat& q) { float len = length(q); return quat{ q.u / len, q.i / len, q.j / len, q.k / len }; }

HOST_DEVICE inline float dot(const vec2& a, const vec2& b) { return a.x * b.y + a.y * b.x; }
HOST_DEVICE inline float dot(const vec3& a, const vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HOST_DEVICE inline float dot(const vec4& a, const vec4& b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

HOST_DEVICE inline bool hasnan(const vec2& v) { return std::isnan(v.x) || std::isnan(v.y); }
HOST_DEVICE inline bool hasnan(const vec3& v) { return std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z); }
HOST_DEVICE inline bool hasnan(const vec4& v) { return std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z) || std::isnan(v.w); }
HOST_DEVICE inline bool hasnan(const quat& q) { return std::isnan(q.u) || std::isnan(q.i) || std::isnan(q.j) || std::isnan(q.k); }

HOST_DEVICE inline vec3 integerize(const vec3& v) { return vec3{ floorf(v.x), floorf(v.y), floorf(v.z) }; }

HOST_DEVICE inline quat conjugate(const quat& q) { return quat{ q.u, -q.i, -q.j, -q.k }; }

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

HOST_DEVICE inline vec3 rotate_vector(const vec3& v, const quat& q) {
    quat v_quat{ 0.0f, v.x, v.y, v.z };
    quat q_conj = conjugate(q);
    quat rotated = q * v_quat * q_conj;
    return vec3{ rotated.i, rotated.j, rotated.k };
}

SHARED_FILE_SUFFIX
