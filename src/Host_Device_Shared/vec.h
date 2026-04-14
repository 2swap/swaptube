// Definition of data structures in the style of GLM
// Including vec2, vec3, vec4, ivec2, ivec3, ivec4 and quat

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
struct ivec4 {
    int x, y, z, w;
    HOST_DEVICE ivec4(int x, int y, int z, int w) : x(x), y(y), z(z), w(w) {}
    HOST_DEVICE ivec4(int v) : x(v), y(v), z(v), w(v) {}
    HOST_DEVICE ivec4(const vec4& v) : x(floorf(v.x)), y(floorf(v.y)), z(floorf(v.z)), w(floorf(v.w)) {}
    HOST_DEVICE ivec4() {}
};
struct vec3 {
    float x, y, z;
    HOST_DEVICE vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    HOST_DEVICE vec3(float v) : x(v), y(v), z(v) {}
    HOST_DEVICE vec3() {}
    HOST_DEVICE vec3(const vec4& v) : x(v.x), y(v.y), z(v.z) {}
};
struct ivec3 {
    int x, y, z;
    HOST_DEVICE ivec3(int x, int y, int z) : x(x), y(y), z(z) {}
    HOST_DEVICE ivec3(int v) : x(v), y(v), z(v) {}
    HOST_DEVICE ivec3(const vec3& v) : x(floorf(v.x)), y(floorf(v.y)), z(floorf(v.z)) {}
    HOST_DEVICE ivec3() {}
    HOST_DEVICE ivec3(const ivec4& v) : x(v.x), y(v.y), z(v.z) {}
};
struct vec2 {
    float x, y;
    HOST_DEVICE vec2(float x, float y) : x(x), y(y) {}
    HOST_DEVICE vec2(float v) : x(v), y(v) {}
    HOST_DEVICE vec2() {}
    HOST_DEVICE vec2(const vec3& v) : x(v.x), y(v.y) {}
    HOST_DEVICE vec2(const vec4& v) : x(v.x), y(v.y) {}
};
struct ivec2 {
    int x, y;
    HOST_DEVICE ivec2(int x, int y) : x(x), y(y) {}
    HOST_DEVICE ivec2(int v) : x(v), y(v) {}
    HOST_DEVICE ivec2(const vec2& v) : x(floorf(v.x)), y(floorf(v.y)) {}
    HOST_DEVICE ivec2() {}
    HOST_DEVICE ivec2(const ivec3& v) : x(v.x), y(v.y) {}
    HOST_DEVICE ivec2(const ivec4& v) : x(v.x), y(v.y) {}
};

struct quat {
    float u, i, j, k;
    HOST_DEVICE quat(float u, float i, float j, float k) : u(u), i(i), j(j), k(k) {}
    HOST_DEVICE quat() {}
};

struct mat2 {
    vec2 a, b;
    HOST_DEVICE mat2(const vec2& a, const vec2& b) : a(a), b(b) {}
    HOST_DEVICE mat2(float v) : a(v, 0), b(0, v) {}
    HOST_DEVICE mat2() {}
};
struct mat3 {
    vec3 a, b, c;
    HOST_DEVICE mat3(const vec3& a, const vec3& b, const vec3& c) : a(a), b(b), c(c) {}
    HOST_DEVICE mat3(float v) : a(v, 0, 0), b(0, v, 0), c(0, 0, v) {}
    HOST_DEVICE mat3() {}
};
struct mat4 {
    vec4 a, b, c, d;
    HOST_DEVICE mat4(const vec4& a, const vec4& b, const vec4& c, const vec4& d) : a(a), b(b), c(c), d(d) {}
    HOST_DEVICE mat4(float v) : a(v, 0, 0, 0), b(0, v, 0, 0), c(0, 0, v, 0), d(0, 0, 0, v) {}
    HOST_DEVICE mat4() {}
};

// Common operators
HOST_DEVICE inline vec2 operator+(const vec2& a, const vec2& b) { return vec2{ a.x + b.x, a.y + b.y }; }
HOST_DEVICE inline vec3 operator+(const vec3& a, const vec3& b) { return vec3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
HOST_DEVICE inline vec4 operator+(const vec4& a, const vec4& b) { return vec4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
HOST_DEVICE inline ivec2 operator+(const ivec2& a, const ivec2& b) { return ivec2{ a.x + b.x, a.y + b.y }; }
HOST_DEVICE inline ivec3 operator+(const ivec3& a, const ivec3& b) { return ivec3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
HOST_DEVICE inline ivec4 operator+(const ivec4& a, const ivec4& b) { return ivec4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
HOST_DEVICE inline quat operator+(const quat& a, const quat& b) { return quat{ a.u + b.u, a.i + b.i, a.j + b.j, a.k + b.k }; }
HOST_DEVICE inline mat2 operator+(const mat2& a, const mat2& b) { return mat2{ a.a + b.a, a.b + b.b }; }
HOST_DEVICE inline mat3 operator+(const mat3& a, const mat3& b) { return mat3{ a.a + b.a, a.b + b.b, a.c + b.c }; }
HOST_DEVICE inline mat4 operator+(const mat4& a, const mat4& b) { return mat4{ a.a + b.a, a.b + b.b, a.c + b.c, a.d + b.d }; }

HOST_DEVICE inline vec2& operator+=(vec2& a, const vec2& b) { a.x += b.x; a.y += b.y; return a; }
HOST_DEVICE inline vec3& operator+=(vec3& a, const vec3& b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
HOST_DEVICE inline vec4& operator+=(vec4& a, const vec4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
HOST_DEVICE inline ivec2& operator+=(ivec2& a, const ivec2& b) { a.x += b.x; a.y += b.y; return a; }
HOST_DEVICE inline ivec3& operator+=(ivec3& a, const ivec3& b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
HOST_DEVICE inline ivec4& operator+=(ivec4& a, const ivec4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
HOST_DEVICE inline quat& operator+=(quat& a, const quat& b) { a.u += b.u; a.i += b.i; a.j += b.j; a.k += b.k; return a; }
HOST_DEVICE inline mat2& operator+=(mat2& a, const mat2& b) { a.a += b.a; a.b += b.b; return a; }
HOST_DEVICE inline mat3& operator+=(mat3& a, const mat3& b) { a.a += b.a; a.b += b.b; a.c += b.c; return a; }
HOST_DEVICE inline mat4& operator+=(mat4& a, const mat4& b) { a.a += b.a; a.b += b.b; a.c += b.c; a.d += b.d; return a; }

HOST_DEVICE inline vec2 operator-(const vec2& a, const vec2& b) { return vec2{ a.x - b.x, a.y - b.y }; }
HOST_DEVICE inline vec3 operator-(const vec3& a, const vec3& b) { return vec3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
HOST_DEVICE inline vec4 operator-(const vec4& a, const vec4& b) { return vec4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
HOST_DEVICE inline ivec2 operator-(const ivec2& a, const ivec2& b) { return ivec2{ a.x - b.x, a.y - b.y }; }
HOST_DEVICE inline ivec3 operator-(const ivec3& a, const ivec3& b) { return ivec3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
HOST_DEVICE inline ivec4 operator-(const ivec4& a, const ivec4& b) { return ivec4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
HOST_DEVICE inline quat operator-(const quat& a, const quat& b) { return quat{ a.u - b.u, a.i - b.i, a.j - b.j, a.k - b.k }; }
HOST_DEVICE inline mat2 operator-(const mat2& a, const mat2& b) { return mat2{ a.a - b.a, a.b - b.b }; }
HOST_DEVICE inline mat3 operator-(const mat3& a, const mat3& b) { return mat3{ a.a - b.a, a.b - b.b, a.c - b.c }; }
HOST_DEVICE inline mat4 operator-(const mat4& a, const mat4& b) { return mat4{ a.a - b.a, a.b - b.b, a.c - b.c, a.d - b.d }; }

HOST_DEVICE inline vec2 operator-(const vec2& v) { return vec2{ -v.x, -v.y }; }
HOST_DEVICE inline vec3 operator-(const vec3& v) { return vec3{ -v.x, -v.y, -v.z }; }
HOST_DEVICE inline vec4 operator-(const vec4& v) { return vec4{ -v.x, -v.y, -v.z, -v.w }; }
HOST_DEVICE inline ivec2 operator-(const ivec2& v) { return ivec2{ -v.x, -v.y }; }
HOST_DEVICE inline ivec3 operator-(const ivec3& v) { return ivec3{ -v.x, -v.y, -v.z }; }
HOST_DEVICE inline ivec4 operator-(const ivec4& v) { return ivec4{ -v.x, -v.y, -v.z, -v.w }; }
HOST_DEVICE inline quat operator-(const quat& q) { return quat{ -q.u, -q.i, -q.j, -q.k }; }
HOST_DEVICE inline mat2 operator-(const mat2& m) { return mat2{ -m.a, -m.b }; }
HOST_DEVICE inline mat3 operator-(const mat3& m) { return mat3{ -m.a, -m.b, -m.c }; }
HOST_DEVICE inline mat4 operator-(const mat4& m) { return mat4{ -m.a, -m.b, -m.c, -m.d }; }

HOST_DEVICE inline vec2& operator-=(vec2& a, const vec2& b) { a.x -= b.x; a.y -= b.y; return a; }
HOST_DEVICE inline vec3& operator-=(vec3& a, const vec3& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
HOST_DEVICE inline vec4& operator-=(vec4& a, const vec4& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
HOST_DEVICE inline ivec2& operator-=(ivec2& a, const ivec2& b) { a.x -= b.x; a.y -= b.y; return a; }
HOST_DEVICE inline ivec3& operator-=(ivec3& a, const ivec3& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
HOST_DEVICE inline ivec4& operator-=(ivec4& a, const ivec4& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
HOST_DEVICE inline quat& operator-=(quat& a, const quat& b) { a.u -= b.u; a.i -= b.i; a.j -= b.j; a.k -= b.k; return a; }
HOST_DEVICE inline mat2& operator-=(mat2& a, const mat2& b) { a.a -= b.a; a.b -= b.b; return a; }
HOST_DEVICE inline mat3& operator-=(mat3& a, const mat3& b) { a.a -= b.a; a.b -= b.b; a.c -= b.c; return a; }
HOST_DEVICE inline mat4& operator-=(mat4& a, const mat4& b) { a.a -= b.a; a.b -= b.b; a.c -= b.c; a.d -= b.d; return a; }

HOST_DEVICE inline vec2 operator*(const vec2& a, float scalar) { return vec2{ a.x * scalar, a.y * scalar }; }
HOST_DEVICE inline vec3 operator*(const vec3& a, float scalar) { return vec3{ a.x * scalar, a.y * scalar, a.z * scalar }; }
HOST_DEVICE inline vec4 operator*(const vec4& a, float scalar) { return vec4{ a.x * scalar, a.y * scalar, a.z * scalar, a.w * scalar }; }
HOST_DEVICE inline ivec2 operator*(const ivec2& a, int scalar) { return ivec2{ a.x * scalar, a.y * scalar }; }
HOST_DEVICE inline ivec3 operator*(const ivec3& a, int scalar) { return ivec3{ a.x * scalar, a.y * scalar, a.z * scalar }; }
HOST_DEVICE inline ivec4 operator*(const ivec4& a, int scalar) { return ivec4{ a.x * scalar, a.y * scalar, a.z * scalar, a.w * scalar }; }
HOST_DEVICE inline quat operator*(const quat& a, float scalar) { return quat{ a.u * scalar, a.i * scalar, a.j * scalar, a.k * scalar }; }

HOST_DEVICE inline vec2& operator*=(vec2& a, float scalar) { a.x *= scalar; a.y *= scalar; return a; }
HOST_DEVICE inline vec3& operator*=(vec3& a, float scalar) { a.x *= scalar; a.y *= scalar; a.z *= scalar; return a; }
HOST_DEVICE inline vec4& operator*=(vec4& a, float scalar) { a.x *= scalar; a.y *= scalar; a.z *= scalar; a.w *= scalar; return a; }
HOST_DEVICE inline ivec2& operator*=(ivec2& a, float scalar) { a.x *= scalar; a.y *= scalar; return a; }
HOST_DEVICE inline ivec3& operator*=(ivec3& a, float scalar) { a.x *= scalar; a.y *= scalar; a.z *= scalar; return a; }
HOST_DEVICE inline ivec4& operator*=(ivec4& a, float scalar) { a.x *= scalar; a.y *= scalar; a.z *= scalar; a.w *= scalar; return a; }
HOST_DEVICE inline quat& operator*=(quat& a, float scalar) { a.u *= scalar; a.i *= scalar; a.j *= scalar; a.k *= scalar; return a; }

HOST_DEVICE inline vec2 operator*(float scalar, const vec2& a) { return a * scalar; }
HOST_DEVICE inline vec3 operator*(float scalar, const vec3& a) { return a * scalar; }
HOST_DEVICE inline vec4 operator*(float scalar, const vec4& a) { return a * scalar; }
HOST_DEVICE inline ivec2 operator*(float scalar, const ivec2& a) { return a * scalar; }
HOST_DEVICE inline ivec3 operator*(float scalar, const ivec3& a) { return a * scalar; }
HOST_DEVICE inline ivec4 operator*(float scalar, const ivec4& a) { return a * scalar; }
HOST_DEVICE inline quat operator*(float scalar, const quat& a) { return a * scalar; }

HOST_DEVICE inline vec2 operator/(const vec2& a, float scalar) { return a * (1.0f / scalar); }
HOST_DEVICE inline vec3 operator/(const vec3& a, float scalar) { return a * (1.0f / scalar); }
HOST_DEVICE inline vec4 operator/(const vec4& a, float scalar) { return a * (1.0f / scalar); }
HOST_DEVICE inline ivec2 operator/(const ivec2& a, float scalar) { return a * (1.0f / scalar); }
HOST_DEVICE inline ivec3 operator/(const ivec3& a, float scalar) { return a * (1.0f / scalar); }
HOST_DEVICE inline ivec4 operator/(const ivec4& a, float scalar) { return a * (1.0f / scalar); }
HOST_DEVICE inline quat operator/(const quat& a, float scalar) { return a * (1.0f / scalar); }

HOST_DEVICE inline vec2& operator/=(vec2& a, float scalar) { return a *= (1.0f / scalar); }
HOST_DEVICE inline vec3& operator/=(vec3& a, float scalar) { return a *= (1.0f / scalar); }
HOST_DEVICE inline vec4& operator/=(vec4& a, float scalar) { return a *= (1.0f / scalar); }
HOST_DEVICE inline ivec2& operator/=(ivec2& a, float scalar) { return a *= (1.0f / scalar); }
HOST_DEVICE inline ivec3& operator/=(ivec3& a, float scalar) { return a *= (1.0f / scalar); }
HOST_DEVICE inline ivec4& operator/=(ivec4& a, float scalar) { return a *= (1.0f / scalar); }
HOST_DEVICE inline quat& operator/=(quat& a, float scalar) { return a *= (1.0f / scalar); }

HOST_DEVICE inline vec2 operator*(const vec2& a, const vec2& b) { return vec2{ a.x * b.x, a.y * b.y }; }
HOST_DEVICE inline vec3 operator*(const vec3& a, const vec3& b) { return vec3{ a.x * b.x, a.y * b.y, a.z * b.z }; }
HOST_DEVICE inline vec4 operator*(const vec4& a, const vec4& b) { return vec4{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w }; }
HOST_DEVICE inline ivec2 operator*(const ivec2& a, const ivec2& b) { return ivec2{ a.x * b.x, a.y * b.y }; }
HOST_DEVICE inline ivec3 operator*(const ivec3& a, const ivec3& b) { return ivec3{ a.x * b.x, a.y * b.y, a.z * b.z }; }
HOST_DEVICE inline ivec4 operator*(const ivec4& a, const ivec4& b) { return ivec4{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w }; }

HOST_DEVICE inline vec2& operator*=(vec2& a, const vec2& b) { a.x *= b.x; a.y *= b.y; return a; }
HOST_DEVICE inline vec3& operator*=(vec3& a, const vec3& b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
HOST_DEVICE inline vec4& operator*=(vec4& a, const vec4& b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
HOST_DEVICE inline ivec2& operator*=(ivec2& a, const ivec2& b) { a.x *= b.x; a.y *= b.y; return a; }
HOST_DEVICE inline ivec3& operator*=(ivec3& a, const ivec3& b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return	a; }
HOST_DEVICE inline ivec4& operator*=(ivec4&	a, const ivec4&	b) {a.x *=	b.x; a.y *=	b.y; a.z *=	b.z; a.w *= b.w; return	a; }

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
HOST_DEVICE inline ivec2 operator/(const ivec2& a, const ivec2& b) { return ivec2{ a.x / b.x, a.y / b.y }; }
HOST_DEVICE inline ivec3 operator/(const ivec3& a, const ivec3& b) { return ivec3{ a.x / b.x, a.y / b.y, a.z / b.z }; }
HOST_DEVICE inline ivec4 operator/(const ivec4& a, const ivec4& b) { return ivec4{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w }; }

HOST_DEVICE inline vec2& operator/=(vec2& a, const vec2& b) { a.x /= b.x; a.y /= b.y; return a; }
HOST_DEVICE inline vec3& operator/=(vec3& a, const vec3& b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; return a; }
HOST_DEVICE inline vec4& operator/=(vec4& a, const vec4& b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w; return a; }
HOST_DEVICE inline ivec2& operator/=(ivec2& a, const ivec2& b) { a.x /= b.x; a.y /= b.y; return a; }
HOST_DEVICE inline ivec3& operator/=(ivec3& a, const ivec3& b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; return	a; }
HOST_DEVICE inline ivec4& operator/=(ivec4&	a, const ivec4&	b) {a.x /=	b.x; a.y /=	b.y; a.z /=	b.z; a.w /= b.w; return	a; }

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
HOST_DEVICE inline float dot(const ivec2& a, const ivec2& b) { return a.x * b.y + a.y * b.x; }
HOST_DEVICE inline float dot(const ivec3& a, const ivec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HOST_DEVICE inline float dot(const ivec4& a, const ivec4& b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

HOST_DEVICE inline bool hasnan(const vec2& v) { return std::isnan(v.x) || std::isnan(v.y); }
HOST_DEVICE inline bool hasnan(const vec3& v) { return std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z); }
HOST_DEVICE inline bool hasnan(const vec4& v) { return std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z) || std::isnan(v.w); }
HOST_DEVICE inline bool hasnan(const ivec2& v) { return std::isnan(v.x) || std::isnan(v.y); }
HOST_DEVICE inline bool hasnan(const ivec3& v) { return std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z); }
HOST_DEVICE inline bool hasnan(const ivec4& v) { return std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z) || std::isnan(v.w); }
HOST_DEVICE inline bool hasnan(const quat& q) { return std::isnan(q.u) || std::isnan(q.i) || std::isnan(q.j) || std::isnan(q.k); }

HOST_DEVICE inline vec3 integerize(const vec3& v) { return vec3{ floorf(v.x), floorf(v.y), floorf(v.z) }; }

HOST_DEVICE inline quat conjugate(const quat& q) { return quat{ q.u, -q.i, -q.j, -q.k }; }

HOST_DEVICE inline mat2 transpose(const mat2& m) { return mat2{ vec2{ m.a.x, m.b.x }, vec2{ m.a.y, m.b.y } }; }
HOST_DEVICE inline mat3 transpose(const mat3& m) { return mat3{ vec3{ m.a.x, m.b.x, m.c.x }, vec3{ m.a.y, m.b.y, m.c.y }, vec3{ m.a.z, m.b.z, m.c.z } }; }
HOST_DEVICE inline mat4 transpose(const mat4& m) { return mat4{ vec4{ m.a.x, m.b.x, m.c.x, m.d.x }, vec4{ m.a.y, m.b.y, m.c.y, m.d.y }, vec4{ m.a.z, m.b.z, m.c.z, m.d.z }, vec4{ m.a.w, m.b.w, m.c.w, m.d.w } }; }

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
HOST_DEVICE inline ivec2 clamp(const ivec2& v, const ivec2& min, const ivec2& max) {
    return ivec2{
        (int) fmaxf(min.x, fminf(max.x, v.x)),
        (int) fmaxf(min.y, fminf(max.y, v.y))
    };
}
HOST_DEVICE inline ivec3 clamp(const ivec3& v, const ivec3& min, const ivec3& max) {
    return ivec3{
        (int) fmaxf(min.x, fminf(max.x, v.x)),
        (int) fmaxf(min.y, fminf(max.y, v.y)),
        (int) fmaxf(min.z, fminf(max.z, v.z))
    };
}
HOST_DEVICE inline ivec4 clamp(const ivec4& v, const ivec4& min, const ivec4& max) {
    return ivec4{
        (int) fmaxf(min.x, fminf(max.x, v.x)),
        (int) fmaxf(min.y, fminf(max.y, v.y)),
        (int) fmaxf(min.z, fminf(max.z, v.z)),
        (int) fmaxf(min.w, fminf(max.w, v.w))
    };
}

HOST_DEVICE inline vec3 cross(const vec3& a, const vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}
HOST_DEVICE inline ivec3 cross(const ivec3& a, const ivec3& b) {
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

HOST_DEVICE inline quat get_quat(const vec3& forward, const vec3& up) {
    vec3 unit_z = normalize(forward);
    vec3 unit_y = normalize(up - dot(up, unit_z) * unit_z);
    vec3 unit_x = cross(unit_y, unit_z);
    float t;
    quat q;
    if (unit_z.z < 0) {
	if (unit_x.x > unit_y.y) {
	    t = 1.0f + unit_x.x - unit_y.y - unit_z.z;
	    q = quat(unit_y.z - unit_z.y, t, unit_x.y + unit_y.x, unit_x.z + unit_z.x);
	} else {
	    t = 1.0f - unit_x.x + unit_y.y - unit_z.z;
	    q = quat(unit_z.x - unit_x.z, unit_x.y + unit_y.x, t, unit_y.z + unit_z.y);
	}
    } else {
	if (unit_x.x < -unit_y.y) {
	    t = 1.0f - unit_x.x - unit_y.y + unit_z.z;
	    q = quat(unit_x.y - unit_y.x, unit_x.z + unit_z.x, unit_y.z + unit_z.y, t);
	} else {
	    t = 1.0f + unit_x.x + unit_y.y + unit_z.z;
	    q = quat(t, unit_y.z - unit_z.y, unit_z.x - unit_x.z, unit_x.y - unit_y.x);
	}
    }
    return q * 0.5f / sqrtf(t);
}

HOST_DEVICE inline vec2 transform(const mat2& m, const vec2& v) {
    mat2 transposed = transpose(m);
    return vec2{
        dot(transposed.a, v),
        dot(transposed.b, v)
    };
}
HOST_DEVICE inline vec3 transform(const mat3& m, const vec3& v) {
    mat3 transposed = transpose(m);
    return vec3{
        dot(transposed.a, v),
        dot(transposed.b, v),
        dot(transposed.c, v)
    };
}
HOST_DEVICE inline vec4 transform(const mat4& m, const vec4& v) {
    mat4 transposed = transpose(m);
    return vec4{
        dot(transposed.a, v),
        dot(transposed.b, v),
        dot(transposed.c, v),
        dot(transposed.d, v)
    };
}

SHARED_FILE_SUFFIX
