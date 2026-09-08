#include "../Scenes/Math/GeometryScene.h"
#include <cmath>

// Construction of Richard Schwartz's Outer Billiards Unbounded Orbit.
void render_video() {
    GeometryScene scene;
    scene.manager.set({
        {"zoom", "-0.2"},
        {"ticks_opacity", "0"},
    });

    const float phi = (1.f + std::sqrt(5.f)) / 2.f;
    auto vtx = [](float radius, float angle_deg) {
        const float a = angle_deg * float(M_PI) / 180.f;
        return vec2(radius * std::cos(a), radius * std::sin(a));
    };
    // Intersection of the infinite line through (p,q) with the line through (u,v).
    auto meet = [](vec2 p, vec2 q, vec2 u, vec2 v) {
        const vec2 d1 = q - p, d2 = v - u;
        const float t = ((u.x - p.x) * d2.y - (u.y - p.y) * d2.x) / (d1.x * d2.y - d1.y * d2.x);
        return p + d1 * t;
    };

    // Outer pentagon: A at the top, then B C D E clockwise.
    const float R = 2.0f;
    const vec2 A = vtx(R,  90), B = vtx(R,  18), C = vtx(R, -54), D = vtx(R, -126), E = vtx(R, 162);

    // Inner pentagon (the diagonal intersections): a at the bottom, then b c d e clockwise.
    const float r = R / (phi * phi);
    const vec2 a = vtx(r, -90), b = vtx(r, -162), c = vtx(r, 126), d = vtx(r, 54), e = vtx(r, -18);

    // --- Outer pentagon: each vertex drops in with the edge that reaches it ---
    stage_macroblock(SilenceBlock(1));
    scene.construction.add(GeometricPoint(A, "A", 1.0f, false, ""));
    scene.construction.add(GeometricPoint(B, "B", 1.0f, false, ""));
    scene.construction.add(GeometricLine(A, B, "AB"));
    scene.construction.add(GeometricPoint(C, "C", 1.0f, false, ""));
    scene.construction.add(GeometricLine(B, C, "BC"));
    scene.construction.add(GeometricPoint(D, "D", 1.0f, false, ""));
    scene.construction.add(GeometricLine(C, D, "CD"));
    scene.construction.add(GeometricPoint(E, "E", 1.0f, false, ""));
    scene.construction.add(GeometricLine(D, E, "DE"));
    scene.construction.add(GeometricLine(E, A, "EA"));
    scene.render_microblock();

    // --- The five diagonals, drawn as one continuous pentagram stroke ---
    stage_macroblock(SilenceBlock(1));
    scene.construction.add(GeometricLine(A, C, "AC"));
    scene.construction.add(GeometricLine(C, E, "CE"));
    scene.construction.add(GeometricPoint(d, "d", 0.7f, false, ""));
    scene.construction.add(GeometricLine(E, B, "EB"));
    scene.construction.add(GeometricPoint(e, "e", 0.7f, false, ""));
    scene.construction.add(GeometricPoint(a, "a", 0.7f, false, ""));
    scene.construction.add(GeometricLine(B, D, "BD"));
    scene.construction.add(GeometricPoint(b, "b", 0.7f, false, ""));
    scene.construction.add(GeometricPoint(c, "c", 0.7f, false, ""));
    scene.construction.add(GeometricLine(D, A, "DA"));
    scene.render_microblock();

    // --- Extend ce to outer edge BC (point y), and ca to outer edge CD (point z) ---
    const vec2 y = meet(c, e, B, C);
    const vec2 z = meet(c, a, C, D);
    stage_macroblock(SilenceBlock(1));
    scene.construction.add(GeometricPoint(y, "y", 0.7f, false, ""));
    scene.construction.add(GeometricLine(c, y, "cy"));
    scene.construction.add(GeometricPoint(z, "z", 0.7f, false, ""));
    scene.construction.add(GeometricLine(c, z, "cz"));
    scene.render_microblock();

    // --- Connect y-z; w is where it crosses diagonal CE ---
    const vec2 w = meet(y, z, C, E);
    stage_macroblock(SilenceBlock(1));
    scene.construction.add(GeometricLine(y, z, "yz"));
    scene.construction.add(GeometricPoint(w, "w", 0.7f, false, ""));
    scene.render_microblock();

    // --- Connect a-y and w-e; P is their intersection ---
    const vec2 P = meet(a, y, w, e);
    stage_macroblock(SilenceBlock(1));
    scene.construction.add(GeometricLine(a, y, "ay"));
    scene.construction.add(GeometricLine(w, e, "we"));
    scene.construction.add(GeometricPoint(P, "P", 1.0f, false, ""));
    scene.construction.add(GeometricLine(A, d, "Ad"));
    scene.construction.add(GeometricLine(E, a, "Ea"));
    scene.render_microblock();

    // --- Strip everything back to lines Ad, ad, EA, Ea and the point P ---
    // Ad and Ea are the A->d and E->a stubs of diagonals AC and CE, so the
    // full diagonals fade out and the stubs grow back in.
    const vector<string> dead_lines = {"AB","BC","CD","DE","AC","CE","EB","BD","DA","cy","cz","yz","ay","we"};
    const vector<string> dead_points = {"A","B","C","D","E","a","b","c","d","e","y","z","w"};
    stage_macroblock(SilenceBlock(1));
    for (const string& id : dead_lines)  scene.construction.fade_line(id);
    for (const string& id : dead_points) scene.construction.fade_point(id);
    scene.construction.add(GeometricLine(a, d, "ad"));
    scene.render_microblock();

    // Hold on the final figure.
    stage_macroblock(SilenceBlock(1));
    scene.render_microblock();
}
