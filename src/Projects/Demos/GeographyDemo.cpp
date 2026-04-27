#include "../Scenes/Common/ThreeDimensionScene.h"

quat lat_long_to_quat(vec2 lat_long) {
    float lon = lat_long.x;
    float lat = lat_long.y;

    // Convert latitude and longitude from degrees to radians
    float lat_rad = lat * (M_PI / 180.0f);
    float lon_rad = lon * (M_PI / 180.0f);

    // Calculate the quaternion components
    float cy = cos(lon_rad * 0.5f);
    float sy = sin(lon_rad * 0.5f);
    float cp = cos(lat_rad * 0.5f);
    float sp = sin(lat_rad * 0.5f);

    return quat(
        cy * cp,
        sy * cp,
        cy * sp,
        sy * sp 
    );
}

void render_video() {
    ThreeDimensionScene gs;
    gs.enable_globe();

    quat netherlands = lat_long_to_quat(vec2(52.1326f, 5.2913f));
    stage_macroblock(SilenceBlock(4), 1);
    gs.manager.set({
        {"d", "2"},
    });
    gs.manager.transition(MICRO, {
        {"q1", to_string(netherlands.u)},
        {"qi", to_string(netherlands.i)},
        {"qj", to_string(netherlands.j)},
        {"qk", to_string(netherlands.k)},
    });
    gs.manager.transition(MICRO, {
        {"d", "1.1"},
    });
    gs.render_microblock();
}
