quat lat_long_to_quat(vec2 lat_long) {
    float lon = -lat_long.x;
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

vec4 lat_long_to_xyz(vec2 lat_long) {
    float lat = lat_long.x;
    float lon = lat_long.y;

    // Convert latitude and longitude from degrees to radians
    float lat_rad = lat * (M_PI / 180.0f);
    float lon_rad = lon * (M_PI / 180.0f);

    // Calculate the Cartesian coordinates
    lon_rad -= M_PI/2;
    float x = cos(lat_rad) * cos(lon_rad);
    float y = sin(lat_rad);
    float z = cos(lat_rad) * sin(lon_rad);

    return vec4(x, y, z, 0);
}

void quat_mult_string(
    const string& ru, const string& ri, const string& rj, const string& rk,
    const string& su, const string& si, const string& sj, const string& sk,
    string& out_u, string& out_i, string& out_j, string& out_k) {

    out_u = ru + " " + su + " * " + ri + " " + si + " * - " + rj + " " + sj + " * - " + rk + " " + sk + " * -";
    out_i = ru + " " + si + " * " + ri + " " + su + " * + " + rj + " " + sk + " * + " + rk + " " + sj + " * -";
    out_j = ru + " " + sj + " * " + ri + " " + sk + " * - " + rj + " " + su + " * + " + rk + " " + si + " * +";
    out_k = ru + " " + sk + " * " + ri + " " + sj + " * + " + rj + " " + si + " * - " + rk + " " + su + " * +";
}

void set_camera_to_lat_long(shared_ptr<GraphScene> gs, vec2 lat_long, bool set, TransitionType tt) {
    vec4 f = lat_long_to_xyz(lat_long);
    StateSet focus({
        {"x",to_string(f.x)},
        {"y",to_string(f.y)},
        {"z",to_string(f.z)},
    });
    if(set) {
        gs->manager.set(focus);
    } else {
        gs->manager.transition(tt, focus);
    }
    gs->manager.set({
        {"theta", ".5"},
        {"phi", "{t} .2 * sin .4 *"},
    });
    quat rot = lat_long_to_quat(lat_long);
    string ru = "<theta> 2 / cos";
    string ri = "<theta> 2 / sin";
    string rj = "0";
    string rk = "0";
    string su = "<phi> 2 / cos";
    string si = "0";
    string sj = "0";
    string sk = "<phi> 2 / sin";
    string au, ai, aj, ak;
    quat_mult_string(ru, ri, rj, rk, su, si, sj, sk, au, ai, aj, ak);
    string cu = to_string(rot.u);
    string ci = to_string(rot.i);
    string cj = to_string(rot.j);
    string ck = to_string(rot.k);
    string q1, qi, qj, qk;
    quat_mult_string(au, ai, aj, ak, cu, ci, cj, ck, q1, qi, qj, qk);
    StateSet rot_state({
        {"q1", q1},
        {"qi", qi + " {t} .15 * sin .1 * +"},
        {"qj", qj + " {t} .25 * sin .1 * +"},
        {"qk", qk},
    });
    if(set) {
        gs->manager.set(rot_state);
    } else {
        gs->manager.transition(tt, rot_state);
    }
}

void trace_path(shared_ptr<GraphScene> gs, vector<string> path, int color) {
    for(int i = 0; i < path.size() - 1; i++) {
        string node1 = path[i];
        string node2 = path[i + 1];
        gs->config->transition_node_color(MICRO, HashableString(node1).get_hash(), color);
        gs->config->transition_edge_color(MICRO, HashableString(node1).get_hash(), HashableString(node2).get_hash(), color);
        gs->render_microblock();
    }
    gs->config->transition_node_color(MICRO, HashableString(path.back()).get_hash(), color);
    gs->render_microblock();
}
