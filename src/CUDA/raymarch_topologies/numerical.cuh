__constant__ char d_w_equation[256];

__device__ glm::vec4 surface(glm::vec3 v) {
    char w_inserted[256];
    insert_tags_xyz(d_w_equation, v.x, v.y, v.z, w_inserted, 256);
    double w = 0;
    if(!calculator(w_inserted, &w)) printf("Error calculating manifold w at (%f,%f,%f): %s\n", v.x, v.y, v.z, w_inserted);
    return glm::vec4(v.x, v.y, v.z, w);
}

// Compute partial derivatives of the surface embedding wrt parameter axes
static __device__ void dsurface_dv_numerical(glm::vec3 v, float delta,
                                   glm::vec4& d_dx, glm::vec4& d_dy, glm::vec4& d_dz) {
    glm::vec4 here   = surface(v);
    glm::vec4 plus_x = surface(v + glm::vec3(delta, 0.0f, 0.0f));
    glm::vec4 plus_y = surface(v + glm::vec3(0.0f, delta, 0.0f));
    glm::vec4 plus_z = surface(v + glm::vec3(0.0f, 0.0f, delta));
    d_dx = (plus_x - here) / delta;
    d_dy = (plus_y - here) / delta;
    d_dz = (plus_z - here) / delta;
}

static __device__ void metric_tensor(glm::vec3 v, float g[3][3]) {
    glm::vec4 d_dx, d_dy, d_dz;
    dsurface_dv_numerical(v, 1e-4f, d_dx, d_dy, d_dz);

    g[0][0] = glm::dot(d_dx, d_dx);
    g[0][1] = glm::dot(d_dx, d_dy);
    g[0][2] = glm::dot(d_dx, d_dz);

    g[1][0] = glm::dot(d_dy, d_dx);
    g[1][1] = glm::dot(d_dy, d_dy);
    g[1][2] = glm::dot(d_dy, d_dz);

    g[2][0] = glm::dot(d_dz, d_dx);
    g[2][1] = glm::dot(d_dz, d_dy);
    g[2][2] = glm::dot(d_dz, d_dz);
}

static __device__ void dmetric_dv_numerical(glm::vec3 v, float dg[3][3][3], float delta) {
    float g_pu[3][3], g_mu[3][3], g_pv[3][3], g_mv[3][3], g_pw[3][3], g_mw[3][3];
    metric_tensor(v + glm::vec3( delta, 0.0f, 0.0f), g_pu);
    metric_tensor(v + glm::vec3(-delta, 0.0f, 0.0f), g_mu);

    metric_tensor(v + glm::vec3(0.0f,  delta, 0.0f), g_pv);
    metric_tensor(v + glm::vec3(0.0f, -delta, 0.0f), g_mv);

    metric_tensor(v + glm::vec3(0.0f, 0.0f,  delta), g_pw);
    metric_tensor(v + glm::vec3(0.0f, 0.0f, -delta), g_mw);

    float g_u[3][3], g_v[3][3], g_w[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            g_u[i][j] = (g_pu[i][j] - g_mu[i][j]) / (2.0f * delta);
            g_v[i][j] = (g_pv[i][j] - g_mv[i][j]) / (2.0f * delta);
            g_w[i][j] = (g_pw[i][j] - g_mw[i][j]) / (2.0f * delta);
        }

    for (int l = 0; l < 3; ++l)
        for (int j = 0; j < 3; ++j) {
            dg[0][l][j] = g_u[l][j];
            dg[1][l][j] = g_v[l][j];
            dg[2][l][j] = g_w[l][j];
        }

    // Γ^i_jk = 1/2 g^{i l} ( ∂_j g_{l k} + ∂_k g_{l j} - ∂_l g_{j k} )
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                float sum = 0.0f;
                for (int l = 0; l < 3; ++l) {
                    float term = dg[j][l][k] + dg[k][l][j] - dg[l][j][k];
                    sum += 0.5f * g_inv[i][l] * term;
                }
                Gamma[i][j][k] = sum;
            }
        }
    }
    return true;
}
