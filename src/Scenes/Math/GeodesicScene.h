#pragma once

#include "../Scene.h"
#include "../../Host_Device_Shared/vec.h"
#include "../../Host_Device_Shared/ManifoldData.c"

class GeodesicScene : public Scene {
public:
    GeodesicScene(const double width = 1, const double height = 1);

    void draw_perspective(ResolvedStateEquation& x_eq,
                          ResolvedStateEquation& y_eq,
                          ResolvedStateEquation& z_eq,
                          ResolvedStateEquation& w_eq);

    void draw_manifold(ResolvedStateEquation& x_eq,
                       ResolvedStateEquation& y_eq,
                       ResolvedStateEquation& z_eq,
                       ResolvedStateEquation& w_eq);

    void draw() override;

    const StateQuery populate_state_query() const override;

    bool check_if_data_changed() const;
    void change_data();
    void mark_data_unchanged();
};
