#pragma once

#include <string>
#include <unordered_set>

#include "../Common/ThreeDimensionScene.h"
#include "../../Host_Device_Shared/ManifoldData.c"

class ManifoldScene : public ThreeDimensionScene {
private:
    std::unordered_set<std::string> manifold_names;

public:
    ManifoldScene(const double width = 1, const double height = 1);

    void add_manifold(const std::string& name,
                      const std::string& x_eq, const std::string& y_eq, const std::string& z_eq, const std::string& r_eq, const std::string& i_eq,
                      const std::string& u_min, const std::string& u_max, const std::string& u_steps,
                      const std::string& v_min, const std::string& v_max, const std::string& v_steps);

    void remove_manifold(const std::string& name);

    void draw() override;

    const StateQuery populate_state_query() const override;
};
