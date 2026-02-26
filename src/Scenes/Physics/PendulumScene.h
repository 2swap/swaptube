#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include "../../DataObjects/Pendulum.h"
#include "../Scene.h"
#include "../../IO/AudioWriter.h"
#include "../../IO/Writer.h"

class PendulumScene : public Scene {
public:
    PendulumScene(PendulumState s, const double width = 1, const double height = 1);

    int alpha_subtract = 2;

    const StateQuery populate_state_query() const override;

    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;
    std::unordered_map<std::string, double> stage_publish_to_global() const override;

    void draw() override;

    void generate_tone();
    void generate_audio(double duration, std::vector<sample_t>& left, std::vector<sample_t>& right, double volume_mult = 1);

    Pendulum pend;

private:
    int tonegen = 0;
    double energy = 0;
    double energy_slew = 0;
    PendulumState start_state;
    Pixels path_background;
    const int pendulum_count = 2;
};
