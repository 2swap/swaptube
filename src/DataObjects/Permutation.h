#pragma once

// list of pieces
// list of orbit, an orbit is a list of PLACES

#include <unordered_map>
#include <vector>
#include <cstdint>
#include <string>
#include "../Host_Device_Shared/vec.h"

class Permutation {
    public:
        Permutation(std::string file_name);
        vec2 get_point(const std::string begin, const std::string end, std::string orbit_name, float t);
        std::unordered_map<std::string, vec2> places;
        std::unordered_map<std::string, uint32_t> pieces;
        std::unordered_map<std::string, std::vector<std::string>> orbits;
};
