#include "Permutation.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include "../Host_Device_Shared/helpers.h"

Permutation::Permutation(std::string file_name) {
    std::ifstream file(file_name);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open permutation file: " + file_name);
        return;
    }

    std::string line;
    std::string current_section = "";

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        ss >> token;
        cout << "Processing line: " << line << endl;

        // Changement de section
        if (token == "places" || token == "pieces" || token == "orbits") {
            current_section = token;
            continue;
        }

        // Section PLACES (Format : Nom X Y)
        if (current_section == "places") {
            std::string name = token;
            float x, y;
            if (ss >> x >> y) {
                places[name] = vec2(x, y);
            }
        } 
        // Section PIECES (Format : Nom A R G B)
        else if (current_section == "pieces") {

            std::string name = token;
            uint32_t a, r, g, b;
            if (ss >> a >> r >> g >> b) {
                // Construction du uint32_t au format 0xAARRGGBB
                uint32_t color = ((a & 0xFF) << 24) | 
                                 ((r & 0xFF) << 16) | 
                                 ((g & 0xFF) << 8)  | 
                                  (b & 0xFF);
                pieces[name] = color;
                // cout << "Loaded piece: " << name << " with color ARGB(" << a << ", " << r << ", " << g << ", " << b << ")" << endl;
            }
        } 
        // Section ORBITS (Format : NomLieu1 Lieu2 Lieu3 ...)
        else if (current_section == "orbits") {
            std::string orbit_name = token;
            std::string place_name;
            std::vector<std::string> path;

            while (ss >> place_name) {
                path.push_back(place_name);
            }
            orbits[orbit_name] = path;
        }
    }

    file.close();
}

vec2 Permutation::get_point(const std::string begin, const std::string end, std::string orbit_name, float t){
    // use bezier in the helper function to get a point between begin and end
    vec2 p2 = places[begin];
    vec2 p3 = places[end];
    return vec2(0,0);//bezier(p2, p3, t);

} 
