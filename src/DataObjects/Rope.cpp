#include "Rope.h"

extern "C" void initialize_nodes_from_file(const std::string& file_name, vec2* d_rope);
extern "C" void allocate_rope_and_pins(vec2** rope_pointer, vec2** pins_pointer);
extern "C" void physics(vec2* rope, const int rope_length, vec2* pins, const int pins_length);

void Rope::tick(const StateReturn& state) {
    physics(d_nodes, 1000, d_pins, 10);
}

Rope::Rope(std::string file_name) {
    allocate_rope_and_pins(&d_nodes, &d_pins);
    initialize_nodes_from_file(file_name, d_nodes);
}