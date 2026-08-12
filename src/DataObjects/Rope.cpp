#include "Rope.h"

extern "C" void initialize_nodes_from_file(const std::string& file_name, vec2* d_rope);
extern "C" void allocate_rope_and_pins(vec2** rope_pointer, vec2** pins_pointer);
extern "C" void physics(vec2* rope, const int rope_length, const vec2* pins, const int pins_length);
extern "C" void copy_pins(const vec2* h_pins, vec2* d_pins, const int pins_length);

void Rope::tick() {
    for (int i = 0; i < 3; ++i) {
        physics(d_nodes, 1000, d_pins, h_pins.size());
    }
}

Rope::Rope(const std::string& file_name) {
    allocate_rope_and_pins(&d_nodes, &d_pins);
    initialize_nodes_from_file(file_name, d_nodes);
}

void Rope::add_pin(const vec2& pos) {
    h_pins.push_back(pos);
    copy_pins(h_pins.data(), d_pins, h_pins.size());
}

void Rope::remove_pin(int pin_index) {
    if (pin_index >= 0 && pin_index < h_pins.size()) {
        h_pins.erase(h_pins.begin() + pin_index);
        copy_pins(h_pins.data(), d_pins, h_pins.size());
    }
}
