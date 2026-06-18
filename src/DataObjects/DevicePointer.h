#pragma once

#include <cstdint>
#include "DataObject.h"

class DevicePointer : public DataObject {
public:
    DevicePointer(const int size);
    ~DevicePointer();
    void tick(const StateReturn& state);
    uint32_t* get_ptr();
    void copy_to_host(uint32_t* host_ptr, const ivec2& wh);
private:
    int size;
    uint32_t* device_ptr;
    void resize(const int new_size);
};
