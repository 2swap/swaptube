#pragma once

#include <cstdint>
#include "../Host_Device_Shared/vec.h"
#include "../Core/State/StateManager.h"

class DevicePointer {
public:
    DevicePointer(const ivec2& wh);
    DevicePointer();
    ~DevicePointer();
    void tick(const ivec2&);
    uint32_t* get_ptr();
    void copy_to_host(uint32_t* host_ptr);
    void copy_to_device(uint32_t* host_ptr);
    ivec2 get_wh() const;
private:
    ivec2 wh;
    uint32_t* device_ptr;
    void resize(const ivec2& new_wh);
};
