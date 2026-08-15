#include <cstdint>
#include "DevicePointer.h"
#include "../IO/Writer.h"

extern "C" void cuda_free_pixels_on_device(uint32_t* d_pixels);
extern "C" uint32_t* cuda_alloc_pixels_on_device(int size);
extern "C" void cuda_copy_pixels_to_host(uint32_t* h_pixels, int size, uint32_t* d_pixels);
extern "C" void cuda_copy_pixels_to_device(uint32_t* h_pixels, int size, uint32_t* d_pixels);

DevicePointer::DevicePointer(const ivec2& wh) : wh(wh) {
    cout << "Allocating device pointer of size " << wh.x << " by " << wh.y << endl;
    device_ptr = cuda_alloc_pixels_on_device(wh.x*wh.y);
}

DevicePointer::DevicePointer() : wh(ivec2(0,0)) { }

DevicePointer::~DevicePointer() {
    //cout << "Freeing device pointer of size " << wh.x << " by " << wh.y << endl;
    cuda_free_pixels_on_device(device_ptr);
}

void DevicePointer::resize(const ivec2& new_wh) {
    cuda_free_pixels_on_device(device_ptr);
    device_ptr = cuda_alloc_pixels_on_device(new_wh.x * new_wh.y);
    wh = new_wh;
}

void DevicePointer::tick(const ivec2& scale) {
    // Reallocate only if size is too small
    // TODO this does not handle nested scenes, assuming all scenes are children of the whole video's frame size
    if (wh.x * wh.y < scale.x * scale.y) {
        resize(scale);
    }
}

void DevicePointer::copy_to_host(uint32_t* host_ptr) {
    cuda_copy_pixels_to_host(host_ptr, wh.x * wh.y, device_ptr);
}

void DevicePointer::copy_to_device(uint32_t* host_ptr) {
    cuda_copy_pixels_to_device(host_ptr, wh.x * wh.y, device_ptr);
}

ivec2 DevicePointer::get_wh() const {
    return wh;
}

uint32_t* DevicePointer::get_ptr() {
    return device_ptr;
}
