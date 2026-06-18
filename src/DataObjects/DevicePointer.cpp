#include <cstdint>
#include "DevicePointer.h"
#include "../IO/Writer.h"

extern "C" void cuda_free_pixels_on_device(uint32_t* d_pixels);
extern "C" uint32_t* cuda_alloc_pixels_on_device(int size);
extern "C" void cuda_copy_pixels_to_host(uint32_t* h_pixels, int size, uint32_t* d_pixels);

DevicePointer::DevicePointer(const int size) : size(size) {
    cout << "Allocating device pointer of size " << size << endl;
    device_ptr = cuda_alloc_pixels_on_device(size);
}

DevicePointer::~DevicePointer() {
    mark_updated();
    cuda_free_pixels_on_device(device_ptr);
}

void DevicePointer::resize(const int new_size) {
    mark_updated();
    cuda_free_pixels_on_device(device_ptr);
    device_ptr = cuda_alloc_pixels_on_device(new_size);
    size = new_size;
}

void DevicePointer::tick(const StateReturn& state) {
    mark_updated();
    // Reallocate only if size is too small
    int width = state["w"] * get_video_width_pixels();
    int height = state["h"] * get_video_height_pixels();
    // TODO this does not handle nested scenes, assuming all scenes are children of the whole video's frame size
    if (size < width * height) {
        resize(width * height);
    }
}

void DevicePointer::copy_to_host(uint32_t* host_ptr, const ivec2& wh) {
    cuda_copy_pixels_to_host(host_ptr, wh.x * wh.y, device_ptr);
}

uint32_t* DevicePointer::get_ptr() {
    return device_ptr;
}
