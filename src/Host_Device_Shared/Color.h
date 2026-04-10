#pragma once

#include <cstdint>

typedef uint32_t Color;

#if defined(__CUDACC__) || defined(__HIPCC__)
namespace Cuda {
typedef uint32_t Color;
}
#endif
