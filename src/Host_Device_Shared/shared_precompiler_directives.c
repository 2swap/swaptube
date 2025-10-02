#pragma once

#ifdef __CUDACC__

#define HOST_DEVICE __host__ __device__
#define SHARED_FILE_PREFIX namespace Cuda {
#define SHARED_FILE_SUFFIX }

#else

#define HOST_DEVICE
#define SHARED_FILE_PREFIX
#define SHARED_FILE_SUFFIX

#endif
