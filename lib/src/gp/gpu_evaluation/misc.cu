#include <iostream>

#include "goblin/gp/gpu_evaluation/misc.h"

namespace goblin {

// Helper function to check for CUDA errors
// https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
#define __CHECK_CUDA_ERR__(err) check((err), #err, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::abort();
    }
}

GpuInfo get_gpu_info() {
    GpuInfo info;
    __CHECK_CUDA_ERR__(cudaGetDevice(&info.device_id));
    __CHECK_CUDA_ERR__(cudaDeviceGetAttribute(&info.num_sms, cudaDevAttrMultiProcessorCount, info.device_id));
    return info;
}

}