#include "goblin/ga-gp/helper.h"
#include "goblin/ga-gp/types.h"

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

template <typename T>
T* allocate_on_gpu(size_t count) {
    T* d_ptr = nullptr;
    __CHECK_CUDA_ERR__(cudaMalloc(&d_ptr, count * sizeof(T)));
    return d_ptr;
}

template <typename T>
void copy_to_gpu(T* d_ptr, const T* host_data, size_t count) {
    __CHECK_CUDA_ERR__(cudaMemcpy(d_ptr, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
T* allocate_and_copy(const T* host_data, size_t count) {
    T* d_ptr = allocate_on_gpu<T>(count);
    copy_to_gpu(d_ptr, host_data, count);
    return d_ptr;
}

template <typename T>
void copy_from_device(T* host_data, T* d_ptr, size_t count) {
    __CHECK_CUDA_ERR__(cudaMemcpy(host_data, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void free_on_gpu(T* d_ptr) {
    __CHECK_CUDA_ERR__(cudaFree(d_ptr));
}

template <typename T>
void zero_mem_on_gpu(T* d_ptr, size_t count) {
    __CHECK_CUDA_ERR__(cudaMemset(d_ptr, 0, count * sizeof(T)));
}

template float* allocate_on_gpu<float>(size_t);
template void copy_to_gpu<float>(float*, const float*, size_t);
template float* allocate_and_copy<float>(const float*, size_t);
template void copy_from_device<float>(float*, float*, size_t);
template void free_on_gpu<float>(float*);
template void zero_mem_on_gpu<float>(float* d_ptr, size_t count);

}