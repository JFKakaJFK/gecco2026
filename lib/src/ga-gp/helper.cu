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
    check(cudaMalloc(&d_ptr, count * sizeof(T)), "cudaMalloc", __FILE__, __LINE__);
    return d_ptr;
}

template <typename T>
void copy_to_gpu(T* d_ptr, const T* host_data, size_t count) {
    check(cudaMemcpy(d_ptr, host_data, count * sizeof(T), cudaMemcpyHostToDevice),
          "cudaMemcpy H2D", __FILE__, __LINE__);
}

template <typename T>
T* allocate_and_copy(const T* host_data, size_t count) {
    T* d_ptr = allocate_on_gpu<T>(count);
    copy_to_gpu(d_ptr, host_data, count);
    return d_ptr;
}

template <typename T>
void copy_from_device(T* host_data, T* d_ptr, size_t count) {
    check(cudaMemcpy(host_data, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H", __FILE__, __LINE__);
}

template <typename T>
void free_on_gpu(T* d_ptr) {
    check(cudaFree(d_ptr), "cudaFree", __FILE__, __LINE__);
}

int compute_block_size(int count) {
    int best_block_size = 32;
    int min_redundant_threads = MAX_THREADS_PER_BLOCK;

    int blocks_needed;
    int redundant_threads;

    // If there are multiple optimal solutions, the largest block size is preferred
    // TODO test if this is beneficial in practice

    // Iterate over possible block sizes (multiples of 32)
    for (int b = MAX_THREADS_PER_BLOCK; b > 0; b -= 32) {
        // Round up division to determine number of blocks needed
        blocks_needed = (count + b - 1) / b;
        redundant_threads = blocks_needed * b - count;

        if (redundant_threads < min_redundant_threads) {
            min_redundant_threads = redundant_threads;
            best_block_size = b;
        }

        // Early exit if perfect fit is found
        if (redundant_threads == 0) {
            break;
        }
    }

    return best_block_size;
};

// Explicit template instantiation for all types you use
template float* allocate_on_gpu<float>(size_t);
template void copy_to_gpu<float>(float*, const float*, size_t);
template float* allocate_and_copy<float>(const float*, size_t);
template void copy_from_device<float>(float*, float*, size_t);
template void free_on_gpu<float>(float*);

template NodeType* allocate_on_gpu<NodeType>(size_t);
template void copy_to_gpu<NodeType>(NodeType*, const NodeType*, size_t);
template NodeType* allocate_and_copy<NodeType>(const NodeType*, size_t);
template void copy_from_device<NodeType>(NodeType*, NodeType*, size_t);
template void free_on_gpu<NodeType>(NodeType*);

}