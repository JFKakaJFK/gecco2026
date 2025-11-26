#pragma once
#ifndef _GOBLIN_GA_GP_HELPER_H
#define _GOBLIN_GA_GP_HELPER_H

#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// Maximum number of threads per CUDA block, currently defined as 1024,
// which is the maximum for modern NVIDIA GPUs.
#define MAX_THREADS_PER_BLOCK 1024

// Helper function to check for CUDA errors
// https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
#ifdef __CUDACC__
#define __CHECK_CUDA_ERR__(err) check((err), #err, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::abort();
    }
}
#endif

namespace goblin {

/// @brief Compute optimal block size for given number of data points
/// @param num_data_points Number of data points to process
/// @return Optimal block size (number of threads per block)
int compute_block_size(int num_data_points) {
    int best_block_size = 32;
    int min_redundant_threads = MAX_THREADS_PER_BLOCK;

    int blocks_needed;
    int redundant_threads;

    // If there are multiple optimal solutions, the largest block size is preferred
    // TODO test if this is beneficial in practice

    // Iterate over possible block sizes (multiples of 32)
    for (int b = MAX_THREADS_PER_BLOCK; b > 0; b -= 32) {
        // Round up division to determine number of blocks needed
        blocks_needed = (num_data_points + b - 1) / b;
        redundant_threads = blocks_needed * b - num_data_points;

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

#ifdef __CUDACC__
template <typename T>
T* allocate_on_gpu(size_t count) {
    T* d_ptr = nullptr;
    size_t bytes = count * sizeof(T);

    __CHECK_CUDA_ERR__(cudaMalloc(&d_ptr, bytes));

    return d_ptr;
}

template <typename T>
void copy_to_gpu(T* d_ptr, const T* host_data, size_t count) {
    size_t bytes = count * sizeof(T);

    __CHECK_CUDA_ERR__(cudaMemcpy(d_ptr, host_data, bytes, cudaMemcpyHostToDevice));
}

template <typename T>
T* allocate_and_copy(const T* host_data, size_t count) {
    T* d_ptr = allocate_on_gpu<T>(count);
    copy_to_gpu(d_ptr, host_data, count);

    return d_ptr;
}

template <typename T>
void free_on_gpu(T* d_ptr) {
    __CHECK_CUDA_ERR__(cudaFree(d_ptr));
}
#endif

};



#endif /* _GOBLIN_GA_GP_HELPER_H */