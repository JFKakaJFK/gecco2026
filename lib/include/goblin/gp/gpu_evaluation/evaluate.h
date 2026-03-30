#pragma once

#ifndef _GOBLIN_GA_GP_EVAL_KERNEL_H
#define _GOBLIN_GA_GP_EVAL_KERNEL_H

#include <cstdint>
#include <vector>

#include "goblin/gp/gpu_evaluation/launch_config.h"

namespace goblin {

using u8 = std::uint8_t;

#ifdef __CUDACC__
__global__
void evaluate_kernel_baseline(
    float* X,
    float* Y,
    const u8* v_type,
    float* v_value,
    float* partial,
    size_t solution_length,
    size_t num_datapoints
);

__global__
void evaluate_kernel_restrict(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const u8* __restrict__ v_type,
    const float* __restrict__ v_value,
    float* __restrict__ v,
    size_t solution_length,
    size_t num_datapoints
);

__global__
void evaluate_kernel_shared_memory(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const u8* __restrict__ v_type,
    const float* __restrict__ v_value,
    float* __restrict__ partial,
    size_t solution_length,
    size_t num_datapoints
);

__device__
float compute_tree_output_baseline(
    float* X,
    const u8* type,
    const float* value,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
);

__device__
float compute_tree_output_restrict(
    const float* __restrict__ X,
    const u8* __restrict__ type,
    const float* __restrict__ value,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
);

__device__
float compute_tree_output_inplace(
    const float* __restrict__ X,
    const u8* __restrict__ type,
    const float* __restrict__ value,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
);

__global__
void compute_mse_kernel_baseline(
    const float* __restrict__ partial,
    float* __restrict__ result,
    size_t num_solutions,
    size_t num_datapoints
);

__global__
void mse_kernel_restrict(
    const float* __restrict__ partial,
    float* __restrict__ result,
    size_t num_solutions,
    size_t num_datapoints
);

__global__
void evaluate_kernel_hybrid(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const u8* __restrict__ v_type,
    const float* __restrict__ v_value,
    float* __restrict__ partial,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoints_per_block,
    size_t datapoints_per_thread
);

__global__
void compute_tree_output_wrapper(
    float* X,
    const u8* type,
    float* value,
    float* result,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index,
    KernelVersion version
);
#endif

void evaluate_kernel_wrapper(
    float* X,
    float* Y,
    u8* type,
    float* value,
    float* partial,
    LaunchConfig config
);

void mse_kernel_wrapper(
    float* partial,
    float* result,
    LaunchConfig config
);

void evaluate_mse_kernel_wrapper(
    float* X,
    float* Y,
    u8* type,
    float* value,
    float* result,
    LaunchConfig config
);

void kernel_wrapper(
    float* X,
    float* Y,
    u8* type,
    float* value,
    float* partial,
    float* result,
    LaunchConfig config
);

float test_compute_output_kernel(
    std::vector<float> h_X,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_datapoints,
    size_t datapoint_index,
    KernelVersion version
);

std::vector<float> test_evaluate_kernel(
    std::vector<float> h_X,
    std::vector<float> h_Y,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_solutions,
    size_t num_datapoints,
    KernelVersion version
);

std::vector<float> test_compute_mse_kernel(
    std::vector<float> partial,
    size_t num_solutions,
    size_t num_datapoints,
    KernelVersion version
);

std::vector<float> test_evaluate_mse_kernel(
    std::vector<float> h_X,
    std::vector<float> h_Y,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_solutions,
    size_t num_datapoints,
    KernelVersion version
);

std::vector<float> test_kernel_hybrid(
    std::vector<float> h_X,
    std::vector<float> h_Y,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_solutions,
    size_t num_datapoints,
    size_t blocks_per_individual
);

}  // namespace goblin

#endif /* _GOBLIN_GA_GP_EVAL_KERNEL_H */