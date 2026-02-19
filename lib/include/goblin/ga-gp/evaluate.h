#pragma once

#ifndef _GOBLIN_GA_GP_EVAL_KERNEL_H
#define _GOBLIN_GA_GP_EVAL_KERNEL_H

#include <vector>

#include "goblin/ga-gp/misc.h"

namespace goblin {

#ifdef __CUDACC__
__global__
void evaluate_kernel_baseline(
    float* X, 
    float* Y, 
    float* v_type, 
    float* v_value, 
    float* partial,
    size_t solution_length, 
    size_t num_datapoints   
);

__global__
void evaluate_kernel_restrict(
    const float* __restrict__ X, 
    const float* __restrict__ Y, 
    const float* __restrict__ v_type, 
    const float* __restrict__ v_value, 
    float* __restrict__ v,
    size_t solution_length, 
    size_t num_datapoints
);

__global__
void evaluate_kernel_shared_memory(
    const float* __restrict__ X, 
    const float* __restrict__ Y, 
    const float* __restrict__ v_type, 
    const float* __restrict__ v_value, 
    float* __restrict__ partial,
    size_t solution_length, 
    size_t num_datapoints
);

__device__
float compute_tree_output_baseline(
    float* X, 
    const float* type,
    const float* value,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
);

__device__
float compute_tree_output_restrict(
    const float* __restrict__ X, 
    const float* __restrict__ type,
    const float* __restrict__ value,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
);

__device__
float compute_tree_output_inplace(
    const float* __restrict__ X, 
    const float* __restrict__ type,
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
void compute_tree_output_wrapper(
    float* X, 
    float* type,
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
    float* type, 
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
    float* type, 
    float* value, 
    float* result,
    LaunchConfig config
);

void kernel_wrapper(
    float* X, 
    float* Y, 
    float* type, 
    float* value, 
    float* partial,
    float* result,
    LaunchConfig config
);

float test_compute_output_kernel(
    std::vector<float> h_X,
    std::vector<float> h_type,
    std::vector<float> h_value,
    size_t num_datapoints,
    size_t datapoint_index,
    KernelVersion version
);

std::vector<float> test_evaluate_kernel(
    std::vector<float> h_X, 
    std::vector<float> h_Y, 
    std::vector<float> h_type, 
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
    std::vector<float> h_type, 
    std::vector<float> h_value, 
    size_t num_solutions,
    size_t num_datapoints
);

}  // namespace goblin

#endif /* _GOBLIN_GA_GP_EVAL_KERNEL_H */