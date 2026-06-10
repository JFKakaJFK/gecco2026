#include "cub/cub.cuh"

#include <cstdint>

#include "goblin/gp/gpu_evaluation/evaluate.h"
#include "goblin/gp/gpu_evaluation/launch_config.h"
#include "goblin/gp/gpu_evaluation/memory.h"
#include "goblin/gp/gpu_evaluation/misc.h"
#include "goblin/gp/gpu_evaluation/types.h"

// CUB BlockReduce requires a compile-time block size, so we instantiate a template for every
// valid multiple of WARP_SIZE and dispatch at runtime via a switch over block.x.
#define BLOCK_SIZE_LIST \
    X(32)  X(64)  X(96)  X(128) X(160) X(192) X(224) X(256) \
    X(288) X(320) X(352) X(384) X(416) X(448) X(480) X(512) \
    X(544) X(576) X(608) X(640) X(672) X(704) X(736) X(768) \
    X(800) X(832) X(864) X(896) X(928) X(960) X(992) X(1024)

namespace goblin {

__device__
float compute_tree_output(
    const float* __restrict__ X,
    const u8* __restrict__ type,
    const float* __restrict__ value,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
) {
    // Per-thread evaluation stack; depth bounded by EVAL_MAX_STACK_DEPTH.
    float stack[EVAL_MAX_STACK_DEPTH];
    int sp = 0;

    // Traverse solution in postfix order.
    for (size_t index = 0; index < solution_length; index++) {
        NodeType node_type = static_cast<NodeType>(type[index]);

        if (node_type == NodeType::Input) {
            size_t input_index = static_cast<size_t>(value[index]);
            stack[sp++] = X[datapoint_index + (input_index * num_datapoints)];
        } else if (node_type == NodeType::Constant) {
            stack[sp++] = value[index];
        } else if (node_type == NodeType::Operator) {
            Operator op_value = static_cast<Operator>(value[index]);

            switch (op_value) {
                case Operator::Add: stack[sp - 1]    += stack[--sp];                        break;
                case Operator::Sub: stack[sp - 1]    -= stack[--sp];                        break;
                case Operator::Mul: stack[sp - 1]    *= stack[--sp];                        break;
                case Operator::Div: stack[sp - 1]    /= stack[--sp];                        break;
                case Operator::Sin: stack[sp - 1]     = sinf(stack[sp - 1]);                break;
                case Operator::Cos: stack[sp - 1]     = cosf(stack[sp - 1]);                break;
                case Operator::Exp: stack[sp - 1]     = expf(stack[sp - 1]);                break;
                case Operator::Log: stack[sp - 1]     = logf(stack[sp - 1]);                break;
                case Operator::Square: stack[sp - 1] *= stack[sp - 1];                      break;
                case Operator::Sqrt: stack[sp - 1]    = sqrtf(stack[sp - 1]);               break;
                case Operator::Pow: {
                    float rhs = stack[--sp];
                    stack[sp - 1] = powf(stack[sp - 1], rhs);
                    break;
                }
                case Operator::Abs: stack[sp - 1]     = fabsf(stack[sp - 1]);               break;
                case Operator::Min: {
                    float rhs = stack[--sp];
                    stack[sp - 1] = fminf(stack[sp - 1], rhs);
                    break;
                }
                case Operator::Max: {
                    float rhs = stack[--sp];
                    stack[sp - 1] = fmaxf(stack[sp - 1], rhs);
                    break;
                }
            }
        } else {
            break;
        }
    }

    return stack[--sp];
}

// https://nvidia.github.io/cccl/cub/index.html#flexible-data-arrangement
template <int BLOCK_THREADS>
__global__
void single_block_kernel(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const u8* __restrict__ v_type,
    const float* __restrict__ v_value,
    float* __restrict__ result,
    const size_t solution_length,
    const size_t num_datapoints,
    const size_t datapoints_per_thread
) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    size_t solution_index  = static_cast<size_t>(blockIdx.x);
    size_t solution_offset = solution_index * solution_length;

    // Load the solution into shared memory cooperatively across all threads.
    __shared__ u8    sh_type [EVAL_MAX_NODES];
    __shared__ float sh_value[EVAL_MAX_NODES];

    for (size_t i = threadIdx.x; i < solution_length; i += blockDim.x) {
        sh_type[i]  = v_type [solution_offset + i];
        sh_value[i] = v_value[solution_offset + i];
    }

    __syncthreads();

    float se = 0.0F;

    for (size_t i = 0; i < datapoints_per_thread; i++) {
        size_t datapoint_index = (i * blockDim.x) + threadIdx.x;

        if (datapoint_index < num_datapoints) {
            float output = compute_tree_output(
                X, sh_type, sh_value,
                solution_length,
                num_datapoints,
                datapoint_index
            );

            float error = output - Y[datapoint_index];
            se = __fmaf_rn(error, error, se);
        }
    }

    float sum = BlockReduce(temp_storage).Sum(se);

    if (threadIdx.x == 0) {
        result[solution_index] = sum / num_datapoints;
    }
}

template <int BLOCK_THREADS>
__global__
void dynamic_block_evaluate_kernel(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const u8* __restrict__ v_type,
    const float* __restrict__ v_value,
    float* __restrict__ partial,
    const size_t solution_length,
    const size_t num_datapoints,
    const size_t datapoints_per_block,
    const size_t datapoints_per_thread
) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // blockIdx.y selects which slice of datapoints this block owns.
    size_t datapoint_index       = (static_cast<size_t>(blockIdx.y) * datapoints_per_block) + threadIdx.x;
    const size_t last_datapoint  = (static_cast<size_t>(blockIdx.y) + 1) * datapoints_per_block;
    const size_t solution_index  = static_cast<size_t>(blockIdx.x);
    const size_t solution_offset = solution_index * solution_length;

    __shared__ u8    sh_type [EVAL_MAX_NODES];
    __shared__ float sh_value[EVAL_MAX_NODES];

    for (size_t i = threadIdx.x; i < solution_length; i += blockDim.x) {
        sh_type[i]  = v_type [solution_offset + i];
        sh_value[i] = v_value[solution_offset + i];
    }

    __syncthreads();

    float se = 0.0F;

    for (size_t i = 0; i < datapoints_per_thread; i++) {
        // Stride by blockDim.x for coalesced global memory access.
        if (datapoint_index < num_datapoints && datapoint_index < last_datapoint) {
            float output = compute_tree_output(
                X, sh_type, sh_value,
                solution_length,
                num_datapoints,
                datapoint_index
            );

            float error = output - Y[datapoint_index];
            se = __fmaf_rn(error, error, se);
        }

        datapoint_index += blockDim.x;
    }

    float block_sum = BlockReduce(temp_storage).Sum(se);

    if (threadIdx.x == 0) {
        // gridDim.y == blocks_per_individual; store into a flat partial-sums array.
        const size_t partial_index = (gridDim.y * solution_index) + blockIdx.y;
        partial[partial_index] = block_sum;
    }
}

// TODO add items_per_thread template
// https://nvidia.github.io/cccl/cub/index.html#flexible-data-arrangement
template <int BLOCK_THREADS, int ITEMS_PER_THREAD = 1>
__global__
void dynamic_block_reduction_kernel(
    const float* __restrict__ partial,
    float* __restrict__ result,
    size_t num_partial,
    size_t num_datapoints
) {
    using BlockLoad   = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_DIRECT>;
    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

    __shared__ union {
        typename BlockLoad::TempStorage   load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    size_t solution_index = static_cast<size_t>(blockIdx.x);
    size_t block_offset   = solution_index * num_partial;

    float thread_data[ITEMS_PER_THREAD];

    BlockLoad(temp_storage.load).Load(partial + block_offset, thread_data, num_partial, 0);

    __syncthreads();

    float sum = BlockReduce(temp_storage.reduce).Sum(thread_data);

    __syncthreads();

    if (threadIdx.x == 0) {
        result[solution_index] = sum / num_datapoints;
    }
}

/// Wrapper Functions

/*
 * compute_tree_output_wrapper exists only to test compute_tree_output from host code.
 * compute_tree_output is __device__-only, so it must be called through a __global__ kernel.
 */
__global__
void compute_tree_output_wrapper(
    float* X,
    u8* type,
    float* value,
    float* result,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
) {
    result[0] = compute_tree_output(
        X, type, value,
        solution_length,
        num_datapoints,
        datapoint_index
    );
}

void single_block_wrapper(
    float* X,
    float* Y,
    u8* type,
    float* value,
    float* result,
    const LaunchConfig config
) {
    dim3 block(config.eval.block.x, config.eval.block.y, config.eval.block.z);
    dim3 grid (config.eval.grid.x,  config.eval.grid.y,  config.eval.grid.z);

    switch (block.x) {
        #define X(BS) \
        case BS: \
            single_block_kernel<BS><<<grid, block>>>( \
                X, Y, type, value, result, config.solution_length, config.num_datapoints, config.datapoints_per_thread); \
            break;
        BLOCK_SIZE_LIST
        #undef X

        default:
            assert(false && "Unsupported block size");
    }

    __CHECK_CUDA_ERR__(cudaGetLastError());
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());
}

void dynamic_block_evaluate_wrapper(
    float* X,
    float* Y,
    u8* type,
    float* value,
    float* partial,
    const LaunchConfig config
) {
    dim3 block(config.eval.block.x, config.eval.block.y, config.eval.block.z);
    dim3 grid (config.eval.grid.x,  config.eval.grid.y,  config.eval.grid.z);

    switch (block.x) {
        #define X(BS) \
        case BS: \
            dynamic_block_evaluate_kernel<BS><<<grid, block>>>( \
                X, Y, type, value, partial, config.solution_length, config.num_datapoints, \
                config.datapoints_per_block, config.datapoints_per_thread); \
            break;
        BLOCK_SIZE_LIST
        #undef X

        default:
            assert(false && "Unsupported block size");
    }

    __CHECK_CUDA_ERR__(cudaGetLastError());
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());
}

void dynamic_block_reduction_wrapper(
    float* partial,
    float* result,
    const LaunchConfig config
) {
    dim3 block(config.mse.block.x, config.mse.block.y, config.mse.block.z);
    dim3 grid (config.mse.grid.x,  config.mse.grid.y,  config.mse.grid.z);

    size_t num_partials = config.eval.grid.y;

    switch (config.mse.block.x) {
        #define X(BS) \
        case BS: \
            dynamic_block_reduction_kernel<BS><<<grid, block>>>( \
                partial, result, num_partials, config.num_datapoints); \
            break;
        BLOCK_SIZE_LIST
        #undef X

        default:
            assert(false && "Unsupported block size");
    }

    __CHECK_CUDA_ERR__(cudaGetLastError());
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());
}

void kernel_wrapper(
    float* X,
    float* Y,
    u8* type,
    float* value,
    float* partial,
    float* result,
    const LaunchConfig config
) {
    if (config.kernel_version == KernelVersion::SingleBlock) {
        single_block_wrapper(X, Y, type, value, result, config);
    } else {
        dynamic_block_evaluate_wrapper(X, Y, type, value, partial, config);
        dynamic_block_reduction_wrapper(partial, result, config);
    }
}


float test_compute_tree_output(
    std::vector<float> h_X,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_datapoints,
    size_t datapoint_index,
    KernelVersion version
) {
    size_t solution_length = h_type.size();

    float* d_X     = allocate_and_copy(h_X.data(), h_X.size());
    u8*    d_type  = allocate_and_copy(h_type.data(), solution_length);
    float* d_value = allocate_and_copy(h_value.data(), solution_length);
    float* d_result = allocate_on_gpu<float>(1);

    dim3 block(1);
    dim3 grid(1, 1);

    compute_tree_output_wrapper<<<grid, block>>>(
        d_X, d_type, d_value, d_result,
        solution_length, num_datapoints, datapoint_index
    );

    __CHECK_CUDA_ERR__(cudaGetLastError());
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());

    float result;
    __CHECK_CUDA_ERR__(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    free_on_gpu(d_type);
    free_on_gpu(d_value);
    free_on_gpu(d_result);

    return result;
}

std::vector<float> test_single_block(
    std::vector<float> h_X,
    std::vector<float> h_Y,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_solutions,
    size_t num_datapoints
) {
    size_t solution_length = h_type.size() / num_solutions;

    float* d_X     = allocate_and_copy(h_X.data(), h_X.size());
    float* d_Y     = allocate_and_copy(h_Y.data(), h_Y.size());
    u8*    d_type  = allocate_and_copy(h_type.data(), h_type.size());
    float* d_value = allocate_and_copy(h_value.data(), h_value.size());
    float* d_result = allocate_on_gpu<float>(num_solutions);

    LaunchConfig config = LaunchConfig::determine(
        KernelVersion::SingleBlock, num_solutions, num_datapoints, solution_length, std::nullopt);

    single_block_wrapper(d_X, d_Y, d_type, d_value, d_result, config);

    std::vector<float> result(num_solutions);
    __CHECK_CUDA_ERR__(cudaMemcpy(result.data(), d_result, num_solutions * sizeof(float), cudaMemcpyDeviceToHost));

    free_on_gpu(d_type);
    free_on_gpu(d_value);
    free_on_gpu(d_result);

    return result;
}

std::vector<float> test_dynamic_block(
    std::vector<float> h_X,
    std::vector<float> h_Y,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_solutions,
    size_t num_datapoints,
    size_t blocks_per_individual
) {
    size_t solution_length = h_type.size() / num_solutions;

    float* d_X       = allocate_and_copy(h_X.data(), h_X.size());
    float* d_Y       = allocate_and_copy(h_Y.data(), h_Y.size());
    u8*    d_type    = allocate_and_copy(h_type.data(), h_type.size());
    float* d_value   = allocate_and_copy(h_value.data(), h_value.size());
    float* d_partial = allocate_on_gpu<float>(num_solutions * blocks_per_individual);
    float* d_result  = allocate_on_gpu<float>(num_solutions);

    LaunchConfig config = LaunchConfig::determine(
        KernelVersion::DynamicBlock, num_solutions, num_datapoints, solution_length, blocks_per_individual);

    kernel_wrapper(d_X, d_Y, d_type, d_value, d_partial, d_result, config);

    std::vector<float> result(num_solutions);
    __CHECK_CUDA_ERR__(cudaMemcpy(result.data(), d_result, num_solutions * sizeof(float), cudaMemcpyDeviceToHost));

    free_on_gpu(d_partial);
    free_on_gpu(d_result);
    free_on_gpu(d_type);
    free_on_gpu(d_value);

    return result;
}

}
