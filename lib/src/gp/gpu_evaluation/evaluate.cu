#include "cub/cub.cuh"

#include <cstdint>

#include "goblin/gp/gpu_evaluation/evaluate.h"
#include "goblin/gp/gpu_evaluation/launch_config.h"
#include "goblin/gp/gpu_evaluation/memory.h"
#include "goblin/gp/gpu_evaluation/types.h"

#define __CHECK_CUDA_ERR__(err) check((err), #err, __FILE__, __LINE__)

#define BLOCK_SIZE_LIST \
    X(32)  X(64)  X(96)  X(128) X(160) X(192) X(224) X(256) \
    X(288) X(320) X(352) X(384) X(416) X(448) X(480) X(512) \
    X(544) X(576) X(608) X(640) X(672) X(704) X(736) X(768) \
    X(800) X(832) X(864) X(896) X(928) X(960) X(992) X(1024)

#define MAX_STACK_DEPTH 16
#define MAX_NUM_NODES 256
#define MAX_ARITY 2

namespace goblin {

__global__
void evaluate_kernel_baseline(
    float* X, 
    float* Y, 
    u8* v_type, 
    float* v_value, 
    float* partial,
    size_t solution_length, 
    size_t num_datapoints
) {
    // Calculate datapoint index
    size_t datapoint_index = (static_cast<size_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    size_t solution_index = static_cast<size_t>(blockIdx.x);

    if (datapoint_index < num_datapoints) {
        // Calculate offset for first element of solution
        size_t solution_offset = solution_index * solution_length;

        // Pointers to first element of solution
        u8* type = v_type + solution_offset;
        float* value = v_value + solution_offset;

        // Compute output of solution
        float output = compute_tree_output_baseline(
            X, type, value, 
            solution_length, 
            num_datapoints, 
            datapoint_index
        );

        // Determine squared error
        float error = output - Y[datapoint_index];
        float se = error * error;

        size_t partial_index = (solution_index * num_datapoints) + datapoint_index;

        // Store squared error in global memory
        partial[partial_index] = se; 
    }
};

__global__
void evaluate_kernel_restrict(
    const float* __restrict__ X, 
    const float* __restrict__ Y, 
    const u8* __restrict__ v_type, 
    const float* __restrict__ v_value, 
    float* __restrict__ partial,
    size_t solution_length, 
    size_t num_datapoints
) {
    // Calculate datapoint index
    size_t datapoint_index = (static_cast<size_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    size_t solution_index = static_cast<size_t>(blockIdx.x);

    if (datapoint_index < num_datapoints) {
        // Calculate offset for first element of solution
        size_t solution_offset = solution_index * solution_length;

        // Pointers to first element of solution
        const u8* type = v_type + solution_offset;
        const float* value = v_value + solution_offset;

        // Compute output of solution
        float output = compute_tree_output_restrict(
            X, type, value, 
            solution_length, 
            num_datapoints, 
            datapoint_index
        );

        // Determine squared error
        float error = output - Y[datapoint_index];
        float se = error * error;

        size_t partial_index = (solution_index * num_datapoints) + datapoint_index;

        // Store squared error in global memory
        partial[partial_index] = se;
    }
};


__global__
void evaluate_kernel_shared_memory(
    const float* __restrict__ X, 
    const float* __restrict__ Y, 
    const u8* __restrict__ v_type, 
    const float* __restrict__ v_value, 
    float* __restrict__ partial,
    size_t solution_length,
    size_t num_datapoints
) {
    // Calculate datapoint index
    size_t datapoint_index = (static_cast<size_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    size_t solution_index = static_cast<size_t>(blockIdx.x);

    // Calculate offset for solution
    size_t solution_offset = solution_index * solution_length;

    // Layout shared memory
    __shared__ u8 sh_type[MAX_NUM_NODES];
    __shared__ float sh_value[MAX_NUM_NODES];

    // Cooperative load of solution data into shared memory
    for (size_t i = threadIdx.x; i < solution_length; i+= blockDim.x) {
        sh_type[i] = v_type[solution_offset + i];
        sh_value[i] = v_value[solution_offset + i];
    }

    // Ensure all solution data has been copied into shared memory
    __syncthreads();

    if (datapoint_index < num_datapoints) {
        // Compute output of solution
        float output = compute_tree_output_restrict(
            X, sh_type, sh_value, 
            solution_length, 
            num_datapoints, 
            datapoint_index
        );

        // Determine squared error
        float error = output - Y[datapoint_index];
        float se = error * error;

        size_t partial_index = (solution_index * num_datapoints) + datapoint_index;

        // Store squared error in global memory
        partial[partial_index] = se;
    }
};

template <int BLOCK_SIZE>
__global__
void evaluate_kernel_block_reduce(
    const float* __restrict__ X, 
    const float* __restrict__ Y, 
    const u8* __restrict__ v_type, 
    const float* __restrict__ v_value, 
    float* __restrict__ partial,
    size_t solution_length, 
    size_t num_datapoints
) {
    // Calculate datapoint index
    // blockIdx.y = block index of solution
    // blockDim.x = number of threads in a block
    const size_t datapoint_index = (static_cast<size_t>(blockIdx.y) * blockDim.x) + threadIdx.x;
    const size_t solution_index = static_cast<size_t>(blockIdx.x);

    // Calculate offset for solution
    const size_t solution_offset = solution_index * solution_length;

    // Layout shared memory
    __shared__ u8 sh_type[MAX_NUM_NODES];
    __shared__ float sh_value[MAX_NUM_NODES];

    // Cooperative load of solution data into shared memory
    for (size_t i = threadIdx.x; i < solution_length; i+= blockDim.x) {
        sh_type[i] = v_type[solution_offset + i];
        sh_value[i] = v_value[solution_offset + i];
    }

    __syncthreads();

    float se = 0.0F;

    // Early exit if thread does not correspond to datapoint
    if (datapoint_index < num_datapoints) {
        // Compute output of solution
        float output = compute_tree_output_restrict(
            X, sh_type, sh_value, 
            solution_length, 
            num_datapoints, 
            datapoint_index
        );

        // Determine squared error
        float error = output - Y[datapoint_index];
        se = error * error;
    }

    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float block_sum = BlockReduce(temp_storage).Sum(se);

    __syncthreads();

    if (threadIdx.x == 0) {
        // gridDim.y = Number of partial results per solution
        // blockIdx.y = Partial result index of current solution
        size_t partial_index = (gridDim.y * solution_index) + blockIdx.y;
        partial[partial_index] = block_sum;
    }
}

// https://nvidia.github.io/cccl/cub/index.html#flexible-data-arrangement
template <int BLOCK_THREADS>
__global__
void evaluate_mse_kernel(
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

    size_t solution_index = static_cast<size_t>(blockIdx.x);

    // Calculate offset for solution
    size_t solution_offset = solution_index * solution_length;

    // Layout shared memory
    __shared__ u8 sh_type[MAX_NUM_NODES];
    __shared__ float sh_value[MAX_NUM_NODES];

    // Cooperative load of solution data into shared memory
    for (size_t i = threadIdx.x; i < solution_length; i+= blockDim.x) {
        sh_type[i] = v_type[solution_offset + i];
        sh_value[i] = v_value[solution_offset + i];
    }

    __syncthreads();

    float se = 0.0F;

    for (size_t i = 0; i < datapoints_per_thread; i++) {
        // Calculate datapoint index
        size_t datapoint_index = (i * blockDim.x) + threadIdx.x;

        // Check if datapoint_index corresponds to actual datapoint
        if (datapoint_index < num_datapoints) {
            // Compute output of solution
            float output = compute_tree_output_restrict(
                X, sh_type, sh_value, 
                solution_length, 
                num_datapoints, 
                datapoint_index
            );

            // Determine squared error
            float error = output - Y[datapoint_index];
            se += error * error;
        }
    }

    float sum = BlockReduce(temp_storage).Sum(se);

    if (threadIdx.x == 0) {
        result[solution_index] = sum / num_datapoints;
    }
}

template <int BLOCK_THREADS>
__global__
void evaluate_mse_kernel_fmaf(
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

    size_t solution_index = static_cast<size_t>(blockIdx.x);

    // Calculate offset for solution
    size_t solution_offset = solution_index * solution_length;

    // Layout shared memory
    __shared__ u8 sh_type[MAX_NUM_NODES];
    __shared__ float sh_value[MAX_NUM_NODES];

    // Cooperative load of solution data into shared memory
    for (size_t i = threadIdx.x; i < solution_length; i+= blockDim.x) {
        sh_type[i] = v_type[solution_offset + i];
        sh_value[i] = v_value[solution_offset + i];
    }

    __syncthreads();

    float se = 0.0F;

    for (size_t i = 0; i < datapoints_per_thread; i++) {
        // Calculate datapoint index
        size_t datapoint_index = (i * blockDim.x) + threadIdx.x;

        // Check if datapoint_index corresponds to actual datapoint
        if (datapoint_index < num_datapoints) {
            // Compute output of solution
            float output = compute_tree_output_restrict(
                X, sh_type, sh_value, 
                solution_length, 
                num_datapoints, 
                datapoint_index
            );

            // Determine squared error
            float error = output - Y[datapoint_index];
            // se += error * error;
            se = __fmaf_rn(error, error, se); // TODO check SASS code
        }
    }

    float sum = BlockReduce(temp_storage).Sum(se);

    if (threadIdx.x == 0) {
        result[solution_index] = sum / num_datapoints;
    }
}

template <int BLOCK_THREADS>
__global__
void evaluate_mse_kernel_inplace(
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

    size_t solution_index = static_cast<size_t>(blockIdx.x);

    // Calculate offset for solution
    size_t solution_offset = solution_index * solution_length;

    // Layout shared memory
    __shared__ u8 sh_type[MAX_NUM_NODES];
    __shared__ float sh_value[MAX_NUM_NODES];

    // Cooperative load of solution data into shared memory
    for (size_t i = threadIdx.x; i < solution_length; i+= blockDim.x) {
        sh_type[i] = v_type[solution_offset + i];
        sh_value[i] = v_value[solution_offset + i];
    }

    __syncthreads();

    float se = 0.0F;

    for (size_t i = 0; i < datapoints_per_thread; i++) {
        // Calculate datapoint index
        size_t datapoint_index = (i * blockDim.x) + threadIdx.x;

        // Check if datapoint_index corresponds to actual datapoint
        if (datapoint_index < num_datapoints) {
            // Compute output of solution
            float output = compute_tree_output_inplace(
                X, sh_type, sh_value, 
                solution_length, 
                num_datapoints, 
                datapoint_index
            );

            // Determine squared error
            float error = output - Y[datapoint_index];
            // se += error * error;
            se = __fmaf_rn(error, error, se); // TODO check SASS code
        }
    }

    float sum = BlockReduce(temp_storage).Sum(se);

    if (threadIdx.x == 0) {
        result[solution_index] = sum / num_datapoints;
    }
}

template <int BLOCK_THREADS>
__global__
void evaluate_kernel_hybrid(
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

    // Calculate first datapoint index for thread
    size_t datapoint_index = (static_cast<size_t>(blockIdx.y) * datapoints_per_block) + threadIdx.x;
    // Calculate last datapoint index for current block to ensure that datapoints are not processed
    // by multiple blocks
    const size_t last_datapoint_index = (static_cast<size_t>(blockIdx.y) + 1) * datapoints_per_block;
    const size_t solution_index = static_cast<size_t>(blockIdx.x);
    
    // Calculate offset for solution
    const size_t solution_offset = solution_index * solution_length;

    // Layout shared memory
    __shared__ u8 sh_type[MAX_NUM_NODES];
    __shared__ float sh_value[MAX_NUM_NODES];

    // Cooperative load of solution data into shared memory
    for (size_t i = threadIdx.x; i < solution_length; i += blockDim.x) {
        sh_type[i] = v_type[solution_offset + i];
        sh_value[i] = v_value[solution_offset + i];
    }

    __syncthreads();

    float se = 0.0F;

    for (size_t i = 0; i < datapoints_per_thread; i++) {
        // To ensure coalesced memory access of the datapoints, datapoints are divided across
        // threads in a strided manner.
        // E.g.: thread 0 in block 0 will compute datapoints at indices 0, 1 * blockDim.x, 2 * blockDim.x,
        // where blockDim.x is the amount of threads in the block
        if (datapoint_index < num_datapoints && datapoint_index < last_datapoint_index) {
            float output = compute_tree_output_inplace(
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
        // gridDim.y = blocks_per_individual
        const size_t partial_index = (gridDim.y * solution_index) + blockIdx.y;
        partial[partial_index] = block_sum;
    }
}

__device__
float compute_tree_output_baseline(
    float* X, 
    const u8* type,
    const float* value,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
) {
    // Evaluation stack (per thread)
    float stack[MAX_STACK_DEPTH];
    int sp = 0;

    // Traverse through solution from left to right
    for (size_t index = 0; index < solution_length; index++) {
        // Get type of current element
        NodeType node_type = static_cast<NodeType>(type[index]);

        if (node_type == NodeType::Input) {
            size_t input_index = size_t(value[index]);
            // Push input variable onto stack and increase stack pointer
            stack[sp++] = X[datapoint_index + (input_index * num_datapoints)];
        } else if (node_type == NodeType::Constant) {
            // Push constant value onto stack and increase stack pointer
            stack[sp++] = value[index];
        } else if (node_type == NodeType::Operator) {
            Operator op_value = static_cast<Operator>(value[index]);
            
            // Determine arity of operator
            int arity;
            switch (op_value) {
                case Operator::Sin:
                case Operator::Cos:
                case Operator::Exp:
                case Operator::Log:
                case Operator::Square:
                case Operator::Sqrt:
                case Operator::Abs:
                    arity = 1;
                    break;
                default:
                    arity = 2;
                    break;
            }
            
            // Get operands from stack
            float args[MAX_ARITY];
            for (int j = 0; j < arity; j++) {
                args[j] = stack[--sp];
            }

            float res = 0.0F;

            switch (op_value) {
                case Operator::Add: res = args[1] + args[0]; break;
                case Operator::Sub: res = args[1] - args[0]; break;
                case Operator::Mul: res = args[1] * args[0]; break;
                case Operator::Div: res = args[1] / args[0]; break;
                case Operator::Sin: res = sinf(args[0]); break;
                case Operator::Cos: res = cosf(args[0]); break;
                case Operator::Exp: res = expf(args[0]); break;
                case Operator::Log: res = logf(args[0]); break;
                case Operator::Square: res = args[0] * args[0]; break;
                case Operator::Sqrt: res = sqrtf(args[0]); break;
                case Operator::Pow: res = powf(args[1], args[0]); break;
                case Operator::Abs: res = fabsf(args[0]); break;
                case Operator::Min: res = fminf(args[1], args[0]); break;
                case Operator::Max: res = fmaxf(args[1], args[0]); break;
            }

            stack[sp++] = res;   
        } else {
            break;
        }
    }

    // Final result for solution is on the top of the stack
    float result = stack[--sp];

    return result;
}

__device__
float compute_tree_output_restrict(
    const float* __restrict__ X, 
    const u8* __restrict__ type,
    const float* __restrict__ value,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
) {
    // Evaluation stack (per thread)
    float stack[MAX_STACK_DEPTH];
    int sp = 0;

    // Traverse through solution from left to right
    for (size_t index = 0; index < solution_length; index++) {
        // Get type of current element
        NodeType node_type = static_cast<NodeType>(type[index]);

        if (node_type == NodeType::Input) {
            size_t input_index = static_cast<size_t>(value[index]);
            // Push input variable onto stack and increase stack pointer
            stack[sp++] = X[datapoint_index + (input_index * num_datapoints)];
        } else if (node_type == NodeType::Constant) {
            // Push constant value onto stack and increase stack pointer
            stack[sp++] = value[index];
        } else if (node_type == NodeType::Operator) { // ValueKind::Operator
            // TODO improve
            Operator op_value = static_cast<Operator>(value[index]);
            
            // Determine arity of operator
            int arity;
            switch (op_value) {
                case Operator::Sin:
                case Operator::Cos:
                case Operator::Exp:
                case Operator::Log:
                case Operator::Square:
                case Operator::Sqrt:
                case Operator::Abs:
                    arity = 1;
                    break;
                default:
                    arity = 2;
                    break;
            }

            // Get operands from stack
            float args[MAX_ARITY];
            for (int j = 0; j < arity; j++) {
                args[j] = stack[--sp];
            }

            float res = 0.0F;

            switch (op_value) {
                case Operator::Add: res = args[1] + args[0]; break;
                case Operator::Sub: res = args[1] - args[0]; break;
                case Operator::Mul: res = args[1] * args[0]; break;
                case Operator::Div: res = args[1] / args[0]; break;
                case Operator::Sin: res = sinf(args[0]); break;
                case Operator::Cos: res = cosf(args[0]); break;
                case Operator::Exp: res = expf(args[0]); break;
                case Operator::Log: res = logf(args[0]); break;
                case Operator::Square: res = args[0] * args[0]; break;
                case Operator::Sqrt: res = sqrtf(args[0]); break;
                case Operator::Pow: res = powf(args[1], args[0]); break;
                case Operator::Abs: res = fabsf(args[0]); break;
                case Operator::Min: res = fminf(args[1], args[0]); break;
                case Operator::Max: res = fmaxf(args[1], args[0]); break;
            }

            stack[sp++] = res;   
        } else {
            break;
        }
    }

    // Final result for solution is on the top of the stack
    float result = stack[--sp];

    return result;
}

__device__
float compute_tree_output_inplace(
    const float* __restrict__ X, 
    const u8* __restrict__ type,
    const float* __restrict__ value,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
) {
    // Evaluation stack (per thread)
    float stack[MAX_STACK_DEPTH];
    int sp = 0;

    // Traverse through solution from left to right
    for (size_t index = 0; index < solution_length; index++) {
        // Get type of current element
        NodeType node_type = static_cast<NodeType>(type[index]);

        if (node_type == NodeType::Input) {
            size_t input_index = static_cast<size_t>(value[index]);
            // Push input variable onto stack and increase stack pointer
            stack[sp++] = X[datapoint_index + (input_index * num_datapoints)];
        } else if (node_type == NodeType::Constant) {
            // Push constant value onto stack and increase stack pointer
            stack[sp++] = value[index];
        } else if (node_type == NodeType::Operator) { // ValueKind::Operator
            Operator op_value = static_cast<Operator>(value[index]);

            // Apply the operator on the operands depending on op_value.
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

    // Final result for solution is on the top of the stack
    float result = stack[--sp];

    return result;
}

__global__
void mse_kernel_baseline(const float* partial, float* result, size_t num_solutions, size_t num_datapoints) {
    size_t solution_index = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;

    if (solution_index < num_solutions) {
        size_t start = solution_index * num_datapoints; 
        size_t end = start + num_datapoints - 1;

        float sum = 0.0F;

        // Loop through datapoints and sum squared errors
        for (size_t i = start; i <= end; i++) {
            sum += partial[i];
        }

        // Divide by num_datapoints to get the mean
        result[solution_index] = sum / num_datapoints;
    }
}

__global__
void mse_kernel_restrict(
    const float* __restrict__ partial, 
    float* __restrict__ result, 
    size_t num_solutions, 
    size_t num_datapoints
) {
    size_t solution_index = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;

    if (solution_index < num_solutions) {
        size_t start = solution_index * num_datapoints; 
        size_t end = start + num_datapoints - 1;

        float sum = 0.0F;

        // Loop through datapoints and sum squared errors
        for (size_t i = start; i <= end; i++) {
            sum += partial[i];
        }

        result[solution_index] = sum / num_datapoints;;
    }
}

// TODO add items_per_thread template
// https://nvidia.github.io/cccl/cub/index.html#flexible-data-arrangement
template <int BLOCK_THREADS, int ITEMS_PER_THREAD = 1>
__global__
void mse_kernel_block_reduce(
    const float* __restrict__ partial, 
    float* __restrict__ result,
    size_t num_partial,
    size_t num_datapoints
) {
    // Specialize BlockLoad and BlockReduce collective types
    using BlockLoad = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_DIRECT>;
    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

    // Allocate type-safe, repurposable shared memory for collectives
    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    size_t solution_index =static_cast<size_t>(blockIdx.x);
    size_t block_offset = solution_index * num_partial;

    float thread_data[ITEMS_PER_THREAD];

    // Use a block load to load the partial results
    BlockLoad(temp_storage.load).Load(partial + block_offset, thread_data, num_partial, 0);

    __syncthreads();

    float sum = BlockReduce(temp_storage.reduce).Sum(thread_data);

    __syncthreads();

    if (threadIdx.x == 0) {
        result[solution_index] = sum / num_datapoints;
    }
}

void evaluate_kernel_wrapper(
    float* X, 
    float* Y, 
    u8* type, 
    float* value, 
    float* partial,
    const LaunchConfig config
) {
    dim3 block(config.eval.block.x, config.eval.block.y, config.eval.block.z);
    dim3 grid(config.eval.grid.x, config.eval.grid.y, config.eval.grid.z);

    // Launch evaluate kernel that calculates the squared error for every solution and datapoint combination
    switch (config.kernel_version) {
        case (KernelVersion::Baseline):
            evaluate_kernel_baseline<<<grid, block>>>(X, Y, type, value, partial, config.solution_length, config.num_datapoints);
            break;
        case (KernelVersion::Restrict):
            evaluate_kernel_restrict<<<grid, block>>>(X, Y, type, value, partial, config.solution_length, config.num_datapoints);
            break;
        case (KernelVersion::SharedMemory):
            evaluate_kernel_shared_memory<<<grid, block>>>(X, Y, type, value, partial, config.solution_length, config.num_datapoints);
            break;
        case (KernelVersion::BlockReduce):
            switch (block.x) {
                #define X(BS) \
                case BS: \
                    evaluate_kernel_block_reduce<BS><<<grid, block>>>( \
                        X, Y, type, value, partial, config.solution_length, config.num_datapoints); \
                    break;
                BLOCK_SIZE_LIST
                #undef X

                default:
                    assert(false && "Unsupported block size");
            }
            break;
        case (KernelVersion::Hybrid):
            switch (block.x) {
                #define X(BS) \
                case BS: \
                    evaluate_kernel_hybrid<BS><<<grid, block>>>( \
                        X, Y, type, value, partial, config.solution_length, config.num_datapoints, \
                        config.datapoints_per_block, config.datapoints_per_thread); \
                    break;
                BLOCK_SIZE_LIST
                #undef X

                default:
                    assert(false && "Unsupported block size");
            }
            break;
        default:
            break;
    }

    __CHECK_CUDA_ERR__(cudaGetLastError());

    // Wait until all blocks and threads are done
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());
}

void mse_kernel_wrapper(
    float* partial, 
    float* result, 
    const LaunchConfig config
) {
    dim3 block(config.mse.block.x, config.mse.block.y, config.mse.block.z);
    dim3 grid(config.mse.grid.x, config.mse.grid.y, config.mse.grid.z);

    size_t num_partials = config.eval.grid.y;
    
    switch(config.kernel_version) {
        case (KernelVersion::Baseline):
            mse_kernel_baseline<<<grid, block>>>(partial, result, config.num_solutions, config.num_datapoints);
            break;
        case (KernelVersion::Restrict):
        case (KernelVersion::SharedMemory):
            mse_kernel_restrict<<<grid, block>>>(partial, result, config.num_solutions, config.num_datapoints);  
            break;
        case (KernelVersion::BlockReduce):
        case (KernelVersion::Hybrid):
            switch (config.mse.block.x) {
                #define X(BS) \
                case BS: \
                    mse_kernel_block_reduce<BS><<<grid, block>>>( \
                        partial, result, num_partials, config.num_datapoints); \
                    break;
                BLOCK_SIZE_LIST
                #undef X

                default:
                    assert(false && "Unsupported block size");
            }
            break;
        default:
            break;
    }

    __CHECK_CUDA_ERR__(cudaGetLastError());

    // Wait until all blocks and threads are done
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());
}

void evaluate_mse_kernel_wrapper(
    float* X, 
    float* Y, 
    u8* type, 
    float* value, 
    float* result,
    const LaunchConfig config
) {
    dim3 block(config.eval.block.x, config.eval.block.y, config.eval.block.z);
    dim3 grid(config.eval.grid.x, config.eval.grid.y, config.eval.grid.z);

    switch(config.kernel_version) {
        case (KernelVersion::SingleKernel):
            switch (block.x) {
                #define X(BS) \
                case BS: \
                    evaluate_mse_kernel<BS><<<grid, block>>>( \
                        X, Y, type, value, result, config.solution_length, config.num_datapoints, config.datapoints_per_thread); \
                    break;
                BLOCK_SIZE_LIST
                #undef X

                default: 
                    assert(false && "Unsupported block size");
            }
            break;
        case (KernelVersion::SingleKernelFMAF):
            switch (block.x) {
                #define X(BS) \
                case BS: \
                    evaluate_mse_kernel_fmaf<BS><<<grid, block>>>( \
                        X, Y, type, value, result, config.solution_length, config.num_datapoints, config.datapoints_per_thread); \
                    break;
                BLOCK_SIZE_LIST
                #undef X

                default: 
                    assert(false && "Unsupported block size");
            }
            break;
        case (KernelVersion::SingleKernelInplace):
            switch (block.x) {
                #define X(BS) \
                case BS: \
                    evaluate_mse_kernel_inplace<BS><<<grid, block>>>( \
                        X, Y, type, value, result, config.solution_length, config.num_datapoints, config.datapoints_per_thread); \
                    break;
                BLOCK_SIZE_LIST
                #undef X

                default: 
                    assert(false && "Unsupported block size");
            }
            break;
        default:
            break;
    }

    __CHECK_CUDA_ERR__(cudaGetLastError());

    // Wait until all blocks and threads are done
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
    if (config.kernel_version == KernelVersion::SingleKernel ||
        config.kernel_version == KernelVersion::SingleKernelFMAF ||
        config.kernel_version == KernelVersion::SingleKernelInplace) {
        evaluate_mse_kernel_wrapper(X, Y, type, value, result, config);
    } else {
        // Launch evaluate kernel that calculates the squared error for every solution and datapoint combination
        evaluate_kernel_wrapper(X, Y, type, value, partial, config);

        // Launch mse kernel that calculates the mse over all datapoints for each solution
        mse_kernel_wrapper(partial, result, config);
    }
}

/* 
 * The wrapper function below is only required to test the compute_tree_output kernel.
 * The compute_tree_output kernel is defined as a device-only function, so it cannot be 
 * called from functions running on the host. However, the wrapper can be called from the device,
 * and can call device-only functions.
*/
__global__
void compute_tree_output_wrapper(
    float* X, 
    u8* type,
    float* value,
    float* result,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index,
    KernelVersion version
) {
    if (version == KernelVersion::Baseline) {
        result[0] = compute_tree_output_baseline(
            X, 
            type, 
            value, 
            solution_length, 
            num_datapoints, 
            datapoint_index
        );
    } else if (version == KernelVersion::Restrict) {
        result[0] = compute_tree_output_restrict(
            X, 
            type, 
            value, 
            solution_length, 
            num_datapoints, 
            datapoint_index
        );
    } else if (version == KernelVersion::SingleKernelInplace) {
        result[0] = compute_tree_output_inplace(
            X, 
            type, 
            value, 
            solution_length, 
            num_datapoints, 
            datapoint_index
        );
    }

    
}

float test_compute_output_kernel(
    std::vector<float> h_X,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_datapoints,
    size_t datapoint_index,
    KernelVersion version
) {
    // Length of a single solution vector
    size_t solution_length = h_type.size();

    // Allocate memory and copy data
    float* d_X = allocate_and_copy(h_X.data(), h_X.size());
    u8* d_type = allocate_and_copy(h_type.data(), solution_length);
    float* d_value = allocate_and_copy(h_value.data(), solution_length);

    // Allocate memory
    float* d_result = allocate_on_gpu<float>(1);

    dim3 block(1);
    dim3 grid(1, 1);

    compute_tree_output_wrapper<<<grid, block>>>(
        d_X, d_type, d_value, d_result, 
        solution_length, num_datapoints, datapoint_index, version
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

std::vector<float> test_evaluate_kernel(
    std::vector<float> h_X, 
    std::vector<float> h_Y, 
    std::vector<u8> h_type, 
    std::vector<float> h_value, 
    size_t num_solutions,
    size_t num_datapoints,
    KernelVersion version
) {
    size_t solution_length = h_type.size() / num_solutions;

    // Allocate memory and copy data
    float* d_X = allocate_and_copy(h_X.data(), h_X.size());
    float* d_Y = allocate_and_copy(h_Y.data(), h_Y.size());
    u8* d_type = allocate_and_copy(h_type.data(), h_type.size());
    float* d_value = allocate_and_copy(h_value.data(), h_value.size());

    size_t partial_size;

    if (version == KernelVersion::BlockReduce) {
        partial_size = num_solutions;
    } else {
        partial_size = num_solutions * num_datapoints;
    }

    float* d_partial = allocate_on_gpu<float>(partial_size);    

    LaunchConfig config = LaunchConfig::determine(version, num_solutions, num_datapoints, solution_length, std::nullopt);

    evaluate_kernel_wrapper(d_X, d_Y, d_type, d_value, d_partial, config);

    std::vector<float> result(partial_size);
    __CHECK_CUDA_ERR__(cudaMemcpy(result.data(), d_partial, partial_size * sizeof(float), cudaMemcpyDeviceToHost));

    free_on_gpu(d_type);
    free_on_gpu(d_value);
    free_on_gpu(d_partial);

    return result;
}

std::vector<float> test_compute_mse_kernel(
    std::vector<float> partial, 
    size_t num_solutions, 
    size_t num_datapoints, 
    KernelVersion version
) {
    // Allocate memory and copy data
    float* d_partial = allocate_and_copy(partial.data(), partial.size());

    // Allocate memory
    float* d_result = allocate_on_gpu<float>(num_solutions);

    LaunchConfig config = LaunchConfig::determine(version, num_solutions, num_datapoints, 0, std::nullopt);

    mse_kernel_wrapper(d_partial, d_result, config);

    std::vector<float> result(num_solutions);
    __CHECK_CUDA_ERR__(cudaMemcpy(result.data(), d_result, num_solutions * sizeof(float), cudaMemcpyDeviceToHost));

    free_on_gpu(d_partial);
    free_on_gpu(d_result);

    return result;
}

std::vector<float> test_evaluate_mse_kernel(
    std::vector<float> h_X, 
    std::vector<float> h_Y, 
    std::vector<u8> h_type, 
    std::vector<float> h_value, 
    size_t num_solutions,
    size_t num_datapoints,
    KernelVersion version
) {
    size_t solution_length = h_type.size() / num_solutions;

    // Allocate memory and copy data
    float* d_X = allocate_and_copy(h_X.data(), h_X.size());
    float* d_Y = allocate_and_copy(h_Y.data(), h_Y.size());
    u8* d_type = allocate_and_copy(h_type.data(), h_type.size());
    float* d_value = allocate_and_copy(h_value.data(), h_value.size());

    float* d_result = allocate_on_gpu<float>(num_solutions);    

    LaunchConfig config = LaunchConfig::determine(KernelVersion::SingleKernel, num_solutions, num_datapoints, solution_length, std::nullopt);

    evaluate_mse_kernel_wrapper(d_X, d_Y, d_type, d_value, d_result, config);

    std::vector<float> result(num_solutions);
    __CHECK_CUDA_ERR__(cudaMemcpy(result.data(), d_result, num_solutions * sizeof(float), cudaMemcpyDeviceToHost));

    free_on_gpu(d_type);
    free_on_gpu(d_value);
    free_on_gpu(d_result);

    return result;
}

std::vector<float> test_kernel_hybrid(
    std::vector<float> h_X,
    std::vector<float> h_Y,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_solutions,
    size_t num_datapoints,
    size_t blocks_per_individual
) {
    size_t solution_length = h_type.size() / num_solutions;

    float* d_X = allocate_and_copy(h_X.data(), h_X.size());
    float* d_Y = allocate_and_copy(h_Y.data(), h_Y.size());
    u8* d_type = allocate_and_copy(h_type.data(), h_type.size());
    float* d_value = allocate_and_copy(h_value.data(), h_value.size());

    float* d_partial = allocate_on_gpu<float>(num_solutions * blocks_per_individual);
    float* d_result = allocate_on_gpu<float>(num_solutions);

    LaunchConfig config = LaunchConfig::determine(
        KernelVersion::Hybrid, num_solutions, num_datapoints, solution_length, blocks_per_individual
    );

    kernel_wrapper(d_X, d_Y, d_type, d_value, d_partial, d_result, config);

    std::vector<float> result(num_solutions);
    __CHECK_CUDA_ERR__(cudaMemcpy(result.data(), d_result, num_solutions * sizeof(float), cudaMemcpyDeviceToHost));

    free_on_gpu(d_partial);
    free_on_gpu(d_result);
    free_on_gpu(d_type);
    free_on_gpu(d_value);

    return result;
}

};