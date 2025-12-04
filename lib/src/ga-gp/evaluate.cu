#include <vector>

#include "goblin/ga-gp/evaluate.h"
#include "goblin/ga-gp/helper.h"
#include "goblin/ga-gp/types.h"

#define __CHECK_CUDA_ERR__(err) check((err), #err, __FILE__, __LINE__)

// Defines the number of elements in the stack that holds temporary values for the current thread
// Depends on the maximum tree depth (e.g. depth = 2 -> 7 temp values, depth = 3 -> 15 temp values)
#define MAX_STACK_DEPTH 64
#define MAX_NUM_NODES 64

namespace goblin {

/* 
 * The evaluation kernel handles the evaluation of the solutions over all the data.
 * The grid is 2 dimensional and the blocks are 1 dimensional. 
 * The grids and block dimensions are used as follows:
 * - The block dimension is based on the amount of datapoints.
 * - The x dimension of the grid is for the solutions (e.g. x = 0 is the first solution, x = 1 the second, etc..)
 * - The y dimension of the grid is for the possibility that the number of datapoints is more than the max amount
 *   of threads per block. If that is the case, multiple blocks are assigned to the same solution. (e.g. y = 0 is for 
 *   the first n datapoints, y = 1 for the second n datapoints, where n is the number of threads per block)
 * 
 * Parameters:
 * - X: pointer to the array containing all the datapoints stored in column-major order. 
 * - Y: pointer to the array containing the expected outputs for the data
 * - v_type: pointer to the array containing the types of the solution nodes
 * - v_value: pointer to the array containing the values of the solution nodes
 * - solution_length: max number of elements in a solution
 * - num_inputs: number of inputs (data columns)
 * - num_datapoints: number of datapoints (data rows)
 * 
 * Indexing:
 * - The solutions are stored contiguously in memory. To access a solution we need to know its index, and how much
 *   memory each solution occupies in memory. The first element of the nth solution, where each solution consists of
 *   m nodes is than located at offset n * m. 
 * - The datapoints are stored in column-major order. This means the nth input of the mth datapoint resides at m + n * rows
 * 
 * 
 * Evaluation
 * - The nodes of a solution are stored in a reverse preorder. So the right most child is the first element, and the root 
 *   is the last element. The expression tree should be evaluated bottom up, which can be done by traversing the elements
 *   in order (from left to right).
 * - Numerical values such as constants and inputs are pushed onto the stack when encountered. For operators, the arity
 *   is determined, and that amount of values are popped from the stack. The result of the operation is pushed onto 
 *   the stack again. This is process continues until there are no more nodes to evaluate.
*/

__global__
void evaluate_kernel(
    float* __restrict__ X, 
    float* __restrict__ Y, 
    float* __restrict__ v_type, 
    float* __restrict__ v_value, 
    int solution_length, 
    int num_datapoints,
    float* result
) {
    // Calculate datapoint index
    int datapoint_index = blockIdx.y * blockDim.x + threadIdx.x;
    int solution_index = blockIdx.x;

    // Calculate offset for solution
    int solution_offset = solution_index * solution_length;

    // Layout shared memory
    __shared__ float shmem[MAX_NUM_NODES * 2];
    float* sh_type = (float*)shmem;
    float* sh_value = (float*)(sh_type + solution_length);

    // Cooperative load of solution data into shared memory
    for (int i = threadIdx.x; i < solution_length; i+= blockDim.x) {
        sh_type[i] = v_type[solution_offset + i];
        sh_value[i] = v_value[solution_offset + i];
    }

    __syncthreads();

    // Early exit if thread does not correspond to datapoint
    if (datapoint_index < num_datapoints) {
        // Compute output of solution
        float output = compute_tree_output(
            X, sh_type, sh_value, 
            solution_length, 
            num_datapoints, 
            datapoint_index
        );

        // Determine squared error
        float error = output - Y[datapoint_index];
        float se = error * error;

        // Store squared error in global memory
        result[solution_index * num_datapoints + datapoint_index] = se;
    }
};

__device__
float compute_tree_output(
    float* __restrict__ X, 
    float* __restrict__ type,
    float* __restrict__ value,
    int solution_length,
    int num_datapoints,
    int datapoint_index
) {
    // Evaluation stack (per thread)
    float stack[MAX_STACK_DEPTH];
    int sp = 0;

    // Traverse through solution from left to right
    for (int index = 0; index < solution_length; index++) {
        // Get type of current element
        NodeType t = static_cast<NodeType>(type[index]);

        if (t == NodeType::Input) {
            int input_index = int(value[index]);
            // Push input variable onto stack and increase stack pointer
            stack[sp++] = X[datapoint_index + input_index * num_datapoints];
        } else if (t == NodeType::Constant) {
            // Push constant value onto stack and increase stack pointer
            stack[sp++] = value[index];
        } else if (t == NodeType::Operator) { // ValueKind::Operator
            // TODO improve
            Operator op_value = static_cast<Operator>(static_cast<int>(value[index]));
            
            // Get operands from stack depending on arity of operator
            int arity = 2; // Currently only arity of 2 is supported
            float args[2];
            for (int j = 0; j < arity; j++) {
                args[j] = stack[--sp];
            }

            float res = 0.0f;

            switch (op_value) {
                case Operator::Add: res = args[0] + args[1]; break;
                case Operator::Sub: res = args[0] - args[1]; break;
                case Operator::Mul: res = args[0] * args[1]; break;
                case Operator::Div: res = (args[1] == 0.0f) ? 0.0f : args[0] / args[1]; break;
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

__global__
void compute_mse_kernel(float* se, float* mse, int num_solutions, int num_datapoints) {
    int solution_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (solution_index < num_solutions) {
        int se_start = solution_index * num_datapoints; 
        int se_end = se_start + num_datapoints - 1;

        float sum = 0.0f;

        // Loop through datapoints and sum squared errors
        for (int i = se_start; i <= se_end; i++) {
            sum += se[i];
        }

        // Divide by num_datapoints to get the mean
        float result = sum / num_datapoints;

        mse[solution_index] = result;
    }
}

void evaluate_kernel_wrapper(
    float* X, 
    float* Y, 
    float* type, 
    float* value, 
    int solution_length, 
    int num_solutions,
    int num_datapoints,
    float* se
) {
    // Dimensions for evaluation kernel
    int eval_block_x = compute_block_size(num_datapoints);
    int eval_grid_y = (num_datapoints + eval_block_x - 1) / eval_block_x;
    
    dim3 block(eval_block_x);
    dim3 grid(num_solutions, eval_grid_y);

    // Launch evaluate kernel that calculates the squared error for every solution and datapoint combination
    evaluate_kernel<<<grid, block>>>(X, Y, type, value, solution_length, num_datapoints, se);
    __CHECK_CUDA_ERR__(cudaGetLastError());

    // Wait until all blocks and threads are done
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());
}

void compute_mse_kernel_wrapper(
    float* se, 
    float* mse, 
    int num_solutions, 
    int num_datapoints
) {
    // Dimensions for mse kernel
    int mse_block_x = compute_block_size(num_solutions);
    int mse_grid_x = (num_solutions + mse_block_x - 1) / mse_block_x;

    dim3 block(mse_block_x);
    dim3 grid(mse_grid_x);

    // Launch mse kernel that calculates the mse over all datapoints for each solution
    compute_mse_kernel<<<grid, block>>>(se, mse, num_solutions, num_datapoints);
    __CHECK_CUDA_ERR__(cudaGetLastError());

    // Wait until all blocks and threads are done
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());
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
    float* type,
    float* value,
    int solution_length,
    int num_datapoints,
    int datapoint_index,
    float* result
) {
    result[0] = compute_tree_output(
        X, 
        type, 
        value, 
        solution_length, 
        num_datapoints, 
        datapoint_index
    );
}

float test_compute_output_kernel(
    std::vector<float> h_X,
    std::vector<float> h_type,
    std::vector<float> h_value,
    int num_datapoints,
    int datapoint_index
) {
    // Length of a single solution vector
    int solution_length = h_type.size();

    // Allocate memory and copy data
    float* d_X = allocate_and_copy(h_X.data(), h_X.size());
    float* d_type = allocate_and_copy(h_type.data(), solution_length);
    float* d_value = allocate_and_copy(h_value.data(), solution_length);

    // Allocate memory
    float* d_result = allocate_on_gpu<float>(1);

    dim3 block(1);
    dim3 grid(1, 1);

    compute_tree_output_wrapper<<<grid, block>>>(d_X, d_type, d_value, solution_length, num_datapoints, datapoint_index, d_result);

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
    std::vector<float> h_type, 
    std::vector<float> h_value, 
    int num_solutions,
    int num_datapoints
) {
    int solution_length = h_type.size() / num_solutions;

    // Allocate memory and copy data
    float* d_X = allocate_and_copy(h_X.data(), h_X.size());
    float* d_Y = allocate_and_copy(h_Y.data(), h_Y.size());
    float* d_type = allocate_and_copy(h_type.data(), h_type.size());
    float* d_value = allocate_and_copy(h_value.data(), h_value.size());

    // Allocate memory
    float* d_result = allocate_on_gpu<float>(num_solutions * num_datapoints);

    dim3 block(num_datapoints);
    dim3 grid(num_solutions, 1);

    evaluate_kernel<<<grid, block>>>(d_X, d_Y, d_type, d_value, solution_length, num_datapoints, d_result);

    __CHECK_CUDA_ERR__(cudaGetLastError());
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());

    std::vector<float> result(num_solutions * num_datapoints);
    __CHECK_CUDA_ERR__(cudaMemcpy(result.data(), d_result, num_solutions * num_datapoints * sizeof(float), cudaMemcpyDeviceToHost));

    free_on_gpu(d_type);
    free_on_gpu(d_value);
    free_on_gpu(d_result);

    return result;
}

std::vector<float> test_compute_mse_kernel(std::vector<float> se, int num_solutions, int num_datapoints) {
    // Allocate memory and copy data
    float* d_se = allocate_and_copy(se.data(), se.size());

    // Allocate memory
    float* d_mse = allocate_on_gpu<float>(num_solutions);

    dim3 block(num_solutions);
    dim3 grid(1);

    compute_mse_kernel<<<grid, block>>>(d_se, d_mse, num_solutions, num_datapoints);

    __CHECK_CUDA_ERR__(cudaGetLastError());
    __CHECK_CUDA_ERR__(cudaDeviceSynchronize());

    std::vector<float> mse(num_solutions);
    __CHECK_CUDA_ERR__(cudaMemcpy(mse.data(), d_mse, num_solutions * sizeof(float), cudaMemcpyDeviceToHost));

    free_on_gpu(d_se);
    free_on_gpu(d_mse);

    return mse;
}

};