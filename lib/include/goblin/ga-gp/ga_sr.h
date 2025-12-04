#pragma once
#ifndef _GOBLIN_GA_GP_SR_H
#define _GOBLIN_GA_GP_SR_H

#include "goblin/ga-gp/evaluate.h"
#include "goblin/ga-gp/helper.h"
#include "goblin/ga-gp/types.h"
#include "goblin/gp/context.h"
#include "goblin/gp/init.h"
#include "goblin/gp/instance.h"
#include "goblin/lib/archive.h"
#include "goblin/lib/init.h"
#include "goblin/lib/types.h"

namespace goblin {

class GASRProblem : public GPInstanceBase {
    public: 
        GASRProblem(
            GPContext ctx,
            Arr2D<CType> X,
            Arr2D<CType> Y,
            bool linear_scaling = false,
            std::optional<AnyInit> init = std::nullopt,
            CType constant_init_lower_bound = -1.0,
            CType constant_init_upper_bound = 1.0
        ) : ctx(ctx),
            linear_scaling(linear_scaling),
            objective("mse"),
            _archive_fitness(MOFitness(1)),
            _fitness(MOFitness(1)),
            _init(from_any_init(init.value_or(std::make_shared<HalfHalfInit>()))),
            _target(_archive_fitness),
            _num_datapoints(X.rows()),
            _solution_length(ctx.max_expression_size) {
            
            _num_continuous = this->ctx.num_continuous;

            _continuous_upper_bounds = Vec<CType>::Constant(_num_continuous, std::numeric_limits<CType>::max());
            _continuous_lower_bounds = -_continuous_upper_bounds;

            _continuous_init_lower_bounds = Vec<CType>::Constant(_num_continuous, constant_init_lower_bound);
            _continuous_init_upper_bounds = Vec<CType>::Constant(_num_continuous, constant_init_upper_bound);

            // Copy data to GPU
            _copy_data_to_gpu(X, Y);
            _num_solutions_allocated = 0;
            _num_results_allocated = 0;
        }

        ~GASRProblem() {
            free_gpu();
        }

        void free_gpu() {
            _free_data_on_gpu();
            _free_solution_on_gpu();
            _free_results_on_gpu();
        }

        usize num_discrete() const override final { return ctx.num_discrete; };
        CRef<Vec<DType>> discrete_domain_sizes() const override final { return ctx.domain_sizes; };

        usize num_continuous() const override final { return _num_continuous; };
        CRef<Vec<CType>> continuous_lower_bounds() const override final { return _continuous_lower_bounds; };
        CRef<Vec<CType>> continuous_upper_bounds() const override final { return _continuous_upper_bounds; };

        CRef<Vec<CType>> continuous_init_lower_bounds() const override final { return _continuous_init_lower_bounds; };
        CRef<Vec<CType>> continuous_init_upper_bounds() const override final { return _continuous_init_upper_bounds; };


        void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
            int num_solutions = indices.size();

            if (num_solutions == 0) {
                return;
            }

            // Transform solutions to GPU compatible representation
            std::vector<float> node_type;
            std::vector<float> node_value;

            for (auto i : indices) {
                ctx.to_gpu_repr(solutions[i], node_type, node_value);
            }

            __goblin_runtime_assert(node_type.size() == node_value.size());

            // Copy solution data to GPU
            _copy_solutions_to_gpu(node_type, node_value);

            // Allocate memory for se and mse on device
            _allocate_results_on_gpu(num_solutions);

            // Launch evaluate kernel that calculates the squared error for every solution and datapoint combination
            evaluate_kernel_wrapper(
                d_X,
                d_Y,
                d_type,
                d_value,
                _solution_length,
                num_solutions,
                _num_datapoints,
                d_se
            );

            // Launch mse kernel that calculates the mse over all datapoints for each solution
            compute_mse_kernel_wrapper(
                d_se,
                d_mse,
                num_solutions,
                _num_datapoints
            );

            // Retrieve the results from the GPU
            std::vector<float> result(num_solutions);
            copy_from_device(result.data(), d_mse, num_solutions);

            size_t k = 0;
            for (auto i : indices) {
                solutions[i].quality().constraint_value = 0.0;
                solutions[i].quality().objectives(0) = result[k++];
            }
        }

        void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override final {
            _init->add_random(rng, *this, solutions, count);
        };

        const FitnessBase& fitness() const override final { return _fitness; };

        const ArchiveFitnessBase& archive_fitness() const override final { return _archive_fitness; };

        const GPContext& context() const override final { return ctx; }

        void register_target(CRefS<Vec<CType>> target_objectives) {
            _target.clear();
            Solution s(
                archive_fitness().worst(),
                num_discrete() > 0 ? std::make_optional<Vec<DType>>(Vec<DType>::Zero(num_discrete())) : std::nullopt,
                num_continuous() > 0 ? std::make_optional<Vec<CType>>(Vec<CType>::Zero(num_continuous())) : std::nullopt);
            s.quality().objectives = target_objectives;
            __goblin_runtime_assert(static_cast<usize>(s.quality().objectives.size()) >= fitness().num_objectives());
            s.quality().constraint_value = 0.0;
            _target.update(s, false);
        };

        void register_target(std::vector<CType> target_objectives) {
            register_target(Eigen::Map<Vec<CType>>(target_objectives.data(), target_objectives.size()));
        };

        bool target_reached(const ArchiveBase& archive) const override final {
            if (!_target.empty()) {
            return archive.covers(_target);
            } else {
            return false;
            }
        };

        void log_header(std::ostream& os) const override final {
            os << "expressions,";
            os << "mse_train,";

            fitness().log_header(os);
        };

        void log(std::ostream& os, SolutionBase& solution) override final {
            os << '"';
            log_solution(os, solution);
            os << "\",";
            for (usize i = 0; i < 1; i++) {
                os << solution.quality().objectives(i) << ',';
            }
        
            fitness().log(os, solution.quality());
        };

        void log_solution(std::ostream& os, const SolutionBase& solution) const override final {
            auto exprs = ctx.to_sympy(solution);
            for (usize i = 0; i < exprs.size(); i++) {
                if (i > 0) {
                    os << " , ";
                }
                if (linear_scaling) {
                    os << solution.continuous_values()(ctx.num_continuous + 2 * i) << " + ("
                    << solution.continuous_values()(ctx.num_continuous + 2 * i + 1) << " * (" << exprs[i] << "))";
                } else {
                    os << exprs[i];
                }
            }
        };

        GPContext ctx;
        bool linear_scaling;
        std::string objective;


    private:
        void _copy_data_to_gpu(Arr2D<CType> X, Arr2D<CType> Y) {
            // Convert double to float
            Arr2D<float> X32 = X.cast<float>();
            Arr2D<float> Y32 = Y.cast<float>();

            d_X = allocate_and_copy(X32.data(), X32.size());
            d_Y = allocate_and_copy(Y32.data(), Y32.size());
        }

        void _copy_solutions_to_gpu(std::vector<float> node_type, std::vector<float> node_value) {
            size_t num_solutions = node_type.size();
            
            // Allocate memory if not allocated or size has increased
            if (_num_solutions_allocated < num_solutions) {
                // Free existing memory
                _free_solution_on_gpu();

                // Allocate and copy new solution data
                d_type = allocate_and_copy(node_type.data(), node_type.size());
                d_value = allocate_and_copy(node_value.data(), node_value.size());
                _num_solutions_allocated = num_solutions;
            } else {
                copy_to_gpu(d_type, node_type.data(), node_type.size());
                copy_to_gpu(d_value, node_value.data(), node_value.size());
            }
        }

        void _allocate_results_on_gpu(size_t num_solutions) {
            if (_num_results_allocated < num_solutions) {
                d_se = allocate_on_gpu<float>(num_solutions * _num_datapoints);
                d_mse = allocate_on_gpu<float>(num_solutions);
                _num_results_allocated = num_solutions;
            }
        }

        void _free_data_on_gpu() {
            free_on_gpu(d_X);
            free_on_gpu(d_Y);
        }

        void _free_solution_on_gpu() {
            free_on_gpu(d_type);
            free_on_gpu(d_value);
            _num_solutions_allocated = 0;
        }

        void _free_results_on_gpu() {
            free_on_gpu(d_se);
            free_on_gpu(d_mse);
            _num_results_allocated = 0;
        }

        // bool _solution_allocated;
        size_t _num_solutions_allocated;
        size_t _num_results_allocated;

        // GPU pointers
        float* d_X = nullptr;
        float* d_Y = nullptr;
        float* d_type = nullptr;
        float* d_value = nullptr;
        float* d_se = nullptr;
        float* d_mse = nullptr;

        MOFitness _archive_fitness;
        MOFitness _fitness;
        std::shared_ptr<InitBase> _init;
        UnboundedArchive _target;
        usize _num_continuous;
        Vec<CType> _continuous_lower_bounds;
        Vec<CType> _continuous_upper_bounds;
        Vec<CType> _continuous_init_lower_bounds;
        Vec<CType> _continuous_init_upper_bounds;

        int _num_datapoints;
        int _solution_length;
};

}


#endif /* _GOBLIN_GA_GP_SR_H */