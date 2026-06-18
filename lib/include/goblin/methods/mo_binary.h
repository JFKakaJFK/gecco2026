#pragma once
#ifndef _GOBLIN_MO_BINARY_GOMEA_H
#define _GOBLIN_MO_BINARY_GOMEA_H

#include <cassert>
#include <functional>
#include <optional>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "goblin/bench/problem.h"
#include "goblin/lib/assert.h"
#include "goblin/lib/budget.h"
#include "goblin/lib/method.h"
#include "goblin/lib/rng.h"
#include "goblin/lib/types.h"

namespace goblin {

class MOBinaryGOMEA final : public MethodBase {
   public:
    MOBinaryGOMEA(usize initial_population_size = 8,
                  usize max_num_populations = 25,
                  usize subgeneration_factor = 2,
                  usize target_archive_size = 100,
                  std::string mutation = "none",
                  bool stop_covered_populations = false)
        : initial_population_size(initial_population_size),
          max_num_populations(max_num_populations),
          subgeneration_factor(subgeneration_factor),
          target_archive_size(target_archive_size),
          mutation(mutation),
          stop_covered_populations(stop_covered_populations) {};

    std::optional<u64> current_generation() const override final;

    std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
        InstanceBase& problem,
        const Budget& budget,
        std::optional<u64> seed = std::nullopt,
        std::optional<usize> population_size = std::nullopt) override final {
        if (problem.num_discrete() < 1 || problem.num_continuous() > 0 ||
            (problem.discrete_domain_sizes().array() > DType(2)).any()) {
            __goblin_runtime_assert(false);  // problem not supported
        }

        auto archive = std::make_shared<UnboundedArchive>(problem.archive_fitness());
        TerminationStatus status = TerminationStatus::Converged;
        Rng rng = seeded_rng(seed);

        AoSSet s;
        s.add(Solution(problem.archive_fitness().worst(), Vec<DType>::Zero(problem.num_discrete()), std::nullopt));
        std::vector<usize> idxs{0};
        std::function<void(char*, double*, double*)> fn = [&](char* solution, double* obj, double* con) {
            for (usize i = 0; i < problem.num_discrete(); i++) {
                s[0].discrete_values()(i) = static_cast<DType>(solution[i]);
            }
            problem.evaluate(rng, s, idxs);

            archive->update(s[0], true);
            if (problem.target_reached(*archive)) {
                status = TerminationStatus::TargetReached;
                throw std::runtime_error("");
            }

            auto& q = s[0].quality_as<MOQuality>();
            for (usize i = 0; i < problem.num_objectives(); i++) {
                obj[i] = q.objectives(i);
            }
            *con = q.constraint_value;
        };

        try {
            run_helper(fn, problem.num_objectives(), problem.num_discrete(),
                       population_size.value_or(initial_population_size),
                       population_size.has_value() ? 1 : max_num_populations, subgeneration_factor, target_archive_size,
                       mutation, stop_covered_populations, seed, budget.max_evaluations);
        } catch (std::runtime_error& e) {
        }
        return std::make_tuple(archive, status);
    };

   private:
    void run_helper(std::function<void(char*, double*, double*)> evaluate,
                    usize num_objectives,
                    usize num_parameters,
                    usize initial_population_size,
                    usize max_num_populations,
                    usize subgeneration_factor,
                    usize target_archive_size,
                    std::string mutation,
                    bool stop_covered_populations,
                    std::optional<u64> seed,
                    std::optional<usize> max_evaluations);

    usize initial_population_size;
    usize max_num_populations;
    usize subgeneration_factor;
    usize target_archive_size;
    std::string mutation;
    bool stop_covered_populations;
};

};  // namespace goblin

#endif /* _GOBLIN_MO_BINARY_GOMEA_H */
