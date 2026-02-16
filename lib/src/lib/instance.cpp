#include "goblin/lib/instance.h"

#include <Cache/Cache.h>
#include <Cache/Policy/FIFO.h>
#include <Cache/Policy/LFU.h>
#include <Cache/Policy/LIFO.h>
#include <Cache/Policy/LRU.h>
#include <Cache/Policy/MRU.h>
#include <Cache/Policy/Random.h>
#include <stdexcept>

namespace goblin {
Mat<CType> InstanceBase::gradients(Rng& rng,
                                   SolutionSetBase& solutions,
                                   SolutionSetBase& parents,
                                   const std::vector<const Subset*>& subsets,
                                   const std::span<const usize>& indices,
                                   u64& evaluations) {
  const CType e = 1e-6;
  const CType ee = e + e;

  Mat<CType> grads = Mat<CType>::Zero(indices.size(), num_continuous());
  if (num_continuous() == 0) {
    return grads;
  }

  // back up qualities
  std::vector<std::unique_ptr<QualityBase>> actual;
  actual.reserve(indices.size());
  for (size_t i = 0; i < indices.size(); i++) {
    actual.push_back(solutions[indices[i]].quality().clone());
  }

  std::vector<std::unique_ptr<QualityBase>> q_e;
  q_e.reserve(indices.size());

  std::vector<usize> solutions_to_evaluate;
  solutions_to_evaluate.reserve(indices.size());
  for (size_t c = 0; c < num_continuous(); c++) {
    solutions_to_evaluate.clear();

    // -e
    for (size_t _i = 0, i; _i < indices.size(); _i++) {
      i = indices[_i];
      if (solutions[i].continuous_active()(c)) {
        solutions[i].continuous_values()(c) -= e;
        solutions_to_evaluate.push_back(i);
      }
    }

    evaluate_partial(rng, solutions, parents, subsets, solutions_to_evaluate);
    evaluations += solutions_to_evaluate.size();

    q_e.clear();
    for (size_t i = 0; i < indices.size(); i++) {
      q_e.push_back(solutions[indices[i]].quality().clone());
    }

    // +e
    for (size_t _i = 0, i; _i < indices.size(); _i++) {
      i = indices[_i];
      if (solutions[i].continuous_active()(c)) {
        solutions[i].continuous_values()(c) += ee;
      }
    }

    evaluate_partial(rng, solutions, parents, subsets, solutions_to_evaluate);
    evaluations += solutions_to_evaluate.size();

    for (size_t i = 0; i < indices.size(); i++) {
      CType dist = fitness().distance(*q_e[i], solutions[indices[i]].quality(), std::nullopt) / ee;
      if (!isna(dist) && dist > 0.0) {
        Ordering o = fitness().cmp(*q_e[i], solutions[indices[i]].quality(), std::nullopt);
        if (o == Ordering::Better) {
          grads(i, c) = dist;
        } else if (o == Ordering::Worse) {
          grads(i, c) = -dist;
        }
      }
    }

    // restore values
    for (size_t _i = 0, i; _i < indices.size(); _i++) {
      i = indices[_i];
      if (solutions[i].continuous_active()(c)) {
        solutions[i].continuous_values()(c) -= e;
      }
    }
  }

  // restore actual quality
  for (size_t i = 0; i < indices.size(); i++) {
    solutions[indices[i]].assign_quality(*actual[i]);
  }
  return grads;
}

std::tuple<std::vector<usize>, u64> InstanceBase::gradient_steps(Rng& rng,
                                                                 SolutionSetBase& solutions,
                                                                 SolutionSetBase& parents,
                                                                 const std::span<const usize>& indices,
                                                                 usize num_steps) {
  const CType e = 1e-6;
  const CType learning_rate = 0.1;

  std::vector<usize> changed_solutions;
  if (num_continuous() == 0) {
    return std::make_tuple(changed_solutions, 0);
  }

  bool any_solution_changed = false;
  std::vector<Subset> _subsets(solutions.size());
  std::vector<const Subset*> subsets(solutions.size());
  for (usize i = 0; i < indices.size(); i++) {
    for (usize j = 0; j < num_continuous(); j++) {
      if (solutions[indices[i]].continuous_active()(j)) {
        _subsets[indices[i]].continuous.push_back(j);
        subsets[indices[i]] = &_subsets[indices[i]];
        any_solution_changed = true;
      }
    }
  }
  if (!any_solution_changed) {
    return std::make_tuple(changed_solutions, 0);
  }

  u64 evaluations = 0;

  std::vector<bool> solution_changed(indices.size(), false);
  for (usize s = 0; s < num_steps && any_solution_changed; s++) {
    any_solution_changed = false;

    auto grads = gradients(rng, solutions, parents, subsets, indices, evaluations);

    for (usize i = 0; i < indices.size(); i++) {
      for (usize j = 0; j < num_continuous(); j++) {
        if (std::abs(grads(i, j)) > e) {
          solutions[indices[i]].continuous_values()(j) -= learning_rate * grads(i, j);
          solution_changed[i] = true;
          any_solution_changed = true;
        }
      }
    }
  }

  changed_solutions.reserve(indices.size());
  for (usize i = 0; i < indices.size(); i++) {
    if (solution_changed[i]) {
      changed_solutions.push_back(indices[i]);
    }
  }
  // update the quality now that the solution has been changed
  evaluate_partial(rng, solutions, parents, subsets, changed_solutions);
  evaluations += changed_solutions.size();
  return std::make_tuple(changed_solutions, evaluations);
}

template <typename Cache>
std::shared_ptr<CachedInstanceBase> cached_impl(std::shared_ptr<InstanceBase> problem, usize cache_size) {
  struct CachedWrapper : public CachedInstanceBase {
    CachedWrapper(std::shared_ptr<InstanceBase>&& problem, usize cache_size)
        : problem(std::move(problem)), cache_size(cache_size), cache(cache_size) {};

    usize num_objectives() const override final { return problem->num_objectives(); }

    usize num_discrete() const override final { return problem->num_discrete(); }
    CRef<Vec<DType>> discrete_domain_sizes() const override final { return problem->discrete_domain_sizes(); }

    usize num_continuous() const override final { return problem->num_continuous(); }
    CRef<Vec<CType>> continuous_lower_bounds() const override final { return problem->continuous_lower_bounds(); }
    CRef<Vec<CType>> continuous_upper_bounds() const override final { return problem->continuous_upper_bounds(); }

    CRef<Vec<CType>> continuous_init_lower_bounds() const override final {
      return problem->continuous_init_lower_bounds();
    }
    CRef<Vec<CType>> continuous_init_upper_bounds() const override final {
      return problem->continuous_init_upper_bounds();
    }

    void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
      std::vector<usize> idxs;
      idxs.reserve(indices.size());
      std::vector<CacheKey> keys;
      keys.reserve(indices.size());

      // 1. cache lookup
      for (usize i : indices) {
        auto key = problem->solution_cache_key(solutions[i]);
        if (cache.contains(key)) {
          solutions[i].assign_quality(*cache[key]);
        } else {
          idxs.push_back(i);
          keys.push_back(key);
        }
      }

      // 2. compute missing
      problem->evaluate(rng, solutions, idxs);

      // 3. cache update
      for (usize i = 0; i < idxs.size(); i++) {
        cache.insert(keys[i], solutions[idxs[i]].quality().clone());
      }
    };
    void evaluate_partial(Rng& rng,
                          SolutionSetBase& solutions,
                          SolutionSetBase& parents,
                          const std::vector<const Subset*>& subsets,
                          const std::span<const usize>& indices) override final {
      std::vector<usize> idxs;
      idxs.reserve(indices.size());
      std::vector<CacheKey> keys;
      keys.reserve(indices.size());

      // 1. cache lookup
      for (usize i : indices) {
        auto key = problem->solution_cache_key(solutions[i]);
        if (cache.contains(key)) {
          solutions[i].assign_quality(*cache[key]);
        } else {
          idxs.push_back(i);
          keys.push_back(key);
        }
      }

      // 2. compute missing
      problem->evaluate_partial(rng, solutions, parents, subsets, idxs);

      // 3. cache update
      for (usize i = 0; i < idxs.size(); i++) {
        cache.insert(keys[i], solutions[idxs[i]].quality().clone());
      }
    }

    bool adapt(Rng& rng) override final {
      if (problem->adapt(rng)) {
        cache.clear();
        return true;
      } else {
        return false;
      }
    };

    Mat<CType> gradients(Rng& rng,
                         SolutionSetBase& solutions,
                         SolutionSetBase& parents,
                         const std::vector<const Subset*>& subsets,
                         const std::span<const usize>& indices,
                         u64& evaluations) override final {
      return problem->gradients(rng, solutions, parents, subsets, indices, evaluations);
    }

    std::tuple<std::vector<usize>, u64> gradient_steps(Rng& rng,
                                                       SolutionSetBase& solutions,
                                                       SolutionSetBase& parents,
                                                       const std::span<const usize>& indices,
                                                       usize num_steps) override final {
      return problem->gradient_steps(rng, solutions, parents, indices, num_steps);
    }

    virtual void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override final {
      return problem->add_random(rng, solutions, count);
    }

    const FitnessBase& fitness() const override final { return problem->fitness(); }
    const ArchiveFitnessBase& archive_fitness() const override final { return problem->archive_fitness(); }
    std::tuple<bool, bool> inherit_discrete(SolutionBase& offspring,
                                            const SolutionBase& donor,
                                            const Subset& subset) const override final {
      return problem->inherit_discrete(offspring, donor, subset);
    }

    std::optional<CType> as_continuous(const SolutionBase& solution, usize discrete_index) const override final {
      return problem->as_continuous(solution, discrete_index);
    }

    bool target_reached(const ArchiveBase& archive) const override final { return problem->target_reached(archive); }

    void log_header(std::ostream& os) const override final { return problem->log_header(os); }
    void log_solution(std::ostream& os, const SolutionBase& solution) const override final {
      return problem->log_solution(os, solution);
    }

    void log(std::ostream& os, const SolutionBase& solution) override final { return problem->log(os, solution); }
    CacheKey solution_cache_key(const SolutionBase& solution) const override final {
      return problem->solution_cache_key(solution);
    }

    usize hit_count() const override final { return cache.hit_count(); };
    usize miss_count() const override final { return cache.miss_count(); };
    usize access_count() const override final { return cache.access_count(); }
    usize entry_invalidation_count() const override final { return cache.entry_invalidation_count(); }
    usize cache_invalidation_count() const override final { return cache.cache_invalidation_count(); }
    usize evicted_count() const override final { return cache.evicted_count(); }

    CType hit_ratio() const override final { return cache.hit_ratio(); }
    CType miss_ratio() const override final { return cache.miss_ratio(); }
    CType utilization() const override final { return cache.utilization(); }

    std::shared_ptr<InstanceBase> problem;
    usize cache_size;
    Cache cache;
  };

  return std::make_shared<CachedWrapper>(std::move(problem), cache_size);
}

std::shared_ptr<CachedInstanceBase> Cached(std::shared_ptr<InstanceBase> problem,
                                           usize cache_size,
                                           std::string cache_policy) {
  using Value = std::unique_ptr<QualityBase>;
  if (cache_policy == "fifo") {
    return cached_impl<Cache<CacheKey, Value, Policy::FIFO>>(std::move(problem), cache_size);
  } else if (cache_policy == "lfu") {
    return cached_impl<Cache<CacheKey, Value, Policy::LFU>>(std::move(problem), cache_size);
  } else if (cache_policy == "lifo") {
    return cached_impl<Cache<CacheKey, Value, Policy::LIFO>>(std::move(problem), cache_size);
  } else if (cache_policy == "lru") {
    return cached_impl<Cache<CacheKey, Value, Policy::LRU>>(std::move(problem), cache_size);
  } else if (cache_policy == "mru") {
    return cached_impl<Cache<CacheKey, Value, Policy::MRU>>(std::move(problem), cache_size);
  } else if (cache_policy == "random") {
    return cached_impl<Cache<CacheKey, Value, Policy::Random>>(std::move(problem), cache_size);
  } else {
    throw std::runtime_error("Unknown cache policy.");
  }
}
};  // namespace goblin
