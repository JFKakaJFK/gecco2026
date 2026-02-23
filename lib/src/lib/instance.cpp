#include "goblin/lib/instance.h"

#include <Cache/Cache.h>
#include <Cache/Policy/FIFO.h>
#include <Cache/Policy/LFU.h>
#include <Cache/Policy/LIFO.h>
#include <Cache/Policy/LRU.h>
#include <Cache/Policy/MRU.h>
#include <Cache/Policy/Random.h>
#include <stdexcept>

template <>
struct std::hash<goblin::CacheKey> {
  std::size_t operator()(const goblin::CacheKey& key) const noexcept { return key.hash_value_; }
};

// TODO at some point consider https://github.com/Nicoshev/rapidhash / https://github.com/Cyan4973/xxHash
// since FNV-1a is slow: https://aras-p.info/blog/2016/08/09/More-Hash-Function-Tests/
inline std::size_t hash_bytes(const std::vector<std::byte>& data) {
  // FNV-1a as per https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
  const std::size_t FNV_BASIS = 0xcbf29ce484222325;
  const std::size_t FNV_PRIME = 0x00000100000001b3;
  std::size_t h = FNV_BASIS;
  for (auto b : data) {
    h ^= static_cast<std::size_t>(b);
    h *= FNV_PRIME;
  }
  return h;
};

namespace goblin {
CacheKey::CacheKey(const std::string& key) {
  const auto* s = reinterpret_cast<const std::byte*>(key.data());
  key_ = std::vector<std::byte>(s, s + key.size());
  hash_value_ = hash_bytes(key_);
};
CacheKey::CacheKey(std::vector<std::byte>&& key) : key_(std::move(key)), hash_value_(hash_bytes(key_)) {};

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
        : CachedInstanceBase(*problem), problem(std::move(problem)), cache_size(cache_size), cache(cache_size) {};

    void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
      std::vector<usize> idxs;
      idxs.reserve(indices.size());
      std::vector<std::optional<CacheKey>> keys;
      keys.reserve(indices.size());

      // 1. cache lookup
      for (usize i : indices) {
        auto key = problem->solution_cache_key(solutions[i]);
        if (key.has_value() && cache.contains(key.value())) {
          solutions[i].assign_quality(*cache[key.value()]);
        } else {
          idxs.push_back(i);
          keys.push_back(key);
        }
      }

      // 2. compute missing
      problem->evaluate(rng, solutions, idxs);

      // 3. cache update
      for (usize i = 0; i < idxs.size(); i++) {
        if (keys[i].has_value()) {
          cache.insert(keys[i].value(), solutions[idxs[i]].quality().clone());
        }
      }
    };
    void evaluate_partial(Rng& rng,
                          SolutionSetBase& solutions,
                          SolutionSetBase& parents,
                          const std::vector<const Subset*>& subsets,
                          const std::span<const usize>& indices) override final {
      std::vector<usize> idxs;
      idxs.reserve(indices.size());
      std::vector<std::optional<CacheKey>> keys;
      keys.reserve(indices.size());

      // 1. cache lookup
      for (usize i : indices) {
        auto key = problem->solution_cache_key(solutions[i]);
        if (key.has_value() && cache.contains(key.value())) {
          solutions[i].assign_quality(*cache[key.value()]);
        } else {
          idxs.push_back(i);
          keys.push_back(key);
        }
      }

      // 2. compute missing
      problem->evaluate_partial(rng, solutions, parents, subsets, idxs);

      // 3. cache update
      for (usize i = 0; i < idxs.size(); i++) {
        if (keys[i].has_value()) {
          cache.insert(keys[i].value(), solutions[idxs[i]].quality().clone());
        }
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

    void log_header(std::ostream& os) const override final {
      inner.log_header(os);
      os << "cache_hit_ratio,cache_utilization";
    };

    void log(std::ostream& os, const SolutionBase& solution) const override final {
      inner.log(os, solution);
      os << hit_ratio() << ',' << utilization();
    };

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
