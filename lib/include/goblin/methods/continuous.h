#pragma once
#ifndef _GOBLIN_METHODS_CONTINUOUS_H
#define _GOBLIN_METHODS_CONTINUOUS_H

#include <limits>
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <concepts>
#include <csetjmp>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>
#include <string_view>
#include <print>
#include <iostream>
#include <span>

#include <Eigen/Cholesky>
#include <Eigen/QR>

#include "goblin/lib/assert.h"
#include "goblin/lib/rng.h"
#include "goblin/lib/types.h"
#include "goblin/lib/solution.h"
#include "goblin/lib/method.h"
#include "goblin/lib/algorithms/mo.h"
#include "goblin/lib/linkage_model.h"
#include "goblin/bench/tracked.h"

namespace goblin {

static Vec<CType> estimate_mean(const SolutionSetBase& solutions,
                                const std::span<const usize> active_indices,
                                const std::span<const usize> active_counts,
                                double selection_percentile,
                                const std::span<const usize> indices,
                                bool intron_aware) {
  Vec<CType> mean = Vec<CType>::Zero(indices.size());
  usize selection_size = 0;
  if (intron_aware) {
    for (usize i : indices) {
      selection_size = std::max(selection_size, static_cast<usize>(selection_percentile * active_counts[i]));
    }

    if (selection_size > 0) {
      bool any_left = true;
      std::vector<usize> num_left(indices.size(), selection_size);
      for (usize i : active_indices) {
        any_left = false;
        for (usize j = 0; j < indices.size(); j++) {
          if (num_left[j] > 0 &&
              (solutions[i].continuous_active()(
                   indices[j])                    // index is active -> we use the value under selection pressure
               || active_counts[indices[j]] == 0  // index is not active -> we still want a non-zero mean to hopefully
                                                  // make the cholesky decomposition work (= we want to keep diversity)
               )) {
            mean(j) += solutions[i].continuous_values()(indices[j]);
            num_left[j]--;
            any_left |= num_left[j] > 0;
          }
        }
        if (!any_left) {
          break;
        }
      }
      for (usize j = 0; j < indices.size(); j++) {
        mean(j) /= static_cast<CType>(selection_size - num_left[j]);
      }
    }
  }

  if (selection_size == 0) {
    selection_size = selection_percentile * active_indices.size();
    for (usize i = 0; i < selection_size; i++) {
      mean += solutions[active_indices[i]].continuous_values()(indices);
    }
    mean /= static_cast<CType>(selection_size);
  }
  return mean;
};

static Mat<CType> estimate_cov(const SolutionSetBase& solutions,
                               const std::span<const usize> active_indices,
                               const std::span<const usize> active_counts,
                               double selection_percentile,
                               const Vec<CType>& mean,
                               const std::span<const usize> indices,
                               bool intron_aware) {
  assert(static_cast<usize>(mean.size()) == indices.size());
  Mat<CType> cov = Mat<CType>::Zero(indices.size(), indices.size());
  if (intron_aware) {
    for (usize i = 0; i < indices.size(); i++) {
      if (active_counts[indices[i]] == 0) {
        // estimate the full univariate covariance if there are no active variables to keep diversity (no selection)
        for (usize l : active_indices) {
          cov(i, i) += std::pow(solutions[l].continuous_values()(indices[i]) - mean(i), 2);
        }
        cov(i, i) /= static_cast<CType>(active_indices.size());
      } else {
        // otherwise estimate the covariance between active variables
        for (usize j = i; j < indices.size(); j++) {
          if (active_counts[indices[j]] > 0) {
            usize selection_size =
                selection_percentile * std::max(active_counts[indices[i]], active_counts[indices[j]]);
            usize actual_size = 0;

            for (usize l : active_indices) {
              if (solutions[l].continuous_active()(indices[i]) && solutions[l].continuous_active()(indices[j])) {
                cov(i, j) += (solutions[l].continuous_values()(indices[i]) - mean(i)) *
                             (solutions[l].continuous_values()(indices[j]) - mean(j));
                actual_size++;
                if (actual_size >= selection_size) {
                  break;
                }
              }
            }
            if (actual_size > 0) {
              // this is fine - that just means that i != j and i and j never appear together so the covariance should
              // be 0
              cov(i, j) /= static_cast<CType>(actual_size);
              cov(j, i) = cov(i, j);
            }
          }
        }
      }
    }
  } else {
    usize selection_size = selection_percentile * active_indices.size();
    for (usize i = 0; i < selection_size; i++) {
      Vec<CType> v = solutions[active_indices[i]].continuous_values()(indices) - mean;
      cov.noalias() += v * v.transpose();
    }
    cov /= static_cast<CType>(selection_size);
  }
  return cov;
};

// Performs an inplace cholesky decomposition. If the decomposition fails, jitter is added to the diagonal to increase
// the rank until finally the univariate diagonal is used.
template <typename Derived>
inline void cholesky_inplace(Eigen::MatrixBase<Derived>& out) {
  using S = typename Derived::Scalar;
  const usize num_tries = 1;
  // const usize num_tries = 10;

#ifdef DEBUG
  Mat<S> cov = out;
#endif

  Eigen::LLT<Mat<S>, Eigen::Lower> cholesky_decomposition(out);
  usize tries = 1;
  S jitter_added = 0.0;
  S epsilon = 1e-10;
  while (cholesky_decomposition.info() != Eigen::Success && tries++ < num_tries) {
    // add epsilon to the diagonal to increase the rank
    out.diagonal().array() += epsilon;
    jitter_added += epsilon;
    epsilon *= 10.0;
    cholesky_decomposition = Eigen::LLT<Mat<S>, Eigen::Lower>(out);
  }

  if (cholesky_decomposition.info() == Eigen::Success) {
    out = cholesky_decomposition.matrixL();

#ifdef DEBUG
    Mat<S> res = out * out.transpose();
    for (isize i = 0; i < cov.rows(); i++) {
      for (isize j = 0; j < cov.cols(); j++) {
        assert(std::abs(res(i, j) - cov(i, j)) < 1e-8);
      }
    }
#endif
  } else {
    // std::println("!!! CHOLESKY FAILED !!!\n{}", log_helper(out, /* escape = */ false, /* indent = */ true));
    // std::println("!!! CHOLESKY FAILED !!!");

    // covariance diagonal without jitter, made positive and sqrt to match the expecation that out * out.T = input
    Vec<S> univariate = (out.diagonal().array() - jitter_added);
    for (isize i = 0; i < univariate.size(); i++) {
      univariate(i) = univariate(i) > 0.0 ? std::sqrt(univariate(i)) : 0.0;
    }
    out.setZero();
    out.diagonal() = univariate;

#ifdef DEBUG
    Mat<S> res = out * out.transpose();
    for (isize i = 0; i < cov.rows(); i++) {
      for (isize j = 0; j < cov.cols(); j++) {
        assert((i == j ? std::abs(res(i, j) - cov(i, j)) : res(i, j)) < 1e-8);
      }
    }
#endif
  }
};

class RvSubsetStateBase {
 public:
  CType distribution_multiplier = 1.0;

  virtual ~RvSubsetStateBase() = default;
};

/// Separate sampling models roughly as per https://ir.cwi.nl/pub/30344/30344.pdf
/// TODO ask anton about API - feels still a bit crude...
class RvSamplingModelBase {
 public:
  virtual std::unique_ptr<RvSubsetStateBase> init(const Subset& subset) const = 0;
  virtual std::unique_ptr<RvSubsetStateBase> inherit(const Subset& subset,
                                                     const Subset& previous_subset,
                                                     const RvSubsetStateBase& previous_state) const = 0;
  virtual void update(const SolutionSetBase& solutions,
                      const std::span<const usize> by_fitness_decreasing,
                      const std::span<const usize> active_counts,
                      bool intron_aware,
                      const Subset& subset,
                      double selection_percentile,
                      RvSubsetStateBase& state) const = 0;
  virtual Vec<CType> sample(Rng& rng, const RvSubsetStateBase& state) const = 0;
  virtual void adapt(const SolutionSetBase& solutions,
                     const std::span<const usize> improved_indices,
                     double oob_ratio,
                     bool max_nis_reached,
                     bool intron_aware,
                     const Subset& subset,
                     RvSubsetStateBase& state) const {};

  virtual bool converged(const RvSubsetStateBase& state) const = 0;

  virtual ~RvSamplingModelBase() = default;
};

class AMaLGaMSubsetState final : public RvSubsetStateBase {
 public:
  bool enable_smoothing = false;
  Vec<CType> mean;
  Mat<CType>
      cov;  // for storing the unscaled covariance, since in the incremental version the scaling happens after the
            // smoothing:
            // https://github.com/renzoscholman/irv-gomea/blob/89c62fcf007f858bd8ab9c2ba7955c7db80df121/src/distribution.cpp#L510
  Mat<CType> L;
};

class AMaLGaMSamplingModel final : public RvSamplingModelBase {
 public:
  AMaLGaMSamplingModel(bool use_mahalanobis_distance_for_sdr =
                           false,  // uses mahalanobis distance instead of max(abs(z)) for computing the SDR
                       CType eta_cov = 1.0,
                       CType std_deviation_ratio_threshold = 1.0,
                       CType distribution_multiplier_decrease = 0.9,
                       CType distribution_multiplier_increase = 1.0 / 0.9,
                       CType min_distribution_multiplier = 1e-10)
      : use_mahalanobis_distance_for_sdr(use_mahalanobis_distance_for_sdr),
        eta_cov(eta_cov),
        std_deviation_ratio_threshold(std_deviation_ratio_threshold),
        distribution_multiplier_decrease(distribution_multiplier_decrease),
        distribution_multiplier_increase(distribution_multiplier_increase),
        min_distribution_multiplier(min_distribution_multiplier) {
    __goblin_runtime_assert(0.0 < eta_cov && eta_cov <= 1.0);
    __goblin_runtime_assert(std_deviation_ratio_threshold >= 0.0);
    __goblin_runtime_assert(0.0 < distribution_multiplier_decrease && distribution_multiplier_decrease <= 1.0);
    __goblin_runtime_assert(1.0 <= distribution_multiplier_increase);
    __goblin_runtime_assert(min_distribution_multiplier >= 0.0);
  };

  std::unique_ptr<RvSubsetStateBase> init(const Subset& subset) const override final {
    usize size = subset.continuous.size();
    auto s = std::make_unique<AMaLGaMSubsetState>();
    s->enable_smoothing = false;
    s->distribution_multiplier = 1.0;
    s->mean = Vec<CType>::Zero(size);
    s->cov = Mat<CType>::Zero(size, size);
    return s;
  };

  std::unique_ptr<RvSubsetStateBase> inherit(const Subset& subset,
                                             const Subset& previous_subset,
                                             const RvSubsetStateBase& previous_state) const override final {
    const auto& prev = static_cast<const AMaLGaMSubsetState&>(previous_state);
    const auto& cs = subset.continuous;
    const auto& ps = previous_subset.continuous;

    auto s = std::make_unique<AMaLGaMSubsetState>();
    s->distribution_multiplier = prev.distribution_multiplier;
    s->mean = Vec<CType>::Zero(cs.size());
    s->cov = Mat<CType>::Zero(cs.size(), cs.size());

    s->enable_smoothing = true;
    for (usize i = 0; i < cs.size(); i++) {
      bool matched = false;
      for (usize pi = 0; pi < ps.size(); pi++) {
        if (cs[i] == ps[pi]) {
          matched = true;
          s->mean(i) = prev.mean(pi);

          for (usize j = 0; j <= i; j++) {
            for (usize pj = 0; pj <= pi; pj++) {
              if (cs[j] == ps[pj]) {
                s->cov(i, j) = prev.cov(pi, pj);
                s->cov(j, i) = s->cov(i, j);
                break;
              }
            }
          }
          break;
        }
      }
      if (!matched) {
        // If not all variables are present in the previous state, a non-zero smoothing factor (eta_cov < 1.0) would
        // shrink the distribution
        s->enable_smoothing = false;
      }
    }

    return s;
  };

  void update(const SolutionSetBase& solutions,
              const std::span<const usize> by_fitness_decreasing,
              const std::span<const usize> active_counts,
              bool intron_aware,
              const Subset& subset,
              double selection_percentile,
              RvSubsetStateBase& state) const override final {
    const auto& cs = subset.continuous;
    auto& s = static_cast<AMaLGaMSubsetState&>(state);

    // usize selection_size = selection_percentile * by_fitness_decreasing.size();

    assert(static_cast<usize>(s.mean.size()) == cs.size());
    s.mean = estimate_mean(solutions, by_fitness_decreasing, active_counts, selection_percentile, cs, intron_aware);

    // Change the focus of the search to the best solution
    if (s.distribution_multiplier < 1.0) {
      for (usize i = 0; i < cs.size(); i++) {
        if (solutions[by_fitness_decreasing[0]].continuous_active()(cs[i])) {
          s.mean(i) = solutions[by_fitness_decreasing[0]].continuous_values()(cs[i]);
        }
      }
    }

    Mat<CType> new_cov =
        estimate_cov(solutions, by_fitness_decreasing, active_counts, selection_percentile, s.mean, cs, intron_aware);

    if (!s.enable_smoothing) {
      s.cov = new_cov;
      s.enable_smoothing = true;
    } else {
      s.cov = eta_cov * new_cov + (1.0 - eta_cov) * s.cov;
    }

    s.L = s.cov * s.distribution_multiplier;

    // // if we don't have enough solutions we fall back to the univariate diagonal
    // if (selection_size < static_cast<usize>(cs.size()) + 1) {
    //   s.L.setZero();
    //   s.L.diagonal() = (s.cov * s.distribution_multiplier).diagonal();
    //   for (isize i = 0; i < s.L.diagonal().size(); i++) {
    //     s.L.diagonal()(i) = s.L.diagonal()(i) > 0.0 ? std::sqrt(s.L.diagonal()(i)) : 0.0;
    //   }
    // } else {
    //   cholesky_inplace(s.L);
    // }

    cholesky_inplace(s.L);
  };

  Vec<CType> sample(Rng& rng, const RvSubsetStateBase& state) const override final {
    thread_local static std::normal_distribution<CType> N(0.0, 1.0);

    const auto& s = static_cast<const AMaLGaMSubsetState&>(state);

    Vec<CType> z(s.mean.size());
    // sample from N(0, I)
    for (isize i = 0; i < z.size(); i++) {
      z(i) = N(rng);
    }
    // scale to N(s.mean, s.cov * s.distribution_multiplier)
    return (s.L.triangularView<Eigen::Lower>() * z) + s.mean;
  };

  void adapt(const SolutionSetBase& solutions,
             const std::span<const usize> improved_indices,
             double oob_ratio,
             bool max_nis_reached,
             bool intron_aware,
             const Subset& subset,
             RvSubsetStateBase& state) const override final {
    auto& s = static_cast<AMaLGaMSubsetState&>(state);

    if (oob_ratio > 0.9) {
      s.distribution_multiplier *= 0.5;
    }

    if (!improved_indices.empty()) {
      if (s.distribution_multiplier < 1.0) {
        s.distribution_multiplier = 1.0;
      }

      CType std_deviation_ratio = compute_SDR(solutions, improved_indices, subset, s.mean, s.L, intron_aware);
      if (std_deviation_ratio > std_deviation_ratio_threshold) {
        s.distribution_multiplier *= distribution_multiplier_increase;
      }
    } else if (max_nis_reached) {
      s.distribution_multiplier *= distribution_multiplier_decrease;
    } else {
      if (s.distribution_multiplier > 1.0) {
        s.distribution_multiplier *= distribution_multiplier_decrease;
      }

      if (s.distribution_multiplier < 1.0) {
        s.distribution_multiplier = 1.0;
      }
    }
  };

  bool converged(const RvSubsetStateBase& state) const override final {
    return state.distribution_multiplier <= min_distribution_multiplier;
  };

 private:
  CType compute_SDR(const SolutionSetBase& solutions,
                    const std::span<const usize> improved_indices,
                    const Subset& subset,
                    const Vec<CType>& mean,
                    const Mat<CType>& L,
                    bool intron_aware) const {
    const auto& s = subset.continuous;

    Vec<CType> avg_params = Vec<CType>::Zero(s.size());
    if (intron_aware) {
      Array<CType> num_active = Array<CType>::Zero(s.size());
      for (usize i : improved_indices) {
        for (usize j = 0; j < s.size(); j++) {
          if (solutions[i].continuous_active()(s[j])) {
            avg_params(j) += solutions[i].continuous_values()(s[j]);
            num_active(j) += 1.0;
          }
        }
      }
      for (usize j = 0; j < s.size(); j++) {
        if (num_active(j) > 0.0) {
          avg_params(j) /= num_active(j);
        }
      }
    } else {
      for (usize i : improved_indices) {
        avg_params += solutions[i].continuous_values()(s);
      }
      avg_params /= static_cast<CType>(improved_indices.size());
    }

    // ? L.triangularView<Eigen::Lower>().solve(avg_params - mean);
    Mat<CType> L_inv = L.completeOrthogonalDecomposition().pseudoInverse();

    avg_params = ((L_inv * avg_params) - mean);

    if (use_mahalanobis_distance_for_sdr) {
      return std::sqrt(avg_params.transpose() * avg_params);  // average std_dev from the mean
    } else {
      return avg_params.array().abs().maxCoeff();  // max std_dev from the mean
    }
  }

  bool use_mahalanobis_distance_for_sdr;
  CType eta_cov;
  CType std_deviation_ratio_threshold;
  CType distribution_multiplier_decrease;
  CType distribution_multiplier_increase;
  CType min_distribution_multiplier;
};

struct RvOptions {
  bool enabled = true;
  bool intron_aware = false;

  double selection_percentile = 0.35;
  double p_accept = 0.05;

  bool init_ams_from_population_mean = true;
  // If randomized, the AMS indices are randomly picked from all active solutions.
  // Otherwise the first `floor(selection_percentile * 0.5)` active solutions are used.
  bool randomize_ams_indices = false;
  bool enable_partial_ams = true;
  CType delta_ams = 2.0;
  CType eta_ams = 1.0;

  usize max_nis = 100;
  usize num_forced_improvement_tries = 8;  // 8 is the RV GOMEA default if I did not miscalculate (1.0 / 2^8 < 0.01)

  // In the GBO setting with partial evaluations, numerical errors of partial fitness updates
  // can accumulate and it might be needed to perform full evaluations once in a while
  //
  // In that case, the default number of generations until re-evaluation is `50`
  //
  // Note that in this setting all archive solutions should be re-evaluated in case the target seems to have been
  // reached to ensure that this is not due to numeric errors. This does not happen in this version.
  std::optional<u64> generations_until_full_evaluation = std::nullopt;

  std::optional<std::string> population_logfile = std::nullopt;
  std::optional<std::string> selection_logfile = std::nullopt;
  std::optional<std::string> subset_logfile = std::nullopt;
  std::optional<std::string> sample_logfile = std::nullopt;

  void validate() {
    __goblin_runtime_assert(0.0 <= p_accept && p_accept < 1.0);
    __goblin_runtime_assert(0.0 <= selection_percentile && selection_percentile < 1.0);
    __goblin_runtime_assert(0.0 < eta_ams && eta_ams <= 1.0);
    __goblin_runtime_assert(0.0 <= delta_ams);
    __goblin_runtime_assert(max_nis > 0);
  };
};

inline std::vector<usize> sort_by_quality_decreasing(const FitnessBase& fitness,
                                                     const SolutionSetBase& solutions,
                                                     const std::vector<usize>& indices,
                                                     std::optional<usize> objective) {
  std::vector<usize> by_fitness;
  if (objective.has_value()) {
    by_fitness = indices;
    std::sort(by_fitness.begin(), by_fitness.end(), [&solutions, &fitness, &objective](usize lhs, usize rhs) {
      return fitness.cmp(solutions[lhs].quality(), solutions[rhs].quality(), objective) == Ordering::Better;
    });
  } else {
    auto [ranks, fronts] = non_dominated_sorting(
        [&solutions, &fitness, &indices](usize lhs, usize rhs) {
          return fitness.cmp(solutions[indices[lhs]].quality(), solutions[indices[rhs]].quality(), std::nullopt);
        },
        indices.size());

    by_fitness.reserve(indices.size());
    for (auto& front : fronts) {
      // TODO re-order fronts to maximize scattering in parameter space?
      for (usize i : front) {
        by_fitness.push_back(indices[i]);
      }
    }
  }

  return by_fitness;
};

class RvState {
 public:
  RvState(RvOptions options, const LinkageModelBase& linkage_model, const RvSamplingModelBase& sampling_model)
      : options(options), linkage_model(linkage_model.clone()), sampling_model(sampling_model) {
    this->options.validate();
  };

  usize perform_generation(Rng& rng,
                           ArchiveBase& archive,
                           InstanceBase& problem,
                           SolutionSetBase& solutions,
                           SolutionSetBase& parents,
                           const std::vector<usize>& solution_clusters,
                           const std::vector<std::vector<usize>>& cluster_solutions) {
    // initialization
    if (generation == 0 || num_clusters != cluster_solutions.size() ||
        static_cast<usize>(num_continuous) != problem.num_continuous() || solution_nis.size() != solutions.size()) {
      linkage_model->init(rng, problem, solutions, VariableSet::Continuous);

      num_clusters = cluster_solutions.size();
      num_continuous = problem.num_continuous();

      solutions_to_evaluate.reserve(solutions.size());
      eval_subsets.resize(solutions.size());

      full.continuous.resize(num_continuous);
      std::iota(full.continuous.begin(), full.continuous.end(), 0);

      improved_indices.resize(num_clusters);

      cluster_active = Array<BType>::Zero(num_clusters);
      mean.resize(num_clusters);
      if (options.init_ams_from_population_mean) {
        for (usize k = 0; k < num_clusters; k++) {
          mean[k] = Vec<CType>::Zero(num_continuous);
          for (usize i : cluster_solutions[k]) {
            mean[k] += solutions[i].continuous_values();
          }
          mean[k] /= static_cast<CType>(cluster_solutions[k].size());
        }
      }
      mean_shift.resize(num_clusters);

      num_oob.resize(num_clusters, 0);
      num_samples.resize(num_clusters, 0);

      no_improvement_stretch.resize(num_clusters, 0);

      solution_improved.resize(solutions.size());
      solution_nis.resize(solutions.size(), 0);

      subsets.resize(num_clusters);
      subset_states.resize(num_clusters);

      // // enable partial ams only if the linkage model is not the full fos - in that case, we stay as close to what
      // // AMaLGaM does as possible...
      // auto ptr = reinterpret_cast<const FullFOS*>(&linkage_model);
      // enable_partial_ams = options.enable_partial_ams_for_full_fos || ptr == nullptr;
    }

    // Re-evaluate the whole population if enabled
    // (for the GBO setting where the partial fitness updates cause numerical drift)
    if (options.generations_until_full_evaluation.has_value() &&
        (generation + 1) % options.generations_until_full_evaluation.value() == 0) {
      solutions_to_evaluate.resize(solutions.size());
      std::iota(solutions_to_evaluate.begin(), solutions_to_evaluate.end(), 0);
      problem.evaluate(rng, solutions, solutions_to_evaluate);
      for (usize i = 0; i < solutions.size(); i++) {
        parents[i] = solutions[i];
      }
    }

    if (options.population_logfile.has_value()) {
      AoSSet p;
      p.add(solutions[0]);
      for (usize i = 0; i < solutions.size(); i++) {
        p[0] = solutions[i];
        debug_log(problem, options.population_logfile.value(), "cluster,", std::format("{},", solution_clusters[i]), p);
      }
    }

    // linkage learning, distribution estimation, ams index assignments per cluster
    auto ams_indices = select_and_learn_linkage(rng, archive, problem, solutions, cluster_solutions);

    if (!(cluster_active.any())) {
      return 0;
    }

    std::fill(solution_improved.begin(), solution_improved.end(), false);

    // shuffle per cluster subset order
    usize max_num_subsets = 0;
    for (usize k = 0; k < num_clusters; k++) {
      max_num_subsets = std::max(max_num_subsets, subsets[k].size());
    }
    Mat<usize> subset_orders(num_clusters, max_num_subsets);
    subset_orders.rowwise() = Vec<usize>::LinSpaced(max_num_subsets, 0, max_num_subsets - 1).transpose();
    for (auto row : subset_orders.rowwise()) {
      std::shuffle(row.begin(), row.end(), rng);
    }

    // variation per solution
    usize evaluations = 0;
    usize subset_idx = 0;
    while (subset_idx < max_num_subsets) {
      evaluations += gom_step(rng, archive, problem, solutions, parents, solution_clusters, cluster_solutions,
                              subset_orders, subset_idx++, ams_indices);
    }

    evaluations += full_ams(rng, archive, problem, solutions, parents, solution_clusters, ams_indices);

    // solution NIS update
    std::vector<bool> any_improved(num_clusters, false);
    for (usize i = 0; i < solutions.size(); i++) {
      if (solution_improved[i]) {
        solution_nis[i] = 0;
        any_improved[solution_clusters[i]] = true;
      } else if (solutions[i].continuous_active().any()) {
        // the nis is only updated for active solutions
        solution_nis[i]++;
      }
    }

    if (options.num_forced_improvement_tries > 0) {
      evaluations += forced_improvements(rng, archive, problem, solutions, parents, solution_clusters, subset_orders,
                                         max_num_subsets);
    }

    // cluster NIS update
    for (usize k = 0; k < num_clusters; k++) {
      if (any_improved[k]) {
        no_improvement_stretch[k] = 0;
      } else {
        bool all_leq_one = true;
        for (const auto& s : subset_states[k]) {
          if (s->distribution_multiplier > 1.0) {
            all_leq_one = false;
            break;
          }
        }
        if (all_leq_one) {
          no_improvement_stretch[k]++;
        }
      }
    }

    generation++;
    return evaluations;
  };

  bool converged() const {
    if (num_clusters > 0) {
      for (usize k = 0; k < num_clusters; k++) {
        for (usize i = 0; i < subsets[k].size(); i++) {
          if (!sampling_model.converged(*subset_states[k][i])) {
            return false;
          }
        }
      }
      return true;
    } else {
      return !options.enabled;
    }
  };

  // private:

  std::vector<std::set<usize>> select_and_learn_linkage(Rng& rng,
                                                        ArchiveBase& archive,
                                                        InstanceBase& problem,
                                                        SolutionSetBase& solutions,
                                                        const std::vector<std::vector<usize>>& cluster_solutions) {
    std::vector<std::vector<usize>> active_indices(num_clusters);
    std::vector<std::set<usize>> ams_indices(num_clusters);  // TODO use an unordered_set?

    // TODO parallel
    for (usize k = 0; k < num_clusters; k++) {
      // sort cluster by fitness
      auto by_fitness = sort_by_quality_decreasing(problem.fitness(), solutions, cluster_solutions[k],
                                                   k < problem.num_objectives() ? std::make_optional(k) : std::nullopt);

      // derive the actual selection sizes from the number of active solutions
      active_indices[k].reserve(cluster_solutions[k].size());
      std::vector<usize> active_counts(num_continuous, 0);
      for (auto i : by_fitness) {
        if (options.intron_aware) {
          bool any_active = false;
          for (isize j = 0; j < num_continuous; j++) {
            if (solutions[i].continuous_active()(j)) {
              active_counts[j]++;
              any_active = true;
            }
          }
          if (any_active) {
            active_indices[k].push_back(i);
          }
        } else {
          active_indices[k].push_back(i);
        }
      }

      bool enough_solutions = static_cast<usize>(options.selection_percentile * active_indices[k].size()) >= 2;
      cluster_active(k) = enough_solutions;

      if (!cluster_active(k)) {
        // std::println("CLUSTER {} INACTIVE", k);
        continue;
      }

      if (!options.intron_aware && options.selection_logfile.has_value()) {
        usize selection_size = options.selection_percentile * active_indices[k].size();
        AoSSet p;
        p.add(solutions[0]);
        for (usize i = 0; i < selection_size; i++) {
          p[0] = solutions[active_indices[k][i]];
          debug_log(problem, options.selection_logfile.value(), "cluster,", std::format("{},", k), p);
        }
      }

      Vec<CType> new_mean = estimate_mean(solutions, active_indices[k], active_counts, options.selection_percentile,
                                          full.continuous, options.intron_aware);

      // initialize the (previous) cluster mean
      if (mean[k].size() != num_continuous) {
        mean[k] = new_mean;
      }

      // updates
      if (mean_shift[k].size() != num_continuous || options.eta_ams >= 1.0) {
        mean_shift[k] = new_mean - mean[k];
      } else {
        mean_shift[k] = options.eta_ams * (new_mean - mean[k]) + (1.0 - options.eta_ams) * mean_shift[k];
      }

      mean[k] = new_mean;

      // update subsets
      if (subsets[k].empty()) {  // init if empty
        Mat<CType> cov = estimate_cov(solutions, active_indices[k], active_counts, options.selection_percentile,
                                      mean[k], full.continuous, options.intron_aware);
        subsets[k] = linkage_model->subsets(rng, problem, solutions, cluster_solutions[k], cov);

        for (usize i = 0; i < subsets[k].size(); i++) {
          subset_states[k].push_back(sampling_model.init(subsets[k][i]));
        }
      } else if (!linkage_model->is_static()) {  // update only if the fos is not static
        Mat<CType> cov = estimate_cov(solutions, active_indices[k], active_counts, options.selection_percentile,
                                      mean[k], full.continuous, options.intron_aware);
        FOS new_fos = linkage_model->subsets(rng, problem, solutions, cluster_solutions[k], cov);

        std::vector<std::unique_ptr<RvSubsetStateBase>> new_subset_states;
        new_subset_states.reserve(new_fos.size());

        // the assumption is that the number of subsets can change, so a 1:1 mapping might not be possible in any case
        // - to avoid any issues, we simply inherit the distribution multiplier from the "closest" previous subset in
        // O(n2) instead of the hungarian algorithm in O(n3) // TODO test if that is not a performance regression over
        // doing the hungarian algorithm - but that does not handle changing numbers of subsets...
        for (usize i = 0; i < new_fos.size(); i++) {
          usize best = 0;
          double similarity = -1.0;
          for (usize j = 0; j < subsets[k].size(); j++) {
            double sim = new_fos[i].similarity(subsets[k][j]);
            if (sim > similarity) {
              similarity = sim;
              best = j;
            }
          }

          new_subset_states.push_back(sampling_model.inherit(new_fos[i], subsets[k][best], *subset_states[k][best]));
        }

        subsets[k] = std::move(new_fos);
        subset_states[k] = std::move(new_subset_states);
      }

      for (usize i = 0; i < subsets[k].size(); i++) {
        sampling_model.update(solutions, active_indices[k], active_counts, options.intron_aware, subsets[k][i],
                              options.selection_percentile, *subset_states[k][i]);

        if (options.subset_logfile.has_value()) {
          const auto& s = subsets[k][i].continuous;
          if (auto p = dynamic_cast<const AMaLGaMSubsetState*>(subset_states[k][i].get()); p != nullptr) {
            debug_log(problem, options.subset_logfile.value(), "cluster,subset,mean,mean_shift,L,dmul,kind,",
                      std::format("{},{},{},{},{},{},amalgam,", k, log_helper(s), log_helper(mean[k](s)),
                                  log_helper(mean_shift[k](s)), log_helper(p->L), p->distribution_multiplier));
          }
        }
      }

      {  // assign ams indices
        usize ams_pool_size = options.intron_aware ? active_indices[k].size() : cluster_solutions[k].size();
        usize num_ams_solutions = 0.5 * options.selection_percentile * ams_pool_size;
        std::vector<usize> perm;
        if (options.randomize_ams_indices) {
          perm = permute(rng, ams_pool_size);
          perm.resize(num_ams_solutions);
        } else {
          perm.resize(num_ams_solutions);
          std::iota(perm.begin(), perm.end(), 0);
        }

        for (auto i : perm) {
          ams_indices[k].insert(active_indices[k][i]);
        }
      }
    }

    return ams_indices;
  };

  std::tuple<bool, bool, bool> should_accept(Rng& rng,
                                             const FitnessBase& fitness,
                                             const ArchiveBase& archive,
                                             const SolutionBase& solution,
                                             const SolutionBase& parent,
                                             std::optional<usize> objective,
                                             bool strict) {
    thread_local static std::uniform_real_distribution<double> p(0.0, 1.0);
    if (objective.has_value()) {
      // improvement in the extreme direction or sideways improvement in another
      // objective
      Ordering o = fitness.cmp(solution.quality(), parent.quality(), objective);
      if (o == Ordering::Better || (fitness.num_objectives() > 1 && fitness.cmp(solution.quality(), parent.quality(),
                                                                                std::nullopt) == Ordering::Better)) {
        return std::make_tuple(true, false,
                               fitness.cmp(solution.quality(), archive.so_solution(objective.value()).quality(),
                                           objective) == Ordering::Better);
      }

      if (!strict && options.p_accept > 0.0 && p(rng) < options.p_accept) {
        return std::make_tuple(true, true, false);
      }
    } else {
      bool is_non_dominated = !archive.dominates(solution, true);
      Ordering o = fitness.cmp(solution.quality(), parent.quality(), std::nullopt);
      if (o == Ordering::Better || is_non_dominated) {
        return std::make_tuple(true, false, is_non_dominated);
      }
    }

    return std::make_tuple(false, false, false);
  };

  template <typename Derived>
  void sample(Rng& rng, const InstanceBase& problem, usize k, usize fos_idx, Eigen::MatrixBase<Derived>&& out) {
    thread_local static std::uniform_real_distribution<CType> U(0.0, 1.0);
    const usize TRIES = 100;

    const auto& s = subsets[k][fos_idx].continuous;

    for (usize i = 0; i < TRIES; i++) {
      num_samples[k]++;

      out = sampling_model.sample(rng, *subset_states[k][fos_idx]);

      // bounds check
      if ((problem.continuous_lower_bounds()(s).array() <= out.array()).all() &&
          (out.array() <= problem.continuous_upper_bounds()(s).array()).all()) {
        return;
      }
      num_oob[k]++;
    }

    // otherwise we sample uniformally in the init bounds
    for (usize j = 0; j < s.size(); j++) {
      out(j) *= U(rng) * (problem.continuous_init_upper_bounds()(s[j]) - problem.continuous_init_lower_bounds()(s[j])) +
                problem.continuous_init_lower_bounds()(s[j]);
    }
  };

  usize gom_step(Rng& rng,
                 ArchiveBase& archive,
                 InstanceBase& problem,
                 SolutionSetBase& solutions,
                 SolutionSetBase& parents,
                 const std::vector<usize>& solution_clusters,
                 const std::vector<std::vector<usize>>& cluster_solutions,
                 const Arr2D<usize>& subset_orders,
                 usize subset_idx,
                 const std::vector<std::set<usize>>& ams_indices) {
    solutions_to_evaluate.clear();

    for (usize k = 0; k < num_clusters; k++) {
      num_oob[k] = 0;
      num_samples[k] = 0;

      improved_indices[k].clear();
    }

    bool do_partial_ams =
        // we do want to do partial ams
        options.enable_partial_ams &&
        // and we have a mean shift
        (generation > 0 || options.init_ams_from_population_mean);

    // TODO parallel?
    for (usize i = 0; i < solutions.size(); i++) {
      auto k = solution_clusters[i];
      if (!cluster_active(k)) {
        continue;
      }

      auto fos_idx = subset_orders(k, subset_idx);

      // some clusters might have more subsets...
      if (fos_idx < subsets[k].size() && subsets[k][fos_idx].continuous.size() > 0) {
        eval_subsets[i] = &subsets[k][fos_idx];
        const auto& s = subsets[k][fos_idx].continuous;

        sample(rng, problem, k, fos_idx, solutions[i].continuous_values()(s));
        if (do_partial_ams && ams_indices[k].contains(i)) {
          Vec<CType> shifted_values;
          CType shift_magnitude = 1.0;
          while (shift_magnitude > 1e-10) {  // 34 times (log2(1/1e-10))
            shifted_values = solutions[i].continuous_values()(s) +
                             shift_magnitude * subset_states[k][fos_idx]->distribution_multiplier * options.delta_ams *
                                 mean_shift[k](s);

            if ((problem.continuous_lower_bounds()(s).array() <= shifted_values.array()).all() &&
                (shifted_values.array() <= problem.continuous_upper_bounds()(s).array()).all()) {
              solutions[i].continuous_values()(s) = shifted_values;
              break;
            }

            shift_magnitude *= 0.5;
          }
        }

        bool evaluation_needed = !options.intron_aware || solutions[i].continuous_active()(s).array().any();
        if (evaluation_needed) {
          solutions_to_evaluate.push_back(i);
        } else {
            parents[i] = solutions[i];
        }
      }
    }

    if (solutions_to_evaluate.empty())
      return 0;

    problem.evaluate_partial(rng, solutions, parents, eval_subsets, solutions_to_evaluate);

    if (options.sample_logfile.has_value()) {
      AoSSet gom_solutions;
      for (usize i : solutions_to_evaluate) {
        gom_solutions.add(solutions[i]);
      }
      debug_log(problem, options.sample_logfile.value(), "step,", "gom,", gom_solutions);
    }

    for (usize i : solutions_to_evaluate) {
      auto k = solution_clusters[i];
      std::optional<usize> objective = k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

      // acceptance - first see what improved, do the intermediate update
      // and then really update the archive
      auto [accept, accept_randomly, improved] =
          should_accept(rng, problem.fitness(), archive, solutions[i], parents[i], objective, false);
      if (accept || accept_randomly) {
        parents[i] = solutions[i];

        if (!accept_randomly) {
          solution_improved[i] = true;
        }

        if (improved) {
          improved_indices[k].push_back(i);
        }
      } else {
        solutions[i].reject(parents[i], problem.always_inherit_continuous(), *eval_subsets[i]);
      }
    }

    for (usize k = 0; k < num_clusters; k++) {
      // intermediate update
      if (cluster_active(k)) {
        auto fos_idx = subset_orders(k, subset_idx);
        if (fos_idx < subsets[k].size()) {
          sampling_model.adapt(solutions, improved_indices[k],
                               static_cast<double>(num_oob[k]) / static_cast<double>(num_samples[k]),
                               no_improvement_stretch[k] >= options.max_nis, options.intron_aware, subsets[k][fos_idx],
                               *subset_states[k][fos_idx]);
        }
      }

      // archive update
      for (usize i : improved_indices[k]) {
        archive.update(solutions[i], true);
      }
    }

    return solutions_to_evaluate.size();
  };

  usize full_ams(Rng& rng,
                 ArchiveBase& archive,
                 InstanceBase& problem,
                 SolutionSetBase& solutions,
                 SolutionSetBase& parents,
                 const std::vector<usize>& solution_clusters,
                 const std::vector<std::set<usize>>& ams_indices) {
    if ((generation == 0 && !options.init_ams_from_population_mean)) {
      return 0;
    }

    solutions_to_evaluate.clear();

    for (usize k = 0; k < num_clusters; k++) {
      if (cluster_active(k)) {
        for (usize i : ams_indices[k]) {
          eval_subsets[i] = &full;

          // Eigen reference, writes back to the solution...
          auto values = solutions[i].continuous_values();

          CType shift_magnitude = 1.0;
          bool in_bounds = false;
          while (shift_magnitude > 1e-10) {
            values = parents[i].continuous_values() + shift_magnitude * options.delta_ams * mean_shift[k];

            if ((problem.continuous_lower_bounds().array() <= values.array()).all() &&
                (values.array() <= problem.continuous_upper_bounds().array()).all()) {
              // always true since the ams indices come from a pre-selection of active only indices...
              // bool evaluation_needed = !options.intron_aware || solutions[i].continuous_active().array().any();
              // if(evaluation_needed){
              //     solutions_to_evaluate.push_back(i);
              // }

              // solutions[i].continuous_values() = values; // not needed since values already is a reference, this
              // just is a self assignment...
              solutions_to_evaluate.push_back(i);
              in_bounds = true;
              break;
            }

            shift_magnitude *= 0.5;
          }
          if (!in_bounds) {
            // std::println("OOB {}, nothing to do.", i);
            values = parents[i].continuous_values();
          }
        }
      }
    }

    if (solutions_to_evaluate.empty()) {
      return 0;
    }

    problem.evaluate_partial(rng, solutions, parents, eval_subsets, solutions_to_evaluate);

    if (options.sample_logfile.has_value()) {
      AoSSet ams_solutions;
      for (usize i : solutions_to_evaluate) {
        ams_solutions.add(solutions[i]);
      }
      debug_log(problem, options.sample_logfile.value(), "step,", "ams,", ams_solutions);
    }

    for (usize i : solutions_to_evaluate) {
      auto k = solution_clusters[i];
      std::optional<usize> objective = k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

      auto [accept, accept_randomly, improved] =
          should_accept(rng, problem.fitness(), archive, solutions[i], parents[i], objective, false);

      if (accept) {
        parents[i] = solutions[i];

        if (!accept_randomly) {
          solution_improved[i] = true;
        }

        if (improved) {
          archive.update(solutions[i], false);
        }
      } else {
        solutions[i].reject(parents[i], problem.always_inherit_continuous(), *eval_subsets[i]);
      }
    }

    return solutions_to_evaluate.size();
  };

  usize forced_improvements(Rng& rng,
                            ArchiveBase& archive,
                            InstanceBase& problem,
                            SolutionSetBase& solutions,
                            SolutionSetBase& parents,
                            const std::vector<usize>& solution_clusters,
                            Mat<usize>& subset_orders,
                            usize max_num_subsets) {
    std::vector<const SolutionBase*> closest_elites;
    closest_elites.reserve(solution_clusters.size());

    std::vector<usize> solutions_to_improve;
    solutions_to_improve.reserve(solutions.size());
    for (usize i = 0; i < solution_clusters.size(); i++) {
      usize k = solution_clusters[i];
      if (solution_nis[i] > options.max_nis && solutions[i].continuous_active().array().any()) {
        // rv fi only works if there the elite solution actually has active rv values...
        if (k < problem.num_objectives()) {
          const auto* e = &archive.so_solution(k);
          if (problem.fitness().cmp(solutions[i].quality(), e->quality(), k) == Ordering::Worse &&
              e->continuous_active().any()) {
            solutions_to_improve.push_back(i);
            closest_elites.push_back(e);
          }
        } else {
          CType closest_dist = std::numeric_limits<CType>::infinity();
          const SolutionBase* closest = nullptr;
          for (usize j = 0; j < archive.size(); j++) {
            const SolutionBase* e = &archive[j];
            if (e->continuous_active().any()) {
              // TODO norm this distance? If so, how?
              CType dist = problem.fitness().distance(solutions[i].quality(), e->quality(), std::nullopt);
              if (dist < closest_dist) {
                closest_dist = dist;
                closest = e;
              }
            }
          }
          if (!isna(closest_dist) && closest) {
            solutions_to_improve.push_back(i);
            closest_elites.push_back(closest);
          }
        }
      }
    }

    for (auto row : subset_orders.rowwise()) {
      std::shuffle(row.begin(), row.end(), rng);
    }

    std::vector<usize> eval2improve_idx;
    eval2improve_idx.reserve(solutions_to_improve.size());

    usize subset_idx = 0;
    usize evaluations = 0;
    usize tries_left = options.num_forced_improvement_tries;
    CType alpha = 0.5;
    while (tries_left > 0 && !solutions_to_improve.empty()) {
      tries_left--;

      while (subset_idx < max_num_subsets && !solutions_to_improve.empty()) {
        eval2improve_idx.clear();
        solutions_to_evaluate.clear();

        for (usize j = 0; j < solutions_to_improve.size(); j++) {
          auto i = solutions_to_improve[j];
          usize k = solution_clusters[i];

          auto fos_idx = subset_orders(k, subset_idx);

          // due to filtering/max_subset_size, some clusters might have more
          // subsets...
          if (fos_idx < subsets[k].size()) {
            const auto& s = subsets[k][fos_idx].continuous;
            bool active_overlap = false;
            for (usize l : s) {
              if (solutions[i].continuous_active()(l)) {
                active_overlap = true;
                break;
              }
            }
            if (active_overlap) {
              eval_subsets[i] = &subsets[k][fos_idx];

              solutions[i].continuous_values()(s) = alpha * parents[i].continuous_values()(s) +
                                                    (CType(1.0) - alpha) * closest_elites[j]->continuous_values()(s);
              solutions_to_evaluate.push_back(i);
              eval2improve_idx.push_back(j);
            }
          }
        }

        problem.evaluate_partial(rng, solutions, parents, eval_subsets, solutions_to_evaluate);
        evaluations += solutions_to_evaluate.size();

        // accept in a random order and remove the improved solutions from the
        // worklist
        {
          std::vector<usize> perm(solutions_to_evaluate.size());
          std::iota(perm.begin(), perm.end(), 0);
          std::shuffle(perm.begin(), perm.end(), rng);

          solutions_to_evaluate.clear();
          auto& indices_to_remove = solutions_to_evaluate;

          for (usize j : perm) {
            auto i = solutions_to_improve[eval2improve_idx[j]];

            auto k = solution_clusters[i];
            std::optional<usize> objective =
                k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

            auto [changed, _, improved] =
                should_accept(rng, problem.fitness(), archive, solutions[i], parents[i], objective, true);

            if (changed) {
              parents[i] = solutions[i];
              indices_to_remove.push_back(eval2improve_idx[j]);

              if (improved) {
                archive.update(solutions[i], false);
              }
            } else {
              solutions[i].reject(parents[i], problem.always_inherit_continuous(), *eval_subsets[i]);
            }
          }
          // sort indices in reverse - otherwise removing smaller indices might
          // cause invalid memory accesses when trying to remove a larger index
          // later on...
          std::sort(indices_to_remove.begin(), indices_to_remove.end(), std::greater{});
          for (usize i : indices_to_remove) {
            std::swap(solutions_to_improve[i], solutions_to_improve.back());
            solutions_to_improve.pop_back();

            std::swap(closest_elites[i], closest_elites.back());
            closest_elites.pop_back();
          }
        }

        subset_idx++;
      }

      alpha *= 0.5;
    }
    return evaluations;
  };

  // std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus>
  // run(InstanceBase &problem, const Budget &budget, std::optional<usize> seed,
  //     std::optional<usize> population_size){
  //         // ims go brr
  //     }

  std::optional<usize> current_generation() const  // override final
  {
    return generation;
  };

  RvOptions options;

 private:
  std::unique_ptr<LinkageModelBase> linkage_model;
  const RvSamplingModelBase& sampling_model;

  usize num_clusters = 0;
  isize num_continuous = 0;
  std::vector<usize> solutions_to_evaluate;
  std::vector<std::vector<usize>> improved_indices;
  std::vector<const Subset*>
      eval_subsets;  // pointers because 1. we want to avoid copies and 2. the view should be nullable
  Subset full;

  Array<BType> cluster_active;
  std::vector<Vec<CType>> mean;        // per cluster mean
  std::vector<Vec<CType>> mean_shift;  // per cluster mean shift

  std::vector<u64> num_oob;
  std::vector<u64> num_samples;

  std::vector<u64> no_improvement_stretch;  // per cluster
  std::vector<bool> solution_improved;
  std::vector<u64> solution_nis;

  std::vector<std::vector<std::unique_ptr<RvSubsetStateBase>>> subset_states;
  std::vector<FOS> subsets;

  u64 generation = 0;
};

};  // namespace goblin

#endif /* _GOBLIN_METHODS_CONTINUOUS_H */
