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

// Disentangle linkage from sampling...



// Performs an inplace cholesky decomposition. If the decomposition fails, jitter is added to the diagonal to increase
// the rank until finally the univariate diagonal is used.
template <typename Derived>
inline void cholesky_inplace(Eigen::MatrixBase<Derived>& out) {
  using S = typename Derived::Scalar;
  const usize num_tries = 10;

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
    std::println("!!! CHOLESKY FAILED !!!\n{}", log_helper(out, /* escape = */ false, /* indent = */ true));

    // covariance diagonal without jitter, made positive and sqrt to match the expecation that out * out.T = input
    Vec<S> univariate = (out.diagonal().array() - jitter_added).max(1e-10).sqrt();
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

// only AMS is managed by the rvstate directly, rest is done by the sampling distribution...
// one samplingmodel per cluster -> parallel updates should be possible
class RvSamplingModelBase {
    public:

    virtual std::unique_ptr<RvSamplingModelBase> clone() const = 0;

    virtual void init(const FOS& fos) = 0;
    virtual void update_subsets(const FOS& new_fos, const FOS& old_fos) {
        init(new_fos);
    };

    virtual void update_distribution(const SolutionSetBase& solutions, const std::span<const usize> by_fitness_decreasing, const std::span<const usize> active_counts, const Subset& subset, double selection_percentile, usize subset_idx) = 0;
    virtual Vec<CType> sample(Rng& rng, usize subset_idx) const = 0;
    virtual void adapt(const SolutionSetBase& solutions, const std::span<const usize>  improved_indices, double oob_ratio, bool max_nis_reached, const Subset& subset, usize subset_idx) {};

    virtual bool converged() const = 0;
    virtual ~RvSamplingModelBase() = default;
};

class AMaLGaMSamplingModel final: public RvSamplingModelBase {
    public:
    AMaLGaMSamplingModel(
        bool intron_aware = true,
        bool use_mahalanobis_distance_for_sdr = false, // uses mahalanobis distance instead of max(abs(z)) for computing the SDR
        CType eta_cov = 1.0,
        CType std_deviation_ratio_threshold = 1.0,
        CType distribution_multiplier_decrease = 0.9,
        CType distribution_multiplier_increase = 1.0 / 0.9,
        CType min_distribution_multiplier = 1e-10):
    intron_aware(intron_aware),
    use_mahalanobis_distance_for_sdr(use_mahalanobis_distance_for_sdr),
    eta_cov(eta_cov),
    std_deviation_ratio_threshold(std_deviation_ratio_threshold),
    distribution_multiplier_decrease(distribution_multiplier_decrease),
    distribution_multiplier_increase(distribution_multiplier_increase),
    min_distribution_multiplier(min_distribution_multiplier)
    {};

    std::unique_ptr<RvSamplingModelBase> clone() const override final {
        return std::make_unique<AMaLGaMSamplingModel>(*this);
    }

    void init(const FOS& fos) override final {
        initialized.clear();
        initialized.resize(fos.size(), false);

        distribution_multiplier.resize(fos.size());
        distribution_multiplier.setConstant(1.0);

        mean.resize(fos.size());
        cov.resize(fos.size());
        L.resize(fos.size());
    }

    void update_subsets(const FOS& new_fos, const FOS& old_fos) override final {
        initialized.clear();
        initialized.resize(new_fos.size(), false);

        Array<CType> new_distribution_multiplier(new_fos.size());
        std::vector<Vec<CType>> new_mean(new_fos.size());
        std::vector<Mat<CType>> new_cov(new_fos.size());

        // the assumption is that the _number_ of subsets can change, so a 1:1 mapping might not be possible in any case - to avoid any issues, we simply inherit the distribution multiplier from the "closest" previous subset in O(n2) instead of the hungarian algorithm in O(n3) (there also is no need for exclusive assignment/no condition for subset state being inherited from only once)

        // TODO test if that is not a performance regression compared to the hungarian algorithm - but that does not handle changing numbers of subsets...

        for (usize i = 0; i < new_fos.size(); i++) {
          usize best = 0;
          double similarity = -1.0;
          for (usize j = 0; j < old_fos.size(); j++) {
            double sim = new_fos[i].similarity(old_fos[j]);
            if (sim > similarity) {
              similarity = sim;
              best = j;
            }
          }

          new_distribution_multiplier(i) = distribution_multiplier(best);
          new_mean[i] = mean[best];
          new_cov[i] = cov[best];
        }

        distribution_multiplier = new_distribution_multiplier;
        mean = new_mean;
        cov = new_cov;
    }

    void update_distribution(const SolutionSetBase& solutions, const std::span<const usize> by_fitness_decreasing, const std::span<const usize> active_counts, const Subset& subset, double selection_percentile, usize subset_idx) override final {
        const auto& s = subset.continuous;
        usize selection_size = selection_percentile * by_fitness_decreasing.size();

        // TODO intron aware ...
        mean[subset_idx] = Vec<CType>::Zero(s.size());
        for(usize i = 0; i < selection_size; i++){
            mean[subset_idx] += solutions[by_fitness_decreasing[i]].continuous_values()(s);
        }
        mean[subset_idx] /= static_cast<CType>(selection_size);

        // Change the focus of the search to the best solution
        if(distribution_multiplier(subset_idx) < 1.0){
            for(usize i: s){
                if(solutions[by_fitness_decreasing[0]].continuous_active()(i)){
                mean[subset_idx](i) = solutions[by_fitness_decreasing[0]].continuous_values()(i);
                }
            }
        }

        // TODO intron aware ...
        Mat<CType> new_cov = Mat<CType>::Zero(s.size(), s.size());
        for (usize i = 0; i < selection_size; i++) {
          Vec<CType> v = solutions[by_fitness_decreasing[i]].continuous_values() - mean[subset_idx];
          new_cov.noalias() += v * v.transpose();
        }
        new_cov /= static_cast<CType>(selection_size);

        if(cov[subset_idx].size() != new_cov.size()){
            cov[subset_idx] = new_cov;
        } else {
            cov[subset_idx] = eta_cov * new_cov + (1.0 - eta_cov) * cov[subset_idx];
        }

        L[subset_idx] = cov[subset_idx];
        cholesky_inplace(L[subset_idx]);
    }

    Vec<CType> sample(Rng& rng, usize subset_idx) const override final {
        assert(initialized[subset_idx]);

        thread_local static std::normal_distribution<CType> N(0.0, 1.0);
        Vec<CType> z(mean[subset_idx].size());
        // sample from N(0, I)
        for (isize i = 0; i < z.size(); i++) {
          z(i) = N(rng);
        }
        // scale to N(mean[subset_idx], cov[subset_idx])
        return (L[subset_idx].triangularView<Eigen::Lower>() * z) + mean[subset_idx];
    };

    void adapt(const SolutionSetBase& solutions, const std::span<const usize> improved_indices, double oob_ratio, bool max_nis_reached, const Subset& subset, usize subset_idx) override final {
        if(oob_ratio > 0.9){
            distribution_multiplier(subset_idx) *= 0.5;
        }

        if(!improved_indices.empty()){
            if(distribution_multiplier(subset_idx) < 1.0){
                distribution_multiplier(subset_idx) = 1.0;
            }

            CType std_deviation_ratio = compute_SDR(solutions, improved_indices, subset, subset_idx);
            if(std_deviation_ratio > std_deviation_ratio_threshold){
                distribution_multiplier(subset_idx) *= distribution_multiplier_increase;
            }
        } else if(max_nis_reached){
            distribution_multiplier(subset_idx) *= distribution_multiplier_decrease;
        } else {
            if(distribution_multiplier(subset_idx) > 1.0){
                distribution_multiplier(subset_idx) *= distribution_multiplier_decrease;
            }

            if(distribution_multiplier(subset_idx) < 1.0){
                distribution_multiplier(subset_idx) = 1.0;
            }
        }
    }


    bool converged() const override final {
        return distribution_multiplier.size() > 0 && (distribution_multiplier <= min_distribution_multiplier).all();
    }

    private:

    CType compute_SDR(const SolutionSetBase& solutions, const std::span<const usize> improved_indices, const Subset& subset, usize subset_idx){
        const auto& s = subset.continuous;

        // TODO intron aware
        if(use_mahalanobis_distance_for_sdr){
            std::unreachable();
        } else {
            // TODO ask anton about z of avg vs avg of z
            Vec<CType> avg_params = Vec<CType>::Zero(s.size());
            for(usize i: improved_indices){
                avg_params += solutions[i].continuous_values()(s);
            }
            avg_params /= static_cast<CType>(improved_indices.size());

            Mat<CType> L_inv = L[subset_idx].completeOrthogonalDecomposition()
            .pseudoInverse();

            return ((L_inv * avg_params) - mean[subset_idx]).array().abs().maxCoeff();
        }
    }

    // options

    bool intron_aware;
    bool use_mahalanobis_distance_for_sdr;
    CType eta_cov;
    CType std_deviation_ratio_threshold;
    CType distribution_multiplier_decrease;
    CType distribution_multiplier_increase;
    CType min_distribution_multiplier;

    // per subset data

    std::vector<bool> initialized;
    Array<CType> distribution_multiplier;
    std::vector<Vec<CType>> mean;
    std::vector<Mat<CType>> cov; // unscaled by dmul
    std::vector<Mat<CType>> L;

};

struct RvOptions {
  bool enabled = true;
  bool intron_aware = true;
  bool init_ams_from_population_mean = true;
  // If randomized, the AMS indices are randomly picked from all active solutions.
  // Otherwise the first `floor(selection_percentile * 0.5)` active solutions are used.
  bool randomize_ams_indices = false;
  // Determines whether partial and full AMS are performed when the full FOS is used.
  bool enable_partial_ams_for_full_fos = true;
  bool use_no_improvement_counts = false;
  usize num_forced_improvement_tries = 8;  // 8 is the RV GOMEA default if I did not miscalculate (1.0 / 2^8 < 0.01)
  usize max_nis = 20;

  double selection_percentile = 0.35;
  double p_accept = 0.05;

  CType delta_ams = 2.0;

  CType eta_ams = 1.0;
  CType eta_cov = 1.0;

  CType std_deviation_ratio_threshold = 1.0;
  CType distribution_multiplier_decrease = 0.9;
  CType distribution_multiplier_increase = 1.0 / 0.9;

  CType min_distribution_multiplier = 1e-10;

  // In the GBO setting with partial evaluations, numerical errors of partial fitness updates
  // can accumulate and it might be needed to perform full evaluations once in a while
  //
  // In that case, the default number of generations until re-evaluation is `50`
  std::optional<u64> generations_until_full_evaluation = std::nullopt;

  std::optional<std::string> population_logfile = std::nullopt;
  std::optional<std::string> selection_logfile = std::nullopt;
  std::optional<std::string> subset_logfile = std::nullopt;
  std::optional<std::string> sample_logfile = std::nullopt;

  void validate() {
    __goblin_runtime_assert(0.0 <= p_accept && p_accept < 1.0);
    __goblin_runtime_assert(0.0 <= selection_percentile && selection_percentile < 1.0);
    __goblin_runtime_assert(0.0 <= eta_ams && eta_ams < 1.0);
    __goblin_runtime_assert(0.0 <= eta_cov && eta_cov < 1.0);
    __goblin_runtime_assert(std_deviation_ratio_threshold > 0.0);
    __goblin_runtime_assert(0.0 < distribution_multiplier_decrease && distribution_multiplier_decrease < 1.0);
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
      for (usize i : front) {
        by_fitness.push_back(indices[i]);
      }
    }
  }

  return by_fitness;
};

class RvState {
 public:
  RvState(RvOptions options, const LinkageModelBase& linkage_model) : options(options), linkage_model(linkage_model.clone()) {};

  usize perform_generation(Rng& rng,
                           ArchiveBase& archive,
                           InstanceBase& problem,
                           SolutionSetBase& solutions,
                           SolutionSetBase& parents,
                           const std::vector<usize>& solution_clusters,
                           const std::vector<std::vector<usize>>& cluster_solutions) {
    // initialization
    if (num_clusters != cluster_solutions.size() || static_cast<usize>(num_continuous) != problem.num_continuous() ||
        no_improvement_counts.size() != solution_clusters.size()) {
            linkage_model->init(rng, problem, solutions, VariableSet::Continuous);

      num_clusters = cluster_solutions.size();
      num_continuous = problem.num_continuous();

      full.continuous.resize(num_continuous);
      std::iota(full.continuous.begin(), full.continuous.end(), 0);

      if (no_improvement_counts.size() != solution_clusters.size()) {
        no_improvement_counts.clear();
        no_improvement_counts.resize(solution_clusters.size(), 0);
        eval_subsets.resize(solution_clusters.size());
        solutions_to_evaluate.reserve(solution_clusters.size());

        solution_nis.clear();
        solution_nis.resize(solution_clusters.size(), 0);

        solution_improved.clear();
        solution_improved.resize(solution_clusters.size(), false);
      }

      if (static_cast<usize>(cluster_active.size()) != num_clusters) {
        cluster_active.resize(num_clusters);
        ms_active.resize(num_clusters);
        distribution_multipliers.resize(num_clusters);
        num_oob.resize(num_clusters);
        num_samples.resize(num_clusters);
        L.resize(num_clusters);
        subsets.resize(num_clusters);
        mean.resize(num_clusters);
        mean_shift.resize(num_clusters);
        cov.resize(num_clusters);
        any_improved.resize(num_clusters, false);
        no_improvement_stretch.resize(num_clusters, 0);
        improved_indices.resize(num_clusters);
      }

      // enable partial ams only if the linkage model is not the full fos - in that case, we stay as close to what
      // AMaLGaM does as possible...
      auto ptr = reinterpret_cast<const FullFOS*>(&linkage_model);
      enable_partial_ams = options.enable_partial_ams_for_full_fos || ptr == nullptr;
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
      any_improved[k] = false;
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

    // normal RV-GOMEA updates the solution NIS here - but in the intron aware case some solutions might not have been
    // subject to variation at all, so this is changed into a no improvement count and updated per variation step, so
    // nothing to do here
    for (usize i = 0; i < solutions.size(); i++) {
      if (solution_improved[i]) {
        solution_nis[i] = 0;
        any_improved[solution_clusters[i]] = true;
      } else {
        solution_nis[i]++;
      }
    }

    if (options.num_forced_improvement_tries > 0) {
      evaluations += forced_improvements(rng, archive, problem, solutions, parents, solution_clusters);
    }

    for (usize k = 0; k < num_clusters; k++) {
      if (any_improved[k]) {
        no_improvement_stretch[k] = 0;
      } else if ((distribution_multipliers[k] <= 1.0).all()) {
        no_improvement_stretch[k]++;
      }
    }

    generation++;
    return evaluations;
  };

  bool converged() const {
    if (num_clusters > 0) {
      for (usize k = 0; k < num_clusters; k++) {
        if ((distribution_multipliers[k] > options.min_distribution_multiplier).any()) {
          return false;
        }
      }
      return true;
    } else {
      return !options.enabled;
    }
  };

  // private:

  Vec<CType> estimate_mean(const SolutionSetBase& solutions,
                           std::vector<usize>& active_indices,
                           std::vector<usize>& active_counts) {
    Vec<CType> new_mean = Vec<CType>::Zero(num_continuous);
    if (options.intron_aware) {
      std::vector<usize> selection_sizes(num_continuous);
      for (isize i = 0; i < num_continuous; i++) {
        selection_sizes[i] = options.selection_percentile * active_counts[i];
      }
      std::vector<usize> num_left = selection_sizes;
      bool any_left = true;
      for (usize i = 0; i < active_indices.size() && any_left; i++) {
        any_left = false;
        for (isize j = 0; j < num_continuous; j++) {
          if (num_left[j] > 0 && solutions[active_indices[i]].continuous_active()(j)) {
            new_mean(j) += solutions[active_indices[i]].continuous_values()(j);
            num_left[j]--;
            any_left |= num_left[j] > 0;
          }
        }
      }
      for (isize j = 0; j < num_continuous; j++) {
        if (selection_sizes[j] > 0) {
          new_mean(j) /= static_cast<CType>(selection_sizes[j]);
        }
      }
    } else {
      usize selection_size = options.selection_percentile * active_indices.size();
      for (usize i = 0; i < selection_size; i++) {
        new_mean += solutions[active_indices[i]].continuous_values();
      }
      new_mean /= static_cast<CType>(selection_size);
    }
    return new_mean;
  };

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

      bool enough_solutions = active_indices[k].size() >= 5;
      bool any_subsets_active = distribution_multipliers[k].size() == 0 ||
                                (distribution_multipliers[k] >= options.min_distribution_multiplier).any();
      cluster_active(k) = enough_solutions && any_subsets_active;

      if (!cluster_active(k)) {
        // std::println("CLUSTER {} INACTIVE ({},{})", k, enough_solutions, any_subsets_active);
        continue;
      }

      {  // selection
        std::vector<usize> selection_sizes(num_continuous, options.selection_percentile * cluster_solutions[k].size());
        if (options.intron_aware) {
          for (isize i = 0; i < num_continuous; i++) {
            selection_sizes[i] = options.selection_percentile * active_counts[i];
          }
        }

        if (!options.intron_aware && options.selection_logfile.has_value()) {
          AoSSet p;
          p.add(solutions[0]);
          for (usize i = 0; i < selection_sizes[0]; i++) {
            p[0] = solutions[active_indices[k][i]];
            debug_log(problem, options.selection_logfile.value(), "cluster,", std::format("{},", k), p);
          }
        }

        Vec<CType> new_mean = estimate_mean(solutions, active_indices[k], active_counts);
        // TODO extract mean estimation (distribution estimation should happen per subset to allow having more
        // up-to-date distributions - not an issue for GP though) mean
        // Vec<CType> new_mean = Vec<CType>::Zero(num_continuous);
        {
          // if (options.intron_aware) {
          //   std::vector<usize> num_left = selection_sizes;
          //   bool any_left = true;
          //   for (usize i = 0; i < active_indices[k].size() && any_left; i++) {
          //     any_left = false;
          //     for (isize j = 0; j < num_continuous; j++) {
          //       if (num_left[j] > 0 && solutions[active_indices[k][i]].continuous_active()(j)) {
          //         new_mean(j) += solutions[active_indices[k][i]].continuous_values()(j);
          //         num_left[j]--;
          //         any_left |= num_left[j] > 0;
          //       }
          //     }
          //   }
          // } else {
          //   for (usize i = 0; i < selection_sizes[0]; i++) {
          //     new_mean += solutions[active_indices[k][i]].continuous_values();
          //   }
          // }
          // for (isize j = 0; j < num_continuous; j++) {
          //   if (selection_sizes[j] > 0) {
          //     new_mean(j) /= static_cast<CType>(selection_sizes[j]);
          //   }
          // }

          // change the focus of the search to the best solution (so only)
          if (k < problem.num_objectives() && distribution_multipliers[k].size() > 0 &&
              (distribution_multipliers[k] < 1.0).all()) {
            const auto& s = archive.so_solution(k);
            if (options.intron_aware) {
              for (isize j = 0; j < num_continuous; j++) {
                if (s.continuous_active()(j)) {
                  new_mean(j) = s.continuous_values()(j);
                }
              }
            } else {
              new_mean = s.continuous_values();
            }
          }

          // initialize the (previous) cluster mean
          if (mean[k].size() != num_continuous) {
            if (options.init_ams_from_population_mean) {
              mean[k] = Vec<CType>::Zero(num_continuous);
              for (usize i : cluster_solutions[k]) {
                mean[k] += solutions[i].continuous_values();
              }
              mean[k] /= cluster_solutions[k].size();
            } else {
              mean[k] = new_mean;
            }
          }
        }

        // TODO extract cov estimation (distribution estimation should happen per subset to allow having more up-to-date
        // distributions - not an issue for GP though) cov
        usize selection_size = selection_sizes[0];
        Mat<CType> new_cov = Mat<CType>::Zero(num_continuous, num_continuous);
        {
          if (options.intron_aware) {
            Vec<CType> tmp(num_continuous);
            Vec<CType> w(num_continuous);
            for (isize j = 0; j < num_continuous; j++) {
              w(j) = 1.0 / std::sqrt(static_cast<CType>(selection_sizes[j]));
            }

            std::vector<usize> num_left = selection_sizes;
            bool any_left = true;
            for (usize i = 0; i < active_indices[k].size() && any_left; i++) {
              tmp.setZero();
              any_left = false;
              for (isize j = 0; j < num_continuous; j++) {
                if (num_left[j] > 0 && solutions[active_indices[k][i]].continuous_active()(j)) {
                  tmp(j) += (solutions[active_indices[k][i]].continuous_values()(j) - new_mean(j)) * w(j);
                  num_left[j]--;
                  any_left |= num_left[j] > 0;
                }
              }
              new_cov += tmp * tmp.transpose();
            }
          } else {
            // for (isize l = 0; l < num_continuous; l++) {
            //   for (isize r = l; r < num_continuous; r++) {
            //     for (usize i = 0; i < selection_size; i++) {
            //       auto vs = solutions[active_indices[k][i]].continuous_values();
            //       new_cov(l, r) += (vs(l) - new_mean(l)) * (vs(r) - new_mean(r));
            //     }
            //     new_cov(l, r) /= static_cast<CType>(selection_size);
            //     new_cov(r, l) = new_cov(l, r);
            //   }
            // }
            for (usize i = 0; i < selection_size; i++) {
              Vec<CType> v = solutions[active_indices[k][i]].continuous_values() - new_mean;
              new_cov.noalias() += v * v.transpose();
            }
            new_cov /= static_cast<CType>(selection_size);
          }

          // TODO regularize cov for small selection sizes selection_size < (n*(n+1))/2+1 where n = num_continuous
        }

        // updates
        if (mean_shift[k].size() != num_continuous) {
          mean_shift[k] = new_mean - mean[k];
        } else if (options.eta_ams < 1.0) {
          mean_shift[k] = options.eta_ams * (new_mean - mean[k]) + (1.0 - options.eta_ams) * mean_shift[k];
        } else {
          mean_shift[k] = new_mean - mean[k];
        }
        // TODO test the effect of this...
        ms_active(k) = true;  // (new_mean - mean[k]).array().abs().sum() > 1e-12;
        mean[k] = new_mean;

        if (cov[k].rows() != num_continuous || options.eta_cov >= 1.0) {
          cov[k] = new_cov;
        } else {
          cov[k] = options.eta_cov * new_cov + (1.0 - options.eta_cov) * cov[k];
        }

        // update subsets
        if (subsets[k].empty()) {  // init if empty
          subsets[k] = linkage_model->subsets(rng, problem, solutions, cluster_solutions[k], cov[k]);

          usize ssize = subsets[k].size();
          distribution_multipliers[k] = Array<CType>::Ones(ssize);
          num_oob[k] = Array<u64>::Zero(ssize);
          num_samples[k] = Array<u64>::Zero(ssize);
          L[k].resize(subsets[k].size());
        } else if (!linkage_model->is_static()) {  // update only if the fos is not static
          FOS new_fos = linkage_model->subsets(rng, problem, solutions, cluster_solutions[k], cov[k]);

          Array<CType> previous_distribution_multipliers = distribution_multipliers[k];

          num_oob[k].resize(new_fos.size());
          num_oob[k].setZero();
          num_samples[k].resize(new_fos.size());
          num_samples[k].setZero();
          L[k].resize(new_fos.size());
          distribution_multipliers[k].resize(new_fos.size());

          // the assumption is that the number of subsets can change, so a 1:1 mapping might not be possible in any case
          // - to avoid any issues, we simply inherit the distribution multiplier from the "closest" previous subset in
          // O(n2) instead of the hungarian algorithm in O(n3) // TODO test if that is not a performance regression over
          // doing the hungarian algorithm - but that does not handle changing numbers of subsets...
          for (usize i = 0; i < new_fos.size(); i++) {
            usize best = 0;
            double similarity = -1.0;
            for (usize j = 0; j < subsets[k].size(); j++) {
              double sim = 0.0;
              for (usize n : new_fos[i].continuous) {
                for (usize o : subsets[k][j].continuous) {
                  if (n == o) {
                    sim += 1.0;
                  }
                }
              }
              sim /= std::max(new_fos[i].continuous.size(), subsets[k][j].continuous.size());

              if (sim > similarity) {
                similarity = sim;
                best = j;
              }
            }

            distribution_multipliers[k][i] = previous_distribution_multipliers[best];
          }

          subsets[k] = std::move(new_fos);
        }

        // std::println("RV Cluster {}", k);
        // std::cout << "  mean: \n" << mean[k] << "\n  mean_shift: \n" << mean_shift[k] << std::endl;

        for (usize i = 0; i < subsets[k].size(); i++) {
          const auto& s = subsets[k][i].continuous;

          bool force_univariate = true;
          if (force_univariate) {
            L[k][i].resize(s.size(), s.size());
            // {
            //     Vec<CType> univariate = (distribution_multipliers[k](i) * cov[k](s, s)).diagonal().array().max(1e-10)
            //     // .sqrt() // why does removing the sqrt make it look like what I expect (univariate FOS that fails
            //     on non-separable problems)
            //     ;
            //     L[k][i].setZero();
            //     L[k][i].diagonal() = univariate;

            //     if (options.subset_logfile.has_value()) {
            //       debug_log(problem, options.subset_logfile.value(), "cluster,subset,mean,mean_shift,L,dmul,kind,",
            //                 std::format("{},{},{},{},{},{},univariate_no_sqrt,", k, log_helper(s),
            //                 log_helper(mean[k](s)),
            //                             log_helper(mean_shift[k](s)), log_helper(L[k][i]),
            //                             distribution_multipliers[k](i)));
            //     }
            // }
            {
              Vec<CType> univariate = (distribution_multipliers[k](i) * cov[k](s, s))
                                          .diagonal()
                                          .array()
                                          .max(1e-10)  //
                                          .sqrt()      // why does removing the sqrt make it look like what I expect
                                                       // (univariate FOS that fails on non-separable problems)
                  ;
              L[k][i].setZero();
              L[k][i].diagonal() = univariate;

              if (options.subset_logfile.has_value()) {
                debug_log(
                    problem, options.subset_logfile.value(), "cluster,subset,mean,mean_shift,L,dmul,kind,",
                    std::format("{},{},{},{},{},{},univariate,", k, log_helper(s), log_helper(mean[k](s)),
                                log_helper(mean_shift[k](s)), log_helper(L[k][i]), distribution_multipliers[k](i)));
              }
            }
          }
          // distribution_multipliers[k](i) = 1.0; // does not make a difference
          L[k][i] = distribution_multipliers[k](i) * cov[k](s, s);
          if (selection_size < static_cast<usize>(num_continuous) + 1) {
            // if we don't have enough solutions we fall back to the univariate diagonal
            Vec<CType> univariate = L[k][i].diagonal().array().max(1e-10).sqrt();
            L[k][i].setZero();
            L[k][i].diagonal() = univariate;
          } else {
            cholesky_inplace(L[k][i]);
          }

          if (options.subset_logfile.has_value()) {
            debug_log(problem, options.subset_logfile.value(), "cluster,subset,mean,mean_shift,L,dmul,univariate,",
                      std::format("{},{},{},{},{},{},false,", k, log_helper(s), log_helper(mean[k](s)),
                                  log_helper(mean_shift[k](s)), log_helper(L[k][i]), distribution_multipliers[k](i)));
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

        // ams_indices[k].reserve(num_ams_solutions);
        if (options.intron_aware) {
          for (auto i : perm) {
            ams_indices[k].insert(active_indices[k][i]);
          }
        } else {
          for (auto i : perm) {
            ams_indices[k].insert(cluster_solutions[k][i]);
          }
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
    thread_local static std::normal_distribution<CType> N(0.0, 1.0);
    thread_local static std::uniform_real_distribution<CType> U(0.0, 1.0);
    const usize TRIES = 100;

    const auto& s = subsets[k][fos_idx].continuous;

    for (usize i = 0; i < TRIES; i++) {
      num_samples[k](fos_idx)++;

      // sample
      for (usize j = 0; j < s.size(); j++) {
        out(j) = N(rng);
      }
      // make multivariate
      out = (L[k][fos_idx].triangularView<Eigen::Lower>() * out) + mean[k](s);
      // bounds check
      if ((problem.continuous_lower_bounds()(s).array() <= out.array()).all() &&
          (out.array() <= problem.continuous_upper_bounds()(s).array()).all()) {
        return;
      }
    }

    // otherwise we sample uniformally in the init bounds
    for (usize j = 0; j < subsets[k][fos_idx].size(); j++) {
      out(j) = U(rng) * (problem.continuous_init_upper_bounds()(j) - problem.continuous_init_lower_bounds()(j)) +
               problem.continuous_init_lower_bounds()(j);
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
      // improved_indices[k].reserve(cluster_solutions.size());
    }

    bool do_partial_ams =
        // we have a mean shift
        (generation > 0 || options.init_ams_from_population_mean) &&
        // and we do want to do partial ams
        enable_partial_ams;

    // TODO parallel?
    for (usize i = 0; i < solutions.size(); i++) {
      auto k = solution_clusters[i];
      if (!cluster_active(k)) {
        continue;
      }

      auto fos_idx = subset_orders(k, subset_idx);

      // some clusters might have more subsets...
      if (fos_idx < subsets[k].size() && L[k][fos_idx].size() > 0 && subsets[k][fos_idx].continuous.size() > 0) {
        eval_subsets[i] = &subsets[k][fos_idx];
        const auto& s = subsets[k][fos_idx].continuous;

        sample(rng, problem, k, fos_idx, solutions[i].continuous_values()(s));
        if (do_partial_ams && ms_active(k) && ams_indices[k].contains(i)) {
          Vec<CType> shifted_values;
          CType shift_magnitude = 1.0;
          while (shift_magnitude > 1e-10) {  // 34 times (log2(1/1e-10))
            shifted_values = solutions[i].continuous_values()(s) + shift_magnitude *
                                                                       distribution_multipliers[k][fos_idx] *
                                                                       options.delta_ams * mean_shift[k](s);

            if ((problem.continuous_lower_bounds()(s).array() <= shifted_values.array()).all() &&
                (shifted_values.array() <= problem.continuous_upper_bounds()(s).array()).all()) {
              solutions[i].continuous_values()(s) = shifted_values;
              break;
            }

            shift_magnitude *= 0.5;
          }
        }

        bool evaluation_needed = options.intron_aware || solutions[i].continuous_active()(s).array().any();
        if (evaluation_needed) {
          solutions_to_evaluate.push_back(i);
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
        if (!accept_randomly) {
          if (options.use_no_improvement_counts) {
            no_improvement_counts[i] = 0;
            any_improved[k] = true;
          } else {
            solution_improved[i] = true;
          }
        }
        parents[i] = solutions[i];

        if (improved) {
          improved_indices[k].push_back(i);
        }
      } else {
        if (options.use_no_improvement_counts) {
          no_improvement_counts[i]++;
        }
        solutions[i].reject(parents[i], problem.always_inherit_continuous(), *eval_subsets[i]);
      }
    }

    for (usize k = 0; k < num_clusters; k++) {
      // intermediate update
      if (cluster_active(k)) {
        auto fos_idx = subset_orders(k, subset_idx);
        if (fos_idx < subsets[k].size()) {
          const auto& s = subsets[k][fos_idx].continuous;

          // too many out-of-bounds samples
          if (static_cast<double>(num_oob[k](fos_idx)) / static_cast<double>(num_samples[k](fos_idx)) > 0.9) {
            distribution_multipliers[k](fos_idx) *= 0.5;
          }

          if (improved_indices[k].empty()) {
            // no_improvement_stretch[k]++;
            bool max_nis_reached = no_improvement_stretch[k] >= options.max_nis;

            if ((distribution_multipliers[k](fos_idx) > 1.0) || max_nis_reached) {
              distribution_multipliers[k](fos_idx) *= options.distribution_multiplier_decrease;
            }

            if (!max_nis_reached && distribution_multipliers[k](fos_idx) < 1.0) {
              distribution_multipliers[k](fos_idx) = 1.0;
            }
          } else {
            // any_improved[k] = true;
            // no_improvement_stretch[k] = 0;

            CType std_deviation_ratio;
            {
              Mat<CType> L_inv = L[k][fos_idx]  // .triangularView<Eigen::Lower>()
                                     .completeOrthogonalDecomposition()
                                     .pseudoInverse();

              Array<CType> num_improvements = Array<CType>::Zero(s.size());
              Vec<CType> average_z_of_improvement = Array<CType>::Zero(s.size());
              Vec<CType> tmp(s.size());
              for (usize i : improved_indices[k]) {
                if (options.intron_aware) {
                  // If we pay attention to introns, then the improvements count only the changes to active values
                  for (usize j = 0; j < s.size(); j++) {
                    // since there was an improvement, we know there is at least one such case
                    if (solutions[i].continuous_active()(s[j])) {
                      tmp(j) = (solutions[i].continuous_values()(s[j]) - mean[k](s[j]));
                      num_improvements(j) += 1.0;
                    } else {
                      tmp(j) = 0.0;
                    }
                  }
                } else {
                  tmp = solutions[i].continuous_values()(s) - mean[k](s);
                  num_improvements += 1.0;
                }
                average_z_of_improvement += L_inv * tmp;
              }

              if (options.intron_aware) {
                // inactive variables will have a deviation of 0 from the mean (i.e. so they don't influence the
                // maximum)
                for (usize j = 0; j < s.size(); j++) {
                  if (num_improvements(j) > 0) {
                    average_z_of_improvement(j) /= num_improvements(j);
                  }
                }
              } else {
                average_z_of_improvement.array() /= num_improvements;
              }
              std_deviation_ratio = average_z_of_improvement.array().abs().maxCoeff();
            }

            if (distribution_multipliers[k](fos_idx) < 1.0) {
              distribution_multipliers[k](fos_idx) = 1.0;
            }

            if (std_deviation_ratio > options.std_deviation_ratio_threshold) {
              distribution_multipliers[k](fos_idx) *= options.distribution_multiplier_increase;
            }
          }
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
    if ((generation == 0 && !options.init_ams_from_population_mean) || !(ms_active.any())) {
      return 0;
    }

    solutions_to_evaluate.clear();

    for (usize k = 0; k < num_clusters; k++) {
      if (cluster_active(k) && ms_active(k)) {
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

              // solutions[i].continuous_values() = values; // not needed since values already is a reference, this just
              // is a self assignment...
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
        if (!accept_randomly) {
          if (options.use_no_improvement_counts) {
            no_improvement_counts[i] = 0;
            any_improved[k] = true;
          } else {
            solution_improved[i] = true;
          }
        }

        parents[i] = solutions[i];

        if (improved) {
          archive.update(solutions[i], false);
        }
      } else {
        if (options.use_no_improvement_counts) {
          no_improvement_counts[i]++;
        }
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
                            const std::vector<usize>& solution_clusters) {
    Array<usize> max_nic(num_clusters);
    for (usize k = 0; k < num_clusters; k++) {
      max_nic(k) = options.max_nis * (subsets[k].size() + /* ams_count */ 1);
    }

    std::vector<const SolutionBase*> closest_elites;
    closest_elites.reserve(solution_clusters.size());
    solutions_to_evaluate.clear();
    for (usize i = 0; i < solution_clusters.size(); i++) {
      usize k = solution_clusters[i];
      bool do_forced_improvements =
          options.use_no_improvement_counts ? no_improvement_counts[i] > max_nic(k) : solution_nis[i] > options.max_nis;
      if (do_forced_improvements && solutions[i].continuous_active().array().any()) {
        // rv fi only works if there the elite solution actually has active rv values...
        if (k < problem.num_objectives()) {
          const auto* e = &archive.so_solution(k);
          if (problem.fitness().cmp(solutions[i].quality(), e->quality(), k) == Ordering::Worse &&
              e->continuous_active().any()) {
            solutions_to_evaluate.push_back(i);
            closest_elites.push_back(e);
            eval_subsets[i] = &full;
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
            solutions_to_evaluate.push_back(i);
            closest_elites.push_back(closest);
            eval_subsets[i] = &full;
          }
        }
      }
    }

    usize evaluations = 0;
    usize tries_left = options.num_forced_improvement_tries;
    CType alpha = 0.5;
    while (tries_left > 0 && !solutions_to_evaluate.empty()) {
      tries_left--;

      for (usize _i = 0, i; _i < solutions_to_evaluate.size(); _i++) {
        i = solutions_to_evaluate[_i];
        solutions[i].continuous_values() =
            alpha * parents[i].continuous_values() + (CType(1.0) - alpha) * closest_elites[_i]->continuous_values();
      }

      problem.evaluate_partial(rng, solutions, parents, eval_subsets, solutions_to_evaluate);
      evaluations += solutions_to_evaluate.size();

      for (usize _i = solutions_to_evaluate.size(), i; _i > 0;) {
        i = solutions_to_evaluate[--_i];
        auto k = solution_clusters[i];
        std::optional<usize> objective = k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

        auto [changed, _accept_randomly, _improved] =
            should_accept(rng, problem.fitness(), archive, solutions[i], parents[i], objective, true);

        if (changed) {
          parents[i] = solutions[i];
          std::swap(solutions_to_evaluate[_i], solutions_to_evaluate.back());
          solutions_to_evaluate.pop_back();

          std::swap(closest_elites[_i], closest_elites.back());
          closest_elites.pop_back();
        }
      }

      alpha *= 0.5;
    }
    return evaluations;
  };

  RvOptions options;
  std::unique_ptr<LinkageModelBase> linkage_model;

  bool enable_partial_ams;
  usize num_clusters;
  isize num_continuous;
  std::vector<usize> solutions_to_evaluate;
  std::vector<std::vector<usize>> improved_indices;
  std::vector<const Subset*>
      eval_subsets;  // pointers because 1. we want to avoid copies and 2. the view should be nullable
  Subset full;

  // per cluster data
  Array<BType> cluster_active;
  Array<BType> ms_active;
  std::vector<Array<CType>> distribution_multipliers;
  std::vector<Array<u64>> num_oob;
  std::vector<Array<u64>> num_samples;
  std::vector<std::vector<Mat<CType>>> L;
  std::vector<FOS> subsets;
  std::vector<Vec<CType>> mean;
  std::vector<Vec<CType>> mean_shift;
  std::vector<Mat<CType>> cov;
  std::vector<usize> any_improved;
  std::vector<usize> no_improvement_stretch;  // per cluster
  std::vector<usize> solution_nis;            // per solution
  std::vector<bool> solution_improved;
  std::vector<usize> no_improvement_counts;  // per solution (count to not penalize inactive solutions)
  usize generation = 0;

  // std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus>
  // run(InstanceBase &problem, const Budget &budget, std::optional<usize> seed,
  //     std::optional<usize> population_size){
  //         // ims go brr
  //     }

  std::optional<usize> current_generation() const  // override final
  {
    return generation;
  };
};

};  // namespace goblin

#endif /* _GOBLIN_METHODS_CONTINUOUS_H */
