#pragma once
#ifndef _GOBLIN_BENCH_TRACKED_H
#define _GOBLIN_BENCH_TRACKED_H

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <ostream>
#include <print>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>
#include <span>

#include "goblin/bench/timer.h"
#include "goblin/lib/assert.h"
#include "goblin/lib/instance.h"
#include "goblin/lib/method.h"

namespace goblin {

template <typename T>
inline void log_helper(std::ostream& os, const std::vector<T>& span, bool escape = true, bool _indent = false) {
  if (escape) {
    os << '"';
  }
  os << '[';
  usize i = 0;
  for (const auto& e : span) {
    if (i++ > 0) {
      os << ',';
    }
    if constexpr (std::same_as<T, char> || std::same_as<T, u8>) {
#ifdef __cpp_lib_print
      std::print(os,
#else
      os << std::format(
#endif
                 "{:d}", e);
    } else {
      os << e;
    }
  }
  os << ']';
  if (escape) {
    os << '"';
  }
};

template <typename EigenLike>
inline void log_helper(std::ostream& os, const EigenLike& m, bool escape = true, bool indent = false) {
  if (escape) {
    os << '"';
  }
  os << '[';
  if (m.rows() > 1 && m.cols() > 1) {
    for (isize r = 0; r < m.rows(); r++) {
      if (r > 0) {
        os << ',';
      }
      if (indent) {
        os << "\n  ";
      }
      os << '[';
      for (isize c = 0; c < m.cols(); c++) {
        if (c > 0) {
          os << ',';
        }

        // fmt to alwyas use the decimal instead of the ascii byte value for
        // (unsigned) chars
        if constexpr (std::same_as<typename EigenLike::Scalar, char> || std::same_as<typename EigenLike::Scalar, u8>) {
#ifdef __cpp_lib_print
          std::print(os,
#else
          os << std::format(
#endif
                     "{:d}", m(r, c));
        } else {
          os << m(r, c);
        }
      }
      os << ']';
    }
    if (indent) {
      os << '\n';
    }
  } else {
    for (isize i = 0; i < m.size(); i++) {
      if (i > 0) {
        os << ',';
      }
      // fmt to alwyas use the decimal instead of the ascii byte value for
      // (unsigned) chars
      if constexpr (std::same_as<typename EigenLike::Scalar, char> || std::same_as<typename EigenLike::Scalar, u8>) {
#ifdef __cpp_lib_print
        std::print(os,
#else
        os << std::format(
#endif
                   "{:d}", m(i));
      } else {
        os << m(i);
      }
    }
  }
  os << ']';
  if (escape) {
    os << '"';
  }
};

template <typename T>
inline std::string log_helper(const T& t, bool escape = true, bool indent = false) {
  std::ostringstream os;
  log_helper(os, t, escape, indent);
  return os.str();
};

class TrackingOptions {
 public:
  TrackingOptions() = delete;
  // TODO at some point think about enabling dynamically setting the logging
  // precision for floating points
  // TODO at some point allow these params on Tracked::run to reduce the amount
  // of config object nesting?
  TrackingOptions(std::filesystem::path logpath,
                  std::optional<std::vector<std::tuple<std::string, std::string>>> log_info = std::nullopt,
                  usize archive_capacity = 100,
                  u64 max_evaluations_until_archive_adaption = 100000,
                  bool consider_evaluation_time = true,
                  bool report_intermediate_results = true,
                  /// Report every time the elitist archive gets updated (for when algorithm behaviour is more
                  /// interesting than the results)
                  bool report_on_archive_change = false,
                  /// If enabled, the raw discrete/continuous values and active information is logged
                  bool report_raw_solutions = false,
                  u64 initial_evaluations_until_next_report = 10,
                  u64 eval_factor = 2,
                  u64 max_evaluations_until_next_report = 1000000,
                  u64 initial_generations_until_next_report = 1,
                  u64 generation_factor = 2,
                  u64 max_generations_until_next_report = 100,
                  std::chrono::nanoseconds initial_time_until_next_report = std::chrono::seconds(1),
                  u64 time_factor = 2,
                  std::chrono::nanoseconds max_time_until_next_report = std::chrono::minutes(10))
      : archive_capacity(archive_capacity),
        max_evaluations_until_archive_adaption(max_evaluations_until_archive_adaption),
        consider_evaluation_time(consider_evaluation_time),
        report_intermediate_results(report_intermediate_results),
        report_on_archive_change(report_on_archive_change),
        report_raw_solutions(report_raw_solutions),
        initial_evaluations_until_next_report(initial_evaluations_until_next_report),
        eval_factor(eval_factor),
        max_evaluations_until_next_report(max_evaluations_until_next_report),
        initial_generations_until_next_report(initial_generations_until_next_report),
        generation_factor(generation_factor),
        max_generations_until_next_report(max_generations_until_next_report),
        initial_time_until_next_report(initial_time_until_next_report),
        time_factor(time_factor),
        max_time_until_next_report(max_time_until_next_report),
        logpath(logpath) {
    if (log_info.has_value()) {
      for (auto& kv : log_info.value()) {
        // TODO escape any '"' here?
        log_info_headers += std::get<0>(kv) + ',';
        log_info_values += std::get<1>(kv) + ',';
      }
    }
  };

  usize archive_capacity;
  u64 max_evaluations_until_archive_adaption;
  bool consider_evaluation_time;
  bool report_intermediate_results;
  bool report_on_archive_change;
  bool report_raw_solutions;

  u64 initial_evaluations_until_next_report;
  u64 eval_factor;  // 1 is linear, >= 2 is exponential spacing
  u64 max_evaluations_until_next_report;

  u64 initial_generations_until_next_report;
  u64 generation_factor;  // 1 is linear, >= 2 is exponential spacing
  u64 max_generations_until_next_report;

  std::chrono::nanoseconds initial_time_until_next_report;
  u64 time_factor;  // 1 is linear, >= 2 is exponential spacing
  std::chrono::nanoseconds max_time_until_next_report;

 private:
  std::filesystem::path logpath;

  // key-value pairs to log, e.g.
  // [(method_name,AMaLGaM),(problem_name,Sphere),(dims,10),(run,99)]
  std::string log_info_headers;
  std::string log_info_values;

  friend class Tracked;
};

/// An instance that intercepts evaluations
class Tracked final : public WrappedInstance {
 public:
  Tracked() = delete;

  void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
    wrap_eval([&](const std::span<const usize>& _indices) { inner.evaluate(rng, solutions, _indices); }, solutions,
              indices);
  };
  void evaluate_partial(Rng& rng,
                        SolutionSetBase& solutions,
                        SolutionSetBase& parents,
                        const std::vector<const Subset*>& subsets,
                        const std::span<const usize>& indices) override final {
    wrap_eval(
        [&](const std::span<const usize>& _indices) {
          inner.evaluate_partial(rng, solutions, parents, subsets, _indices);
        },
        solutions, indices);
  };

  bool adapt(Rng& rng) override final {
    if (inner.adapt(rng)) {
      alg_timer.stop();
      // TODO Should these evaluations count (to both time/evals)? They are tracking only so for now they are not
      // counted, but with the consequence that if you track evaluations more evaluations than allowed by the budget
      // will be performed to arguably more correctly track the algorithm performance...
      inner.reevaluate_and_rebuild_archive(rng, archive);
      alg_timer.start();
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
    u64 evals_before = this->evaluations, _evals = evaluations;
    Mat<CType> res = inner.gradients(rng, solutions, parents, subsets, indices, evaluations);
    this->evaluations = std::max(this->evaluations, evals_before + evaluations - _evals);
    return res;
  }

  std::tuple<std::vector<usize>, u64> gradient_steps(Rng& rng,
                                                     SolutionSetBase& solutions,
                                                     SolutionSetBase& parents,
                                                     const std::span<const usize>& indices,
                                                     usize num_steps) override final {
    u64 evals_before = evaluations;
    auto res = inner.gradient_steps(rng, solutions, parents, indices, num_steps);

    alg_timer.stop();
    evaluations = std::max(evaluations, evals_before + /* evaluations */ std::get<1>(res));

    bool archive_changed = false;
    for (usize i : /* changed_indices */ std::get<0>(res)) {
      archive_changed |= archive.update(solutions[i], true);
    }
    // there is not enough information about the behaviour of gradient_steps for more granular reports
    if (archive_changed && config.report_on_archive_change) {
      report(archive);
    }

    if (inner.target_reached(archive)) {
      status = TerminationStatus::TargetReached;
      throw TrackingException("");
    }

    alg_timer.start();
    return res;
  }

  /// This can be used by the algorithm to log when debugging to log an
  /// `ArchiveBase`/`SolutionSetBase`-like type. Both the passed headers and
  /// values need to be empty, or valid csv columns with a ',' separator at the
  /// end (possibly escaping other ',' occurrences with '"').
  template <typename P>
  void request_debug_report(std::filesystem::path debug_logpath,
                            const P& solutions,
                            std::string_view debug_headers,
                            std::string_view debug_values) {
    // close the actual logfile to open up the debug logpath next
    if (logfile.is_open()) {
      logfile.close();
    }

    // temporarily change the logpath
    auto tracked_generation = generation;
    generation = method.current_generation();
    std::swap(debug_logpath, config.logpath);
    report(solutions, debug_headers, debug_values);
    std::swap(debug_logpath, config.logpath);
    generation = tracked_generation;

    // close the debug logfile to open up the actual logpath again next
    if (logfile.is_open()) {
      logfile.close();
    }
  };

  void request_debug_report(std::filesystem::path debug_logpath,
                            std::string_view debug_headers,
                            std::string_view debug_values) {
    request_debug_report(debug_logpath, archive, debug_headers, debug_values);
  };

  static std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& instance,
      MethodBase& method,
      Budget& budget,
      TrackingOptions config,
      std::optional<usize> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) {
    std::random_device rd;
    usize _seed = seed.has_value() ? seed.value()
                                   : std::uniform_int_distribution<usize>(1, std::numeric_limits<usize>::max())(rd);
    Tracked ti(instance, method, budget, config, _seed);
    try {
      ti.alg_timer.start();
      auto [_, alg_status] = method.run(ti, budget, _seed, population_size);
      ti.alg_timer.stop();
      ti.status = alg_status;

      // TODO why is this necessary? (alg should not be able to hit the target without the exception being thrown...)
      if (instance.target_reached(ti.archive)) {
        ti.status = TerminationStatus::TargetReached;
      }

      // // effectively guess the reason for stopping - assume convergence
      // // unless the budget is exhausted or the target was reached
      // auto elapsed = ti.alg_timer.elapsed() + ti.eval_timer.elapsed();
      // auto ts =
      //     budget.exhausted(ti.generation.value_or(0), ti.evaluations,
      //     elapsed);
      // if (ts.has_value()) {
      //   ti.status = ts.value();
      // } else if (instance.target_reached(ti.archive)) {
      //   ti.status = TerminationStatus::TargetReached;
      // } else {
      //   ti.status = TerminationStatus::Converged;
      // }
    } catch (const TrackingException& e) {
    }

    ti.report(ti.archive);

    return std::make_tuple(std::make_shared<AdaptiveGridArchive>(std::move(ti.archive)), ti.status);
  };

 private:
  struct TrackingException : std::runtime_error {
    using std::runtime_error::runtime_error;
  };

  Tracked(InstanceBase& instance, MethodBase& method, Budget& budget, TrackingOptions config, usize seed)
      : WrappedInstance(instance),
        // instance(instance),
        method(method),
        budget(budget),
        config(config),
        seed(seed),
        status(TerminationStatus::Running),
        archive(inner.archive_fitness(), config.archive_capacity),
        generation(std::nullopt),
        last_generation(0),
        generations_at_next_report(config.initial_generations_until_next_report),
        evaluations(0),
        evaluations_at_next_report(config.initial_evaluations_until_next_report),
        time_elapsed_at_next_report(config.initial_time_until_next_report) {};

  template <typename E>
  void wrap_eval(E eval, SolutionSetBase& solutions, const std::span<const usize>& indices) {
    alg_timer.stop();

    // TODO if evaluations == 0 optionally add all solutions to the archive, not just indices => correct tracking for
    // warm start scenarios where some solutions already have a fitness

    // check if the budget was exhausted while the algorithm was running
    auto elapsed = config.consider_evaluation_time ? alg_timer.elapsed() + eval_timer.elapsed() : alg_timer.elapsed();
    generation = method.current_generation();
    auto ts = budget.exhausted(generation.value_or(0), evaluations, elapsed);
    if (ts.has_value()) {
      status = ts.value();
      throw TrackingException("");
    }

    // actually evaluate, but ensure we don't go beyond the evaluation limit
    u64 evaluations_performed;
    if (budget.max_evaluations.has_value() && evaluations + indices.size() > budget.max_evaluations.value()) {
      auto evals_left = budget.max_evaluations.value() - evaluations;
      eval_timer.start();
      eval(indices.first(evals_left));
      eval_timer.stop();
      evaluations_performed = evals_left;

      assert(budget.exhausted(0, evaluations + evaluations_performed, elapsed).has_value() &&
             "Evaluation limit was not reached!");
    } else {
      eval_timer.start();
      eval(indices);
      eval_timer.stop();
      evaluations_performed = indices.size();
    }

    // update the internal archive, and possibly stop if the target was reached
    for (usize i = 0; i < evaluations_performed; i++) {
      bool archive_changed = archive.update(solutions[indices[i]], true);
      evaluations++;  // update the evaluations one at time to be "truthful" in case of an early return before all
                      // evaluations performed were considered...

      if (archive_changed && config.report_on_archive_change) {
        report(archive);
      }

      // the vtr is checked for each solution to level the playing field between batched algorithms and algorithms
      // evaluating one by one
      if (inner.target_reached(archive)) {
        status = TerminationStatus::TargetReached;
        throw TrackingException("");
      }
    }

    evaluations_since_last_archive_adaption += evaluations_performed;

    // check if we need to stop because the evaluation time/evaluations
    // exhausted the budget
    elapsed = config.consider_evaluation_time ? alg_timer.elapsed() + eval_timer.elapsed() : alg_timer.elapsed();
    ts = budget.exhausted(0, evaluations, elapsed);
    if (ts.has_value()) {
      status = ts.value();
      throw TrackingException("");
    }

    // if we made it here, possibly report (if not, a final report
    // will be generated - no need for doing the final report twice)
    if (should_report()) {
      report(archive);
    }
    if (evaluations_since_last_archive_adaption >= config.max_evaluations_until_archive_adaption) {
      evaluations_since_last_archive_adaption = 0;
      archive.adapt();
    }

    alg_timer.start();
  };

  bool should_report() {
    if (!config.report_intermediate_results)
      return false;

    bool report_needed = false;
    if (evaluations >= evaluations_at_next_report) {
      report_needed = true;
      evaluations_at_next_report =
          std::min(config.eval_factor > 1 ? evaluations * config.eval_factor
                                          : evaluations + config.initial_evaluations_until_next_report,
                   evaluations + config.max_evaluations_until_next_report);
    }

    u64 g = generation.value_or(0);
    if (g >= generations_at_next_report && g != last_generation) {
      last_generation = g;
      report_needed = true;
      generations_at_next_report =
          std::min(config.generation_factor > 1 ? g * config.generation_factor
                                                : g + config.initial_generations_until_next_report,
                   g + config.max_generations_until_next_report);
    }

    auto elapsed = config.consider_evaluation_time ? alg_timer.elapsed() + eval_timer.elapsed() : alg_timer.elapsed();
    if (elapsed > time_elapsed_at_next_report) {
      report_needed = true;
      time_elapsed_at_next_report = std::min(
          config.time_factor > 1
              ? std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed * config.time_factor)
              : std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed + config.initial_time_until_next_report),
          elapsed + config.max_time_until_next_report);
    }

    return report_needed;
  };

  template <typename A>
  void report(const A& solutions, std::string_view debug_headers = "", std::string_view debug_values = "") {
    namespace fs = std::filesystem;
    typedef std::chrono::duration<double> Seconds;

    if (config.logpath == "/dev/null") {
      return;
    }

    if (!logfile.is_open()) {
      if (!config.logpath.parent_path().empty()) {
        fs::create_directories(config.logpath.parent_path());
      }

      // clear the file if it was not cleared before
      if (truncated_files.contains(config.logpath)) {
        logfile.open(config.logpath, std::ios::out | std::ios::app | std::ios::ate);
      } else {
        truncated_files.insert(config.logpath);
        logfile.open(config.logpath, std::ios::out | std::ios::trunc);
      }

      if (fs::is_empty(config.logpath)) {
        // clang-format off
        logfile <<
            "status,"
            "evaluations,"
            "generation,"
            "total_time_seconds,"
            "alg_time_seconds,"
            "eval_time_seconds,"
            "current_population_size,"
            "current_population_generation,"
            "current_population_restarts,"
            << config.log_info_headers
            << debug_headers <<
            "seed,"
            "discrete,"
            "discrete_active,"
            "continuous,"
            "continuous_active,"
        ;
        // clang-format on
        inner.log_header(logfile);
        logfile << std::endl;  // here we want to flush
      }
    }

    std::string gen = generation.has_value() ? std::to_string(generation.value()) : "", pop_size = "", pop_gen = "",
                pop_restarts = "";
    auto pop_info = method.current_population();
    if (pop_info.has_value()) {
      auto [p_size, p_gen, p_restarts] = pop_info.value();
      pop_size = std::to_string(p_size);
      pop_gen = std::to_string(p_gen);
      pop_restarts = std::to_string(p_restarts);
    }
    Seconds alg_time = alg_timer.elapsed();
    Seconds eval_time = eval_timer.elapsed();
    Seconds total_time = alg_time + eval_time;

    auto common = std::format("{},{},{},{},{},{},{},{},{},{}{}{},", format_as(status), evaluations, gen,
                              total_time.count(), alg_time.count(), eval_time.count(), pop_size, pop_gen, pop_restarts,
                              config.log_info_values, debug_values, seed);

    for (usize i = 0; i < solutions.size(); i++) {
      const auto& s = solutions[i];

      logfile << common;
      if (config.report_raw_solutions) {
        // clang-format off
        log_helper(logfile,   s.discrete_values(), true); logfile << ',';
        log_helper(logfile,   s.discrete_active(), true); logfile << ',';
        log_helper(logfile, s.continuous_values(), true); logfile << ',';
        log_helper(logfile, s.continuous_active(), true); logfile << ',';
        // clang-format on
      } else {
        logfile << ",,,,";
      }
      inner.log(logfile, s);
      logfile << "\n";
    }
    logfile << std::flush;
  };

  // InstanceBase& instance;
  MethodBase& method;
  Budget& budget;
  TrackingOptions config;
  usize seed;

  TerminationStatus status;
  AdaptiveGridArchive archive;

  Timer alg_timer;
  Timer eval_timer;

  std::optional<u64> generation;
  u64 last_generation;
  u64 generations_at_next_report;

  u64 evaluations;
  u64 evaluations_at_next_report;
  std::chrono::nanoseconds time_elapsed_at_next_report;

  usize evaluations_since_last_archive_adaption;

  std::ofstream logfile;
  std::set<std::filesystem::path> truncated_files;
};

/// Tracked running was intended to unify reporting across algorithms
/// - this method abuses that functionality to re-use that logging for
/// other purposes controlled by the algorithm, not the tracking
inline void debug_log(InstanceBase& problem,
                      std::string_view path,
                      std::string_view headers = "",
                      std::string_view values = "",
                      std::optional<std::reference_wrapper<const SolutionSetBase>> population = std::nullopt) {
  if (auto ti = dynamic_cast<Tracked*>(&problem); ti != nullptr) {
    if (population.has_value()) {
      ti->request_debug_report(path, population.value().get(), headers, values);
    } else {
      ti->request_debug_report(path, headers, values);
    }
  } else {
    throw std::runtime_error(
        "Debug log called on an incompatible problem instance. Try using "
        "`Tracked::run` to enable logging.");
  }
};

template <typename T>
inline std::string iterator2str(T&& it) {
  std::ostringstream os;
  os << '[';
  usize i = 0;
  for (const auto& e : it) {
    if (i++ > 0) {
      os << ',';
    }
    os << e;
  }
  os << ']';
  return os.str();
};

};  // namespace goblin

#endif /* _GOBLIN_BENCH_TRACKED_H */
