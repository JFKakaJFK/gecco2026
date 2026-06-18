#pragma once
#ifndef _GOBLIN_LIB_BUDGET_H
#define _GOBLIN_LIB_BUDGET_H

#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <string_view>

#include <Eigen/Dense>

#include "goblin/lib/types.h"

namespace goblin {
enum struct TerminationStatus : u8 {
    TimeLimitReached,
    GenerationLimitReached,
    EvaluationLimitReached,
    TargetReached,
    Converged,
    Aborted,
    Running
};

inline constexpr std::string_view format_as(const TerminationStatus& s) noexcept {
    switch (s) {
        case TerminationStatus::TimeLimitReached:
            return std::string_view{"TimeLimitReached"};
        case TerminationStatus::GenerationLimitReached:
            return std::string_view{"GenerationLimitReached"};
        case TerminationStatus::EvaluationLimitReached:
            return std::string_view{"EvaluationLimitReached"};
        case TerminationStatus::TargetReached:
            return std::string_view{"TargetReached"};
        case TerminationStatus::Converged:
            return std::string_view{"Converged"};
        case TerminationStatus::Aborted:
            return std::string_view{"Aborted"};
        default:
            return std::string_view{"Running"};
    };
}

inline std::ostream& operator<<(std::ostream& os, TerminationStatus s) {
    return os << format_as(s);
}

struct Budget {
    std::optional<u64> max_evaluations = std::nullopt;
    std::optional<u64> max_generations = std::nullopt;
    std::optional<std::chrono::nanoseconds> max_time = std::nullopt;
    std::optional<std::function<bool()>> termination_callback = std::nullopt;

    Budget(std::optional<u64> max_evaluations = std::nullopt,
           std::optional<u64> max_generations = std::nullopt,
           std::optional<u64> max_time_seconds = std::nullopt,
           std::optional<std::chrono::nanoseconds> max_duration = std::nullopt,
           std::optional<std::function<bool()>> termination_callback = std::nullopt)
        : max_evaluations(max_evaluations),
          max_generations(max_generations),
          max_time(max_time_seconds.has_value() ? std::make_optional<std::chrono::seconds>(max_time_seconds.value())
                                                : max_duration),
          termination_callback(termination_callback) {};

    std::optional<TerminationStatus> constexpr exhausted(usize generations,
                                                         usize evaluations,
                                                         std::chrono::nanoseconds elapsed) const {
        if (max_generations.has_value() && generations > max_generations.value()) {
            return std::make_optional(TerminationStatus::GenerationLimitReached);
        }
        if (max_evaluations.has_value() && evaluations >= max_evaluations.value()) {
            return std::make_optional(TerminationStatus::EvaluationLimitReached);
        }
        if (max_time.has_value() && elapsed >= max_time.value()) {
            return std::make_optional(TerminationStatus::TimeLimitReached);
        }

        return std::nullopt;
    }

    std::optional<TerminationStatus> constexpr exhausted_or_external_criterion_met(
        usize generations,
        usize evaluations,
        std::chrono::nanoseconds elapsed) const {
        if (termination_callback.has_value() && termination_callback.value()()) {
            return std::make_optional(TerminationStatus::Aborted);
        }

        return exhausted(generations, evaluations, elapsed);
    }
};
};  // namespace goblin

#endif /* _GOBLIN_LIB_BUDGET_H */
