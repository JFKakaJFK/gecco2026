#ifndef _GOBLIN_BENCH_TIMER_H
#define _GOBLIN_BENCH_TIMER_H

#pragma once

#include <chrono>
#include <optional>

namespace goblin {
class Timer {
   public:
    void start() { start_time_ = std::make_optional(std::chrono::high_resolution_clock::now()); }

    void stop() {
        auto now = std::chrono::high_resolution_clock::now();
        if (start_time_.has_value()) {
            total_time_ += std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time_.value());
            start_time_ = std::nullopt;
        }
    }

    std::chrono::nanoseconds elapsed() const { return total_time_; }

   private:
    std::chrono::nanoseconds total_time_{0};
    std::optional<std::chrono::high_resolution_clock::time_point> start_time_ = std::nullopt;
};
};  // namespace goblin

#endif /* _GOBLIN_BENCH_TIMER_H */
