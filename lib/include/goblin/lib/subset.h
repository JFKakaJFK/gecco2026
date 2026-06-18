#pragma once
#ifndef _GOBLIN_LIB_SUBSET_H
#define _GOBLIN_LIB_SUBSET_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>

#include <Eigen/Dense>

#include "goblin/lib/assert.h"
#include "goblin/lib/fitness.h"
#include "goblin/lib/rng.h"
#include "goblin/lib/types.h"

namespace goblin {

struct Subset {
    std::vector<usize> discrete;
    std::vector<usize> continuous;

    inline void clear() {
        discrete.clear();
        continuous.clear();
    };

    inline usize size() const noexcept { return discrete.size() + continuous.size(); }

    inline bool empty() const noexcept { return discrete.empty() && continuous.empty(); }

    CType similarity(const Subset& other) const {
        usize matches = 0;

        for (usize i : discrete) {
            for (usize j : other.discrete) {
                if (i == j) {
                    matches++;
                }
            }
        }
        for (usize i : continuous) {
            for (usize j : other.continuous) {
                if (i == j) {
                    matches++;
                }
            }
        }

        /* // should be correct, but for now let's keep it simple
        usize i = 0, j = 0;
        // subset indices are sorted, so we only need to check until the next index
        // in the other set is higher
        while (i < discrete.size() && j < other.discrete.size()) {
          if (discrete[i] < other.discrete[j]) {
            i++;
          } else if (discrete[i] > other.discrete[j]) {
            j++;
          } else {
            i++;
            j++;
            matches++;
          }
        }
        i = 0;
        j = 0;
        while (i < continuous.size() && j < other.continuous.size()) {
          if (continuous[i] < other.continuous[j]) {
            i++;
          } else if (continuous[i] > other.continuous[j]) {
            j++;
          } else {
            i++;
            j++;
            matches++;
          }
        } */
        return static_cast<CType>(matches) / static_cast<CType>(std::max(size(), other.size()));
    };

    Subset merge(const Subset& other) const {
        assert(std::is_sorted(discrete.begin(), discrete.end()) &&
               std::is_sorted(other.discrete.begin(), other.discrete.end()) && "Discrete indices are not sorted!");
        assert(std::is_sorted(continuous.begin(), continuous.end()) &&
               std::is_sorted(other.continuous.begin(), other.continuous.end()) &&
               "Continuous indices are not sorted!");

        Subset s;
        usize this_i = 0, other_i = 0, idx;
        while (this_i < discrete.size() || other_i < other.discrete.size()) {
            if (this_i >= discrete.size()) {
                idx = other.discrete[other_i++];
            } else if (other_i >= other.discrete.size()) {
                idx = discrete[this_i++];
            } else if (discrete[this_i] <= other.discrete[other_i]) {
                idx = discrete[this_i++];
            } else {
                idx = other.discrete[other_i++];
            }

            if (s.discrete.empty() || s.discrete.back() != idx) {
                s.discrete.push_back(idx);
            }
        }
        this_i = 0;
        other_i = 0;
        while (this_i < continuous.size() || other_i < other.continuous.size()) {
            if (this_i >= continuous.size()) {
                idx = other.continuous[other_i++];
            } else if (other_i >= other.continuous.size()) {
                idx = continuous[this_i++];
            } else if (continuous[this_i] <= other.continuous[other_i]) {
                idx = continuous[this_i++];
            } else {
                idx = other.continuous[other_i++];
            }

            if (s.continuous.empty() || s.continuous.back() != idx) {
                s.continuous.push_back(idx);
            }
        }
        return s;
    };
};

inline bool operator==(const Subset& lhs, const Subset& rhs) {
    return lhs.discrete == rhs.discrete && lhs.continuous == rhs.continuous;
}
inline bool operator!=(const Subset& lhs, const Subset& rhs) {
    return !(lhs == rhs);
}

using FOS = std::vector<Subset>;

}  // namespace goblin

#endif /* _GOBLIN_LIB_SUBSET_H */
