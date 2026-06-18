#pragma once
#ifndef _GOBLIN_BENCH_FUNCTIONS_MIXED_H
#define _GOBLIN_BENCH_FUNCTIONS_MIXED_H

#include <cassert>
#include <cmath>
#include <numbers>

#include "goblin/bench/functions.h"
#include "goblin/lib/assert.h"

namespace goblin {

class LeadingSpheres final : public ObjectiveBase {
   public:
    LeadingSpheres(usize ndims) : dims(ndims) {};

    usize num_discrete() const override final { return dims; };
    usize num_continuous() const override final { return dims; };

    std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                      RefS<Vec<CType>> continuous_values,
                                      RefS<Array<BType>> discrete_active,
                                      RefS<Array<BType>> continuous_active) override final {
        CType ov = CType(0.0);
        CType cv = CType(0.0);
        bool active = true;
        for (usize i = 0; i < dims; i++) {
            if (active) {
                discrete_active(i) = true;
                if (discrete_values(i) != DType(1)) {
                    active = false;
                }

                continuous_active(i) = true;

                // use a scaled sigmoid to map from [0, oo) to [0, 1)
                auto scaled = 2.0 / (1.0 + std::exp(-continuous_values(i))) - 1.0;
                if (isna(scaled)) {
                    cv = 1.0;
                }
                ov += scaled * scaled;
            } else {
                ov += 1.0;
            }
        }
        return std::make_tuple(ov, cv);
    };

   private:
    usize dims;
};

};  // namespace goblin

#endif /* _GOBLIN_BENCH_FUNCTIONS_MIXED_H */
