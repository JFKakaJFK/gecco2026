#include "goblin/lib/fitness.h"

namespace goblin {

Ordering MOFitness::cmp(const QualityBase& lhs, const QualityBase& rhs, std::optional<usize> objective) const {
    const auto& ql = lhs.as<MOQuality>();
    const auto& qr = rhs.as<MOQuality>();
    // Constraints are always minimized
    Ordering o = cmp(ql.constraint_value, qr.constraint_value, _epsilon, true);

    if (o == Ordering::Equal || o == Ordering::NonDominated) {
        if (objective.has_value()) {
            o = cmp(ql.objectives(objective.value()), qr.objectives(objective.value()), _epsilon, _minimize);
        } else {
            for (usize i = 0; i < _num_objectives && o != Ordering::NonDominated; i++) {
                o = o | cmp(ql.objectives(i), qr.objectives(i), _epsilon, _minimize);
            }
        }
    }
    return o;
};

CType MOFitness::distance(const QualityBase& lhs, const QualityBase& rhs, std::optional<usize> objective) const {
    const auto& ql = lhs.as<MOQuality>();
    const auto& qr = rhs.as<MOQuality>();
    CType dist;
    if (objective.has_value()) {
        dist = distance(ql.objectives(objective.value()), qr.objectives(objective.value()));
    } else {
        dist = (ql.objectives - qr.objectives).norm();
    }
    return isna(dist) ? std::numeric_limits<CType>::infinity() : dist;
};
};  // namespace goblin
