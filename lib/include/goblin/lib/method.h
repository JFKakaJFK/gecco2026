#pragma once
#ifndef _GOBLIN_LIB_METHOD_H
#define _GOBLIN_LIB_METHOD_H

#include <optional>
#include <tuple>

#include "goblin/lib/archive.h"
#include "goblin/lib/budget.h"
#include "goblin/lib/instance.h"

namespace goblin {
class MethodBase {
   public:
    virtual std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(InstanceBase& problem,
                                                                            const Budget& budget,
                                                                            std::optional<u64> seed,
                                                                            std::optional<usize> population_size) = 0;

    // TODO scipy.optimize compatible .minimize function

    /// Evaluations used and time elapsed can be collected without knowing about
    /// method details, but not the generation/iteraction. This method provides a
    /// hook for tracking the progress over generations
    virtual std::optional<u64> current_generation() const { return std::nullopt; };

    /// Size, generations and restarts of the currently active population if available for multi-start schemes
    virtual std::optional<std::tuple<usize, u64, u64>> current_population() const { return std::nullopt; };

    virtual ~MethodBase() {};
};
};  // namespace goblin

#endif /* _GOBLIN_LIB_METHOD_H */
