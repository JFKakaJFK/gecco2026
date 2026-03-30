#ifndef _GOBLIN_H
#define _GOBLIN_H

#pragma once

// clang-format off

#include "goblin/lib/assert.h"
#include "goblin/lib/types.h"
#include "goblin/lib/rng.h"
#include "goblin/lib/ordering.h"
#include "goblin/lib/fitness.h"
#include "goblin/lib/subset.h"
#include "goblin/lib/solution.h"
#include "goblin/lib/archive.h"
#include "goblin/lib/budget.h"
#include "goblin/lib/instance.h"
#include "goblin/lib/algorithms/upgma.h"
#include "goblin/lib/algorithms/subset_selection.h"
#include "goblin/lib/algorithms/mo.h"
#include "goblin/lib/linkage_model.h"
#include "goblin/lib/method.h"
#include "goblin/lib/init.h"

#include "goblin/gp/context.h"
#include "goblin/gp/init.h"
#include "goblin/gp/instance.h"
#include "goblin/gp/operator.h"
#include "goblin/gp/sr.h"
#include "goblin/gp/template.h"

#include "goblin/gp/gpu_evaluation/launch_config.h"
#include "goblin/gp/gpu_evaluation/types.h"

#ifdef GOBLIN_HAS_CUDA
#include "goblin/gp/gpu_evaluation/evaluate.h"
#include "goblin/gp/gpu_evaluation/memory.h"
#include "goblin/gp/gpu_evaluation/misc.h"
#endif

inline bool has_gpu_support() {
#ifdef GOBLIN_HAS_CUDA
  return true;
#else
  return false;
#endif
}

#include "goblin/bench/functions.h"
#include "goblin/bench/functions/combinators.h"
#include "goblin/bench/functions/discrete.h"
#include "goblin/bench/functions/continuous.h"
#include "goblin/bench/functions/mixed.h"
#include "goblin/bench/problem.h"
#include "goblin/bench/timer.h"
#include "goblin/bench/tracked.h"

#include "goblin/methods/ims.h"
#include "goblin/methods/amalgam.h"
#include "goblin/methods/library.h"
#include "goblin/methods/mo_binary.h"
#include "goblin/methods/mixed.h"
#include "goblin/methods/classic/common.h"
#include "goblin/methods/classic/de.h"
#include "goblin/methods/classic/es.h"
#include "goblin/methods/classic/pso.h"
#include "goblin/methods/classic/simple_ga.h"
#include "goblin/methods/classic/standard_gp.h"

#include "goblin/examples/voronoi.h"

// clang-format on

#endif /* _GOBLIN_H */
