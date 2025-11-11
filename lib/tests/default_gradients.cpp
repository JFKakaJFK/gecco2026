#include "doctest/doctest.h"
#include <Eigen/Dense>

#include "goblin/lib/instance.h"
#include "goblin/bench/functions/continuous.h"
#include "goblin/bench/problem.h"

using namespace goblin;

TEST_CASE("goblin::lib::instance_default_gradient") {
  CType vtr = 1e-8;
  BenchmarkInstance sphere(std::make_shared<Sphere>(2));
  sphere.register_target({vtr});
  sphere.set_initial_bounds(100.0, 110.0);

  REQUIRE(sphere.num_objectives() == 1);
  REQUIRE(sphere.num_continuous() == 2);

  Rng rng(42, 0);
  AoSSet solutions, parents;
  std::vector<usize> indices{0};

  sphere.add_random(rng, solutions, 1);
  sphere.evaluate(rng, solutions, indices);
  parents = solutions;

  UnboundedArchive archive(sphere.archive_fitness());
  archive.update(solutions[0], false);

  REQUIRE(!sphere.target_reached(archive));  // the starting point is not good

  Quality prev = solutions[0].quality();

  SUBCASE("gradients()") {
    Subset full;
    for (usize i = 0; i < sphere.num_continuous(); i++) {
      full.continuous.push_back(i);
    }
    std::vector<const Subset*> subsets{&full};

    CType lr = 0.1;

    u64 evaluations;
    for (usize i = 0; i < 100; i++) {
      auto grads = sphere.gradients(rng, solutions, parents, subsets, indices, evaluations);

      if (grads.row(0).array().abs().sum() < 1e-10) {
        break;
      }

      solutions[0].continuous_values() -= lr * grads.row(0);

      if ((i + 1) % 10 == 0) {
        sphere.evaluate(rng, solutions, indices);
        REQUIRE_MESSAGE(sphere.fitness().cmp(solutions[0].quality(), prev, std::nullopt) == Ordering::Better, i,
                        ", P:", sphere.fitness().format(prev), ", C:", sphere.fitness().format(solutions[0].quality()),
                        ", V: ", solutions[0].continuous_values(),
                        ", G: ", grads.row(0));  // the solution improves after a few gradient steps
        prev = solutions[0].quality();
      }
    }

    sphere.evaluate(rng, solutions, indices);
  }

  SUBCASE("gradient_steps()") {
    sphere.gradient_steps(rng, solutions, parents, indices, 100);
    REQUIRE(sphere.fitness().cmp(solutions[0].quality(), prev, std::nullopt) ==
            Ordering::Better);  // the solution improves after a few gradient steps
  }

  archive.update(solutions[0], false);
  REQUIRE(sphere.target_reached(archive));  // and at some point the target is reached...
}
