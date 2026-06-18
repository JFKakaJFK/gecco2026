#include <iostream>

#include <Eigen/Dense>
#include "doctest/doctest.h"

#include "goblin/bench/functions/continuous.h"
#include "goblin/bench/problem.h"

using namespace goblin;

TEST_CASE("goblin::bench::problem") {
    auto fn = std::make_shared<Sphere>(2);
    REQUIRE(fn->num_discrete() == 0);
    REQUIRE(fn->num_continuous() == 2);

    auto objectives = std::make_shared<Objectives>(std::vector<std::shared_ptr<ObjectiveBase>>{fn});
    REQUIRE(objectives->num_objectives() == 1);
    REQUIRE(objectives->num_discrete() == 0);
    REQUIRE(objectives->num_continuous() == 2);

    BenchmarkInstance sphere(objectives);

    REQUIRE(sphere.num_objectives() == 1);
    REQUIRE(sphere.num_continuous() == 2);

    Rng rng = seeded_rng(42);

    AoSSet s;
    s.add(Solution(sphere.archive_fitness().worst(), std::nullopt, Vec<double>::Zero(2)));
    std::vector<usize> idxs{0};

    sphere.evaluate(rng, s, idxs);
    CHECK(s[0].quality_as<MOQuality>().objectives == Vec<double>::Zero(1));

    s[0].continuous_values() = Vec<double>::Constant(2, 2.0);
    sphere.evaluate(rng, s, idxs);
    CHECK(s[0].quality_as<MOQuality>().objectives == Vec<double>::Constant(1, 8.0));
}
