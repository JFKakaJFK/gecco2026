#include <Eigen/Dense>
#include <iostream>
#include "doctest/doctest.h"

#include "goblin/bench/functions/continuous.h"
#include "goblin/bench/functions/discrete.h"
#include "goblin/bench/problem.h"
#include "goblin/bench/tracked.h"
#include "goblin/methods/library.h"

using namespace goblin;

void print_front(InstanceBase& problem, const ArchiveBase& archive) {
    for (usize i = 0; i < archive.size(); i++) {
        std::cout << problem.format_solution(archive[i]) << " @ " << archive.fitness().format(archive[i].quality())
                  << std::endl;
    }
};

TEST_CASE("goblin::methods::so::gomea") {
    const usize N = 5;
    const CType VTR = static_cast<CType>(N);
    BenchmarkInstance one_max(std::vector<std::shared_ptr<ObjectiveBase>>{std::make_shared<OneMax>(N)});
    one_max.add_target(VTR);
    Budget budget(/* max_evaluations = */ 10000);

    auto _f1 = one_max.archive_fitness().worst(), _f2 = one_max.archive_fitness().worst();
    auto &f1 = static_cast<MOQuality&>(*_f1), f2 = static_cast<MOQuality&>(*_f2);
    f1.objectives(0) = 10.0;
    f1.constraint_value = 0;
    REQUIRE(one_max.fitness().cmp(f1, f2, std::nullopt) == Ordering::Better);  // anything better than worst

    f2.objectives(0) = 1.0;
    f2.constraint_value = 0;
    REQUIRE(one_max.fitness().cmp(f1, f2, std::nullopt) == Ordering::Better);  // is maximisation

    auto dgomea = DiscreteGOMEA();
    auto [front, _] = dgomea.run(one_max, budget);

    REQUIRE(front->empty() == false);
    print_front(one_max, *front);
    REQUIRE(front->so_solution(0).quality_as<MOQuality>().objectives[0] == VTR);

    std::cout << "GI-GOMEA" << std::endl;

    dgomea = DiscreteGOMEA("Univariate", "NMI", /* gene_invariant */ true);
    front = std::get<0>(dgomea.run(one_max, budget));

    REQUIRE(front->empty() == false);
    print_front(one_max, *front);
    REQUIRE(front->so_solution(0).quality_as<MOQuality>().objectives[0] == VTR);

    BenchmarkInstance sphere(std::vector<std::shared_ptr<ObjectiveBase>>{std::make_shared<Sphere>(2)});
    sphere.set_initial_bounds(100.0, 110.0);
    sphere.add_target(1e-8);

    auto rvgomea = RvGOMEA(
        /* linkage_model = */ "Full",
        /* base_population_size = */ 10,
        /* max_num_populations = */ 1);
    front = std::get<0>(rvgomea.run(sphere, budget));

    REQUIRE(front->empty() == false);
    print_front(one_max, *front);
    REQUIRE(front->so_solution(0).quality_as<MOQuality>().objectives[0] <= 1e-8);
}
