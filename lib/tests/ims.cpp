#include <iostream>
#include <functional>
#include <print>

#include "doctest/doctest.h"
#include <Eigen/Dense>

#include "goblin/bench/functions/discrete.h"
#include "goblin/bench/problem.h"
#include "goblin/lib/ims.h"
#include "goblin/lib/misc.h"

using namespace std::chrono_literals;
using namespace goblin;

struct DummyPopulation {
  InstanceBase& problem;
  ArchiveBase& global_archive;
  usize population_size;
  usize num_clusters;
  std::vector<usize>& generation_population_sizes;
  std::optional<std::reference_wrapper<ArchiveBase>> local_archive;
  bool is_converged = false;

  DummyPopulation(InstanceBase& problem,
                  ArchiveBase& archive,
                  usize population_size,
                  usize num_clusters,
                  std::vector<usize>& generation_population_sizes)
      : problem(problem),
        global_archive(archive),
        population_size(population_size),
        num_clusters(num_clusters),
        generation_population_sizes(generation_population_sizes) {
    if (global_archive.empty()) {
      Rng rng = seeded_rng(42);
      AoSSet p;
      problem.add_random(rng, p, 1);
      global_archive.update(p[0], true);
    }
  };

  CType avg_dist_to_global_so_elite() const { return 1.0; }

  void restart() {};

  template <typename T>
  usize perform_generation(Rng& rng, T should_terminate) {
    std::println("{:>5d} | {:>3d}", population_size, num_clusters);
    generation_population_sizes.push_back(population_size);
    return population_size;
  }

  ArchiveBase& archive() const {
    if (local_archive.has_value()) {
      return local_archive.value().get();
    } else {
      return global_archive;
    }
  };

  bool converged() { return is_converged; }
};

TEST_CASE("goblin::lib::entropy") {
  BenchmarkInstance instance(std::make_shared<OneMax>(1));
  Budget budget(std::nullopt, /* max_generations = */ 12);

  std::vector<usize> generation_population_sizes;

  auto create_population = [&](InstanceBase& problem, ArchiveBase& archive, usize population_size,
                               usize num_clusters) -> DummyPopulation {
    std::println("+> {:>5d} | {:>3d}", population_size, num_clusters);
    return DummyPopulation(problem, archive, population_size, num_clusters, generation_population_sizes);
  };

  std::vector<usize> initial_population_sizes{2, 4};
  std::vector<usize> max_num_populations{1, 25};
  std::vector<usize> subgeneration_factors{2, 3};

  std::vector<std::vector<std::vector<std::vector<usize>>>> expected_generation_population_sizes{
      // initial_population_size = 2
      {// max_num_populations = 1
       {
           // subgeneration_factor = 2
           {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
           // subgeneration_factor = 3
           {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
       },
       // max_num_populations = 25
       {// subgeneration_factor = 2
        {2, 2, 4, 2, 2, 4, 8, 2, 2, 4, 2, 2},
        // subgeneration_factor = 3
        {2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4}}},
      // initial_population_size = 4
      {// max_num_populations = 1
       {
           // subgeneration_factor = 2
           {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
           // subgeneration_factor = 3
           {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
       },
       // max_num_populations = 25
       {// subgeneration_factor = 2
        {4, 4, 8, 4, 4, 8, 16, 4, 4, 8, 4, 4},
        // subgeneration_factor = 3
        {4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8}}}};

  for (usize i = 0; i < initial_population_sizes.size(); i++) {
    for (usize j = 0; j < max_num_populations.size(); j++) {
      for (usize k = 0; k < subgeneration_factors.size(); k++) {
        generation_population_sizes.clear();

        auto ims = IMS<DummyPopulation>(
            create_population,
            IMSOptions(initial_population_sizes[i], max_num_populations[j], subgeneration_factors[k]));

        ims.run(instance, budget);

        std::println("I={}, #P={}, SF={}", initial_population_sizes[i], max_num_populations[j],
                     subgeneration_factors[k]);
        std::println("Actual:   {}", generation_population_sizes);
        std::println("Expected: {}", expected_generation_population_sizes[i][j][k]);
        for (usize l = 0; l < expected_generation_population_sizes[i][j][k].size(); l++) {
          REQUIRE(generation_population_sizes[l] == expected_generation_population_sizes[i][j][k][l]);
        }
      }
    }
  }

  // REQUIRE(false);
}
