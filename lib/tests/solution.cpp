#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>

#include <doctest/doctest.h>
#include <Eigen/Dense>

#include "goblin/lib/fitness.h"
#include "goblin/lib/solution.h"

using namespace goblin;

void check_eq(SolutionBase& lhs, SolutionBase& rhs, usize i) {
    REQUIRE_MESSAGE(lhs.num_discrete() == rhs.num_discrete(), "idx: ", i);
    REQUIRE_MESSAGE(lhs.num_continuous() == rhs.num_continuous(), "idx: ", i);

    REQUIRE_MESSAGE(lhs.discrete_values() == rhs.discrete_values(), "idx: ", i);
    REQUIRE_MESSAGE((lhs.discrete_active() == rhs.discrete_active()).all(), "idx: ", i,
                    ", lhs: ", lhs.discrete_active(), ", rhs: ", rhs.discrete_active());

    REQUIRE_MESSAGE(lhs.continuous_values() == rhs.continuous_values(), "idx: ", i);
    REQUIRE_MESSAGE((lhs.continuous_active() == rhs.continuous_active()).all(), "idx: ", i,
                    ", lhs: ", lhs.continuous_active(), ", rhs: ", rhs.continuous_active());

    const auto &lq = lhs.quality_as<MOQuality>(), rq = rhs.quality_as<MOQuality>();
    REQUIRE_MESSAGE(lq.objectives == rq.objectives, "idx: ", i);
    REQUIRE_MESSAGE(lq.constraint_value == rq.constraint_value, "idx: ", i);
}

void check_set(SolutionSetBase& set) {
    auto f = MOFitness(2);
    auto discrete = Vec<DType>::Zero(2);
    auto continuous = Vec<CType>::Zero(3);
    Solution s(f.worst(), discrete, continuous);
    s.quality_as<MOQuality>().objectives = Vec<CType>::Zero(2);

    const usize N = 64;  // larger than 32 (initial allocation size)
    for (usize i = 0; i < N; i++) {
        set.add(s);
        CHECK(set.size() == i + 1);
        check_eq(set[set.size() - 1], s, i);
    }
    for (usize i = 0; i < N; i++) {
        set.add(set[i]);
        CHECK(set.size() == N + i + 1);
        check_eq(set[set.size() - 1], s, N + i);
    }

    // Assignment works
    s.discrete_values()(0) = 1;
    set[0] = s;
    check_eq(set[0], s, 2 * N);

    std::vector<usize> indices{0, 1, 2, 3, N};
    // set.remove_at(0);
    set.remove_indices_sorted(indices);
    CHECK(set.size() == 2 * N - indices.size());

    // adding still works after removing...
    for (usize i = 0; i < N; i++) {
        set.add(s);
        check_eq(set[set.size() - 1], s, i);
    }
    for (usize i = 0; i < N; i++) {
        set.add(set[i]);
        check_eq(set[set.size() - 1], set[i], N + i);
    }
};

TEST_CASE("goblin::lib::solution") {
    AoSSet aos_set;
    SoASet<Eigen::ColMajor> soa_set;
    SoASet<Eigen::RowMajor> soar_set;

    SUBCASE("AoSSet") {
        check_set(aos_set);
    };
    SUBCASE("SoASet_colmajor") {
        check_set(soa_set);
    };
    SUBCASE("SoASet_rowmajor") {
        check_set(soar_set);
    };

    SUBCASE("Mixing Sets") {
        auto f = MOFitness(2);
        auto discrete = Vec<DType>::Zero(2);
        auto continuous = Vec<CType>::Zero(3);
        Solution s(f.worst(), discrete, continuous);
        s.quality_as<MOQuality>().objectives = Vec<CType>::Zero(2);

        aos_set.add(s);
        soa_set.add(s);
        soar_set.add(s);

        aos_set[0].discrete_values()(0) = 1;
        soa_set[0].discrete_values()(0) = 2;
        soar_set[0].discrete_values()(0) = 3;

        // can add solutions from other sets
        aos_set.add(soa_set[0]);
        check_eq(aos_set[aos_set.size() - 1], soa_set[0], 1);
        aos_set.add(soar_set[0]);
        check_eq(aos_set[aos_set.size() - 1], soar_set[0], 2);

        soa_set.add(aos_set[0]);
        check_eq(soa_set[soa_set.size() - 1], aos_set[0], 3);
        soa_set.add(soar_set[0]);
        check_eq(soa_set[soa_set.size() - 1], soar_set[0], 4);

        soar_set.add(aos_set[0]);
        check_eq(soar_set[soar_set.size() - 1], aos_set[0], 5);
        soar_set.add(soa_set[0]);
        check_eq(soar_set[soar_set.size() - 1], soa_set[0], 6);

        //  assign solutions from other sets
        aos_set[1] = soar_set[0];
        check_eq(aos_set[1], soar_set[0], 7);
        aos_set[1] = soa_set[0];
        check_eq(aos_set[1], soa_set[0], 8);

        soa_set[2] = aos_set[0];
        check_eq(soa_set[2], aos_set[0], 9);
        soa_set[2] = soar_set[0];
        check_eq(soa_set[2], soar_set[0], 10);

        soar_set[2] = aos_set[0];
        check_eq(soar_set[2], aos_set[0], 11);
        soar_set[2] = soa_set[0];
        check_eq(soar_set[2], soa_set[0], 12);
    }
}
