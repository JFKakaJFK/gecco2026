#include <iostream>

#include <Eigen/Dense>
#include "doctest/doctest.h"

#include "goblin/lib/archive.h"
#include "goblin/lib/misc.h"

using namespace goblin;

void print_objectives(const ArchiveBase& a) {
    for (usize i = 0; i < a.size(); i++) {
        std::println("[{}]:{}", i, a[i].quality_as<MOQuality>().objectives);
    }
};

std::vector<Solution> generate_front_2d(MOFitness& f, usize size, double intercept = 1.0, double slope = -1.0) {
    std::vector<Solution> v;
    v.reserve(size);
    for (usize i = 0; i < size; i++) {
        v.emplace_back(f.worst());
        double x = intercept * static_cast<double>(i) / static_cast<double>(size - 1);
        double y = intercept + slope * x;
        auto& q = v[i].quality_as<MOQuality>();
        q.constraint_value = 0.0;
        q.objectives(0) = x;
        q.objectives(1) = y;
    }
    return v;
};

TEST_CASE("goblin::lib::archive") {
    MOFitness f(2);
    UnboundedArchive ua(f);
    AdaptiveGridArchive aa(f, 50);

    REQUIRE(aa.covers(ua) == true);
    REQUIRE(ua.covers(aa) == true);

    // can add solutions
    auto front = generate_front_2d(f, 100);
    for (const auto& s : front) {
        REQUIRE(ua.update(s, true) == true);
        // re-adding the same solution only returns true if not strict
        REQUIRE(ua.update(s, false) == true);
        REQUIRE(ua.update(s, true) == false);

        REQUIRE(aa.update(s, true) == true);
        // re-adding the same solution only returns true if not strict
        REQUIRE(aa.update(s, false) == true);
        REQUIRE(aa.update(s, true) == false);
    }

    REQUIRE(ua.size() == 100);
    REQUIRE(aa.size() == 100);

    auto dominated = generate_front_2d(f, 10, 1.1);
    UnboundedArchive _ua(f);
    for (const auto& s : dominated) {
        _ua.update(s, false);

        REQUIRE(ua.dominates(s, true) == true);
        REQUIRE(aa.dominates(s, true) == true);
    }
    REQUIRE(_ua.size() == 10);

    REQUIRE(ua.covers(_ua) == true);
    REQUIRE(aa.covers(_ua) == true);

    ua.adapt();
    aa.adapt();
    REQUIRE(ua.size() == 100);
    REQUIRE(aa.size() < 100);

    // print_objectives(aa);
    // print_objectives(_ua);
    REQUIRE(aa.covers(_ua) == true);

    // https://www.desmos.com/calculator/4urzqsrtzm
    UnboundedArchive _ua1(f);
    auto partially_non_dominated = generate_front_2d(f, 25, 2.5, -4.0);
    for (const auto& s : partially_non_dominated) {
        _ua1.update(s, false);
    }

    // print_objectives(ua);
    // print_objectives(_ua1);

    CHECK(ua.covers(_ua1) == false);
    CHECK(aa.covers(_ua1) == false);
    CHECK(_ua1.covers(ua) == false);
    CHECK(_ua1.covers(aa) == false);
}
