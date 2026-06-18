#include <doctest/doctest.h>
#include <Eigen/Dense>
#include <iostream>

#include "goblin/lib/fitness.h"

using namespace goblin;

TEST_CASE("goblin::lib::fitness") {
    auto f = MOFitness(3);

    Eigen::Vector3d a(0.0, 1.0, 0.0);
    Eigen::Vector3d b(1.0, 0.0, 0.0);
    Eigen::Vector3d c(0.0, 0.0, 0.1);

    auto _qa = f.worst();
    auto& qa = static_cast<MOQuality&>(*_qa);
    qa.objectives = a.cast<CType>();
    qa.constraint_value = 0.0;
    auto _qb = f.worst();
    auto& qb = static_cast<MOQuality&>(*_qb);
    qb.objectives = b.cast<CType>();
    qb.constraint_value = 0.0;
    auto _qc = f.worst();
    auto& qc = static_cast<MOQuality&>(*_qc);
    qc.objectives = c.cast<CType>();
    qc.constraint_value = 0.0;

    CHECK(f.num_objectives() == 3);

    CHECK(f.distance(qa, qb, 0) == doctest::Approx(1.0));
    CHECK(f.distance(qa, qb, 1) == doctest::Approx(1.0));
    CHECK(f.distance(qa, qb, 2) == doctest::Approx(0.0));

    CHECK(f.distance(qa, qc, 0) == doctest::Approx(0.0));
    CHECK(f.distance(qa, qc, 1) == doctest::Approx(1.0));
    CHECK(f.distance(qa, qc, 2) == doctest::Approx(0.1));

    CHECK(f.distance(qb, qc, 0) == doctest::Approx(1.0));
    CHECK(f.distance(qb, qc, 1) == doctest::Approx(0.0));
    CHECK(f.distance(qb, qc, 2) == doctest::Approx(0.1));

    CHECK(f.distance(qa, qb) == doctest::Approx(std::sqrt(2.0)));
    CHECK(f.distance(qa, qc) == doctest::Approx(std::sqrt(1.01)));
    CHECK(f.distance(qb, qc) == doctest::Approx(std::sqrt(1.01)));

    CHECK(f.cmp(qa, qb, 0) == Ordering::Better);
    CHECK(f.cmp(qa, qb, 1) == Ordering::Worse);
    CHECK(f.cmp(qa, qb, 2) == Ordering::Equal);

    CHECK(f.cmp(qa, qc, 0) == Ordering::Equal);
    CHECK(f.cmp(qa, qc, 1) == Ordering::Worse);
    CHECK(f.cmp(qa, qc, 2) == Ordering::Better);

    CHECK(f.cmp(qb, qc, 0) == Ordering::Worse);
    CHECK(f.cmp(qb, qc, 1) == Ordering::Equal);
    CHECK(f.cmp(qb, qc, 2) == Ordering::Better);

    CHECK(f.cmp(qa, qb, std::nullopt) == Ordering::NonDominated);
    CHECK(f.cmp(qa, qb, std::nullopt) == Ordering::NonDominated);

    CHECK(f.cmp(qa, qc, std::nullopt) == Ordering::NonDominated);
    CHECK(f.cmp(qb, qc, std::nullopt) == Ordering::NonDominated);
}
