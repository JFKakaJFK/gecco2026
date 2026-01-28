#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <iostream>

#include <Eigen/Dense>
#include <doctest/doctest.h>

#include "goblin/lib/fitness.h"
#include "goblin/lib/solution.h"

using namespace goblin;

struct AExt : SolutionExtension<AExt> {
   usize idx{};

   AExt(usize i): idx(i) {};

   bool operator==(const SolutionExtensionBase& other) const {
       return typeid(*this) == typeid(other) && idx == static_cast<const AExt&>(other).idx;
   };

   std::unique_ptr<SolutionExtensionBase> clone() const override {
     return std::make_unique<AExt>(*this);
   };
};

struct BExt : SolutionExtension<BExt> {
    double cv{};

    BExt(double cv): cv(cv) {};

    bool operator==(const SolutionExtensionBase& other) const {
        return typeid(*this) == typeid(other) && cv == static_cast<const BExt&>(other).cv;
    };

    std::unique_ptr<SolutionExtensionBase> clone() const override {
      return std::make_unique<BExt>(*this);
    };
};

struct CExt : SolutionExtension<CExt> {
    bool value{};

    CExt(bool value): value(value) {};

    bool operator==(const SolutionExtensionBase& other) const {
        return typeid(*this) == typeid(other) && value == static_cast<const CExt&>(other).value;
    };

    std::unique_ptr<SolutionExtensionBase> clone() const override {
      return std::make_unique<CExt>(*this);
    };
};

void check_eq(SolutionBase& lhs, SolutionBase& rhs, usize i) {
  REQUIRE_MESSAGE(lhs.num_discrete() == rhs.num_discrete(), "idx: ", i);
  REQUIRE_MESSAGE(lhs.num_continuous() == rhs.num_continuous(), "idx: ", i);

  REQUIRE_MESSAGE(lhs.discrete_values() == rhs.discrete_values(), "idx: ", i);
  REQUIRE_MESSAGE((lhs.discrete_active() == rhs.discrete_active()).all(), "idx: ", i, ", lhs: ", lhs.discrete_active(),
                  ", rhs: ", rhs.discrete_active());

  REQUIRE_MESSAGE(lhs.continuous_values() == rhs.continuous_values(), "idx: ", i);
  REQUIRE_MESSAGE((lhs.continuous_active() == rhs.continuous_active()).all(), "idx: ", i,
                  ", lhs: ", lhs.continuous_active(), ", rhs: ", rhs.continuous_active());

  REQUIRE_MESSAGE(lhs.quality().objectives == rhs.quality().objectives, "idx: ", i);
  REQUIRE_MESSAGE(lhs.quality().constraint_value == rhs.quality().constraint_value, "idx: ", i);

  REQUIRE_MESSAGE(lhs.num_extensions() == rhs.num_extensions(), "idx: ", i);
  for(const auto& _l: lhs.extensions()){
      SolutionExtensionBase& l = _l.get();
      auto _r = rhs.get_extension(l.key());
      REQUIRE_MESSAGE(_r.has_value(), "idx: ", i);
      SolutionExtensionBase& r = _r.value().get();

      // no general comparison required - all point where a specific extension is needed know about the concrete type
      if(typeid(l) == typeid(AExt)){
          REQUIRE_MESSAGE(static_cast<const AExt&>(l) == r, "idx: ", i);
      }
      if(typeid(l) == typeid(BExt)){
          REQUIRE_MESSAGE(static_cast<const BExt&>(l) == r, "idx: ", i);
      }
      if(typeid(l) == typeid(CExt)){
          REQUIRE_MESSAGE(static_cast<const CExt&>(l) == r, "idx: ", i);
      }
  }
}

void check_set(SolutionSetBase& set) {
  auto f = MOFitness(2);
  auto discrete = Vec<DType>::Zero(2);
  auto continuous = Vec<CType>::Zero(3);
  Solution s(f.worst(), discrete, continuous);
  s.quality().objectives = Vec<CType>::Zero(2);

  REQUIRE(s.num_extensions() == 0);
  REQUIRE(!s.has_extension(AExt::type_key()));
  REQUIRE(!s.has_extension(BExt::type_key()));
  REQUIRE(!s.has_extension(CExt::type_key()));

  s.get_or_insert_extension(AExt(1));
  REQUIRE(static_cast<AExt&>(s.get_or_insert_extension(AExt(2))).idx == 1);
  s.get_or_insert_extension(BExt(1.0));
  REQUIRE(static_cast<BExt&>(s.get_or_insert_extension(BExt(2.0))).cv == 1.0);
  s.get_or_insert_extension(CExt(true));
  REQUIRE(static_cast<CExt&>(s.get_or_insert_extension(CExt(false))).value == true);

  REQUIRE(s.num_extensions() == 3);
  REQUIRE(s.has_extension(AExt::type_key()));
  REQUIRE(s.has_extension(BExt::type_key()));
  REQUIRE(s.has_extension(CExt::type_key()));

  static_cast<AExt&>(s.get_extension(AExt::type_key()).value().get()).idx = 2;
  REQUIRE(static_cast<AExt&>(s.get_extension(AExt::type_key()).value().get()).idx == 2);

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

    static_cast<AExt&>(set[set.size() - 1].get_extension(AExt::type_key()).value().get()).idx = i;
    static_cast<BExt&>(set[set.size() - 1].get_extension(BExt::type_key()).value().get()).cv = i;
    static_cast<CExt&>(set[set.size() - 1].get_extension(CExt::type_key()).value().get()).value = i % 2 == 0;
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

TEST_CASE("goblin::lib::solution_extension") {
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
    s.quality().objectives = Vec<CType>::Zero(2);

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
