#pragma once
#ifndef _GOBLIN_CLASSIC_PSO_H
#define _GOBLIN_CLASSIC_PSO_H

#include "goblin/methods/classic/common.h"

namespace goblin {
namespace classic {

class PSOTopologyBase {
   public:
    virtual std::vector<usize> neighbours(Rng& rng, const SolutionSetBase& population, usize idx) const = 0;
    virtual ~PSOTopologyBase() = default;
};

class RingTopology : public PSOTopologyBase {
    usize num_neighbours{};

   public:
    /// Considers a population of size N as ring where the ends wrap (nodes 0 and N-1 are neighbours). The number of
    /// neighbours determines the (symmetric) reach of each node around the ring, where `num_neighbours >= floor(N-1) /
    /// 2` corresponds to a fully connected star topology.
    RingTopology(usize num_neighbours = 2) : num_neighbours(num_neighbours) {
        if (num_neighbours < 1) {
            throw std::runtime_error("The neighbourhood must extend to at least 1 neighbour in each direction.");
        }
    };

    std::vector<usize> neighbours(Rng& rng, const SolutionSetBase& population, usize idx) const {
        const usize N = population.size();
        const usize n = std::min(2 * num_neighbours + 1, N);
        std::vector<usize> nbs;
        if (n >= N) {
            nbs.resize(N);
            std::iota(nbs.begin(), nbs.end(), 0);
            return nbs;
        }

        nbs.reserve(n);
        isize i = static_cast<isize>(idx) - num_neighbours;
        while (nbs.size() < n) {
            if (i < 0) {
                i += N;
            } else if (i >= N) {
                i -= N;
            }
            nbs.push_back(i);
            i++;
        }

        return nbs;
    };
};

class PSOState : public SolutionExtension<PSOState> {
    std::unique_ptr<QualityBase> _previous_best_quality{};

   public:
    Vec<CType> velocity{};
    Vec<CType> previous_best{};

    const QualityBase& previous_best_quality() const { return *_previous_best_quality; }
    QualityBase& previous_best_quality() { return *_previous_best_quality; }

    void assign_previous_best_quality(const QualityBase& quality) { _previous_best_quality = quality.clone(); }

    PSOState() = default;
    ~PSOState() = default;
    PSOState(const PSOState& other)
        : _previous_best_quality(other._previous_best_quality->clone()),
          velocity(other.velocity),
          previous_best(other.previous_best) {};
    PSOState(PSOState&& other)
        : _previous_best_quality(std::move(other._previous_best_quality)),
          velocity(std::move(other.velocity)),
          previous_best(std::move(other.previous_best)) {};
    PSOState& operator=(const PSOState& other) {
        if (&other != this) {
            velocity = other.velocity;
            previous_best = other.previous_best;
            _previous_best_quality = other._previous_best_quality->clone();
        }
        return *this;
    }
    PSOState& operator=(PSOState&& other) {
        if (&other != this) {
            velocity = std::move(other.velocity);
            previous_best = std::move(other.previous_best);
            _previous_best_quality = std::move(other._previous_best_quality);
        }
        return *this;
    }

    std::unique_ptr<SolutionExtensionBase> clone() const override final { return std::make_unique<PSOState>(*this); };
};

class PSO : public EABase {
   private:
    // options
    std::shared_ptr<PSOTopologyBase> topology;
    double inertia{};
    double cognitive{};
    double social{};

    // temporary buffers
    mutable std::vector<usize>
        solutions_to_evaluate;  // mutable because the buffer does not contain persistent state, and there is little
                                // point in re-allocating the memory each iteration

   public:
    PSO(usize population_size = 25,
        double inertia = 0.729,
        double cognitive = 1.494,
        double social = 1.494,
        std::shared_ptr<PSOTopologyBase> topology = std::make_shared<RingTopology>())
        : EABase(population_size), topology(topology), inertia(inertia), cognitive(cognitive), social(social) {
        if (population_size < 2) {
            throw std::runtime_error("PSO requires a population size >= 2!");
        }
    };

    u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const override final {
        if (problem.num_continuous() < 1) {
            return 0;
        }
        std::uniform_real_distribution<double> U(0.0, 1.0);
        const usize n = population.size();
        const usize D = problem.num_continuous();
        if (n < 2) {
            throw std::runtime_error("PSO requires a population size >= 2!");
        }

        // housekeeping
        solutions_to_evaluate.clear();
        solutions_to_evaluate.reserve(n);

        // update the previous best value
        for (usize i = 0; i < n; i++) {
            // initialize if not set
            if (!population[i].has_extension(PSOState::type_key())) {
                PSOState ext;
                ext.velocity.resize(D);
                for (usize j = 0; j < D; j++) {
                    ext.velocity(j) = 0.0;
                };
                ext.previous_best = population[i].continuous_values();
                ext.assign_previous_best_quality(population[i].quality());
                population[i].get_or_insert_extension(ext);
            } else {
                auto& ext = population[i].extension<PSOState>();
                if (problem.fitness().cmp(population[i].quality(), ext.previous_best_quality(), std::nullopt) !=
                    Ordering::Worse) {
                    ext.assign_previous_best_quality(population[i].quality());
                    ext.previous_best = population[i].continuous_values();
                }
            }
        }

        // update the positions of each particle
        Vec<CType> r0 = Vec<CType>::Zero(D);
        Vec<CType> r1 = Vec<CType>::Zero(D);
        for (usize i = 0; i < n; i++) {
            auto& ext = population[i].extension<PSOState>();

            // social update
            auto neighbours = topology->neighbours(rng, population, i);
            if (neighbours.size() < 1) {
                throw std::runtime_error("Invalid, empty neighbourhood");
            }
            usize best_nb_idx = 0;
            for (usize j = 1; j < neighbours.size(); j++) {
                if (problem.fitness().cmp(
                        population[neighbours[j]].extension<PSOState>().previous_best_quality(),
                        population[neighbours[best_nb_idx]].extension<PSOState>().previous_best_quality(),
                        std::nullopt) != Ordering::Worse) {
                    best_nb_idx = j;
                }
            }

            // velocity update
            auto x = population[i].continuous_values();
            const auto pb = ext.previous_best;
            const auto gb = population[neighbours[best_nb_idx]].extension<PSOState>().previous_best;

            for (usize j = 0; j < D; j++) {
                r0(j) = U(rng);
                r1(j) = U(rng);
            }

            ext.velocity =
                inertia * ext.velocity + cognitive * r0.cwiseProduct(pb - x) + social * r1.cwiseProduct(gb - x);

            // position update
            x += ext.velocity;

            // clamp to boundaries
            const auto lb = problem.continuous_lower_bounds();
            const auto ub = problem.continuous_upper_bounds();
            x = x.cwiseMax(lb).cwiseMin(ub);

            solutions_to_evaluate.push_back(i);
        }

        // evaluation (no partial evaluations since the velocity updates all variables at once)
        problem.evaluate(rng, population, solutions_to_evaluate);

        for (usize i : solutions_to_evaluate) {
            archive.update(population[i], false);
        }

        return solutions_to_evaluate.size();
    };
};

};  // namespace classic
};  // namespace goblin

#endif /* _GOBLIN_CLASSIC_PSO_H */
