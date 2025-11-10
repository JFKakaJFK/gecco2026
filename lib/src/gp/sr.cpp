#include "goblin/gp/sr.h"

namespace goblin {
std::tuple<std::vector<usize>, u64> SRProblem::gradient_steps(Rng& rng,
                                                              SolutionSetBase& solutions,
                                                              SolutionSetBase& parents,
                                                              const std::span<const usize>& indices,
                                                              usize num_steps) {
  struct LMFunctor {
    typedef Scalar Scalar;
    typedef Vec<Scalar> InputType;
    typedef Vec<Scalar> ValueType;
    typedef Mat<Scalar> JacobianType;

    enum [[maybe_unused]] { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };

    // TODO add this as option in the constructor
    enum Mode {
      Forward,
      Central
    };  // TODO add autodiff once SRProblem is templated - only available if std::same_as<Scalar,
        // Eigen::AutoDiffScalar<??>>

    Rng& rng;
    SRProblem* p;

    Array<Scalar> params;
    SolutionBase& solution;
    Subset& s;

    u64& evaluations;

    Mode mode = Mode::Forward;

    int inputs() const { return s.continuous.size(); }
    int values() const { return p->Y_train.size(); }

    int operator()(const InputType& x, ValueType& residual) const {
      for (usize i = 0; i < x.size(); i++) {
        solution.continuous_values()(s.continuous[i]) = x(i);
      }

      usize _;
      Arr2D<Scalar> pred = p->ctx.compute_outputs(p->_eval_buffer, solution, p->X_train, params, _);

      for (usize i = 0; i < p->ctx.num_outputs; i++) {
        residual(Eigen::seqN(i * p->Y_train.rows(), p->Y_train.rows())).array() = p->Y_train.col(i) - pred.col(i);
      }

      evaluations += 1;

      return 0;
    }

    int df(const InputType& x, JacobianType& fjac) const {
      const Scalar e = 1e-6;

      ValueType fwd(values()), bwd(values());
      InputType perturbed = x;

      if (mode == Mode::Forward) {
        operator()(perturbed, bwd);
      }
      for (size_t i = 0; i < x.size(); i++) {
        const Scalar d = e + e * std::abs(x(i));
        perturbed(i) += d;
        operator()(perturbed, fwd);
        if (mode == Mode::Central) {
          perturbed(i) -= d + d;
          operator()(perturbed, bwd);
          perturbed(i) += d;
        } else {
          perturbed(i) -= d;
        }

        fjac.col(i) = (fwd - bwd).array() / (d + d);
      }

      return 2 * x.size();  // = number of evaluations done
    }
  };

  u64 evaluations = 0;
  std::vector<usize> changed_indices;
  changed_indices.reserve(indices.size());
  for (usize i : indices) {
    // for linear scaling, we can just ignore the scaling terms here
    // - if we treat the problem as "unscaled" during gradient optimization,
    // then there are fewer parameters to optimize and the fitness evaluation
    // at the end automatically updates the scaling terms again...
    Subset active;
    for (usize j = 0; j < ctx.num_continuous; j++) {
      if (solutions[i].continuous_active()(j)) {
        active.continuous.push_back(j);
      }
    }
    if (active.continuous.front() < ctx.num_continuous) {
      LMFunctor functor{.rng = rng, .p = this, .solution = solutions[i], .s = active, .evaluations = evaluations};

      // Eigen::NumericalDiff<LMFunctor, Eigen::NumericalDiffMode::Central> diff(functor);
      // Eigen::LevenbergMarquardt<decltype(diff), Scalar> lm(diff);
      Eigen::LevenbergMarquardt<LMFunctor, Scalar> lm(functor);
      lm.parameters.maxfev = num_steps * 2 * active.continuous.size();
      // gradient tolerance
      lm.parameters.gtol = 1e-8;
      // function tolerance
      lm.parameters.ftol = 1e-8;
      // parameter tolerance
      lm.parameters.xtol = 1e-8;

      Vec<Scalar> x = solutions[i].continuous_values()(active.continuous).cast<Scalar>();
      // Status is enum containing reason for termination, > 0 is ok
      // https://libeigen.gitlab.io/eigen/docs-nightly/unsupported/LevenbergMarquardt_2LevenbergMarquardt_8h_source.html
      /* Eigen::LevenbergMarquardtSpace::Status status = */ lm.minimize(x);

      solutions[i].continuous_values()(active.continuous) = x;

      // ensure the fitness is up-to-date
      std::vector<usize> idxs{i};
      evaluate(rng, solutions, idxs);
      changed_indices.push_back(i);
    }
  }
  return std::make_tuple(changed_indices, evaluations);
};
}  // namespace goblin
