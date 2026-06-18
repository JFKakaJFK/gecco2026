#include <iostream>

#include <Eigen/Dense>
#include "doctest/doctest.h"

#include "goblin/gp/operator.h"

using namespace goblin;

#define check_gradient(x) _check_gradient(x, #x)

void _check_gradient(OperatorBase& op, std::string_view opname) {
    if (!op.has_gradient()) {
        return;
    }

    const usize nrows = 25;
    const CType e = 1e-6;
    const CType ee = e + e;

    for (usize a = op.min_arity(); a < std::min(op.max_arity(), static_cast<usize>(5)); a++) {
        Array<CType> d_out(nrows), d_expected(nrows), out(nrows);
        Arr2D<CType> args = Arr2D<CType>::Random(nrows, a);
        Arr2D<CType> d_args = Arr2D<CType>::Zero(nrows, a);
        for (usize i = 0; i < a; i++) {
            d_args.col(i) = 1.0;

            // analytical gradient
            op.apply_grad(out, d_out, args, d_args);

            // vs numerical finite diffferences
            args.col(i) -= e;
            auto f_l = op(args);
            args.col(i) += ee;
            auto f_r = op(args);
            args.col(i) -= e;

            auto d_expected = (f_r - f_l) / ee;

            std::cout << "expected: " << d_expected << "\n"
                      << "actual:   " << d_out << std::endl;

            for (usize r = 0; r < nrows; r++) {
                REQUIRE_MESSAGE(d_out(r) == doctest::Approx(d_expected(r)), opname);
            }

            d_args.col(i) = 0.0;
        };
    }
};

TEST_CASE("goblin::gp::operators") {
    OpAdd add;
    check_gradient(add);

    OpSub sub;
    check_gradient(sub);

    OpMul mul;
    check_gradient(mul);

    OpDiv div;
    check_gradient(div);

    OpSin sin;
    check_gradient(sin);

    OpCos cos;
    check_gradient(cos);

    OpExp exp;
    check_gradient(exp);

    OpLog log;
    check_gradient(log);

    OpSquare square;
    check_gradient(square);

    OpSqrt sqrt;
    check_gradient(sqrt);

    OpPow pow;
    check_gradient(pow);

    OpAbs abs;
    check_gradient(abs);

    OpMin min;
    check_gradient(min);

    OpMax max;
    check_gradient(max);

    const usize nrows = 25;
    for (usize arity = 1; arity <= 5; arity++) {
        Arr2D<CType> args = Arr2D<CType>::Random(nrows, arity);

        // add
        if (add.min_arity() <= arity && arity <= add.max_arity()) {
            auto res = add(args);
            auto expected = args.col(0);
            for (isize i = 1; i < args.cols(); i++) {
                expected += args.col(i);
            }

            for (usize r = 0; r < nrows; r++) {
                REQUIRE(res(r) == doctest::Approx(expected(r)));
            }
        }

        // sub
        if (sub.min_arity() <= arity && arity <= sub.max_arity()) {
            auto res = sub(args);
            auto expected = arity > 1 ? args.col(0).eval() : -args.col(0).eval();
            for (isize i = 1; i < args.cols(); i++) {
                expected -= args.col(i);
            }

            for (usize r = 0; r < nrows; r++) {
                REQUIRE(res(r) == doctest::Approx(expected(r)));
            }
        }

        // mul
        if (mul.min_arity() <= arity && arity <= mul.max_arity()) {
            auto res = mul(args);
            auto expected = args.col(0);
            for (isize i = 1; i < args.cols(); i++) {
                expected *= args.col(i);
            }

            for (usize r = 0; r < nrows; r++) {
                REQUIRE(res(r) == doctest::Approx(expected(r)));
            }
        }

        // div
        if (div.min_arity() <= arity && arity <= div.max_arity()) {
            auto res = div(args);
            auto expected = args.col(0);
            for (isize i = 1; i < args.cols(); i++) {
                expected /= args.col(i);
            }

            for (usize r = 0; r < nrows; r++) {
                REQUIRE(res(r) == doctest::Approx(expected(r)));
            }
        }

        // sin
        if (sin.min_arity() <= arity && arity <= sin.max_arity()) {
            auto res = sin(args);
            auto expected = args.col(0).sin();

            for (usize r = 0; r < nrows; r++) {
                REQUIRE(res(r) == doctest::Approx(expected(r)));
            }
        }

        // cos
        if (cos.min_arity() <= arity && arity <= cos.max_arity()) {
            auto res = cos(args);
            auto expected = args.col(0).cos();

            for (usize r = 0; r < nrows; r++) {
                REQUIRE(res(r) == doctest::Approx(expected(r)));
            }
        }

        // exp
        if (exp.min_arity() <= arity && arity <= exp.max_arity()) {
            auto res = exp(args);
            auto expected = args.col(0).exp();

            for (usize r = 0; r < nrows; r++) {
                REQUIRE(res(r) == doctest::Approx(expected(r)));
            }
        }

        // log
        if (log.min_arity() <= arity && arity <= log.max_arity()) {
            auto res = log(args.abs());
            auto expected = args.abs().col(0).log();

            for (usize r = 0; r < nrows; r++) {
                REQUIRE(res(r) == doctest::Approx(expected(r)));
            }
        }

        // NOTE the following tests are a bit useless since they use the same
        // definition as the operators internally so nothing is verified...

        // square
        if (square.min_arity() <= arity && arity <= square.max_arity()) {
            auto res = square(args);
            auto expected = args.col(0).square();

            for (usize r = 0; r < nrows; r++) {
                REQUIRE(res(r) == doctest::Approx(expected(r)));
            }
        }

        // sqrt
        if (sqrt.min_arity() <= arity && arity <= sqrt.max_arity()) {
            auto res = sqrt(args.abs());
            auto expected = args.abs().col(0).sqrt();

            for (usize r = 0; r < nrows; r++) {
                REQUIRE(res(r) == doctest::Approx(expected(r)));
            }
        }

        // pow
        if (pow.min_arity() <= arity && arity <= pow.max_arity()) {
            auto res = pow(args);
            auto expected = args.col(0).pow(args.col(1));

            for (usize r = 0; r < nrows; r++) {
                if (isna(res(r))) {
                    REQUIRE(isna(expected(r)));
                } else {
                    REQUIRE(res(r) == doctest::Approx(expected(r)));
                }
            }
        }

        // abs
        if (abs.min_arity() <= arity && arity <= abs.max_arity()) {
            auto res = abs(args);
            auto expected = args.col(0).abs();

            for (usize r = 0; r < nrows; r++) {
                REQUIRE(res(r) == doctest::Approx(expected(r)));
            }
        }

        // min
        if (min.min_arity() <= arity && arity <= min.max_arity()) {
            auto res = min(args);
            auto expected = args.rowwise().minCoeff();

            for (usize r = 0; r < nrows; r++) {
                REQUIRE(res(r) == doctest::Approx(expected(r)));
            }
        }

        // max
        if (max.min_arity() <= arity && arity <= max.max_arity()) {
            auto res = max(args);
            auto expected = args.rowwise().maxCoeff();

            for (usize r = 0; r < nrows; r++) {
                REQUIRE(res(r) == doctest::Approx(expected(r)));
            }
        }
    }
}
