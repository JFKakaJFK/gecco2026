#ifndef _GOBLIN_LIB_MISC_H
#define _GOBLIN_LIB_MISC_H

#pragma once

#include <format>
#include <Eigen/Dense>

template<>
struct std::formatter<Eigen::VectorXd, char> {
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const Eigen::VectorXd& v, std::format_context& ctx) const {
        auto out = ctx.out();
        *out++ = '[';
        for (int i = 0; i < v.size(); ++i) {
            out = std::format_to(out, "{}", v[i]);
            if (i + 1 < v.size())
                out = std::format_to(out, ", ");
        }
        *out++ = ']';
        return out;
    }
};

#endif /* _GOBLIN_LIB_MISC_H */
