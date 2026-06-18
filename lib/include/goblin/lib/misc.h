#ifndef _GOBLIN_LIB_MISC_H
#define _GOBLIN_LIB_MISC_H

#pragma once

#include <Eigen/Dense>
#include <format>
#include <vector>

template <>
struct std::formatter<Eigen::VectorXd, char> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

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

template <typename T, typename Alloc>
struct std::formatter<std::vector<T, Alloc>, char> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

    auto format(const std::vector<T, Alloc>& v, std::format_context& ctx) const {
        auto out = ctx.out();
        *out++ = '[';
        for (std::size_t i = 0; i < v.size(); ++i) {
            out = std::format_to(out, "{}", v[i]);
            if (i + 1 < v.size())
                out = std::format_to(out, ", ");
        }
        *out++ = ']';
        return out;
    }
};

#endif /* _GOBLIN_LIB_MISC_H */
