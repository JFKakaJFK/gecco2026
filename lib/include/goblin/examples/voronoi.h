#pragma once
#ifndef _GOBLIN_EXAMPLES_VORONOI_H
#define _GOBLIN_EXAMPLES_VORONOI_H

#include <stdexcept>
#include <cmath>
#include <limits>
#include <format>

#include <print>

#include "goblin/lib/instance.h"
#include "goblin/lib/init.h"
#include "goblin/methods/classic/simple_ga.h"

// struct KDTree {
//     struct Node {
//         usize idx{};
//         std::unique_ptr<Node> left{};
//         std::unique_ptr<Node> right{};
//     };
// };

// as per https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
template<typename C>
void rgb2shifted_oklab(C&& c)
{
    float l = 0.4122214708f * c(0) + 0.5363325363f * c(1) + 0.0514459929f * c(2);
    float m = 0.2119034982f * c(0) + 0.6806995451f * c(1) + 0.1073969566f * c(2);
    float s = 0.0883024619f * c(0) + 0.2817188376f * c(1) + 0.6299787005f * c(2);

    float l_ = std::cbrtf(l);
    float m_ = std::cbrtf(m);
    float s_ = std::cbrtf(s);

    c(0) = 0.2104542553f*l_ + 0.7936177850f*m_ - 0.0040720468f*s_;
    c(1) = 1.9779984951f*l_ - 2.4285922050f*m_ + 0.4505937099f*s_ + 0.5f;
    c(2) = 0.0259040371f*l_ + 0.7827717662f*m_ - 0.8086757660f*s_ + 0.5f;
}

// as per https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
// but assuming values in [0, 1] for a & b instead of [-0.5, 0.5]
template<typename C>
void shifted_oklab2rgb(C&& c)
{
    c(1) -= 0.5f; c(2) -= 0.5f;
    float l_ = c(0) + 0.3963377774f * c(1) + 0.2158037573f * c(2);
    float m_ = c(0) - 0.1055613458f * c(1) - 0.0638541728f * c(2);
    float s_ = c(0) - 0.0894841775f * c(1) - 1.2914855480f * c(2);

    float l = l_*l_*l_;
    float m = m_*m_*m_;
    float s = s_*s_*s_;

    c(0) = +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
	c(1) = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
	c(2) = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
}

namespace goblin {

class VoronoiImageReconstruction : public InstanceBase {
    const usize VARS_PER_CELL = 6;
    const usize ENABLED = 0;
    const usize X_COORD = 1;
    const usize Y_COORD = 2;
    const usize OKLAB_L = 3;
    const usize OKLAB_A = 4;
    const usize OKLAB_B = 5;

    const float MAX_COLOR_VALUE = 255.0f;

    public:

    VoronoiImageReconstruction(
        const Mat<DType>& target_image,
        usize width,
        usize height,
        usize min_num_cells = 10,
        usize max_num_cells = 100,
        std::optional<AnyInit> init = std::nullopt,
        bool complexity_objective = false,
        bool use_oklab = false
    ):
        _fitness(
            /* num_objectives = */ complexity_objective ? 2 : 1,
            /* minimize = */true
        ),
        target_image(target_image.cast<float>()),
        init(from_any_init(init.value_or(std::make_shared<CompleteInit>()))),
        width(width),
        height(height),
        min_num_cells(min_num_cells),
        max_num_cells(max_num_cells),
        complexity_objective(complexity_objective),
        use_oklab(use_oklab)
    {
        const usize num_pixels = target_image.rows();
        if(num_pixels != width * height){
            throw std::runtime_error(std::format("Image data ({}pixels) does not match withd and height ({} * {} = {})", num_pixels, width, height, width * height));
        }

        if(min_num_cells > max_num_cells){
            std::swap(min_num_cells, max_num_cells);
            std::swap(this->min_num_cells, this->max_num_cells);
        }

        if(min_num_cells < 1){
            throw std::runtime_error("At least one cell is required!");
        }

        if(max_num_cells >= num_pixels){
            throw std::runtime_error("More voronoi cells than pixels in the image!");
        }

        for(usize i = 0; i < num_pixels; i++){
            this->target_image.row(i) /= MAX_COLOR_VALUE;
            if(use_oklab){
                // convert from rgb to oklab
                rgb2shifted_oklab(this->target_image.row(i));
            }
        }

        // set up domain for each variable as [0, num_values)
        _discrete_domain_sizes.resize(max_num_cells * VARS_PER_CELL);
        for(usize i = 0; i < max_num_cells; i++){
            usize j = i * VARS_PER_CELL;

            // (enabled, X, Y, R, G, B) or (enabled, X, Y, L, a, b)
            _discrete_domain_sizes[j + ENABLED] = i < min_num_cells ? 1 : 2;
            _discrete_domain_sizes[j + X_COORD] = width;
            _discrete_domain_sizes[j + Y_COORD] = height;
            DType num_color_values = std::floor(MAX_COLOR_VALUE + 1.1);
            _discrete_domain_sizes[j + OKLAB_L] = num_color_values;
            _discrete_domain_sizes[j + OKLAB_A] = num_color_values;
            _discrete_domain_sizes[j + OKLAB_B] = num_color_values;
        }
    };

    CRef<Vec<DType>> discrete_domain_sizes() const override final {
        return _discrete_domain_sizes;
    };

    CRef<Vec<CType>> continuous_lower_bounds() const override final {
        return _continuous_lower_bounds;
    };
    CRef<Vec<CType>> continuous_upper_bounds() const override final {
        return _continuous_upper_bounds;
    };

    CRef<Vec<CType>> continuous_init_lower_bounds() const override final {
        return _continuous_init_lower_bounds;
    };
    CRef<Vec<CType>> continuous_init_upper_bounds() const override final {
        return _continuous_init_upper_bounds;
    };

    std::tuple<Mat<u8>, usize, usize> image_data(const SolutionBase& solution, float scale = 1.0) const {
        if(scale <= 0.0){
            throw std::runtime_error("The image scale must be > 0!");
        }

        const usize w = scale * width;
        const usize h = scale * height;

        Mat<float> image(w * h, 3);

        // extract and scale centers
        usize num_cells = 0;
        Mat<float> centers(max_num_cells, VARS_PER_CELL);
        for(usize j = 0; j < max_num_cells; j++){
            usize k = j * VARS_PER_CELL;
            if(j < min_num_cells || solution.discrete_values()(k + ENABLED)){
                centers.row(num_cells) = solution.discrete_values()(Eigen::seqN(k, VARS_PER_CELL)).cast<float>();

                if(use_oklab){
                    centers(num_cells, Eigen::seqN(OKLAB_L, 3)) /= MAX_COLOR_VALUE;
                    shifted_oklab2rgb(centers(num_cells, Eigen::seqN(OKLAB_L, 3)));
                    centers(num_cells, Eigen::seqN(OKLAB_L, 3)) *= MAX_COLOR_VALUE;
                }
                num_cells++;
            }
        }
        centers(Eigen::seqN(0, num_cells), X_COORD) *= scale;
        centers(Eigen::seqN(0, num_cells), Y_COORD) *= scale;

        // TODO use KDTree
        auto closest = [&](float x, float y){
            float dist = std::numeric_limits<float>::infinity();
            usize closest_idx = 0;
            for(usize k = 0; k < num_cells; k++){
                float d = std::pow(x - centers(k, X_COORD), 2) + std::pow(y - centers(k, Y_COORD), 2);
                if(d < dist){
                    dist = d;
                    closest_idx = k;
                }
            }
            return closest_idx;
        };

        for(usize x = 0; x < w; x++){
            for(usize y = 0; y < h; y++){
                usize i = y * w + x;

                usize cell_idx = closest(x, y);

                image.row(i) = centers(cell_idx, Eigen::seqN(OKLAB_L, 3));
            }
        }

        return std::make_tuple(image.cast<u8>(), w, h);
    };

    void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
        Mat<float> centers(max_num_cells, VARS_PER_CELL);

        for(usize i: indices){
            auto& s = solutions[i];

            // extract centers, mark inactive cells as inactive
            usize num_cells = 0;
            for(usize j = 0; j < max_num_cells; j++){
                usize k = j * VARS_PER_CELL;
                if(j < min_num_cells || s.discrete_values()(k + ENABLED)){
                    s.discrete_active()(Eigen::seqN(k, VARS_PER_CELL)) = true;
                    if(j < min_num_cells){
                        s.discrete_active()(k + ENABLED) = false;
                    }

                    centers.row(num_cells) = s.discrete_values()(Eigen::seqN(k, VARS_PER_CELL)).cast<float>();
                    centers(num_cells, Eigen::seqN(OKLAB_L, 3)) /= MAX_COLOR_VALUE;
                    num_cells++;
                } else {
                    s.discrete_active()(Eigen::seqN(k, VARS_PER_CELL)) = false;
                }
            }

            // TODO use KDTree
            auto closest = [&](float x, float y){
                float dist = std::numeric_limits<float>::infinity();
                usize closest_idx = 0;
                for(usize k = 0; k < num_cells; k++){
                    float d = std::pow(x - centers(k, X_COORD), 2) + std::pow(y - centers(k, Y_COORD), 2);
                    if(d < dist){
                        dist = d;
                        closest_idx = k;
                    }
                }
                return closest_idx;
            };

            // compute per-pixel mismatch
            float reconstruction_error = 0.0;
            for(usize x = 0; x < width; x++){
                for(usize y = 0; y < height; y++){
                    usize j = y * width + x;

                    usize cell_idx = closest(x, y);

                    reconstruction_error += (target_image.row(j) - centers(cell_idx, Eigen::seqN(OKLAB_L, 3))).array().square().sum();
                }
            }
            reconstruction_error /= static_cast<float>(width * height);

            s.quality_as<MOQuality>().objectives(0) = reconstruction_error;
            if(complexity_objective){
                s.quality_as<MOQuality>().objectives(1) = static_cast<float>(num_cells);
            }
            s.quality_as<MOQuality>().constraint_value = 0.0;
        }
    }

    void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override final {
        init->add_random(rng, *this, solutions, count);
    }

    const FitnessBase& fitness() const override final { return _fitness;};
    const ArchiveFitnessBase& archive_fitness() const override final { return _fitness;};

    void log_solution(std::ostream& os, const SolutionBase& solution) const override final {
        Vec<float> c(3);
        os << '{';
        bool first = true;
        for(usize i = 0; i < max_num_cells; i++){
            usize j = i * VARS_PER_CELL;
            if(j < min_num_cells || solution.discrete_values()(j + ENABLED)){
                if(!first){
                   os << ", ";
                }

                c = solution.discrete_values()(Eigen::seqN(j + OKLAB_L, 3)).cast<float>();

                if(use_oklab){
                    c /= MAX_COLOR_VALUE;
                    shifted_oklab2rgb(c);
                    c *= MAX_COLOR_VALUE;
                }

                // (x, y): (r, g, b)
                os << '('
                << usize(solution.discrete_values()(j + X_COORD)) << ", "
                << usize(solution.discrete_values()(j + Y_COORD)) << "): ("
                << c(0) << ", " << c(1) << ", " << c(2)
                << ")";

                first = false;
            }
        }
        os << '}';
    }

    private:
    MOFitness _fitness;
    Mat<float> target_image;
    std::shared_ptr<InitBase> init;
    usize width;
    usize height;
    usize min_num_cells;
    usize max_num_cells;
    bool complexity_objective;
    bool use_oklab;

    Vec<DType> _discrete_domain_sizes{};
    Vec<CType> _continuous_lower_bounds{};
    Vec<CType> _continuous_upper_bounds{};

    Vec<CType> _continuous_init_lower_bounds{};
    Vec<CType> _continuous_init_upper_bounds{};
};
};

#endif /* _GOBLIN_EXAMPLES_VORONOI_H */
