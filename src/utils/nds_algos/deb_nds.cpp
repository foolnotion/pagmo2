#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>

namespace pagmo {

/// Fast non dominated sorting
/**
 * An implementation of the fast non dominated sorting algorithm. Complexity is \f$ O(MN^2)\f$ where \f$M\f$ is the
 * number of objectives
 * and \f$N\f$ is the number of individuals.
 *
 * See: Deb, Kalyanmoy, et al. "A fast elitist non-dominated sorting genetic algorithm
 * for multi-objective optimization: NSGA-II." Parallel problem solving from nature PPSN VI. Springer Berlin Heidelberg,
 * 2000.
 *
 * @param points An std::vector containing the objectives of different individuals. Example
 * {{1,2,3},{-2,3,7},{-1,-2,-3},{0,0,0}}
 *
 * @return an std::tuple containing:
 *  - the non dominated fronts, an <tt>std::vector<std::vector<pop_size_t>></tt>
 * containing the non dominated fronts. Example {{1,2},{3},{0}}
 *  - the domination list, an <tt>std::vector<std::vector<pop_size_t>></tt>
 * containing the domination list, i.e. the indexes of all individuals
 * dominated by the individual at position \f$i\f$. Example {{},{},{0,3},{0}}
 *  - the domination count, an <tt>std::vector<pop_size_t></tt> containing the number of individuals
 * that dominate the individual at position \f$i\f$. Example {2, 0, 0, 1}
 *  - the non domination rank, an <tt>std::vector<pop_size_t></tt> containing the index of the non
 * dominated front to which the individual at position \f$i\f$ belongs. Example {2,0,0,1}
 *
 * @throws std::invalid_argument If the size of \p points is not at least 2
 */
fnds_return_type fast_non_dominated_sorting(std::span<vector_double const> points)
{
    auto N = points.size();
    // We make sure to have two points at least (one could also be allowed)
    if (N < 2u) {
        pagmo_throw(std::invalid_argument, "At least two points are needed for fast_non_dominated_sorting: "
                                               + std::to_string(N) + " detected.");
    }
    // Initialize the return values
    std::vector<std::vector<pop_size_t>> non_dom_fronts(1u);
    std::vector<std::vector<pop_size_t>> dom_list(N);
    std::vector<pop_size_t> dom_count(N);
    std::vector<pop_size_t> non_dom_rank(N);

    // Start the fast non dominated sort algorithm
    for (decltype(N) i = 0u; i < N; ++i) {
        dom_list[i].clear();
        dom_count[i] = 0u;
        for (decltype(N) j = 0u; j < i; ++j) {
            if (pareto_dominance(points[i], points[j])) {
                dom_list[i].push_back(j);
                ++dom_count[j];
            } else if (pareto_dominance(points[j], points[i])) {
                dom_list[j].push_back(i);
                ++dom_count[i];
            }
        }
    }
    for (decltype(N) i = 0u; i < N; ++i) {
        if (dom_count[i] == 0u) {
            non_dom_rank[i] = 0u;
            non_dom_fronts[0].push_back(i);
        }
    }
    // we copy dom_count as we want to output its value at this point
    auto dom_count_copy(dom_count);
    auto current_front = non_dom_fronts[0];
    std::vector<std::vector<pop_size_t>>::size_type front_counter(0u);
    while (current_front.size() != 0u) {
        std::vector<pop_size_t> next_front;
        for (decltype(current_front.size()) p = 0u; p < current_front.size(); ++p) {
            for (decltype(dom_list[current_front[p]].size()) q = 0u; q < dom_list[current_front[p]].size(); ++q) {
                --dom_count_copy[dom_list[current_front[p]][q]];
                if (dom_count_copy[dom_list[current_front[p]][q]] == 0u) {
                    non_dom_rank[dom_list[current_front[p]][q]] = front_counter + 1u;
                    next_front.push_back(dom_list[current_front[p]][q]);
                }
            }
        }
        ++front_counter;
        current_front = next_front;
        if (current_front.size() != 0u) {
            non_dom_fronts.push_back(current_front);
        }
    }
    return std::make_tuple(std::move(non_dom_fronts), std::move(dom_list), std::move(dom_count),
                           std::move(non_dom_rank));
}

} // namespace pagmo