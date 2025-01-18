#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>

namespace pagmo {

/// Non dominated front 2D (Kung's algorithm)
/**
 * Finds the non dominated front of a set of two dimensional objectives. Complexity is O(N logN) and is thus lower than
 * the
 * complexity of calling pagmo::fast_non_dominated_sorting
 *
 * See: Jensen, Mikkel T. "Reducing the run-time complexity of multiobjective EAs: The NSGA-II and other algorithms."
 * IEEE Transactions on Evolutionary Computation 7.5 (2003): 503-515.
 *
 * @param input_objs an <tt>std::vector</tt> containing the points (i.e. vector of objectives)
 *
 * @return A <tt>std::vector</tt> containing the indexes of the points in the non-dominated front
 *
 * @throws std::invalid_argument If the objective vectors are not all containing two-objectives
 */
std::vector<pop_size_t> non_dominated_front_2d(std::span<vector_double const> input_objs)
{
    // If the input is empty return an empty vector
    if (input_objs.size() == 0u) {
        return {};
    }
    // How many objectives? M, of course.
    auto M = input_objs[0].size();
    // We make sure all input_objs contain M objectives
    if (!std::all_of(input_objs.begin(), input_objs.end(),
                     [M](const vector_double &item) { return item.size() == M; })) {
        pagmo_throw(std::invalid_argument, "Input contains vector of objectives with heterogeneous dimensionalities");
    }
    // We make sure this function is only requested for two objectives.
    if (M != 2u) {
        pagmo_throw(std::invalid_argument, "The number of objectives detected is " + std::to_string(M)
                                               + ", while Kung's algorithm only works for two objectives.");
    }
    // Sanity checks are over. We may run Kung's algorithm.
    std::vector<pop_size_t> front;
    std::vector<pop_size_t> indexes(input_objs.size());
    std::iota(indexes.begin(), indexes.end(), pop_size_t(0u));
    // Sort in ascending order with respect to the first component
    std::sort(indexes.begin(), indexes.end(), [&input_objs](pop_size_t idx1, pop_size_t idx2) {
        if (detail::equal_to_f(input_objs[idx1][0], input_objs[idx2][0])) {
            return detail::less_than_f(input_objs[idx1][1], input_objs[idx2][1]);
        }
        return detail::less_than_f(input_objs[idx1][0], input_objs[idx2][0]);
    });
    for (auto i : indexes) {
        bool flag = false;
        for (auto j : front) {
            if (pareto_dominance(input_objs[j], input_objs[i])) {
                flag = true;
                break;
            }
        }
        if (!flag) {
            front.push_back(i);
        }
    }
    return front;
}

} // namespace pagmo