#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>

#include <ranges>
#include <eve/module/algo.hpp>

namespace pagmo {
template<bool BINARY_SEARCH = false>
fnds_return_type efficient_sorting(std::span<vector_double const> points)
{
    auto const M = static_cast<int>(std::ssize(points[0]));

    // check if individual i is dominated by any individual in the front f
    auto dominated = [&](auto const& f, size_t i) {
        return std::ranges::any_of(f, [&](size_t j) {
            auto const& a = points[j];
            auto const& b = points[i];
            return M == 2
                ? std::ranges::all_of(std::ranges::iota_view{0, M}, [&](auto k) { return a[k] <= b[k]; })
                : eve::algo::all_of(eve::views::zip(a, b), [](auto t) { auto [x, y] = t; return x <= y; });
        });
    };

    std::vector<std::vector<size_t>> fronts;
    for (size_t i = 0; i < points.size(); ++i) {
        decltype(fronts)::iterator it;
        if constexpr (BINARY_SEARCH) { // binary search
            it = std::partition_point(fronts.begin(), fronts.end(), [&](auto const& f) { return dominated(f, i); });
        } else { // sequential search
            it = std::find_if(fronts.begin(), fronts.end(), [&](auto const& f) { return !dominated(f, i); });
        }
        if (it == fronts.end()) { fronts.push_back({i}); }
        else                    { it->push_back(i);          }
    }
    std::vector<pop_size_t> rank(points.size());
    for (auto i = 0; i < std::ssize(fronts); ++i) {
        for (auto j : fronts[i]) { rank[j] = i; }
    }
    std::vector<std::vector<pop_size_t>> dom_list;
    std::vector<pop_size_t> dom_count;
    return std::make_tuple(std::move(fronts), std::move(dom_list), std::move(dom_count), std::move(rank));
}

fnds_return_type efficient_sorting_binary(std::span<const vector_double> points) {
    return efficient_sorting<true>(points);
}

fnds_return_type efficient_sorting_sequential(std::span<const vector_double> points) {
    return efficient_sorting<false>(points);
}
} // namespace pagmo

#include <ranges>
#include <eve/module/algo.hpp>