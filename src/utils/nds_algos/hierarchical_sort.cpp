#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>

#include <deque>
#include <numeric>
#include <ranges>
#include <cpp-sort/sorters/merge_sorter.h>
#include <eve/module/algo.hpp>

namespace pagmo {
    fnds_return_type hierarchical_sorting(std::span<vector_double const> points)
    {
        auto const N = static_cast<int>(std::ssize(points));
        auto const M = static_cast<int>(std::ssize(points[0]));

        std::deque<size_t> q(points.size());
        std::iota(q.begin(), q.end(), 0UL);
        std::vector<size_t> dominated;
        dominated.reserve(points.size());

        std::vector<std::vector<size_t>> fronts;

        auto dominates = [&](auto const& a, auto const& b) {
            return M == 2
                ? std::ranges::all_of(std::ranges::iota_view{0, M}, [&](auto k) { return a[k] <= b[k]; })
                : eve::algo::all_of(eve::views::zip(a, b), [](auto t) { auto [x, y] = t; return x <= y; });
        };

        cppsort::merge_sorter sorter;

        while (!q.empty()) {
            std::vector<size_t> front;

            while (!q.empty()) {
                auto q1 = q.front(); q.pop_front();
                front.push_back(q1);
                auto nonDominatedCount = 0UL;
                auto const& f1 = points[q1];
                while (q.size() > nonDominatedCount) {
                    auto qj = q.front(); q.pop_front();
                    auto const& f2 = points[qj];
                    if (!dominates(points[q1], points[qj])) {
                        q.push_back(qj);
                        ++nonDominatedCount;
                    } else {
                        dominated.push_back(qj);
                    }
                }
            }
            sorter(dominated);
            std::copy(dominated.begin(), dominated.end(), std::back_inserter(q));
            dominated.clear();
            fronts.push_back(front);
        }
        std::vector<std::vector<pop_size_t>> dom_list(N);
        std::vector<pop_size_t> dom_count(N);
        std::vector<pop_size_t> rank(N);
        for (auto i = 0; i < std::ssize(fronts); ++i) {
            for (auto j : fronts[i]) { rank[j] = i; }
        }
        return std::make_tuple(std::move(fronts), std::move(dom_list), std::move(dom_count), std::move(rank));
    }
}
