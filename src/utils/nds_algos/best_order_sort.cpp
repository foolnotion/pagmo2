#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>

#include <ranges>
#include <cpp-sort/sorters/merge_sorter.h>
#include <eve/module/algo.hpp>

namespace pagmo {

fnds_return_type best_order_sorting(std::span<vector_double const> points)
{
    auto const N = static_cast<int>(points.size());
    auto const M = static_cast<int>(points.front().size());

    // initialization
    std::vector<std::vector<std::vector<int>>> solutionSets(M);

    std::vector<bool> isRanked(N, false); // rank status
    std::vector<pop_size_t> rank(N, 0);          // rank of solutions

    int sc{0}; // number of solutions already ranked
    int rc{1}; // number of fronts so far (at least one front)

    std::vector<std::vector<int>> sortedByObjective(M);
    std::vector<std::vector<int>> sortedIndices(N);

    auto& idx = sortedByObjective[0];
    idx.resize(N);
    std::iota(idx.begin(), idx.end(), 0);
    for(auto i : idx) { sortedIndices[i].push_back(i); }

    // sort the individuals for each objective
    cppsort::merge_sorter sorter;
    for (auto j = 1; j < M; ++j) {
        sortedByObjective[j] = sortedByObjective[j-1];
        sorter(sortedByObjective[j], [&](auto i) { return points[i][j]; });

        for (int i = 0; i < N; ++i) {
            sortedIndices[sortedByObjective[j][i]].push_back(i);
        }
    }

    // utility method
    auto addSolutionToRankSet = [&](auto s, auto j) {
        auto r = rank[s];
        auto& ss = solutionSets[j];
        if (r >= std::ssize(ss)) {
            ss.resize(r+1UL);
        }
        ss[r].push_back(s);
    };

    // algorithm 4 in the original paper
    auto check_dominance = [&](auto s, auto t) {
        auto const& a = sortedIndices[s];
        auto const& b = sortedIndices[t];
        return M == 2
            ? std::ranges::none_of(std::ranges::iota_view{0, M}, [&](auto i) { return a[i] < b[i]; })
            : eve::algo::none_of(eve::views::zip(a, b), [](auto t) { auto [x, y] = t; return x < y; });
    };

    // algorithm 3 in the original paper
    auto findRank = [&](auto s, auto j) {
        bool done{false};

        for (auto k = 0; k < rc; ++k) {
            bool dominated = false;

            if (k >= std::ssize(solutionSets[j])) {
                solutionSets[j].resize(k+1UL);
            }

            for (auto t : solutionSets[j][k]) {
                // check if s is dominated by t
                if (dominated = check_dominance(s, t); dominated) {
                    break;
                }
            }

            if (!dominated) {
                rank[s] = k;
                done = true;
                addSolutionToRankSet(s, j);
                break;
            }
        }

        if (!done) {
            rank[s] = rc;
            addSolutionToRankSet(s, j);
            ++rc;
        }
    };

    // main loop
    for (auto i = 0; i < N; ++i) {
        for (auto j = 0; j < M; ++j) {
            auto s = sortedByObjective[j][i]; // take i-th element from qj
            // auto cs = comparisonSets[s];
            // std::ranges::remove(cs, j); // reduce comparison set
            if (isRanked[s]) {
                addSolutionToRankSet(s, j);
            } else {
                findRank(s, j);
                isRanked[s] = true;
                ++sc;
            }
        }

        if (sc == N) {
            break; // all done, sorting ended
        }
    }

    // return fronts
    std::vector<std::vector<pop_size_t>> fronts;
    fronts.resize(*std::max_element(rank.begin(), rank.end()) + 1UL);
    for (std::size_t i = 0UL; i < rank.size(); ++i) {
        fronts[rank[i]].push_back(i);
    }
    std::vector<std::vector<pop_size_t>> dom_list;
    std::vector<pop_size_t> dom_count;
    return std::make_tuple(std::move(fronts), std::move(dom_list), std::move(dom_count), std::move(rank));
}

} // namespace pagmo