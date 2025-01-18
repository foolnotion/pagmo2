#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>

#include <cpp-sort/sorters/merge_sorter.h>
#include <Eigen/Core>
#include <ranges>
#include <eve/module/algo.hpp>

namespace pagmo {

fnds_return_type rank_ordinal_sorting(std::span<vector_double const> points)
{
    static_assert(EIGEN_VERSION_AT_LEAST(3, 4, 0), "RankOrdinal requires Eigen >= 3.4.0");
    using Vec = Eigen::Array<int, -1, 1>;
    using Mat = Eigen::Array<int, -1, -1>;

    const auto N = static_cast<int>(points.size());
    const auto M = static_cast<int>(points.front().size());

    // We make sure to have two points at least (one could also be allowed)
    if (N < 2u) {
        pagmo_throw(std::invalid_argument,
                    "At least two points are needed for non-dominated sorting: " + std::to_string(N) + " detected.");
    }

    // 1) sort indices according to the stable sorting rules
    Mat p(N, M); // permutation matrix
    Mat r(M, N); // ordinal rank matrix
    p.col(0) = Vec::LinSpaced(N, 0, N-1);
    r(0, p.col(0)) = Vec::LinSpaced(N, 0, N-1);

    std::vector<double> buf(N); // buffer to store fitness values to avoid pointer indirections during sorting
    cppsort::merge_sorter sorter;
    for (auto i = 1; i < M; ++i) {
        std::transform(points.begin(), points.end(), buf.begin(), [i](auto const& pt) { return pt[i]; });
        p.col(i) = p.col(i - 1); // this is a critical part of the approach
        sorter(p.col(i), [&](auto j) { return buf[j]; });
        r(i, p.col(i)) = Vec::LinSpaced(N, 0, N-1);
    }

    // 2) save min and max positions as well as the column index for the max position
    Vec maxc(N);
    Vec maxp(N);
    for (auto i = 0; i < N; ++i) {
        auto c = r.col(i);
        auto max = std::max_element(c.begin(), c.end());
        maxp(i) = *max;
        maxc(i) = std::distance(c.begin(), max);
    }

    // 3) compute ranks / fronts
    std::vector<pop_size_t> rank(N, 0);
    for (auto i : p(Eigen::seq(0, N-2), 0)) {
        if (maxp(i) == N-1) {
            continue;
        }
        for (auto j : p(Eigen::seq(maxp(i)+1, N-1), maxc(i))) {
            if (rank[i] != rank[j]) { continue; }
            auto k = M == 2
                ? (r.col(i) < r.col(j)).all()
                : eve::algo::all_of(eve::views::zip(std::span<int>(r.col(i).data(), r.col(i).size()), std::span<int>(r.col(j).data(), r.col(j).size())),
                                    [](auto t) { auto [a, b] = t; return a < b; });
            rank[j] += static_cast<int>(k);
        }
    }
    std::vector<std::vector<pop_size_t>> fronts(*std::max_element(rank.begin(), rank.end()) + 1);
    for (auto i = 0; i < N; ++i) {
        fronts[rank[i]].push_back(i);
    }
    std::vector<std::vector<pop_size_t>> dom_list(N);
    std::vector<pop_size_t> dom_count(N);
    return std::make_tuple(std::move(fronts), std::move(dom_list), std::move(dom_count), std::move(rank));
}
} // namespace pagmo