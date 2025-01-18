#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>

#include <bit>
#include <stdexcept>
#include <cpp-sort/sorters/merge_sorter.h>

namespace pagmo {

namespace detail {
    template<typename T>
    inline auto make_unique(std::size_t n, std::optional<typename std::remove_extent_t<T>> init = std::nullopt)
    {
        using E = typename std::remove_extent_t<T>;
        auto ptr = std::make_unique<E[]>(n);
        if (init) { std::fill_n(ptr.get(), n, init.value()); }
        return ptr;
    }

    auto update_ranks(auto i, auto const& item, auto& rank, auto& rankset)
    {
        // if we're at the last objective, we can update the ranks
        auto const& [s, lo, hi] = item;
        auto r = rank[i];
        auto const n = std::ssize(rank);
        using E = std::remove_extent_t<typename decltype(s)::element_type>;
        auto constexpr D = std::numeric_limits<E>::digits;
        auto const nb = n / D + static_cast<std::size_t>(n % D != 0);
        if (r+1UL == rankset.size()) {                     // new rankset if necessary
            auto p = detail::make_unique<uint64_t[]>(nb, E{0}); // NOLINT
            rankset.push_back(std::move(p));
        }
        auto& curr = rankset[r];                                           // the pareto front of the current individual
        auto& next = rankset[r+1UL];                                       // next (worse rank) pareto front

        auto b = static_cast<int>(i / D) + lo;

        for (auto j = b, k = lo; k <= hi; ++k, ++j) {                      // iterate over bitset blocks
            auto x = s[k] & curr[j];                                       // final set as intersection of dominance set and rank set
            if (x == 0UL) { continue; }
            auto o = j * D;
            curr[j] &= ~x;                                                 // remove intersection result from current rank set
            next[j] |= x;                                                  // add intersection result to next rank set
            for (; x != 0; x &= (x - 1)) {                                 // iterate over set bits of v
                ++rank[static_cast<std::size_t>(o) + std::countr_zero(x)]; // increment rank
            }
        }
    }

    auto get_fronts(std::vector<pop_size_t> const& rank)
    {
        std::vector<std::vector<std::size_t>> fronts;
        auto rmax = *std::max_element(rank.begin(), rank.end());
        fronts.resize(rmax + 1UL);
        for (std::size_t i = 0UL; i < rank.size(); ++i) {
            fronts[rank[i]].push_back(i);
        }
        return fronts;
    }

    struct item {
        int index;
        double value;

        friend auto operator<(item a, item b) { return a.value < b.value; }
    };
} // namespace detail


fnds_return_type rank_intersect_sorting(std::span<vector_double const> points)
{
    int const N = static_cast<int>(points.size());
    int const M = static_cast<int>(points.front().size());

    // constants
    std::size_t constexpr ZEROS{uint64_t{0}};
    std::size_t constexpr ONES{~ZEROS};
    std::size_t constexpr DIGITS{std::numeric_limits<uint64_t>::digits};
    auto const nb { static_cast<int>(N / DIGITS) + static_cast<int>(N % DIGITS != 0) };
    std::size_t const ub = DIGITS * nb - N; // number of unused bits at the end of the last block (must be set to zero)

    using bitset = std::unique_ptr<uint64_t[]>; // NOLINT
    std::vector<std::tuple<bitset, int, int>> bitsets(N);

    cppsort::merge_sorter sorter;
    std::vector<detail::item> items(N);
    for (auto i = 0; i < N; ++i) {
        items[i] = {i, points[i][1] };
    }
    sorter(items);
    auto mask = detail::make_unique<uint64_t[]>(nb, ONES); // NOLINT
    mask[nb-1] >>= ub;

    for (auto i = 0; i < N; ++i) {
        auto const j = items[i].index;
        auto [q, r] = std::div(j, DIGITS);
        mask[q] &= ~(1UL << r);

        auto lo = 0;
        auto hi = nb-q-1;
        bitset p;
        if (N-1 == i || N-1 == j) {
            lo = hi+1;
            bitsets[j] = { std::move(p), lo, hi };
            continue;
        }
        auto* ptr = mask.get() + q;
        while(hi >= lo && *(ptr + hi) == ZEROS) { --hi; }

        auto sz = hi-lo+1;
        if (sz == 0) { lo = hi+1; } else {
            p = detail::make_unique<uint64_t[]>(sz);
            p[0] = (ONES << r) & mask[q];
            std::copy_n(mask.get()+q+1, sz-1, p.get()+1);
            while (lo <= hi && (p[lo] == ZEROS)) { ++lo; }
            while (lo <= hi && (p[hi] == ZEROS)) { --hi; }
        }
        bitsets[j] = { std::move(p), lo, hi };
    }

    std::vector<bitset> rs; // NOLINT
    rs.push_back(detail::make_unique<uint64_t[]>(nb, ONES)); // vector of sets keeping track of individuals whose rank was updated NOLINT
    rs[0][nb-1] >>= ub; // zero unused region

    for (auto obj = 2; obj < M; ++obj) {
        for (auto& [i, v] : items) { v = points[i][obj]; }
        sorter(items);

        std::fill_n(mask.get(), nb, ONES); // reset q bitset to all ones
        mask[nb-1] >>= ub; // zero unused region

        auto done = 0;
        auto first = items.front().index;
        auto last = items.back().index;

        std::get<1>(bitsets[last]) = std::get<2>(bitsets[last])+1;
        mask[first / DIGITS] &= ~(1UL << static_cast<uint>(first % DIGITS));

        auto mmin = static_cast<int>(first / DIGITS);
        auto mmax = static_cast<int>(first / DIGITS);

        for (auto [i, _] : std::span{items.begin()+1, items.end()-1}) {
            auto [q, r] = std::div(i, DIGITS);
            mask[q] &= ~(1UL << static_cast<uint>(r)); // reset bit i

            mmin = std::min(q, mmin);
            mmax = std::max(q, mmax);

            auto& [bits, lo, hi] = bitsets[i];
            if (lo > hi) { ++done; continue; }

            auto a = std::max(mmin, lo + q);
            auto b = std::min(mmax, hi + q);

            if (b < a) { continue; }

            std::span<uint64_t> pb(bits.get() + a-q, b-a+1);
            std::span<uint64_t const> pm(mask.get() + a, b-a+1);

            std::ranges::transform(pb, pm, std::begin(pb), std::bit_and{});
            while (lo <= hi && (bits[lo] == ZEROS)) { ++lo; }
            while (lo <= hi && (bits[hi] == ZEROS)) { --hi; }
        }
        if (done == N) { break; }
    }

    std::vector<pop_size_t> rank(N, 0);
    for (auto i = 0; i < N; ++i) {
        detail::update_ranks(i, bitsets[i], rank, rs);
    }

    auto fronts = detail::get_fronts(rank);

    std::vector<std::vector<pop_size_t>> dom_list(N);
    std::vector<pop_size_t> dom_count(N);
    return std::make_tuple(std::move(fronts), std::move(dom_list), std::move(dom_count), std::move(rank));
}
} // namespace pagmo
