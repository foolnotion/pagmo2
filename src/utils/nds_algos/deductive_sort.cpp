#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>

namespace pagmo {

namespace detail {

enum dominance_result {
    equal, left, right, none
};

auto dominance(auto const& a, auto const& b) -> dominance_result {
    if (a.size() != b.size()) {
        pagmo_throw(std::invalid_argument,
                    "Different number of objectives found in input fitnesses: " + std::to_string(a.size()) + " and "
                        + std::to_string(b.size()) + ". I cannot define dominance");
    }
    uint8_t r{0};
    uint8_t v{0};
    for (auto i = 0; i < std::ssize(a); ++i) {
        r |= detail::less_than_f(a[i], b[i]);
        v |= detail::greater_than_f(a[i], b[i]);
    }
    return static_cast<dominance_result>(r | (v << 1U));
}
}

fnds_return_type deductive_sorting(std::span<vector_double const> points)
{
    auto n = 0UL; // total number of sorted solutions
    std::vector<std::vector<pop_size_t>> fronts;

    std::size_t constexpr d = std::numeric_limits<uint64_t>::digits;
    auto const s = static_cast<int>(points.size());
    auto const nb = s / d + (s % d != 0);

    std::vector<uint64_t> dominated(nb);
    std::vector<uint64_t> sorted(nb);

    auto set = [](auto&& range, auto i) { range[i / d] |= (1UL << (d - i % d));}; // set bit i
    [[maybe_unused]] auto reset = [](auto&& range, auto i) { range[i / d] &= ~(1UL << (i % d)); }; // unset bit i
    auto get = [](auto&& range, auto i) -> bool { return range[i / d] & (1UL << (d - i % d)); };

    auto dominatedOrSorted = [&](std::size_t i) { return get(sorted, i) || get(dominated, i); };

    while (n < points.size()) {
        std::vector<size_t> front;

        for (size_t i = 0; i < points.size(); ++i) {
            if (dominatedOrSorted(i)) { continue; }

            for (size_t j = i + 1; j < points.size(); ++j) {
                if (dominatedOrSorted(j)) { continue; }

                auto res = detail::dominance(points[i], points[j]);
                if (res == detail::dominance_result::right) { set(dominated, i); }
                if (res == detail::dominance_result::left) { set(dominated, j); }

                if (get(dominated, i)) { break; }
            }

            if (!get(dominated, i)) {
                front.push_back(i);
                set(sorted, i);
            }
        }

        std::fill(dominated.begin(), dominated.end(), 0UL);
        n += front.size();
        fronts.push_back(front);
    }
    std::vector<std::vector<pop_size_t>> dom_list;
    std::vector<pop_size_t> dom_count;
    std::vector<pop_size_t> rank(points.size());
    for (auto i = 0; i < std::ssize(fronts); ++i) {
        for (auto j : fronts[i]) { rank[j] = i; }
    }
    return std::make_tuple(std::move(fronts), std::move(dom_list), std::move(dom_count), std::move(rank));
}
} // namespace pagmo