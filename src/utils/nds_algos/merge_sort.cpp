#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>

#include <bit>
#include <cpp-sort/sorters/merge_sorter.h>

namespace pagmo {
namespace detail {
class BitsetManager {
    using word_t = uint64_t; // NOLINT

    static constexpr size_t FIRST_WORD_RANGE = 0;
    static constexpr size_t LAST_WORD_RANGE = 1;
    static constexpr word_t WORD_MASK = ~word_t{0UL};
    static constexpr size_t WORD_SIZE = std::numeric_limits<word_t>::digits;

    std::vector<std::vector<word_t>> bitsets_;
    std::vector<std::array<size_t, 2>> bsRanges_;
    std::vector<pop_size_t> wordRanking_; //Ranking of each bitset word. A bitset word contains 64 solutions.
    std::vector<pop_size_t> ranking_, ranking0_;
    pop_size_t maxRank_ = 0;
    std::vector<word_t> incrementalBitset_;
    size_t incBsFstWord_{std::numeric_limits<pop_size_t>::max()};
    size_t incBsLstWord_{0};

public:
    [[nodiscard]] auto GetRanking() const -> std::vector<pop_size_t> const& { return ranking0_; }

    auto UpdateSolutionDominance(size_t solutionId) -> bool
    {
        size_t fw = bsRanges_[solutionId][FIRST_WORD_RANGE];
        size_t lw = bsRanges_[solutionId][LAST_WORD_RANGE];
        if (lw > incBsLstWord_) {
            lw = incBsLstWord_;
        }
        if (fw < incBsFstWord_) {
            fw = incBsFstWord_;
        }

        while (fw <= lw && 0 == (bitsets_[solutionId][fw] & incrementalBitset_[fw])) {
            fw++;
        }
        while (fw <= lw && 0 == (bitsets_[solutionId][lw] & incrementalBitset_[lw])) {
            lw--;
        }
        bsRanges_[solutionId][FIRST_WORD_RANGE] = fw;
        bsRanges_[solutionId][LAST_WORD_RANGE] = lw;

        if (fw > lw) {
            return false;
        }
        for (; fw <= lw; fw++) {
            bitsets_[solutionId][fw] &= incrementalBitset_[fw];
        }
        return true;
    }

    void ComputeSolutionRanking(size_t solutionId, size_t initSolId)
    {
        auto fw = bsRanges_[solutionId][FIRST_WORD_RANGE];
        auto lw = bsRanges_[solutionId][LAST_WORD_RANGE];

        if (lw > incBsLstWord_) {
            lw = incBsLstWord_;
        }
        if (fw < incBsFstWord_) {
            fw = incBsFstWord_;
        }
        if (fw > lw) {
            return;
        }
        word_t word{};
        size_t i = 0;
        int rank = 0;
        size_t offset = 0;

        for (; fw <= lw; fw++) {
            word = bitsets_[solutionId][fw] & incrementalBitset_[fw];

            if (word != 0) {
                i = std::countr_zero(static_cast<word_t>(word));
                offset = fw * WORD_SIZE;
                do {
                    auto r = ranking_[offset+i];
                    if (r >= rank) { rank = ranking_[offset + i] + 1; }
                    i++;
                    i += std::countr_zero(word >> i);
                } while (i < WORD_SIZE && rank <= wordRanking_[fw]);
                if (rank > maxRank_) {
                    maxRank_ = rank;
                    break;
                }
            }
        }
        ranking_[solutionId] = rank;
        ranking0_[initSolId] = rank;
        i = solutionId / WORD_SIZE;
        if (rank > wordRanking_[i]) {
            wordRanking_[i] = rank;
        }
    }

    void UpdateIncrementalBitset(size_t solutionId)
    {
        auto wordIndex = solutionId / WORD_SIZE;
        incrementalBitset_[wordIndex] |= (word_t{1} << solutionId);
        if (incBsLstWord_ < wordIndex) {
            incBsLstWord_ = static_cast<int>(wordIndex);
        }
        if (incBsFstWord_ > wordIndex) {
            incBsFstWord_ = wordIndex;
        }
    }

    auto InitializeSolutionBitset(size_t solutionId) -> bool
    {
        auto wordIndex = solutionId / WORD_SIZE;
        if (wordIndex < incBsFstWord_ || 0 == solutionId) {
            bsRanges_[solutionId][FIRST_WORD_RANGE] = std::numeric_limits<int>::max();
            return false;
        }
        if (wordIndex == incBsFstWord_) { //only 1 word in common
            bitsets_[solutionId].resize(wordIndex + 1);
            auto intersection = incrementalBitset_[incBsFstWord_] & ~(WORD_MASK << solutionId);
            if (intersection != 0) {
                bsRanges_[solutionId][FIRST_WORD_RANGE] = wordIndex;
                bsRanges_[solutionId][LAST_WORD_RANGE] = wordIndex;
                bitsets_[solutionId][wordIndex] = intersection;
            }
            return intersection != 0;
        }
        // more than one word in common
        auto lw = incBsLstWord_ < wordIndex ? incBsLstWord_ : wordIndex;
        bsRanges_[solutionId][FIRST_WORD_RANGE] = incBsFstWord_;
        bsRanges_[solutionId][LAST_WORD_RANGE] = lw;
        bitsets_[solutionId] = std::vector<word_t>(lw + 1);
        std::copy_n(incrementalBitset_.data() + incBsFstWord_, lw - incBsFstWord_ + 1, bitsets_[solutionId].data() + incBsFstWord_);
        if (incBsLstWord_ >= wordIndex) { // update (compute intersection) the last word
            bitsets_[solutionId][lw] = incrementalBitset_[lw] & ~(WORD_MASK << solutionId);
            if (bitsets_[solutionId][lw] == 0) {
                bsRanges_[solutionId][LAST_WORD_RANGE]--;
            }
        }
        return true;
    }

    void ClearIncrementalBitset()
    {
        std::fill(incrementalBitset_.begin(), incrementalBitset_.end(), 0UL);
        incBsLstWord_ = 0;
        incBsFstWord_ = std::numeric_limits<int>::max();
        maxRank_ = 0;
    }

    BitsetManager() = default;

    // constructor
    explicit BitsetManager(size_t nSolutions)
    {
        ranking_.resize(nSolutions, 0);
        ranking0_.resize(nSolutions, 0);
        wordRanking_.resize(nSolutions, 0);
        bitsets_.resize(nSolutions);
        bsRanges_.resize(nSolutions);
        incrementalBitset_.resize(nSolutions / WORD_SIZE + static_cast<uint64_t>(nSolutions % WORD_SIZE != 0));
    }
};
} // namespace detail

fnds_return_type merge_sorting(std::span<vector_double const> points)
{
    auto const N = points.size();
    auto const M = points.front().size();
    detail::BitsetManager bsm(N);

    std::vector<std::pair<int, double>> items;
    items.reserve(N);
    for (auto i = 0; i < N; ++i) {
        items.emplace_back(i, points[i][0]);
    }

    cppsort::merge_sorter sorter;
    for (auto obj = 1; obj < M; ++obj) {
        for (auto& [i, v] : items) { v = points[i][obj]; }
        sorter(items, [](auto t){ return t.second; });
        if (obj > 1) { bsm.ClearIncrementalBitset(); }

        auto dominance{false};
        for (auto i = 0; i < N; ++i) {
            auto [j, v] = items[i];
            if (obj == 1) {
                dominance |= bsm.InitializeSolutionBitset(j);
            } else if (obj < M-1) {
                dominance |= bsm.UpdateSolutionDominance(j);
            }
            if (obj == M-1) {
                bsm.ComputeSolutionRanking(j, j);
            }
            bsm.UpdateIncrementalBitset(j);
        }

        if (!dominance) { break; }
    }

    auto ranking = bsm.GetRanking();
    auto rmax = *std::max_element(ranking.begin(), ranking.end());
    std::vector<std::vector<size_t>> fronts(rmax + 1);
    for (auto i = 0UL; i < N; i++) {
        fronts[ranking[i]].push_back(i);
    }

    std::vector<std::vector<pop_size_t>> dom_list(N);
    std::vector<pop_size_t> dom_count(N);
    return std::make_tuple(std::move(fronts), std::move(dom_list), std::move(dom_count), std::move(ranking));
}
} // namespace pagmo
