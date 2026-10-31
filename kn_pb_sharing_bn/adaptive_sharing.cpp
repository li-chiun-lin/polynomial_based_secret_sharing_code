#include "adaptive_sharing.h"

#include "../finite_field/binary_field.h"

#include <algorithm>
#include <cassert>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace std;

namespace
{
    struct AdaptiveMaskInfo
    {
        uint32_t mask = 0;
        vector<int> selIdx;
        vector<unsigned char> xs;
        vector<vector<unsigned char>> evalWeights;
    };

    struct RankedMask
    {
        const AdaptiveMaskInfo* info = nullptr;
        long long score = LLONG_MAX;
    };

    int clamp_int(int value, int low, int high)
    {
        return max(low, min(value, high));
    }

    uint64_t make_lsb_limit(int lsb, size_t selectedCount)
    {
        if (lsb <= 0 || selectedCount == 0)
            return 1ULL;

        const int shift = lsb * static_cast<int>(selectedCount);
        if (shift >= 63)
            return UINT64_MAX;

        return 1ULL << shift;
    }

    uint8_t make_clear_mask(int lsb)
    {
        if (lsb <= 0)
            return 0xFFu;
        if (lsb >= 8)
            return 0x00u;
        return static_cast<uint8_t>(~((1u << lsb) - 1u));
    }

    unsigned char lagrange_eval_weight(
        const vector<unsigned char>& xs,
        size_t basisIdx,
        unsigned char x)
    {
        unsigned char numer = 1;
        unsigned char denom = 1;

        for (size_t j = 0; j < xs.size(); ++j)
        {
            if (j == basisIdx)
                continue;

            numer = gf_mul(numer, static_cast<unsigned char>(x ^ xs[j]));
            denom = gf_mul(denom, static_cast<unsigned char>(xs[basisIdx] ^ xs[j]));
        }

        return gf_mul(numer, gf_inv(denom));
    }

    unsigned char eval_from_weights(
        const vector<unsigned char>& weights,
        const vector<unsigned char>& ys)
    {
        unsigned char value = 0;
        for (size_t j = 0; j < ys.size(); ++j)
            value ^= gf_mul(weights[j], ys[j]);
        return value;
    }

    vector<AdaptiveMaskInfo> build_masks(int n, int k)
    {
        const uint32_t limit = (n >= 32) ? 0xFFFFFFFFu : (1u << n);
        vector<AdaptiveMaskInfo> masks;

        for (uint32_t mask = 0; mask < limit; ++mask)
        {
            if (popcount(mask) != (k - 1))
                continue;

            AdaptiveMaskInfo mi;
            mi.mask = mask;
            for (int idx = 0; idx < n; ++idx)
            {
                if (mask & (1u << idx))
                {
                    mi.selIdx.push_back(idx);
                    mi.xs.push_back(static_cast<unsigned char>(idx + 1));
                }
            }
            mi.xs.push_back(0);
            mi.evalWeights.assign(n, vector<unsigned char>(k, 0));
            for (int idx = 0; idx < n; ++idx)
            {
                const unsigned char x = static_cast<unsigned char>(idx + 1);
                for (int basis = 0; basis < k; ++basis)
                    mi.evalWeights[idx][basis] = lagrange_eval_weight(mi.xs, basis, x);
            }
            masks.push_back(mi);
        }

        return masks;
    }

    double estimate_texture(
        const vector<vector<unsigned char>>& cover,
        int n,
        uint32_t width,
        uint32_t height,
        size_t pos)
    {
        if (width == 0 || height == 0)
            return 0.0;

        const uint32_t x = static_cast<uint32_t>(pos % width);
        const uint32_t y = static_cast<uint32_t>(pos / width);
        double total = 0.0;
        int samples = 0;

        for (int idx = 0; idx < n; ++idx)
        {
            const unsigned char center = cover[idx][pos];
            if (x > 0)
            {
                total += abs(static_cast<int>(center) - static_cast<int>(cover[idx][pos - 1]));
                ++samples;
            }
            if (x + 1 < width)
            {
                total += abs(static_cast<int>(center) - static_cast<int>(cover[idx][pos + 1]));
                ++samples;
            }
            if (y > 0)
            {
                total += abs(static_cast<int>(center) - static_cast<int>(cover[idx][pos - width]));
                ++samples;
            }
            if (y + 1 < height)
            {
                total += abs(static_cast<int>(center) - static_cast<int>(cover[idx][pos + width]));
                ++samples;
            }
        }

        return samples == 0 ? 0.0 : total / static_cast<double>(samples);
    }

    int choose_adaptive_lsb(double texture, const AdaptiveSearchOptions& options)
    {
        const int minLsb = clamp_int(options.minLsb, 0, 8);
        const int maxLsb = clamp_int(options.maxLsb, minLsb, 8);

        if (texture <= options.smoothThreshold)
            return maxLsb;
        if (texture >= options.texturedThreshold)
            return minLsb;

        return clamp_int((minLsb + maxLsb + 1) / 2, minLsb, maxLsb);
    }

    long long evaluate_candidate_sse(
        const vector<vector<unsigned char>>& cover,
        size_t pos,
        int n,
        const AdaptiveMaskInfo& mi,
        const vector<unsigned char>& ys)
    {
        long long sumSq = 0;

        for (int idx = 0; idx < n; ++idx)
        {
            if (mi.mask & (1u << idx))
                continue;

            unsigned char val = eval_from_weights(mi.evalWeights[idx], ys);

            const long long diff =
                static_cast<long long>(val) -
                static_cast<long long>(cover[idx][pos]);
            sumSq += diff * diff;
        }

        for (size_t s = 0; s < mi.selIdx.size(); ++s)
        {
            const int idx = mi.selIdx[s];
            const long long diff =
                static_cast<long long>(ys[s]) -
                static_cast<long long>(cover[idx][pos]);
            sumSq += diff * diff;
        }

        return sumSq;
    }

    long long rank_mask(
        const vector<unsigned char>& secret,
        const vector<vector<unsigned char>>& cover,
        size_t pos,
        int n,
        int lsb,
        const AdaptiveMaskInfo& mi,
        vector<unsigned char>& ys)
    {
        const uint8_t clearMask = make_clear_mask(lsb);

        for (size_t s = 0; s < mi.selIdx.size(); ++s)
        {
            const int idx = mi.selIdx[s];
            ys[s] = static_cast<unsigned char>(cover[idx][pos] & clearMask);
        }
        ys[mi.selIdx.size()] = secret[pos];

        return evaluate_candidate_sse(cover, pos, n, mi, ys);
    }
}

void adjust_cover_with_secret_adaptive(
    const vector<unsigned char>& secret,
    vector<vector<unsigned char>>& cover,
    int n,
    int k,
    uint32_t width,
    uint32_t height,
    const AdaptiveSearchOptions& options)
{
    const size_t m = secret.size();
    assert(n >= 1 && k >= 1 && k <= n);
    assert(static_cast<int>(cover.size()) == n);
    for (int idx = 0; idx < n; ++idx)
        assert(cover[idx].size() == m);

    const vector<AdaptiveMaskInfo> masks = build_masks(n, k);
    const vector<vector<unsigned char>> textureCover = cover;

#pragma omp parallel for schedule(dynamic)
    for (long long i = 0; i < static_cast<long long>(m); ++i)
    {
        const size_t pos = static_cast<size_t>(i);
        const double texture = estimate_texture(textureCover, n, width, height, pos);
        const int localLsb = choose_adaptive_lsb(texture, options);
        const uint8_t clearMask = make_clear_mask(localLsb);

        vector<unsigned char> ys(k);
        const AdaptiveMaskInfo* bestInfo = nullptr;

        vector<RankedMask> ranked;
        ranked.reserve(masks.size());

        for (const AdaptiveMaskInfo& mi : masks)
        {
            const long long score = rank_mask(
                secret, cover, pos, n, localLsb, mi, ys);
            ranked.push_back(RankedMask{ &mi, score });
        }

        sort(ranked.begin(), ranked.end(),
            [](const RankedMask& a, const RankedMask& b) {
                return a.score < b.score;
            });

        const int retainedMasks =
            options.beamWidth <= 0
            ? static_cast<int>(ranked.size())
            : min(options.beamWidth, static_cast<int>(ranked.size()));

        long long bestSSE = LLONG_MAX;
        uint32_t bestMask = 0;
        uint64_t bestLsbChoice = 0;

        for (int r = 0; r < retainedMasks; ++r)
        {
            const AdaptiveMaskInfo& mi = *ranked[r].info;
            const vector<int>& selIdx = mi.selIdx;
            const uint64_t lsbLimit = make_lsb_limit(localLsb, selIdx.size());

            for (uint64_t lsbChoice = 0; lsbChoice < lsbLimit; ++lsbChoice)
            {
                for (size_t s = 0; s < selIdx.size(); ++s)
                {
                    const int idx = selIdx[s];
                    const uint8_t bits = localLsb == 0
                        ? 0
                        : static_cast<uint8_t>(
                            (lsbChoice >> (localLsb * s)) &
                            ((1u << localLsb) - 1u));

                    ys[s] = static_cast<unsigned char>(
                        (cover[idx][pos] & clearMask) | bits);
                }
                ys[selIdx.size()] = secret[pos];

                long long sumSq = 0;

                for (int idx = 0; idx < n; ++idx)
                {
                    if (mi.mask & (1u << idx))
                        continue;

                    unsigned char val = eval_from_weights(mi.evalWeights[idx], ys);

                    const long long diff =
                        static_cast<long long>(val) -
                        static_cast<long long>(cover[idx][pos]);
                    sumSq += diff * diff;
                    if (sumSq >= bestSSE)
                        goto prune_candidate;
                }

                for (size_t s = 0; s < selIdx.size(); ++s)
                {
                    const int idx = selIdx[s];
                    const long long diff =
                        static_cast<long long>(ys[s]) -
                        static_cast<long long>(cover[idx][pos]);
                    sumSq += diff * diff;
                    if (sumSq >= bestSSE)
                        goto prune_candidate;
                }

                if (sumSq < bestSSE)
                {
                    bestSSE = sumSq;
                    bestMask = mi.mask;
                    bestLsbChoice = lsbChoice;
                    bestInfo = &mi;

                    if (bestSSE <= options.earlyStopSSE)
                        goto finish_pixel_search;
                }

            prune_candidate:
                ;
            }
        }

    finish_pixel_search:
        if (bestSSE == LLONG_MAX || bestInfo == nullptr)
            continue;

        const size_t bestSelCount = bestInfo->selIdx.size();
        for (size_t s = 0; s < bestSelCount; ++s)
        {
            const int idx = bestInfo->selIdx[s];
            const uint8_t bits = localLsb == 0
                ? 0
                : static_cast<uint8_t>(
                    (bestLsbChoice >> (localLsb * s)) &
                    ((1u << localLsb) - 1u));
            ys[s] = static_cast<unsigned char>(
                (cover[idx][pos] & clearMask) | bits);
        }
        ys[bestSelCount] = secret[pos];

        for (int idx = 0; idx < n; ++idx)
            cover[idx][pos] = eval_from_weights(bestInfo->evalWeights[idx], ys);

        size_t s = 0;
        for (int idx = 0; idx < n; ++idx)
        {
            if (!(bestMask & (1u << idx)))
                continue;

            const uint8_t bits = localLsb == 0
                ? 0
                : static_cast<uint8_t>(
                    (bestLsbChoice >> (localLsb * s)) &
                    ((1u << localLsb) - 1u));
            cover[idx][pos] = static_cast<unsigned char>(
                (cover[idx][pos] & clearMask) | bits);
            ++s;
        }
    }
}
