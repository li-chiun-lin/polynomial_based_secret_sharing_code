#pragma once

#include <cstdint>
#include <vector>

struct AdaptiveSearchOptions
{
    int maxLsb = 1;
    int minLsb = 0;
    int beamWidth = 0;              // 0 means keep all masks after ranking.
    long long earlyStopSSE = 0;     // Stop a pixel search when SSE <= this value.
    double smoothThreshold = 12.0;  // Lower texture uses maxLsb.
    double texturedThreshold = 32.0;// Higher texture uses minLsb.
};

void adjust_cover_with_secret_adaptive(
    const std::vector<unsigned char>& secret,
    std::vector<std::vector<unsigned char>>& cover,
    int n,
    int k,
    uint32_t width,
    uint32_t height,
    const AdaptiveSearchOptions& options);
