#include "tiff_io.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <string>

using namespace std;

// Discrimination function: sum of absolute differences in a block
double get_smoothness(const vector<double>& block) {
    double smoothness = 0;
    for (size_t i = 0; i < block.size() - 1; ++i) {
        smoothness += abs(block[i + 1] - block[i]);
    }
    return smoothness;
}

// Flipping function F1 (Positive): 0<->1, 2<->3...
int flip_positive(int val) {
    return (val % 2 == 0) ? (val + 1) : (val - 1);
}

// Flipping function F-1 (Negative): -1<->0, 1<->2...
int flip_negative(int val) {
    return (val % 2 != 0) ? (val + 1) : (val - 1);
}

void rs_analysis(const TiffInfo& info) {
    if (info.data.empty()) {
        cerr << "Error: Image data is empty." << endl;
        return;
    }

    int rm = 0, sm = 0, r_m = 0, s_m = 0;
    int total_blocks = 0;
    const int blockSize = 4;

    // Standard Mask M = [0, 1, 1, 0]
    const vector<int> mask = { 0, 1, 1, 0 };

    // Iterate through rows and columns using linear indexing
    for (uint32_t r = 0; r < info.height; ++r) {
        // Stop before the end to ensure a full block of 4 can be read
        for (uint32_t c = 0; c <= info.width - blockSize; c += blockSize) {

            // 1. Extract block of 4 pixels
            vector<double> block(blockSize);
            for (int i = 0; i < blockSize; ++i) {
                // Linear indexing for vector<Pixel>
                block[i] = static_cast<double>(info.data[r * info.width + (c + i)]);
            }

            double f_orig = get_smoothness(block);

            // 2. Positive Flipping Analysis
            vector<double> block_p = block;
            for (int i = 0; i < blockSize; ++i) {
                if (mask[i] == 1) block_p[i] = flip_positive(static_cast<int>(block_p[i]));
            }
            double f_p = get_smoothness(block_p);
            if (f_p > f_orig) rm++;
            else if (f_p < f_orig) sm++;

            // 3. Negative Flipping Analysis
            vector<double> block_n = block;
            for (int i = 0; i < blockSize; ++i) {
                if (mask[i] == 1) block_n[i] = flip_negative(static_cast<int>(block_n[i]));
            }
            double f_n = get_smoothness(block_n);
            if (f_n > f_orig) r_m++;
            else if (f_n < f_orig) s_m++;

            total_blocks++;
        }
    }

    // Output formatted for easy copying into LaTeX tables
    double RM = (double)rm / total_blocks;
    double SM = (double)sm / total_blocks;
    double R_M = (double)r_m / total_blocks;
    double S_M = (double)s_m / total_blocks;

    cout << "--- RS Analysis for " << info.width << "x" << info.height << " ---" << endl;
    cout << "Rm:   " << RM << endl;
    cout << "Sm:   " << SM << endl;
    cout << "R-m:  " << R_M << endl;
    cout << "S-m:  " << S_M << endl;
    cout << "Linearity Gap: " << abs((RM - SM) - (R_M - S_M)) << endl;
}