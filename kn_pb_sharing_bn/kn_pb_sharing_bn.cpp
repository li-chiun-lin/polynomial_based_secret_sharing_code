#include "../tiff_io/tiff_io.h"
#include "../finite_field/binary_field.h"
#include "adaptive_sharing.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <set>
#include <vector>
#include <cassert>
#include <string>
#include <cmath>
#include <climits>
#include <bit>
#include <cstdint>
#include <limits>
#include <sstream>
#include <utility>

using namespace std;

enum class EmbedMode
{
    FullSearch,
    AdaptiveSearch
};

static string mode_name(EmbedMode mode)
{
    return mode == EmbedMode::AdaptiveSearch ? "adaptive" : "full";
}

static string strip_extension(const string& fn)
{
    size_t pos = fn.find_last_of('.');
    if (pos == string::npos) return fn;
    return fn.substr(0, pos);
}

static string embedded_prefix(EmbedMode mode)
{
    return mode == EmbedMode::AdaptiveSearch ? "embedded_adaptive_b_lsb" : "embedded_b_lsb";
}

static string recovered_prefix(EmbedMode mode)
{
    return mode == EmbedMode::AdaptiveSearch ? "recovered_adaptive_b_lsb" : "recovered_b_lsb";
}

struct MaskInfo
{
    uint32_t mask;
    vector<int> selIdx;              // indices in [0..n)
    vector<unsigned char> xs;        // xs = idx+1, ..., 0
    vector<vector<unsigned char>> evalWeights; // [share idx][point idx]
};

static unsigned char lagrange_eval_weight(
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

static unsigned char eval_from_weights(
    const vector<unsigned char>& weights,
    const vector<unsigned char>& ys)
{
    unsigned char value = 0;
    for (size_t j = 0; j < ys.size(); ++j)
        value ^= gf_mul(weights[j], ys[j]);
    return value;
}

// adjust cover using secret, but arithmetic in GF(2^8)
// lsb: number of least-significant bits to try per selected index (allowed 0..8)
// If lsb == 0 -> no modification for selected pixels (use original cover bytes)
static void adjust_cover_with_secret(vector<unsigned char> const& secret,
    vector<vector<unsigned char>>& cover,
    int n,
    int k,
    int lsb)
{
    if (lsb < 0) lsb = 0;
    if (lsb > 8) lsb = 8;

    const size_t m = secret.size();
    assert(n >= 1 && k >= 1 && k <= n + 1);
    assert(static_cast<int>(cover.size()) == n);
    for (int idx = 0; idx < n; ++idx)
        assert(cover[idx].size() == m);

    const uint8_t clearMask = static_cast<uint8_t>(~((1u << lsb) - 1u));
    const uint32_t limit = (n >= 32) ? 0xFFFFFFFFu : (1u << n);

    vector<MaskInfo> masks;
    masks.reserve(64);

    for (uint32_t mask = 0; mask < limit; ++mask)
    {
        if (popcount(mask) != (k - 1)) continue;

        MaskInfo mi;
        mi.mask = mask;

        for (int idx = 0; idx < n; ++idx)
            if (mask & (1u << idx))
            {
                mi.selIdx.push_back(idx);
                mi.xs.push_back(static_cast<unsigned char>(idx + 1));
            }

        mi.xs.push_back(0); // secret point
        mi.evalWeights.assign(n, vector<unsigned char>(k, 0));
        for (int idx = 0; idx < n; ++idx)
        {
            const unsigned char x = static_cast<unsigned char>(idx + 1);
            for (int basis = 0; basis < k; ++basis)
                mi.evalWeights[idx][basis] = lagrange_eval_weight(mi.xs, basis, x);
        }
        masks.push_back(mi);
    }


#pragma omp parallel for schedule(dynamic)
    for (long long i = 0; i < static_cast<long long>(m); ++i)
    {
        long long bestSSE = LLONG_MAX;
        uint32_t bestMask = 0;
        uint64_t bestLsbChoice = 0;
        const MaskInfo* bestInfo = nullptr;

        vector<unsigned char> ys(k);

        for (const MaskInfo& mi : masks)
        {
            const vector<int>& selIdx = mi.selIdx;
            const size_t selCount = selIdx.size();

            const uint64_t lsbLimit =
                (lsb == 0 || selCount == 0) ? 1ULL : (1ULL << (lsb * selCount));

            for (uint64_t lsbChoice = 0; lsbChoice < lsbLimit; ++lsbChoice)
            {
                // build ys (selected + secret)
                for (size_t s = 0; s < selCount; ++s)
                {
                    const int idx = selIdx[s];
                    const uint8_t bits = (lsb == 0)
                        ? 0
                        : static_cast<uint8_t>(
                            (lsbChoice >> (lsb * s)) & ((1u << lsb) - 1u));

                    ys[s] = static_cast<unsigned char>(
                        (cover[idx][i] & clearMask) | bits);
                }
                ys[selCount] = secret[i];

                long long sumSq = 0;

                // unselected pixels (early prune)
                for (int idx = 0; idx < n; ++idx)
                {
                    if (mi.mask & (1u << idx)) continue;

                    unsigned char val = eval_from_weights(mi.evalWeights[idx], ys);

                    long long diff =
                        static_cast<long long>(val) -
                        static_cast<long long>(cover[idx][i]);

                    sumSq += diff * diff;
                    if (sumSq >= bestSSE) goto prune;
                }

                // selected pixels (incremental)
                for (size_t s = 0; s < selCount; ++s)
                {
                    const int idx = selIdx[s];
                    long long diff =
                        static_cast<long long>(ys[s]) -
                        static_cast<long long>(cover[idx][i]);
                    sumSq += diff * diff;
                    if (sumSq >= bestSSE) goto prune;
                }

                if (sumSq < bestSSE)
                {
                    bestSSE = sumSq;
                    bestMask = mi.mask;
                    bestLsbChoice = lsbChoice;
                    bestInfo = &mi;
                }

            prune:
                ;
            }
        }

        if (bestSSE == LLONG_MAX || bestInfo == nullptr)
            continue;

        // apply polynomial values using precomputed Lagrange weights
        const size_t bestSelCount = bestInfo->selIdx.size();
        for (size_t s = 0; s < bestSelCount; ++s)
        {
            const int idx = bestInfo->selIdx[s];
            const uint8_t bits = (lsb == 0)
                ? 0
                : static_cast<uint8_t>(
                    (bestLsbChoice >> (lsb * s)) & ((1u << lsb) - 1u));
            ys[s] = static_cast<unsigned char>(
                (cover[idx][i] & clearMask) | bits);
        }
        ys[bestSelCount] = secret[i];

        for (int idx = 0; idx < n; ++idx)
            cover[idx][i] = eval_from_weights(bestInfo->evalWeights[idx], ys);

        // apply best LSBs
        size_t s = 0;
        for (int idx = 0; idx < n; ++idx)
        {
            if (!(bestMask & (1u << idx))) continue;
            uint8_t bits = (lsb == 0)
                ? 0
                : static_cast<uint8_t>(
                    (bestLsbChoice >> (lsb * s)) & ((1u << lsb) - 1u));
            cover[idx][i] = static_cast<unsigned char>(
                (cover[idx][i] & clearMask) | bits);
            ++s;
        }
    }
}

// embed using GF(2^8)
// secret_name is a parameter
static int embed(const string& inputDir,
    const vector<string>& cover_names,
    int n,
    int k,     // minimum number of shares required for reconstruction
    int lsb,
    const string& outputDir,
    const string& secret_name,
    EmbedMode mode = EmbedMode::FullSearch,
    const AdaptiveSearchOptions& adaptiveOptions = AdaptiveSearchOptions{})
{
    string inDir = inputDir;
    if (!inDir.empty() && inDir.back() != '/' && inDir.back() != '\\') inDir += '/';
    string outDir = outputDir;
    if (!outDir.empty() && outDir.back() != '/' && outDir.back() != '\\') outDir += '/';

    if (n != static_cast<int>(cover_names.size()))
    {
        cerr << "Error: n != cover_names.size()\n";
        return 1;
    }

    // NEW: explicit threshold semantics
    if (k <= 0 || k > n)
    {
        cerr << "Error: require 1 <= k <= n\n";
        return 1;
    }

    TiffInfo info_secret = readTiff(inDir + secret_name);
    vector<unsigned char> secret = info_secret.data;
    if (secret.empty())
    {
        cerr << "Error: secret image '" << secret_name << "' has no data\n";
        return 1;
    }

    vector<TiffInfo> coverInfos;
    coverInfos.reserve(n);
    for (const string& name : cover_names)
    {
        TiffInfo info = readTiff(inDir + name);
        if (info.data.empty())
        {
            cerr << "Error: cover image '" << name << "' has no data\n";
            return 1;
        }
        coverInfos.push_back(info);
    }

    size_t expected = secret.size();
    for (int i = 0; i < n; ++i)
    {
        if (coverInfos[i].data.size() != expected)
        {
            cerr << "Error: size mismatch between secret and cover '"
                << cover_names[i] << "'\n";
            return 1;
        }
    }

    cout << "embed: using GF(2^8), secret bytes=" << expected
        << ", covers=" << n
        << ", threshold k=" << k
        << ", lsb=" << lsb
        << ", mode=" << mode_name(mode)
        << "\n";

    vector<vector<unsigned char>> cover;
    cover.reserve(n);
    for (int i = 0; i < n; ++i)
        cover.emplace_back(coverInfos[i].data);

    // Security guarantee:
    // Any subset of size <= k-1 reveals no information about the secret.
    // Adaptive mode only limits the embedding search; extraction remains unchanged.
    if (mode == EmbedMode::AdaptiveSearch)
    {
        AdaptiveSearchOptions options = adaptiveOptions;
        options.maxLsb = lsb;
        adjust_cover_with_secret_adaptive(
            secret, cover, n, k, info_secret.width, info_secret.height, options);
    }
    else
    {
        adjust_cover_with_secret(secret, cover, n, k, lsb);
    }

    cout << fixed << setprecision(5);
    for (int i = 0; i < n; ++i)
    {
        TiffInfo outInfo = coverInfos[i];
        outInfo.data = cover[i];

        string base = strip_extension(cover_names[i]);
        string outFileName =
            embedded_prefix(mode) + to_string(lsb) + "_" +
            to_string(k) + "_" + to_string(n) + "_" +
            base + ".tiff";

        string outPath = outDir + outFileName;

        writeTiff(outInfo, outPath);
        //diff(inDir + cover_names[i], outPath,
        //    outDir + "diff_" + outFileName);

        double psnr = compute_psnr(coverInfos[i].data, cover[i]);
        if (psnr == INFINITY)
            cout << "PSNR(original -> embedded) for " << cover_names[i] << " : inf\n";
        else if (psnr == -1.0)
            cout << "PSNR(original -> embedded) for " << cover_names[i] << " : error\n";
        else
            cout << "PSNR(original -> embedded) for " << cover_names[i] << " : " << psnr << " dB\n";

        double ssim = compute_ssim(coverInfos[i].data, cover[i]);
        if (ssim == -1.0)
            cout << "SSIM(original -> embedded) for " << cover_names[i] << " : error\n";
        else
            cout << "SSIM(original -> embedded) for " << cover_names[i] << " : " << ssim << "\n";
    }

    return 0;
}

// extract using GF(2^8)
static int extract(const string& inputDir,
    const vector<string>& embed_names,
    const vector<unsigned char>& xs,
    const string& outputDir,
    const string& output_name)
{
    string inDir = inputDir;
    if (!inDir.empty() && inDir.back() != '/' && inDir.back() != '\\')
        inDir += '/';

    string outDir = outputDir;
    if (!outDir.empty() && outDir.back() != '/' && outDir.back() != '\\')
        outDir += '/';

    const int k = static_cast<int>(embed_names.size());
    if (k <= 0)
    {
        cerr << "Error: empty embed_names\n";
        return 1;
    }
    if (static_cast<int>(xs.size()) != k)
    {
        cerr << "Error: xs.size() != embed_names.size()\n";
        return 1;
    }

    // validate xs
    for (int i = 0; i < k; ++i)
    {
        if (xs[i] == 0)
        {
            cerr << "Error: xs[" << i << "] = 0 not allowed\n";
            return 1;
        }
        for (int j = i + 1; j < k; ++j)
            if (xs[i] == xs[j])
            {
                cerr << "Error: duplicate xs\n";
                return 1;
            }
    }

    vector<TiffInfo> infos;
    infos.reserve(k);
    for (const string& name : embed_names)
    {
        TiffInfo info = readTiff(inDir + name);
        if (info.data.empty())
        {
            cerr << "Error: embed image '" << name << "' has no data\n";
            return 1;
        }
        infos.push_back(info);
    }

    const size_t expected = infos[0].data.size();
    for (int i = 1; i < k; ++i)
    {
        if (infos[i].data.size() != expected)
        {
            cerr << "Error: size mismatch among embed images\n";
            return 1;
        }
    }

    vector<unsigned char> secret(expected);

    // --- parallel over byte positions ---
#pragma omp parallel
    {
        vector<unsigned char> ys(k);
        vector<unsigned char> coef(k);

        // reusable interpolation buffers
        vector<unsigned char> numer;
        vector<unsigned char> tmp;
        vector<unsigned char> factor;

#pragma omp for schedule(static)
        for (long long pos = 0; pos < static_cast<long long>(expected); ++pos)
        {
            for (int j = 0; j < k; ++j)
                ys[j] = infos[j].data[pos];

            lagrange_interpolate(xs, ys, coef, numer, tmp, factor);

            // secret is f(0) = a0
            secret[pos] = coef[0];
        }
    }

    TiffInfo outInfo = infos[0];
    outInfo.data = secret;

    const string outPath = outDir + output_name;
    writeTiff(outInfo, outPath);

    return 0;
}

// k_out_of_n: reorder parameters to (input_dir, secret_name, cover_names, output_dir, k, n, lsb)
static int k_out_of_n(const string& inputDir,
    const string& secret_name,
    const vector<string>& cover_names,
    const string& outputDir,
    int k,
    int n,
    int lsb,
    EmbedMode mode = EmbedMode::FullSearch,
    const AdaptiveSearchOptions& adaptiveOptions = AdaptiveSearchOptions{})
{
    if (k < 1 || k > n)
    {
        cerr << "Error: invalid k (must satisfy 1 <= k <= n)\n";
        return 1;
    }

    if (static_cast<int>(cover_names.size()) < n)
    {
        cerr << "Error: provided cover_names contains fewer than n entries\n";
        return 1;
    }

    vector<string> use_covers(cover_names.begin(), cover_names.begin() + n);

    int rc = embed(inputDir, use_covers, n, k, lsb, outputDir, secret_name, mode, adaptiveOptions);
    if (rc != 0) { cerr << "embed failed\n"; return rc; }

    // select first k embedded cover filenames for extraction
    vector<string> extract_names;
    extract_names.reserve(k);
    for (int i = 0; i < k; ++i)
    {
        string base = use_covers[i];
        size_t p = base.find_last_of('.');
        string base_no_ext = (p == string::npos) ? base : base.substr(0, p);
        // embedded filename now includes lsb and the selected embedding mode.
        string embedded_fn = embedded_prefix(mode) +
            to_string(lsb) + "_" + to_string(k) + "_" + to_string(n) + "_" + base_no_ext + ".tiff";
        extract_names.push_back(embedded_fn);
    }

    vector<unsigned char> xs;
    xs.reserve(k);
    for (int i = 1; i <= k; ++i) xs.push_back(static_cast<unsigned char>(i));

    size_t dot = secret_name.find_last_of('.');
    string secret_base = (dot == string::npos) ? secret_name : secret_name.substr(0, dot);
    string out_name = recovered_prefix(mode) +
        to_string(lsb) + "_" + to_string(k) + "_" + to_string(n) + "_" + secret_base + ".tiff";

    rc = extract(outputDir, extract_names, xs, outputDir, out_name);
    if (rc != 0) { cerr << "extract failed\n"; return rc; }

    TiffInfo orig = readTiff(inputDir + "/" + secret_name);
    TiffInfo rec = readTiff(outputDir + "/" + out_name);
    if (orig.data.empty() || rec.data.empty()) { cerr << "Error reading images for PSNR\n"; return 1; }
    if (orig.data.size() != rec.data.size()) { cerr << "Size mismatch for PSNR\n"; return 1; }
    double psnr_secret = compute_psnr(orig.data, rec.data);
    cout << fixed << setprecision(5);
    if (psnr_secret == INFINITY) cout << "PSNR(" << secret_name << " -> " << out_name << ") : inf\n";
    else if (psnr_secret < 0.0) cout << "PSNR(" << secret_name << " -> " << out_name << ") : error\n";
    else cout << "PSNR(" << secret_name << " -> " << out_name << ") : " << psnr_secret << " dB\n";

    return 0;
}

struct ExperimentSpec
{
    string group;
    int size = 512;
    int n = 4;
    int k = 4;
    int lsb = 1;
    EmbedMode mode = EmbedMode::FullSearch;
    AdaptiveSearchOptions adaptiveOptions;
};

static string csv_escape(const string& value)
{
    if (value.find_first_of(",\"\n") == string::npos)
        return value;

    string out = "\"";
    for (char ch : value)
    {
        if (ch == '"') out += "\"\"";
        else out += ch;
    }
    out += "\"";
    return out;
}

static string embedded_file_name(
    EmbedMode mode,
    int lsb,
    int k,
    int n,
    const string& coverName)
{
    return embedded_prefix(mode) + to_string(lsb) + "_" +
        to_string(k) + "_" + to_string(n) + "_" +
        strip_extension(coverName) + ".tiff";
}

static string recovered_file_name(
    EmbedMode mode,
    int lsb,
    int k,
    int n,
    const string& secretName)
{
    return recovered_prefix(mode) + to_string(lsb) + "_" +
        to_string(k) + "_" + to_string(n) + "_" +
        strip_extension(secretName) + ".tiff";
}

static double compute_ber(const Pixels& a, const Pixels& b)
{
    if (a.size() != b.size() || a.empty())
        return -1.0;

    long long bitErrors = 0;
    for (size_t i = 0; i < a.size(); ++i)
        bitErrors += popcount(static_cast<unsigned int>(a[i] ^ b[i]));

    return static_cast<double>(bitErrors) /
        static_cast<double>(a.size() * 8ULL);
}

static int run_experiment_case(
    int caseId,
    const ExperimentSpec& spec,
    const string& inputDir,
    const string& secretName,
    const vector<string>& coverNames,
    const string& outputDir,
    ofstream& csv)
{
    cout << "==== Experiment case " << caseId
        << " [" << spec.group << "]"
        << " mode=" << mode_name(spec.mode)
        << ", n=" << spec.n
        << ", k=" << spec.k
        << ", lsb=" << spec.lsb
        << " ====\n";

    const auto start = chrono::steady_clock::now();
    const int rc = k_out_of_n(
        inputDir,
        secretName,
        coverNames,
        outputDir,
        spec.k,
        spec.n,
        spec.lsb,
        spec.mode,
        spec.adaptiveOptions);
    const auto finish = chrono::steady_clock::now();

    const long long elapsedMs =
        chrono::duration_cast<chrono::milliseconds>(finish - start).count();

    if (rc != 0)
    {
        cerr << "Experiment case " << caseId << " failed\n";
        return rc;
    }

    double psnrSum = 0.0;
    double ssimSum = 0.0;
    double minPsnr = numeric_limits<double>::infinity();
    int metricCount = 0;

    for (int i = 0; i < spec.n; ++i)
    {
        const string coverPath = inputDir + "/" + coverNames[i];
        const string stegoPath = outputDir + "/" +
            embedded_file_name(spec.mode, spec.lsb, spec.k, spec.n, coverNames[i]);

        TiffInfo cover = readTiff(coverPath);
        TiffInfo stego = readTiff(stegoPath);
        if (cover.data.empty() || stego.data.empty())
        {
            cerr << "Error reading experiment output: " << stegoPath << "\n";
            return 1;
        }

        const double psnr = compute_psnr(cover.data, stego.data);
        const double ssim = compute_ssim(cover.data, stego.data);
        if (psnr < 0.0 || ssim < 0.0)
        {
            cerr << "Error computing experiment metrics for " << stegoPath << "\n";
            return 1;
        }

        psnrSum += psnr;
        ssimSum += ssim;
        minPsnr = min(minPsnr, psnr);
        ++metricCount;
    }

    const string recoveredPath = outputDir + "/" +
        recovered_file_name(spec.mode, spec.lsb, spec.k, spec.n, secretName);
    TiffInfo originalSecret = readTiff(inputDir + "/" + secretName);
    TiffInfo recoveredSecret = readTiff(recoveredPath);
    if (originalSecret.data.empty() || recoveredSecret.data.empty())
    {
        cerr << "Error reading recovered secret: " << recoveredPath << "\n";
        return 1;
    }

    const double secretPsnr = compute_psnr(originalSecret.data, recoveredSecret.data);
    const double secretBer = compute_ber(originalSecret.data, recoveredSecret.data);

    const double avgPsnr = metricCount == 0 ? -1.0 : psnrSum / metricCount;
    const double avgSsim = metricCount == 0 ? -1.0 : ssimSum / metricCount;

    csv << caseId << ","
        << csv_escape(spec.group) << ","
        << spec.size << ","
        << spec.n << ","
        << spec.k << ","
        << spec.lsb << ","
        << mode_name(spec.mode) << ","
        << spec.adaptiveOptions.minLsb << ","
        << spec.adaptiveOptions.beamWidth << ","
        << spec.adaptiveOptions.earlyStopSSE << ","
        << spec.adaptiveOptions.smoothThreshold << ","
        << spec.adaptiveOptions.texturedThreshold << ","
        << elapsedMs << ","
        << fixed << setprecision(6)
        << avgPsnr << ","
        << minPsnr << ","
        << avgSsim << ","
        << secretPsnr << ","
        << secretBer << "\n";

    csv.flush();
    return 0;
}

static vector<ExperimentSpec> build_paper_experiment_set()
{
    AdaptiveSearchOptions adaptiveFast;
    adaptiveFast.minLsb = 0;
    adaptiveFast.beamWidth = 2;
    adaptiveFast.earlyStopSSE = 0;
    adaptiveFast.smoothThreshold = 12.0;
    adaptiveFast.texturedThreshold = 32.0;

    AdaptiveSearchOptions adaptiveBalanced = adaptiveFast;
    adaptiveBalanced.beamWidth = 3;

    AdaptiveSearchOptions adaptiveWide = adaptiveFast;
    adaptiveWide.beamWidth = 0; // Ranked masks, but no beam truncation.

    vector<ExperimentSpec> cases;

    // Adaptive-only main comparison for the paper table. Compare these rows
    // against the full-search results already collected in earlier experiments.
    for (int k : { 3, 4 })
    {
        for (int lsb : { 1, 2, 3 })
        {
            cases.push_back(ExperimentSpec{ "main_adaptive_balanced", 512, 4, k, lsb, EmbedMode::AdaptiveSearch, adaptiveBalanced });
        }
    }

    // Beam-width ablation at the strongest common setting.
    for (const auto& item : {
        pair<string, AdaptiveSearchOptions>{ "adaptive_beam_2", adaptiveFast },
        pair<string, AdaptiveSearchOptions>{ "adaptive_beam_3", adaptiveBalanced },
        pair<string, AdaptiveSearchOptions>{ "adaptive_ranked_all_masks", adaptiveWide } })
    {
        cases.push_back(ExperimentSpec{ item.first, 512, 4, 4, 3, EmbedMode::AdaptiveSearch, item.second });
    }

    // Scalability checks for larger share sets.
    cases.push_back(ExperimentSpec{ "n5_adaptive_balanced", 512, 5, 4, 1, EmbedMode::AdaptiveSearch, adaptiveBalanced });
    cases.push_back(ExperimentSpec{ "n5_k5_adaptive_balanced", 512, 5, 5, 1, EmbedMode::AdaptiveSearch, adaptiveBalanced });

    // Optional full-search references. Keep these disabled unless a fresh
    // baseline run is needed.
    //cases.push_back(ExperimentSpec{ "main_full", 512, 4, 3, 1, EmbedMode::FullSearch, AdaptiveSearchOptions{} });
    //cases.push_back(ExperimentSpec{ "main_full", 512, 4, 4, 1, EmbedMode::FullSearch, AdaptiveSearchOptions{} });
    //cases.push_back(ExperimentSpec{ "n5_full_reference", 512, 5, 4, 1, EmbedMode::FullSearch, AdaptiveSearchOptions{} });

    return cases;
}

static int run_paper_experiments()
{
    const int size = 512;
    const string sizeStr = to_string(size);
    const string inputDir = "../../images/USC-SIPI/" + sizeStr;
    const string outputDir = "../out/" + sizeStr + "/experiments";
    const string secretName = "woman.tiff";
    const vector<string> coverNames = {
        "jetplane.tiff",
        "house.tiff",
        "girl.tiff",
        "baboon.tiff",
        "airplane.tiff",
        "boat.tiff",
        "bridge.tiff",
        "couple.tiff",
        "car.tiff",
        "lake.tiff",
        "milkdrop.tiff",
        "peppers.tiff",
        "tiffany.tiff"
    };

    filesystem::create_directories(outputDir);

    const string csvPath = outputDir + "/adaptive_complexity_experiments.csv";
    ofstream csv(csvPath, ios::out | ios::trunc);
    if (!csv)
    {
        cerr << "Error opening CSV output: " << csvPath << "\n";
        return 1;
    }

    csv << "case_id,group,size,n,k,lsb,mode,min_lsb,beam_width,early_stop_sse,"
        << "smooth_threshold,textured_threshold,elapsed_ms,avg_cover_psnr,"
        << "min_cover_psnr,avg_cover_ssim,secret_psnr,secret_ber\n";

    vector<ExperimentSpec> cases = build_paper_experiment_set();
    for (size_t i = 0; i < cases.size(); ++i)
    {
        int rc = run_experiment_case(
            static_cast<int>(i + 1),
            cases[i],
            inputDir,
            secretName,
            coverNames,
            outputDir,
            csv);
        if (rc != 0)
            return rc;
    }

    cout << "Experiment CSV written to: " << csvPath << "\n";
    return 0;
}

int run(int n, int size,
    EmbedMode mode = EmbedMode::FullSearch,
    const AdaptiveSearchOptions& adaptiveOptions = AdaptiveSearchOptions{})
{
	assert(n >= 1 && n <= 6);
	assert(size == 128 || size == 256 || size == 512 || size == 1024);

	cout << "==== Running n = " << n
        << ", size = " << size
        << ", mode = " << mode_name(mode)
        << " ====\n";

	const string size_str = to_string(size);
    const string inputDir = "../../images/USC-SIPI/" + size_str;
    const string outputDir = "../out/" + size_str;
    const string secret_name = "woman.tiff";
    const vector<string> cover_names = {
        "jetplane.tiff",
        "house.tiff",
        "girl.tiff",
        "baboon.tiff",
        "airplane.tiff",
        "boat.tiff",
        "bridge.tiff",
        "couple.tiff",
        "car.tiff",
        "lake.tiff",
        //"lena.tiff", 
        "milkdrop.tiff",
        "peppers.tiff",
        "tiffany.tiff"
    };


    // perform the full task for lsb = 1, 2, 3
    const vector<int> lsbs = { 0, 1, 2, 3 };

    for (int lsb : lsbs)
    {
        cout << "==== Running all k for lsb=" << lsb << " ====\n";
        for (int k = 1; k <= n; ++k)
        {
            cout << "==== Running k = " << k << " (lsb=" << lsb << ") ====\n";
            int rc = k_out_of_n(inputDir, secret_name, cover_names, outputDir, k, n, lsb, mode, adaptiveOptions);
            if (rc != 0) return rc;

#if 1
            for (int i = 0; i < n; ++i)
            {
                string base = cover_names[i];
                size_t p = base.find_last_of('.');
                string base_no_ext = (p == string::npos) ? base : base.substr(0, p);
                string embedded_fn = embedded_prefix(mode) +
                    to_string(lsb) + "_" + to_string(k) + "_" + to_string(n) + "_" + base_no_ext + ".tiff";
#if 0
                string diff_fn = "diff0_" + embedded_fn;
                string diff_inv_fn = "diff_" + embedded_fn;
                string hist_fn = "hist_" + embedded_fn;
                string hist_diff_fn = "hist_diff_" + embedded_fn;

                diff(inputDir + "/" + cover_names[i], outputDir + "/" + embedded_fn, outputDir + "/" + diff_fn, 20);
                diff_inv(inputDir + "/" + cover_names[i], outputDir + "/" + embedded_fn, outputDir + "/" + diff_inv_fn, 20);
                histogram(outputDir + "/" + embedded_fn, outputDir + "/" + hist_fn);
                //histogram(outputDir + "/" + diff_fn, outputDir + "/" + hist_diff_fn);
				diff_histogram(inputDir + "/" + cover_names[i], outputDir + "/" + embedded_fn, outputDir + "/" + hist_diff_fn);
#endif
            }
#endif
        }
    }

    return 0;
}

int main()
{
    AdaptiveSearchOptions adaptiveOptions;
    adaptiveOptions.minLsb = 0;
    adaptiveOptions.beamWidth = 3;
    adaptiveOptions.earlyStopSSE = 0;
    adaptiveOptions.smoothThreshold = 12.0;
    adaptiveOptions.texturedThreshold = 32.0;

    //run(5, 128);
    //run(4, 256);
	//run(5, 256);
    //run(4, 512);
	//run(5, 512);
    //run(4, 512, EmbedMode::AdaptiveSearch, adaptiveOptions);
    //run(5, 512, EmbedMode::AdaptiveSearch, adaptiveOptions);
    run_paper_experiments();
    //run(5, 1024);

	return 0;
}
