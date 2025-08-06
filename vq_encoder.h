#pragma once

#include "vq_bcn_types.h"
#include "bcn_compressor.h"
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>
#include <execution>
#include <numeric>
#include <immintrin.h> // For SIMD intrinsics (AVX2)

// A block is 16 pixels (4x4) of 4 floats (L, a, b, alpha)
using CielabBlock = std::vector<float>;

namespace ColorLuts {
    // OPTIMIZATION: Pre-calculate lookup tables for expensive color conversions.
    static const std::array<float, 256> sRGB_to_Linear = [] {
        std::array<float, 256> table{};
        for (int i = 0; i < 256; ++i) {
            float v = i / 255.0f;
            table[i] = (v > 0.04045f) ? powf((v + 0.055f) / 1.055f, 2.4f) : v / 12.92f;
        }
        return table;
        }();

    static const std::array<uint8_t, 4096> Linear_to_sRGB = [] {
        std::array<uint8_t, 4096> table{};
        for (int i = 0; i < 4096; ++i) {
            float v = i / 4095.0f;
            v = (v > 0.0031308f) ? (1.055f * powf(v, 1.0f / 2.4f) - 0.055f) : (v * 12.92f);
            table[i] = static_cast<uint8_t>(std::clamp(v * 255.0f, 0.0f, 255.0f));
        }
        return table;
        }();
}


class VQEncoder {
public:
    enum class DistanceMetric {
        RGB_SIMD,      // Fastest: Euclidean distance on RGB values, accelerated with AVX2.
        PERCEPTUAL_LAB // High Quality: Euclidean distance in CIELAB color space, now with LUTs.
    };

    struct Config {
        uint32_t codebookSize = 256;
        uint32_t maxIterations = 20;
        DistanceMetric metric = DistanceMetric::PERCEPTUAL_LAB;
    };

private:
    Config config;
    BCFormat bcFormat;
    BCnCompressor bcnCompressor;
    std::mt19937 rng;

    // --- Color Space Conversion ---

    // OPTIMIZATION: Use lookup tables to accelerate sRGB -> Linear conversion
    inline void RgbToCielab(const uint8_t* rgb, float* lab) const {
        float r_lin = ColorLuts::sRGB_to_Linear[rgb[0]];
        float g_lin = ColorLuts::sRGB_to_Linear[rgb[1]];
        float b_lin = ColorLuts::sRGB_to_Linear[rgb[2]];

        float x = r_lin * 0.4124f + g_lin * 0.3576f + b_lin * 0.1805f;
        float y = r_lin * 0.2126f + g_lin * 0.7152f + b_lin * 0.0722f;
        float z = r_lin * 0.0193f + g_lin * 0.1192f + b_lin * 0.9505f;

        x /= 0.95047f; y /= 1.00000f; z /= 1.08883f;

        auto f = [](float t) { return (t > 0.008856f) ? cbrtf(t) : (7.787f * t + 16.0f / 116.0f); };
        float fx = f(x); float fy = f(y); float fz = f(z);

        lab[0] = (116.0f * fy) - 16.0f;
        lab[1] = 500.0f * (fx - fy);
        lab[2] = 200.0f * (fy - fz);
    }

    // BUGFIX: Added Cielab -> RGBA conversion to correctly save refined centroids.
    inline void CielabToRgb(const float* lab, uint8_t* rgb) const {
        float y = (lab[0] + 16.0f) / 116.0f;
        float x = lab[1] / 500.0f + y;
        float z = y - lab[2] / 200.0f;

        auto f_inv = [](float t) { return (t * t * t > 0.008856f) ? (t * t * t) : ((t - 16.0f / 116.0f) / 7.787f); };
        x = f_inv(x) * 0.95047f;
        y = f_inv(y) * 1.00000f;
        z = f_inv(z) * 1.08883f;

        float r_lin = x * 3.2406f + y * -1.5372f + z * -0.4986f;
        float g_lin = x * -0.9689f + y * 1.8758f + z * 0.0415f;
        float b_lin = x * 0.0557f + y * -0.2040f + z * 1.0570f;

        // OPTIMIZATION: Use a lookup table for linear to sRGB conversion.
        rgb[0] = ColorLuts::Linear_to_sRGB[static_cast<int>(std::clamp(r_lin, 0.0f, 1.0f) * 4095.0f)];
        rgb[1] = ColorLuts::Linear_to_sRGB[static_cast<int>(std::clamp(g_lin, 0.0f, 1.0f) * 4095.0f)];
        rgb[2] = ColorLuts::Linear_to_sRGB[static_cast<int>(std::clamp(b_lin, 0.0f, 1.0f) * 4095.0f)];
    }

    CielabBlock RgbaBlockToCielabBlock(const std::vector<uint8_t>& rgbaBlock) const {
        CielabBlock labBlock(16 * 4);
        for (size_t i = 0; i < 16; ++i) {
            RgbToCielab(&rgbaBlock[i * 4], &labBlock[i * 4]);
            labBlock[i * 4 + 3] = rgbaBlock[i * 4 + 3];
        }
        return labBlock;
    }

    std::vector<uint8_t> CielabBlockToRgbaBlock(const CielabBlock& labBlock) const {
        std::vector<uint8_t> rgbaBlock(16 * 4);
        for (size_t i = 0; i < 16; ++i) {
            CielabToRgb(&labBlock[i * 4], &rgbaBlock[i * 4]);
            rgbaBlock[i * 4 + 3] = static_cast<uint8_t>(labBlock[i * 4 + 3]);
        }
        return rgbaBlock;
    }

    // --- Distance Functions ---
    float RgbaBlockDistance_SIMD(const uint8_t* rgbaA, const uint8_t* rgbaB) const {
        __m256i diff_sum = _mm256_setzero_si256();
        __m256i a1 = _mm256_loadu_si256((__m256i*)(rgbaA));
        __m256i b1 = _mm256_loadu_si256((__m256i*)(rgbaB));
        diff_sum = _mm256_add_epi64(diff_sum, _mm256_sad_epu8(a1, b1));
        __m256i a2 = _mm256_loadu_si256((__m256i*)(rgbaA + 32));
        __m256i b2 = _mm256_loadu_si256((__m256i*)(rgbaB + 32));
        diff_sum = _mm256_add_epi64(diff_sum, _mm256_sad_epu8(a2, b2));
        return (float)(_mm256_extract_epi64(diff_sum, 0) + _mm256_extract_epi64(diff_sum, 2));
    }

    float CielabBlockDistance_SIMD(const CielabBlock& labA, const CielabBlock& labB) const {
        __m256 sum_sq_diff = _mm256_setzero_ps();
        for (size_t i = 0; i < 64; i += 8) {
            __m256 a = _mm256_loadu_ps(labA.data() + i);
            __m256 b = _mm256_loadu_ps(labB.data() + i);
            __m256 diff = _mm256_sub_ps(a, b);
            sum_sq_diff = _mm256_fmadd_ps(diff, diff, sum_sq_diff);
        }
        __m128 lo_half = _mm256_castps256_ps128(sum_sq_diff);
        __m128 hi_half = _mm256_extractf128_ps(sum_sq_diff, 1);
        __m128 sum_128 = _mm_add_ps(lo_half, hi_half);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        return _mm_cvtss_f32(sum_128);
    }

    // --- Helper Functions ---
    std::vector<uint8_t> CompressSingleBlock(const uint8_t* rgbaBlock) {
        return bcnCompressor.CompressRGBA(rgbaBlock, 4, 4, bcFormat, 1.0f);
    }

public:
    VQEncoder(const Config& cfg = Config())
        : config(cfg), rng(std::random_device{}()) {
    }

    void SetFormat(BCFormat format) { bcFormat = format; }

    VQCodebook BuildCodebook(
        const std::vector<std::vector<uint8_t>>& rgbaBlocks,
        std::vector<std::vector<uint8_t>>& outRgbaCentroids
    );

    std::vector<uint32_t> QuantizeBlocks(
        const std::vector<std::vector<uint8_t>>& rgbaBlocks,
        const std::vector<std::vector<uint8_t>>& rgbaCentroids
    );
};


// --- VQEncoder Method Implementations ---

inline VQCodebook VQEncoder::BuildCodebook(const std::vector<std::vector<uint8_t>>& rgbaBlocks, std::vector<std::vector<uint8_t>>& outRgbaCentroids) {
    size_t numBlocks = rgbaBlocks.size();
    if (numBlocks == 0) return VQCodebook(BCBlockSize::GetSize(bcFormat), 0);
    if (numBlocks < config.codebookSize) {
        config.codebookSize = static_cast<uint32_t>(numBlocks);
    }

    // 1. K-Means++ Initialization (always done in RGB space for speed)
    std::vector<std::vector<uint8_t>> rgbaCentroids(config.codebookSize);
    std::vector<float> minDistSq(numBlocks, std::numeric_limits<float>::max());
    std::uniform_int_distribution<size_t> distrib(0, numBlocks - 1);
    rgbaCentroids[0] = rgbaBlocks[distrib(rng)];
    for (uint32_t i = 1; i < config.codebookSize; ++i) {
        double current_sum = 0.0;
#pragma omp parallel for reduction(+:current_sum)
        for (int64_t j = 0; j < numBlocks; ++j) {
            float d = RgbaBlockDistance_SIMD(rgbaBlocks[j].data(), rgbaCentroids[i - 1].data());
            minDistSq[j] = std::min(d * d, minDistSq[j]);
            current_sum += minDistSq[j];
        }
        if (current_sum <= 0) {
            for (uint32_t k = i; k < config.codebookSize; ++k) rgbaCentroids[k] = rgbaCentroids[0];
            break;
        }
        std::uniform_real_distribution<double> p_distrib(0.0, current_sum);
        double p = p_distrib(rng);
        double cumulative_p = 0.0;
        for (size_t j = 0; j < numBlocks; ++j) {
            cumulative_p += minDistSq[j];
            if (cumulative_p >= p) {
                rgbaCentroids[i] = rgbaBlocks[j];
                break;
            }
        }
    }

    // 2. K-Means Iterations
    std::vector<uint32_t> assignments(numBlocks, 0);

    if (config.metric == DistanceMetric::PERCEPTUAL_LAB) {
        std::vector<CielabBlock> labBlocks(numBlocks);
        std::vector<CielabBlock> labCentroids(config.codebookSize);
#pragma omp parallel for
        for (int64_t i = 0; i < numBlocks; ++i) labBlocks[i] = RgbaBlockToCielabBlock(rgbaBlocks[i]);
#pragma omp parallel for
        for (int64_t i = 0; i < config.codebookSize; ++i) labCentroids[i] = RgbaBlockToCielabBlock(rgbaCentroids[i]);

        for (uint32_t iter = 0; iter < config.maxIterations; ++iter) {
            std::atomic<bool> hasChanged = false;
#pragma omp parallel for
            for (int64_t i = 0; i < numBlocks; ++i) {
                float min_d = std::numeric_limits<float>::max();
                uint32_t best_c = 0;
                for (uint32_t c = 0; c < config.codebookSize; ++c) {
                    float d = CielabBlockDistance_SIMD(labBlocks[i], labCentroids[c]);
                    if (d < min_d) { min_d = d; best_c = c; }
                }
                if (assignments[i] != best_c) { assignments[i] = best_c; hasChanged = true; }
            }
            if (!hasChanged) break;

            std::vector<CielabBlock> newCentroids(config.codebookSize, CielabBlock(64, 0.0f));
            std::vector<uint32_t> counts(config.codebookSize, 0);
            for (size_t i = 0; i < numBlocks; ++i) {
                uint32_t c_idx = assignments[i];
                counts[c_idx]++;
                for (size_t j = 0; j < 64; ++j) newCentroids[c_idx][j] += labBlocks[i][j];
            }
#pragma omp parallel for
            for (int64_t c = 0; c < config.codebookSize; ++c) {
                if (counts[c] > 0) {
                    float inv_count = 1.0f / counts[c];
                    for (size_t j = 0; j < 64; ++j) labCentroids[c][j] = newCentroids[c][j] * inv_count;
                }
            }
        }
        // BUGFIX: Convert the final, refined LAB centroids back to RGBA for output.
#pragma omp parallel for
        for (int64_t i = 0; i < config.codebookSize; ++i) {
            rgbaCentroids[i] = CielabBlockToRgbaBlock(labCentroids[i]);
        }
    }
    else { // RGB_SIMD path
        for (uint32_t iter = 0; iter < config.maxIterations; ++iter) {
            std::atomic<bool> hasChanged = false;
#pragma omp parallel for
            for (int64_t i = 0; i < numBlocks; ++i) {
                float min_d = std::numeric_limits<float>::max();
                uint32_t best_c = 0;
                for (uint32_t c = 0; c < config.codebookSize; ++c) {
                    float d = RgbaBlockDistance_SIMD(rgbaBlocks[i].data(), rgbaCentroids[c].data());
                    if (d < min_d) { min_d = d; best_c = c; }
                }
                if (assignments[i] != best_c) { assignments[i] = best_c; hasChanged = true; }
            }
            if (!hasChanged) break;

            std::vector<std::vector<uint64_t>> newCentroids(config.codebookSize, std::vector<uint64_t>(64, 0));
            std::vector<uint32_t> counts(config.codebookSize, 0);
            for (size_t i = 0; i < numBlocks; ++i) {
                uint32_t c_idx = assignments[i];
                counts[c_idx]++;
                for (size_t j = 0; j < 64; ++j) newCentroids[c_idx][j] += rgbaBlocks[i][j];
            }
#pragma omp parallel for
            for (int64_t c = 0; c < config.codebookSize; ++c) {
                if (counts[c] > 0) {
                    for (size_t j = 0; j < 64; ++j) rgbaCentroids[c][j] = static_cast<uint8_t>(newCentroids[c][j] / counts[c]);
                }
            }
        }
    }

    outRgbaCentroids = rgbaCentroids;
    VQCodebook finalCodebook(BCBlockSize::GetSize(bcFormat), config.codebookSize);
    finalCodebook.entries.resize(config.codebookSize);
#pragma omp parallel for
    for (int64_t i = 0; i < config.codebookSize; ++i) {
        finalCodebook.entries[i] = CompressSingleBlock(rgbaCentroids[i].data());
    }
    return finalCodebook;
}

inline std::vector<uint32_t> VQEncoder::QuantizeBlocks(const std::vector<std::vector<uint8_t>>& rgbaBlocks, const std::vector<std::vector<uint8_t>>& rgbaCentroids) {
    size_t numBlocks = rgbaBlocks.size();
    if (numBlocks == 0) return {};
    std::vector<uint32_t> indices(numBlocks);
    uint32_t codebookSize = static_cast<uint32_t>(rgbaCentroids.size());

#pragma omp parallel for
    for (int64_t i = 0; i < numBlocks; ++i) {
        float minDist = std::numeric_limits<float>::max();
        uint32_t bestIdx = 0;
        for (uint32_t j = 0; j < codebookSize; ++j) {
            float dist = RgbaBlockDistance_SIMD(rgbaBlocks[i].data(), rgbaCentroids[j].data());
            if (dist < minDist) {
                minDist = dist;
                bestIdx = j;
            }
        }
        indices[i] = bestIdx;
    }
    return indices;
}