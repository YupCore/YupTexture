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
#include <atomic>
#include <vector>
#include <stdexcept> // For std::runtime_error
#include "oklab_math.h"

// A block is 16 pixels (4x4) of 4 floats (L, a, b, alpha)
using CielabBlock = std::vector<float>;
// A sub-vector for Product Quantization
using SubVector = std::vector<float>;

namespace ColorLuts {
    // Pre-calculate lookup tables for expensive color conversions.
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

// Structure for the Product Quantization codebook
struct ProductCodebook {
    // A vector of codebooks, one for each sub-vector space.
    // Each sub-codebook is a vector of centroid SubVectors.
    std::vector<std::vector<SubVector>> sub_codebooks;
    uint32_t num_subvectors = 8;
    uint32_t sub_codebook_size = 256;
};


class VQEncoder {
public:
    enum class DistanceMetric {
        RGB_SIMD,       // Fastest: SAD on RGB values, accelerated with AVX2.
        PERCEPTUAL_LAB  // High Quality: Euclidean distance in CIELAB color space.
    };

    struct Config {
    public:
        uint32_t maxIterations = 32;
        DistanceMetric metric = DistanceMetric::PERCEPTUAL_LAB;

        // --- Product Quantization (PQ) Settings ---
        bool use_product_quantization = true;
        uint32_t pq_subvectors = 8; // Split each 64-dim block into 8 sub-vectors
        uint32_t pq_sub_codebook_size = 256; // 256 centroids for each sub-vector

        // --- General Settings ---
        float fastModeSampleRatio = 1.0f;
        uint32_t min_cb_power = 4; // 2^4 = 16 entries at quality=0
        uint32_t max_cb_power = 10; // 2^10 = 1024 entries at quality=1

        Config(float quality_level = 0.5f) {
            SetQuality(quality_level);
        }

        void SetQuality(float quality_level) {
            quality = std::clamp(quality_level, 0.0f, 1.0f);

            // Map quality non-linearly to the power of the codebook size
            uint32_t power = min_cb_power + static_cast<uint32_t>(roundf(quality * (max_cb_power - min_cb_power)));
            codebookSize = 1 << power; // 2^power
        }

        // Quality meter (0.0 = low, 1.0 = high) drives other settings.
        float quality = 0.5f;
        // --- Standard VQ Settings ---
        uint32_t codebookSize;

    };

private:
    Config config;
    BCFormat bcFormat;
    BCnCompressor bcnCompressor;
    std::mt19937 rng;

    // --- Color Space Conversion ---
    inline void RgbToCielab(const uint8_t* rgb, float* lab) const;
    inline void CielabToRgb(const float* lab, uint8_t* rgb) const;
    CielabBlock RgbaBlockToCielabBlock(const std::vector<uint8_t>& rgbaBlock) const;
    std::vector<uint8_t> CielabBlockToRgbaBlock(const CielabBlock& labBlock) const;

    // --- sRGB color space math ---

    // --- Distance Functions ---
    float RgbaBlockDistanceSAD_SIMD(const uint8_t* rgbaA, const uint8_t* rgbaB) const {
        // Sum of Absolute Differences (SAD) is a good proxy for L1 distance and very fast.
        __m256i diff_sum = _mm256_setzero_si256();

        // Process 64 bytes in two 32-byte chunks
        __m256i a1 = _mm256_loadu_si256((__m256i*)(rgbaA));
        __m256i b1 = _mm256_loadu_si256((__m256i*)(rgbaB));
        diff_sum = _mm256_add_epi64(diff_sum, _mm256_sad_epu8(a1, b1));

        __m256i a2 = _mm256_loadu_si256((__m256i*)(rgbaA + 32));
        __m256i b2 = _mm256_loadu_si256((__m256i*)(rgbaB + 32));
        diff_sum = _mm256_add_epi64(diff_sum, _mm256_sad_epu8(a2, b2));

        // The result is stored in two 64-bit lanes, sum them up.
        return (float)(_mm256_extract_epi64(diff_sum, 0) + _mm256_extract_epi64(diff_sum, 2));
    }

    // Squared Euclidean distance for CIELAB blocks
    float CielabBlockDistanceSq_SIMD(const CielabBlock& labA, const CielabBlock& labB) const {
        __m256 sum_sq_diff = _mm256_setzero_ps();
        for (size_t i = 0; i < 64; i += 8) {
            __m256 a = _mm256_loadu_ps(labA.data() + i);
            __m256 b = _mm256_loadu_ps(labB.data() + i);
            __m256 diff = _mm256_sub_ps(a, b);
            sum_sq_diff = _mm256_fmadd_ps(diff, diff, sum_sq_diff);
        }
        // Horizontal sum of the 8 floats in the accumulator
        __m128 lo_half = _mm256_castps256_ps128(sum_sq_diff);
        __m128 hi_half = _mm256_extractf128_ps(sum_sq_diff, 1);
        __m128 sum_128 = _mm_add_ps(lo_half, hi_half);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        return _mm_cvtss_f32(sum_128);
    }

    // Distance function for Product Quantization sub-vectors
    float SubVectorDistanceSq_SIMD(const SubVector& a, const SubVector& b) const {
        // Assuming sub-vector size is a multiple of 8 for optimal AVX2 usage
        __m256 sum_sq_diff = _mm256_setzero_ps();
        size_t size = a.size();
        for (size_t i = 0; i < size; i += 8) {
            __m256 va = _mm256_loadu_ps(a.data() + i);
            __m256 vb = _mm256_loadu_ps(b.data() + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum_sq_diff = _mm256_fmadd_ps(diff, diff, sum_sq_diff);
        }
        __m128 lo_half = _mm256_castps256_ps128(sum_sq_diff);
        __m128 hi_half = _mm256_extractf128_ps(sum_sq_diff, 1);
        __m128 sum_128 = _mm_add_ps(lo_half, hi_half);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        return _mm_cvtss_f32(sum_128);
    }

    std::vector<uint8_t> CompressSingleBlock(const uint8_t* rgbaBlock, uint8_t channels, uint8_t alphaThreshold = 128);

	// HDR oklab color space math

    // --- OPTIMIZATION: HELPER FUNCTIONS ---
    // These helpers convert an Oklab block's L-channel to/from a perceptual space.
    // This is now done once per block, instead of repeatedly in the distance function.
    inline void OklabBlockToPerceptual(OklabBlock& block) const {
        for (size_t i = 0; i < 16; ++i) {
            block[i * 4 + 0] = Oklab::L_to_perceptual(block[i * 4 + 0]);
        }
    }

    inline void PerceptualBlockToOklab(OklabBlock& block) const {
        for (size_t i = 0; i < 16; ++i) {
            block[i * 4 + 0] = Oklab::perceptual_to_L(block[i * 4 + 0]);
        }
    }

    // --- OPTIMIZATION: DISTANCE FUNCTION ---
    // The distance function now operates on pre-transformed "perceptual" Oklab data.
    // It is much faster as it no longer contains expensive math operations.
    float OklabBlockDistanceSq_SIMD(const OklabBlock& labA, const OklabBlock& labB) const {
        __m256 sum_sq_diff = _mm256_setzero_ps();

        // Weight for the difference. L-channel's squared difference will be weighted by 4.0.
        // This is a tunable parameter for quality.
        const __m256 weight = _mm256_set_ps(1.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 1.0f, 2.0f);

        for (size_t i = 0; i < 64; i += 8) {
            __m256 a = _mm256_loadu_ps(&labA[i]);
            __m256 b = _mm256_loadu_ps(&labB[i]);

            __m256 diff = _mm256_sub_ps(a, b);
            __m256 weighted_diff = _mm256_mul_ps(diff, weight);

            // Fused multiply-add for: sum_sq_diff += weighted_diff * weighted_diff
            sum_sq_diff = _mm256_fmadd_ps(weighted_diff, weighted_diff, sum_sq_diff);
        }

        // Horizontal sum of the 8 floats in the accumulator
        __m128 lo_half = _mm256_castps256_ps128(sum_sq_diff);
        __m128 hi_half = _mm256_extractf128_ps(sum_sq_diff, 1);
        __m128 sum_128 = _mm_add_ps(lo_half, hi_half);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        return _mm_cvtss_f32(sum_128);
    }

    std::vector<uint8_t> CompressSingleBlockHDR(const std::vector<float>& rgbaBlock);

public:
    VQEncoder(const Config& cfg = Config())
        : config(cfg), rng(std::random_device{}()) {
    }

    void SetFormat(BCFormat format) { bcFormat = format; }

    // --- Public API ---
    VQCodebook BuildCodebook(
        const std::vector<std::vector<uint8_t>>& rgbaBlocks,
        uint8_t channels,
        std::vector<std::vector<uint8_t>>& outRgbaCentroids,
        uint8_t alphaThreshold = 128
    );

    std::vector<uint32_t> QuantizeBlocks(
        const std::vector<std::vector<uint8_t>>& rgbaBlocks,
        const std::vector<std::vector<uint8_t>>& rgbaCentroids
    );

    // Product Quantization Methods
    ProductCodebook BuildProductCodebook(
        const std::vector<std::vector<uint8_t>>& rgbaBlocks
    );

    std::vector<std::vector<uint8_t>> QuantizeProductBlocks(
        const std::vector<std::vector<uint8_t>>& rgbaBlocks,
        const ProductCodebook& pq_codebook
    );

    // --- ADDED: Public API for HDR VQ ---
    VQCodebook BuildCodebookHDR(
        const std::vector<std::vector<float>>& rgbaFloatBlocks,
        std::vector<std::vector<float>>& outRgbaCentroids
    );

    std::vector<uint32_t> QuantizeBlocksHDR(
        const std::vector<std::vector<float>>& rgbaFloatBlocks,
        const std::vector<std::vector<float>>& rgbaCentroids
    );
    ProductCodebook BuildProductCodebookHDR(const std::vector<std::vector<float>>& rgbaFloatBlocks);
    std::vector<std::vector<uint8_t>> QuantizeProductBlocksHDR(const std::vector<std::vector<float>>& rgbaFloatBlocks, const ProductCodebook& pq_codebook);
};


// --- Method Implementations ---

inline void VQEncoder::RgbToCielab(const uint8_t* rgb, float* lab) const {
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

inline void VQEncoder::CielabToRgb(const float* lab, uint8_t* rgb) const {
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

    rgb[0] = ColorLuts::Linear_to_sRGB[static_cast<int>(std::clamp(r_lin, 0.0f, 1.0f) * 4095.0f)];
    rgb[1] = ColorLuts::Linear_to_sRGB[static_cast<int>(std::clamp(g_lin, 0.0f, 1.0f) * 4095.0f)];
    rgb[2] = ColorLuts::Linear_to_sRGB[static_cast<int>(std::clamp(b_lin, 0.0f, 1.0f) * 4095.0f)];
}

inline CielabBlock VQEncoder::RgbaBlockToCielabBlock(const std::vector<uint8_t>& rgbaBlock) const {
    CielabBlock labBlock(16 * 4);
    for (size_t i = 0; i < 16; ++i) {
        RgbToCielab(&rgbaBlock[i * 4], &labBlock[i * 4]);
        labBlock[i * 4 + 3] = static_cast<float>(rgbaBlock[i * 4 + 3]); // Alpha
    }
    return labBlock;
}

inline std::vector<uint8_t> VQEncoder::CielabBlockToRgbaBlock(const CielabBlock& labBlock) const {
    std::vector<uint8_t> rgbaBlock(16 * 4);
    for (size_t i = 0; i < 16; ++i) {
        CielabToRgb(&labBlock[i * 4], &rgbaBlock[i * 4]);
        rgbaBlock[i * 4 + 3] = static_cast<uint8_t>(std::clamp(labBlock[i * 4 + 3], 0.0f, 255.0f)); // Alpha
    }
    return rgbaBlock;
}

inline std::vector<uint8_t> VQEncoder::CompressSingleBlock(const uint8_t* rgbaBlock, uint8_t channels, uint8_t alphaThreshold) {
    return bcnCompressor.CompressRGBA(rgbaBlock, 4, 4, channels, bcFormat, 1.0f, alphaThreshold);
}

inline VQCodebook VQEncoder::BuildCodebook(const std::vector<std::vector<uint8_t>>& allRgbaBlocks, uint8_t channels, std::vector<std::vector<uint8_t>>& outRgbaCentroids, uint8_t alphaThreshold) {
    // 1. Sampling
    std::vector<const std::vector<uint8_t>*> sampledBlocksPtrs;
    if (config.fastModeSampleRatio < 1.0f && config.fastModeSampleRatio > 0.0f) {
        size_t numToSample = static_cast<size_t>(allRgbaBlocks.size() * config.fastModeSampleRatio);
        numToSample = std::max(static_cast<size_t>(config.codebookSize), numToSample);
        sampledBlocksPtrs.reserve(numToSample);

        std::vector<size_t> indices(allRgbaBlocks.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t i = 0; i < numToSample && i < allRgbaBlocks.size(); ++i) {
            sampledBlocksPtrs.push_back(&allRgbaBlocks[indices[i]]);
        }
    }
    else {
        sampledBlocksPtrs.reserve(allRgbaBlocks.size());
        for (const auto& block : allRgbaBlocks) {
            sampledBlocksPtrs.push_back(&block);
        }
    }
    const auto& blocksToProcess = sampledBlocksPtrs;
    size_t numBlocks = blocksToProcess.size();

    // 2. K-Means++ Initialization (always in RGB space for speed)
    std::vector<std::vector<uint8_t>> rgbaCentroids(config.codebookSize);
    std::vector<float> minDistSq(numBlocks, std::numeric_limits<float>::max());
    std::uniform_int_distribution<size_t> distrib(0, numBlocks - 1);
    rgbaCentroids[0] = *blocksToProcess[distrib(rng)];
    for (uint32_t i = 1; i < config.codebookSize; ++i) {
        double current_sum = 0.0;
#pragma omp parallel for reduction(+:current_sum)
        for (int64_t j = 0; j < numBlocks; ++j) {
            float d = RgbaBlockDistanceSAD_SIMD((*blocksToProcess[j]).data(), rgbaCentroids[i - 1].data());
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
                rgbaCentroids[i] = *blocksToProcess[j];
                break;
            }
        }
    }

    // 3. K-Means Iterations
    std::vector<uint32_t> assignments(numBlocks, 0);
    std::vector<float> errors(numBlocks);

    if (config.metric == DistanceMetric::PERCEPTUAL_LAB) {
        std::vector<CielabBlock> labBlocks(numBlocks);
#pragma omp parallel for
        for (int64_t i = 0; i < numBlocks; ++i) labBlocks[i] = RgbaBlockToCielabBlock(*blocksToProcess[i]);

        std::vector<CielabBlock> labCentroids(config.codebookSize);
#pragma omp parallel for
        for (int64_t i = 0; i < config.codebookSize; ++i) labCentroids[i] = RgbaBlockToCielabBlock(rgbaCentroids[i]);

        for (uint32_t iter = 0; iter < config.maxIterations; ++iter) {
            std::atomic<bool> hasChanged = false;
#pragma omp parallel for
            for (int64_t i = 0; i < numBlocks; ++i) {
                float min_d = std::numeric_limits<float>::max();
                uint32_t best_c = 0;
                for (uint32_t c = 0; c < config.codebookSize; ++c) {
                    float d = CielabBlockDistanceSq_SIMD(labBlocks[i], labCentroids[c]);
                    if (d < min_d) { min_d = d; best_c = c; }
                }
                errors[i] = min_d;
                if (assignments[i] != best_c) { assignments[i] = best_c; hasChanged = true; }
            }
            if (!hasChanged && iter > 0) break;

            std::vector<CielabBlock> newCentroids(config.codebookSize, CielabBlock(64, 0.0f));
            std::vector<uint32_t> counts(config.codebookSize, 0);
#pragma omp parallel
            {
                std::vector<CielabBlock> localNewCentroids(config.codebookSize, CielabBlock(64, 0.0f));
                std::vector<uint32_t> localCounts(config.codebookSize, 0);
#pragma omp for nowait
                for (int64_t i = 0; i < numBlocks; ++i) {
                    uint32_t c_idx = assignments[i];
                    localCounts[c_idx]++;
                    for (size_t j = 0; j < 64; ++j) localNewCentroids[c_idx][j] += labBlocks[i][j];
                }
#pragma omp critical
                {
                    for (uint32_t c = 0; c < config.codebookSize; ++c) {
                        counts[c] += localCounts[c];
                        for (size_t j = 0; j < 64; ++j) newCentroids[c][j] += localNewCentroids[c][j];
                    }
                }
            }

#pragma omp parallel for
            for (int64_t c = 0; c < config.codebookSize; ++c) {
                if (counts[c] > 0) {
                    float inv_count = 1.0f / counts[c];
                    for (size_t j = 0; j < 64; ++j) labCentroids[c][j] = newCentroids[c][j] * inv_count;
                }
                else {
                    size_t worstBlockIdx = std::distance(errors.begin(), std::max_element(std::execution::par, errors.begin(), errors.end()));
                    labCentroids[c] = labBlocks[worstBlockIdx];
                    errors[worstBlockIdx] = 0.0f;
                }
            }
        }
#pragma omp parallel for
        for (int64_t i = 0; i < config.codebookSize; ++i) rgbaCentroids[i] = CielabBlockToRgbaBlock(labCentroids[i]);
    }
    else { // RGB_SIMD Path
        for (uint32_t iter = 0; iter < config.maxIterations; ++iter) {
            std::atomic<bool> hasChanged = false;
#pragma omp parallel for
            for (int64_t i = 0; i < numBlocks; ++i) {
                float min_d = std::numeric_limits<float>::max();
                uint32_t best_c = 0;
                for (uint32_t c = 0; c < config.codebookSize; ++c) {
                    float d = RgbaBlockDistanceSAD_SIMD((*blocksToProcess[i]).data(), rgbaCentroids[c].data());
                    if (d < min_d) { min_d = d; best_c = c; }
                }
                errors[i] = min_d;
                if (assignments[i] != best_c) { assignments[i] = best_c; hasChanged = true; }
            }
            if (!hasChanged && iter > 0) break;

            std::vector<std::vector<uint64_t>> newCentroids(config.codebookSize, std::vector<uint64_t>(64, 0));
            std::vector<uint32_t> counts(config.codebookSize, 0);
#pragma omp parallel
            {
                std::vector<std::vector<uint64_t>> localNewCentroids(config.codebookSize, std::vector<uint64_t>(64, 0));
                std::vector<uint32_t> localCounts(config.codebookSize, 0);
#pragma omp for nowait
                for (int64_t i = 0; i < numBlocks; ++i) {
                    uint32_t c_idx = assignments[i];
                    localCounts[c_idx]++;
                    for (size_t j = 0; j < 64; ++j) localNewCentroids[c_idx][j] += (*blocksToProcess[i])[j];
                }
#pragma omp critical
                {
                    for (uint32_t c = 0; c < config.codebookSize; ++c) {
                        counts[c] += localCounts[c];
                        for (size_t j = 0; j < 64; ++j) newCentroids[c][j] += localNewCentroids[c][j];
                    }
                }
            }

#pragma omp parallel for
            for (int64_t c = 0; c < config.codebookSize; ++c) {
                if (counts[c] > 0) {
                    for (size_t j = 0; j < 64; ++j) rgbaCentroids[c][j] = static_cast<uint8_t>(newCentroids[c][j] / counts[c]);
                }
                else {
                    size_t worstBlockIdx = std::distance(errors.begin(), std::max_element(std::execution::par, errors.begin(), errors.end()));
                    rgbaCentroids[c] = *blocksToProcess[worstBlockIdx];
                    errors[worstBlockIdx] = 0.0f;
                }
            }
        }
    }

    // 4. Finalize Codebook
    outRgbaCentroids = rgbaCentroids;
    VQCodebook finalCodebook(BCBlockSize::GetSize(bcFormat), config.codebookSize);
    finalCodebook.entries.resize(config.codebookSize);
#pragma omp parallel for
    for (int64_t i = 0; i < config.codebookSize; ++i) {
        finalCodebook.entries[i] = CompressSingleBlock(rgbaCentroids[i].data(), channels, alphaThreshold);
    }
    return finalCodebook;
}


inline std::vector<uint32_t> VQEncoder::QuantizeBlocks(const std::vector<std::vector<uint8_t>>& rgbaBlocks, const std::vector<std::vector<uint8_t>>& rgbaCentroids) {
    size_t numBlocks = rgbaBlocks.size();
    if (numBlocks == 0) return {};
    std::vector<uint32_t> indices(numBlocks);
    uint32_t codebookSize = static_cast<uint32_t>(rgbaCentroids.size());

    if (config.metric == DistanceMetric::PERCEPTUAL_LAB) {
        std::vector<CielabBlock> labBlocks(numBlocks);
#pragma omp parallel for
        for (int64_t i = 0; i < numBlocks; ++i) labBlocks[i] = RgbaBlockToCielabBlock(rgbaBlocks[i]);

        std::vector<CielabBlock> labCentroids(codebookSize);
#pragma omp parallel for
        for (int64_t i = 0; i < codebookSize; ++i) labCentroids[i] = RgbaBlockToCielabBlock(rgbaCentroids[i]);

#pragma omp parallel for
        for (int64_t i = 0; i < numBlocks; ++i) {
            float minDist = std::numeric_limits<float>::max();
            uint32_t bestIdx = 0;
            for (uint32_t j = 0; j < codebookSize; ++j) {
                float dist = CielabBlockDistanceSq_SIMD(labBlocks[i], labCentroids[j]);
                if (dist < minDist) { minDist = dist; bestIdx = j; }
            }
            indices[i] = bestIdx;
        }
    }
    else { // RGB_SIMD path
#pragma omp parallel for
        for (int64_t i = 0; i < numBlocks; ++i) {
            float minDist = std::numeric_limits<float>::max();
            uint32_t bestIdx = 0;
            for (uint32_t j = 0; j < codebookSize; ++j) {
                float dist = RgbaBlockDistanceSAD_SIMD(rgbaBlocks[i].data(), rgbaCentroids[j].data());
                if (dist < minDist) { minDist = dist; bestIdx = j; }
            }
            indices[i] = bestIdx;
        }
    }
    return indices;
}


inline ProductCodebook VQEncoder::BuildProductCodebook(const std::vector<std::vector<uint8_t>>& rgbaBlocks) {
    if (config.metric != DistanceMetric::PERCEPTUAL_LAB) {
        throw std::runtime_error("Product Quantization is only implemented for PERCEPTUAL_LAB metric.");
    }
    size_t numBlocks = rgbaBlocks.size();
    const uint32_t num_subvectors = config.pq_subvectors;
    const size_t full_dim = 64; // 16 pixels * 4 floats
    const size_t sub_dim = full_dim / num_subvectors;

    if (full_dim % num_subvectors != 0) {
        throw std::runtime_error("Full vector dimension must be divisible by number of subvectors.");
    }

    std::vector<CielabBlock> labBlocks(numBlocks);
#pragma omp parallel for
    for (int64_t i = 0; i < numBlocks; ++i) {
        labBlocks[i] = RgbaBlockToCielabBlock(rgbaBlocks[i]);
    }

    ProductCodebook pq_codebook;
    pq_codebook.num_subvectors = num_subvectors;
    pq_codebook.sub_codebook_size = config.pq_sub_codebook_size;
    pq_codebook.sub_codebooks.resize(num_subvectors);

#pragma omp parallel for
    for (int64_t s = 0; s < num_subvectors; ++s) {
        std::vector<SubVector> sub_vectors(numBlocks, SubVector(sub_dim));
        for (size_t i = 0; i < numBlocks; ++i) {
            std::copy(labBlocks[i].begin() + s * sub_dim,
                labBlocks[i].begin() + (s + 1) * sub_dim,
                sub_vectors[i].begin());
        }

        std::vector<SubVector> centroids(config.pq_sub_codebook_size);
        std::vector<float> minDistSq(numBlocks, std::numeric_limits<float>::max());
        std::uniform_int_distribution<size_t> local_distrib(0, numBlocks - 1);
        std::mt19937 local_rng(std::random_device{}() + s); // Seed each thread differently
        centroids[0] = sub_vectors[local_distrib(local_rng)];

        for (uint32_t i = 1; i < config.pq_sub_codebook_size; ++i) {
            double current_sum = 0;
            for (size_t j = 0; j < numBlocks; ++j) {
                float d = SubVectorDistanceSq_SIMD(sub_vectors[j], centroids[i - 1]);
                minDistSq[j] = std::min(d, minDistSq[j]);
                current_sum += minDistSq[j];
            }
            if (current_sum <= 0) break;
            std::uniform_real_distribution<double> p_distrib(0.0, current_sum);
            double p = p_distrib(local_rng);
            double cumulative_p = 0.0;
            for (size_t j = 0; j < numBlocks; ++j) {
                cumulative_p += minDistSq[j];
                if (cumulative_p >= p) {
                    centroids[i] = sub_vectors[j];
                    break;
                }
            }
        }

        std::vector<uint32_t> assignments(numBlocks, 0);
        for (uint32_t iter = 0; iter < config.maxIterations; ++iter) {
            bool changed = false;
            for (size_t i = 0; i < numBlocks; ++i) {
                float min_d = std::numeric_limits<float>::max();
                uint32_t best_c = 0;
                for (uint32_t c = 0; c < config.pq_sub_codebook_size; ++c) {
                    float d = SubVectorDistanceSq_SIMD(sub_vectors[i], centroids[c]);
                    if (d < min_d) { min_d = d; best_c = c; }
                }
                if (assignments[i] != best_c) { assignments[i] = best_c; changed = true; }
            }
            if (!changed && iter > 0) break;

            std::vector<SubVector> newCentroids(config.pq_sub_codebook_size, SubVector(sub_dim, 0.0f));
            std::vector<uint32_t> counts(config.pq_sub_codebook_size, 0);
            for (size_t i = 0; i < numBlocks; ++i) {
                uint32_t c_idx = assignments[i];
                counts[c_idx]++;
                for (size_t j = 0; j < sub_dim; ++j) newCentroids[c_idx][j] += sub_vectors[i][j];
            }
            for (uint32_t c = 0; c < config.pq_sub_codebook_size; ++c) {
                if (counts[c] > 0) {
                    float inv_count = 1.0f / counts[c];
                    for (size_t j = 0; j < sub_dim; ++j) centroids[c][j] = newCentroids[c][j] * inv_count;
                }
            }
        }
        pq_codebook.sub_codebooks[s] = centroids;
    }
    return pq_codebook;
}

inline std::vector<std::vector<uint8_t>> VQEncoder::QuantizeProductBlocks(const std::vector<std::vector<uint8_t>>& rgbaBlocks, const ProductCodebook& pq_codebook) {
    size_t numBlocks = rgbaBlocks.size();
    const uint32_t num_subvectors = pq_codebook.num_subvectors;
    const size_t sub_dim = (16 * 4) / num_subvectors;

    std::vector<std::vector<uint8_t>> all_indices(numBlocks, std::vector<uint8_t>(num_subvectors));

    std::vector<CielabBlock> labBlocks(numBlocks);
#pragma omp parallel for
    for (int64_t i = 0; i < numBlocks; ++i) {
        labBlocks[i] = RgbaBlockToCielabBlock(rgbaBlocks[i]);
    }

#pragma omp parallel for
    for (int64_t i = 0; i < numBlocks; ++i) {
        for (uint32_t s = 0; s < num_subvectors; ++s) {
            SubVector sub_vec(sub_dim);
            std::copy(labBlocks[i].begin() + s * sub_dim,
                labBlocks[i].begin() + (s + 1) * sub_dim,
                sub_vec.begin());

            float min_d = std::numeric_limits<float>::max();
            uint8_t best_idx = 0;
            const auto& sub_codebook = pq_codebook.sub_codebooks[s];
            for (uint32_t c = 0; c < pq_codebook.sub_codebook_size; ++c) {
                float d = SubVectorDistanceSq_SIMD(sub_vec, sub_codebook[c]);
                if (d < min_d) { min_d = d; best_idx = c; }
            }
            all_indices[i][s] = best_idx;
        }
    }

    return all_indices;
}

// -- HDR VQ Methods ---

inline std::vector<uint8_t> VQEncoder::CompressSingleBlockHDR(const std::vector<float>& rgbaBlock) {
    // This assumes the format is BC6H
    return bcnCompressor.CompressHDR(rgbaBlock.data(), 4, 4, bcFormat, 1.0f);
}

inline VQCodebook VQEncoder::BuildCodebookHDR(const std::vector<std::vector<float>>& allRgbaFloatBlocks, std::vector<std::vector<float>>& outRgbaCentroids) {
    // 1. Sampling: Run expensive K-Means on a sample, not the entire dataset.
    std::vector<const std::vector<float>*> sampledBlocksPtrs;
    if (config.fastModeSampleRatio < 1.0f && config.fastModeSampleRatio > 0.0f) {
        size_t numToSample = static_cast<size_t>(allRgbaFloatBlocks.size() * config.fastModeSampleRatio);
        numToSample = std::max(static_cast<size_t>(config.codebookSize * 4), numToSample);
        numToSample = std::min(numToSample, allRgbaFloatBlocks.size());

        sampledBlocksPtrs.reserve(numToSample);
        std::vector<size_t> indices(allRgbaFloatBlocks.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        for (size_t i = 0; i < numToSample; ++i) {
            sampledBlocksPtrs.push_back(&allRgbaFloatBlocks[indices[i]]);
        }
    }
    else {
        sampledBlocksPtrs.reserve(allRgbaFloatBlocks.size());
        for (const auto& block : allRgbaFloatBlocks) {
            sampledBlocksPtrs.push_back(&block);
        }
    }

    const auto& blocksToProcess = sampledBlocksPtrs;
    size_t numBlocks = blocksToProcess.size();
    if (numBlocks == 0) return {};

    // 2. Pre-transformation: Convert to perceptual Oklab *once*.
    std::vector<OklabBlock> perceptualOklabBlocks(numBlocks);
#pragma omp parallel for
    for (int64_t i = 0; i < numBlocks; ++i) {
        perceptualOklabBlocks[i] = Oklab::RgbaFloatBlockToOklabBlock(*blocksToProcess[i]);
        OklabBlockToPerceptual(perceptualOklabBlocks[i]);
    }

    // 3. K-Means++ Initialization (on pre-transformed perceptual data)
    std::vector<OklabBlock> perceptualOklabCentroids(config.codebookSize);
    std::vector<float> minDistSq(numBlocks, std::numeric_limits<float>::max());
    std::uniform_int_distribution<size_t> distrib(0, numBlocks - 1);
    perceptualOklabCentroids[0] = perceptualOklabBlocks[distrib(rng)];

    for (uint32_t i = 1; i < config.codebookSize; ++i) {
        double current_sum = 0.0;
#pragma omp parallel for reduction(+:current_sum)
        for (int64_t j = 0; j < numBlocks; ++j) {
            float d = OklabBlockDistanceSq_SIMD(perceptualOklabBlocks[j], perceptualOklabCentroids[i - 1]);
            minDistSq[j] = std::min(d, minDistSq[j]);
            current_sum += minDistSq[j];
        }
        if (current_sum <= 0) {
            for (uint32_t k = i; k < config.codebookSize; ++k) perceptualOklabCentroids[k] = perceptualOklabCentroids[0];
            break;
        }
        std::uniform_real_distribution<double> p_distrib(0.0, current_sum);
        double p = p_distrib(rng);
        double cumulative_p = 0.0;
        for (size_t j = 0; j < numBlocks; ++j) {
            cumulative_p += minDistSq[j];
            if (cumulative_p >= p) {
                perceptualOklabCentroids[i] = perceptualOklabBlocks[j];
                break;
            }
        }
    }

    // 4. K-Means Iterations
    std::vector<uint32_t> assignments(numBlocks, 0);
    std::vector<float> errors(numBlocks);
    for (uint32_t iter = 0; iter < config.maxIterations; ++iter) {
        std::atomic<bool> hasChanged = false;
#pragma omp parallel for
        for (int64_t i = 0; i < numBlocks; ++i) {
            float min_d = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < config.codebookSize; ++c) {
                float d = OklabBlockDistanceSq_SIMD(perceptualOklabBlocks[i], perceptualOklabCentroids[c]);
                if (d < min_d) { min_d = d; best_c = c; }
            }
            errors[i] = min_d;
            if (assignments[i] != best_c) { assignments[i] = best_c; hasChanged = true; }
        }
        if (!hasChanged && iter > 1) break;

        std::vector<OklabBlock> newCentroids(config.codebookSize, OklabBlock(64, 0.0f));
        std::vector<uint32_t> counts(config.codebookSize, 0);
#pragma omp parallel
        {
            std::vector<OklabBlock> localNewCentroids(config.codebookSize, OklabBlock(64, 0.0f));
            std::vector<uint32_t> localCounts(config.codebookSize, 0);
#pragma omp for nowait
            for (int64_t i = 0; i < numBlocks; ++i) {
                uint32_t c_idx = assignments[i];
                localCounts[c_idx]++;
                for (size_t j = 0; j < 64; ++j) localNewCentroids[c_idx][j] += perceptualOklabBlocks[i][j];
            }
#pragma omp critical
            {
                for (uint32_t c = 0; c < config.codebookSize; ++c) {
                    counts[c] += localCounts[c];
                    for (size_t j = 0; j < 64; ++j) newCentroids[c][j] += localNewCentroids[c][j];
                }
            }
        }
#pragma omp parallel for
        for (int64_t c = 0; c < config.codebookSize; ++c) {
            if (counts[c] > 0) {
                float inv_count = 1.0f / counts[c];
                for (size_t j = 0; j < 64; ++j) perceptualOklabCentroids[c][j] = newCentroids[c][j] * inv_count;
            }
            else {
                size_t worstBlockIdx = std::distance(errors.begin(), std::max_element(std::execution::par, errors.begin(), errors.end()));
                perceptualOklabCentroids[c] = perceptualOklabBlocks[worstBlockIdx];
                errors[worstBlockIdx] = 0.0f;
            }
        }
    }

    // 5. Finalize Codebook
    outRgbaCentroids.resize(config.codebookSize);
#pragma omp parallel for
    for (int64_t i = 0; i < config.codebookSize; ++i) {
        OklabBlock finalOklabCentroid = perceptualOklabCentroids[i];
        PerceptualBlockToOklab(finalOklabCentroid);
        outRgbaCentroids[i] = Oklab::OklabBlockToRgbaFloatBlock(finalOklabCentroid);
    }

    VQCodebook finalCodebook(BCBlockSize::GetSize(bcFormat), config.codebookSize);
    finalCodebook.entries.resize(config.codebookSize);
#pragma omp parallel for
    for (int64_t i = 0; i < config.codebookSize; ++i) {
        finalCodebook.entries[i] = CompressSingleBlockHDR(outRgbaCentroids[i]);
    }
    return finalCodebook;
}


inline std::vector<uint32_t> VQEncoder::QuantizeBlocksHDR(const std::vector<std::vector<float>>& rgbaFloatBlocks, const std::vector<std::vector<float>>& rgbaCentroids) {
    size_t numBlocks = rgbaFloatBlocks.size();
    if (numBlocks == 0) return {};
    std::vector<uint32_t> indices(numBlocks);
    uint32_t codebookSize = static_cast<uint32_t>(rgbaCentroids.size());

    // 1. Convert all final centroids to perceptual Oklab space once.
    std::vector<OklabBlock> perceptualLabCentroids(codebookSize);
#pragma omp parallel for
    for (int64_t i = 0; i < codebookSize; ++i) {
        perceptualLabCentroids[i] = Oklab::RgbaFloatBlockToOklabBlock(rgbaCentroids[i]);
        OklabBlockToPerceptual(perceptualLabCentroids[i]);
    }

    // 2. Find best index for each block, converting to perceptual space as we go.
#pragma omp parallel for
    for (int64_t i = 0; i < numBlocks; ++i) {
        OklabBlock perceptualLabBlock = Oklab::RgbaFloatBlockToOklabBlock(rgbaFloatBlocks[i]);
        OklabBlockToPerceptual(perceptualLabBlock);

        float minDist = std::numeric_limits<float>::max();
        uint32_t bestIdx = 0;
        for (uint32_t j = 0; j < codebookSize; ++j) {
            float dist = OklabBlockDistanceSq_SIMD(perceptualLabBlock, perceptualLabCentroids[j]);
            if (dist < minDist) {
                minDist = dist;
                bestIdx = j;
            }
        }
        indices[i] = bestIdx;
    }
    return indices;
}

inline ProductCodebook VQEncoder::BuildProductCodebookHDR(const std::vector<std::vector<float>>& rgbaFloatBlocks) {
    size_t numBlocks = rgbaFloatBlocks.size();
    const uint32_t num_subvectors = config.pq_subvectors;
    const size_t full_dim = 64; // 16 pixels * 4 floats (L,a,b,A)
    const size_t sub_dim = full_dim / num_subvectors;

    if (full_dim % num_subvectors != 0) {
        throw std::runtime_error("Full vector dimension must be divisible by number of subvectors.");
    }

    // 1. Convert all blocks to perceptual oklab space
    std::vector<OklabBlock> perceptualOklabBlocks(numBlocks);
#pragma omp parallel for
    for (int64_t i = 0; i < numBlocks; ++i) {
        perceptualOklabBlocks[i] = Oklab::RgbaFloatBlockToOklabBlock(rgbaFloatBlocks[i]);
        OklabBlockToPerceptual(perceptualOklabBlocks[i]);
    }

    ProductCodebook pq_codebook;
    pq_codebook.num_subvectors = num_subvectors;
    pq_codebook.sub_codebook_size = config.pq_sub_codebook_size;
    pq_codebook.sub_codebooks.resize(num_subvectors);

    // 2. For each sub-vector space, run a separate K-Means
#pragma omp parallel for
    for (int64_t s = 0; s < num_subvectors; ++s) {
        // Collect all sub-vectors for the current space
        std::vector<SubVector> sub_vectors(numBlocks, SubVector(sub_dim));
        for (size_t i = 0; i < numBlocks; ++i) {
            const auto& block = perceptualOklabBlocks[i];
            std::copy(block.begin() + s * sub_dim,
                block.begin() + (s + 1) * sub_dim,
                sub_vectors[i].begin());
        }

        // K-Means++ Initialization for this subspace
        std::vector<SubVector> centroids(config.pq_sub_codebook_size);
        std::vector<float> minDistSq(numBlocks, std::numeric_limits<float>::max());
        std::mt19937 local_rng(std::random_device{}() + s); // Seed each thread differently
        std::uniform_int_distribution<size_t> local_distrib(0, numBlocks - 1);

        centroids[0] = sub_vectors[local_distrib(local_rng)];

        for (uint32_t i = 1; i < config.pq_sub_codebook_size; ++i) {
            double current_sum = 0;
            for (size_t j = 0; j < numBlocks; ++j) {
                float d = SubVectorDistanceSq_SIMD(sub_vectors[j], centroids[i - 1]);
                minDistSq[j] = std::min(d, minDistSq[j]);
                current_sum += minDistSq[j];
            }
            if (current_sum <= 0) {
                for (uint32_t k = i; k < config.pq_sub_codebook_size; ++k) centroids[k] = centroids[0];
                break;
            }
            std::uniform_real_distribution<double> p_distrib(0.0, current_sum);
            double p = p_distrib(local_rng);
            double cumulative_p = 0.0;
            for (size_t j = 0; j < numBlocks; ++j) {
                cumulative_p += minDistSq[j];
                if (cumulative_p >= p) {
                    centroids[i] = sub_vectors[j];
                    break;
                }
            }
        }

        // K-Means Iterations for this subspace
        std::vector<uint32_t> assignments(numBlocks, 0);
        for (uint32_t iter = 0; iter < config.maxIterations; ++iter) {
            bool changed = false;
            // Assignment Step
            for (size_t i = 0; i < numBlocks; ++i) {
                float min_d = std::numeric_limits<float>::max();
                uint32_t best_c = 0;
                for (uint32_t c = 0; c < config.pq_sub_codebook_size; ++c) {
                    float d = SubVectorDistanceSq_SIMD(sub_vectors[i], centroids[c]);
                    if (d < min_d) { min_d = d; best_c = c; }
                }
                if (assignments[i] != best_c) {
                    assignments[i] = best_c;
                    changed = true;
                }
            }
            if (!changed && iter > 1) break;

            // Update Step
            std::vector<SubVector> newCentroids(config.pq_sub_codebook_size, SubVector(sub_dim, 0.0f));
            std::vector<uint32_t> counts(config.pq_sub_codebook_size, 0);
            for (size_t i = 0; i < numBlocks; ++i) {
                uint32_t c_idx = assignments[i];
                counts[c_idx]++;
                for (size_t j = 0; j < sub_dim; ++j) {
                    newCentroids[c_idx][j] += sub_vectors[i][j];
                }
            }
            for (uint32_t c = 0; c < config.pq_sub_codebook_size; ++c) {
                if (counts[c] > 0) {
                    float inv_count = 1.0f / counts[c];
                    for (size_t j = 0; j < sub_dim; ++j) {
                        centroids[c][j] = newCentroids[c][j] * inv_count;
                    }
                }
                // Not re-seeding empty clusters here for simplicity, but could be added if needed.
            }
        }
        pq_codebook.sub_codebooks[s] = centroids;
    }
    return pq_codebook;
}

inline std::vector<std::vector<uint8_t>> VQEncoder::QuantizeProductBlocksHDR(const std::vector<std::vector<float>>& rgbaFloatBlocks, const ProductCodebook& pq_codebook) {
    size_t numBlocks = rgbaFloatBlocks.size();
    const uint32_t num_subvectors = pq_codebook.num_subvectors;
    const size_t sub_dim = (16 * 4) / num_subvectors;

    std::vector<std::vector<uint8_t>> all_indices(numBlocks, std::vector<uint8_t>(num_subvectors));

#pragma omp parallel for
    for (int64_t i = 0; i < numBlocks; ++i) {
        // Convert block to perceptual space for comparison
        OklabBlock p_block = Oklab::RgbaFloatBlockToOklabBlock(rgbaFloatBlocks[i]);
        OklabBlockToPerceptual(p_block);

        for (uint32_t s = 0; s < num_subvectors; ++s) {
            SubVector sub_vec(sub_dim);
            std::copy(p_block.begin() + s * sub_dim,
                p_block.begin() + (s + 1) * sub_dim,
                sub_vec.begin());

            float min_d = std::numeric_limits<float>::max();
            uint8_t best_idx = 0;
            const auto& sub_codebook = pq_codebook.sub_codebooks[s];
            for (uint32_t c = 0; c < pq_codebook.sub_codebook_size; ++c) {
                float d = SubVectorDistanceSq_SIMD(sub_vec, sub_codebook[c]);
                if (d < min_d) { min_d = d; best_idx = c; }
            }
            all_indices[i][s] = best_idx;
        }
    }
    return all_indices;
}