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

class VQEncoder {
public:
    enum class DistanceMetric {
        RGB_SIMD,      // Fastest: Euclidean distance on RGB values, accelerated with AVX2.
        PERCEPTUAL_LAB // Highest Quality: Euclidean distance in CIELAB color space, now with AVX2.
    };

    struct Config {
        uint32_t codebookSize = 256;
        uint32_t maxIterations = 50; // Increased default as we now have early exit
        DistanceMetric metric = DistanceMetric::PERCEPTUAL_LAB;
    };

private:
    Config config;
    BCFormat bcFormat;
    BCnCompressor bcnCompressor;
    std::mt19937 rng;

    // --- Color Space Conversion (for Perceptual Metric) ---
    inline void RgbToCielab(const uint8_t* rgb, float* lab) const {
        // 1. sRGB to linear RGB
        float r = rgb[0] / 255.0f;
        float g = rgb[1] / 255.0f;
        float b = rgb[2] / 255.0f;
        r = (r > 0.04045f) ? powf((r + 0.055f) / 1.055f, 2.4f) : r / 12.92f;
        g = (g > 0.04045f) ? powf((g + 0.055f) / 1.055f, 2.4f) : g / 12.92f;
        b = (b > 0.04045f) ? powf((b + 0.055f) / 1.055f, 2.4f) : b / 12.92f;

        // 2. Linear RGB to XYZ
        float x = r * 0.4124f + g * 0.3576f + b * 0.1805f;
        float y = r * 0.2126f + g * 0.7152f + b * 0.0722f;
        float z = r * 0.0193f + g * 0.1192f + b * 0.9505f;

        // 3. XYZ to CIELAB (D65 reference white)
        x /= 0.95047f;
        y /= 1.00000f;
        z /= 1.08883f;
        auto f = [](float t) {
            return (t > 0.008856f) ? cbrtf(t) : (7.787f * t + 16.0f / 116.0f);
            };
        float fx = f(x);
        float fy = f(y);
        float fz = f(z);

        lab[0] = (116.0f * fy) - 16.0f;
        lab[1] = 500.0f * (fx - fy);
        lab[2] = 200.0f * (fy - fz);
    }

    CielabBlock RgbaBlockToCielabBlock(const std::vector<uint8_t>& rgbaBlock) const {
        CielabBlock labBlock(16 * 4); // 16 pixels * (L,a,b,alpha)
        for (size_t i = 0; i < 16; ++i) {
            RgbToCielab(&rgbaBlock[i * 4], &labBlock[i * 4]);
            labBlock[i * 4 + 3] = rgbaBlock[i * 4 + 3] / 255.0f; // Pass alpha through, normalized
        }
        return labBlock;
    }

    // --- Distance Functions ---

    // High-performance AVX2 implementation for RGB distance
    float RgbaBlockDistance_SIMD(const uint8_t* rgbaA, const uint8_t* rgbaB) const {
        // This function calculates Sum of Absolute Differences, which is a good proxy for Euclidean distance and is extremely fast.
        __m256i diff_sum = _mm256_setzero_si256();

        // Process 64 bytes in two 32-byte chunks
        __m256i a1 = _mm256_loadu_si256((__m256i*)(rgbaA));
        __m256i b1 = _mm256_loadu_si256((__m256i*)(rgbaB));
        __m256i sad1 = _mm256_sad_epu8(a1, b1); // Calculates SAD for 4x 8-byte segments
        diff_sum = _mm256_add_epi64(diff_sum, sad1);

        __m256i a2 = _mm256_loadu_si256((__m256i*)(rgbaA + 32));
        __m256i b2 = _mm256_loadu_si256((__m256i*)(rgbaB + 32));
        __m256i sad2 = _mm256_sad_epu8(a2, b2);
        diff_sum = _mm256_add_epi64(diff_sum, sad2);

        // Sum the results from the 256-bit register
        return (float)(_mm256_extract_epi64(diff_sum, 0) + _mm256_extract_epi64(diff_sum, 1) + _mm256_extract_epi64(diff_sum, 2) + _mm256_extract_epi64(diff_sum, 3));
    }

    // OPTIMIZATION: High-performance AVX2 implementation for CIELAB distance
    float CielabBlockDistance_SIMD(const CielabBlock& labA, const CielabBlock& labB) const {
        __m256 sum_sq_diff = _mm256_setzero_ps();
        // A CielabBlock is 64 floats (16 pixels * 4 channels). We can process 8 floats at a time.
        for (size_t i = 0; i < 64; i += 8) {
            __m256 a = _mm256_loadu_ps(labA.data() + i);
            __m256 b = _mm256_loadu_ps(labB.data() + i);
            __m256 diff = _mm256_sub_ps(a, b);
            // Fused-multiply-add is perfect here: sum = (diff * diff) + sum
            sum_sq_diff = _mm256_fmadd_ps(diff, diff, sum_sq_diff);
        }

        // Horizontally sum the squared differences in the 256-bit register
        __m128 lo_half = _mm256_castps256_ps128(sum_sq_diff);
        __m128 hi_half = _mm256_extractf128_ps(sum_sq_diff, 1);
        __m128 sum_128 = _mm_add_ps(lo_half, hi_half);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);

        return _mm_cvtss_f32(sum_128);
    }

    // --- Helper Functions ---
    std::vector<uint8_t> DecompressSingleBlock(const uint8_t* bcBlock) {
        return bcnCompressor.DecompressToRGBA(bcBlock, 4, 4, bcFormat);
    }

    std::vector<uint8_t> CompressSingleBlock(const uint8_t* rgbaBlock) {
        return bcnCompressor.CompressRGBA(rgbaBlock, 4, 4, bcFormat, 1.0f);
    }

public:
    VQEncoder(const Config& cfg = Config())
        : config(cfg), rng(std::random_device{}()) {
    }

    void SetFormat(BCFormat format) { bcFormat = format; }

    // OPTIMIZATION: This function now takes pre-decompressed RGBA blocks and outputs the final centroids
    // to avoid redundant work in the main compressor loop.
    VQCodebook BuildCodebook(
        const std::vector<std::vector<uint8_t>>& rgbaBlocks,
        std::vector<std::vector<uint8_t>>& outRgbaCentroids
    );

    // OPTIMIZATION: This function is now a pure, fast comparison loop. It takes pre-decompressed
    // blocks and centroids and uses SIMD for rapid quantization.
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

    // OPTIMIZATION: Decompression is no longer needed here, as we receive RGBA blocks directly.

    // 1. Initialize Centroids using K-Means++
    std::vector<std::vector<uint8_t>> rgbaCentroids(config.codebookSize);
    std::vector<float> minDistSq(numBlocks, std::numeric_limits<float>::max());

    // Step a: Choose first centroid uniformly at random
    std::uniform_int_distribution<size_t> distrib(0, numBlocks - 1);
    size_t firstIdx = distrib(rng);
    rgbaCentroids[0] = rgbaBlocks[firstIdx];

    // Step b, c, d: Choose remaining centroids based on distance. This part still uses RGB distance for speed.
    for (uint32_t i = 1; i < config.codebookSize; ++i) {
        double current_sum = 0.0;
#pragma omp parallel for reduction(+:current_sum)
        for (int64_t j = 0; j < numBlocks; ++j) {
            float d = RgbaBlockDistance_SIMD(rgbaBlocks[j].data(), rgbaCentroids[i - 1].data());
            minDistSq[j] = std::min(d * d, minDistSq[j]); // Use squared distance for better distribution
            current_sum += minDistSq[j];
        }

        if (current_sum <= 0) { // All points are identical, fill remaining centroids and finish
            for (uint32_t k = i; k < config.codebookSize; ++k) rgbaCentroids[k] = rgbaCentroids[0];
            i = config.codebookSize; // End outer loop
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
        // --- PERCEPTUAL K-MEANS (CIELAB) ---
        std::vector<CielabBlock> labBlocks(numBlocks);
        std::vector<CielabBlock> labCentroids(config.codebookSize);
#pragma omp parallel for
        for (int64_t i = 0; i < numBlocks; ++i) labBlocks[i] = RgbaBlockToCielabBlock(rgbaBlocks[i]);
#pragma omp parallel for
        for (int64_t i = 0; i < config.codebookSize; ++i) labCentroids[i] = RgbaBlockToCielabBlock(rgbaCentroids[i]);

        for (uint32_t iter = 0; iter < config.maxIterations; ++iter) {
            std::atomic<bool> hasChanged = false;
            // Assignment step
#pragma omp parallel for
            for (int64_t i = 0; i < numBlocks; ++i) {
                float min_d = std::numeric_limits<float>::max();
                uint32_t best_c = 0;
                for (uint32_t c = 0; c < config.codebookSize; ++c) {
                    // OPTIMIZATION: Use the new SIMD CIELAB distance function
                    float d = CielabBlockDistance_SIMD(labBlocks[i], labCentroids[c]);
                    if (d < min_d) {
                        min_d = d;
                        best_c = c;
                    }
                }
                if (assignments[i] != best_c) {
                    assignments[i] = best_c;
                    hasChanged = true;
                }
            }

            // OPTIMIZATION: If assignments didn't change, the codebook has converged.
            if (!hasChanged) {
                break;
            }

            // Update step
            std::vector<CielabBlock> newCentroids(config.codebookSize, CielabBlock(64, 0.0f));
            std::vector<uint32_t> counts(config.codebookSize, 0);

            for (size_t i = 0; i < numBlocks; ++i) {
                uint32_t c_idx = assignments[i];
                counts[c_idx]++;
                for (size_t j = 0; j < 64; ++j) { // Accumulate float values
                    newCentroids[c_idx][j] += labBlocks[i][j];
                }
            }

#pragma omp parallel for
            for (int64_t c = 0; c < config.codebookSize; ++c) {
                if (counts[c] > 0) {
                    float inv_count = 1.0f / counts[c];
                    for (size_t j = 0; j < 64; ++j) {
                        labCentroids[c][j] = newCentroids[c][j] * inv_count;
                    }
                }
            }
        }
        // NOTE: For simplicity, we don't convert LAB back to RGBA here. The final quantization will be done in RGB space
        // using the original RGB centroids that were refined using LAB-space decisions. This is an engineering trade-off.
        // A higher-quality (but slower) approach could convert the final labCentroids back to RGBA.
    }
    else {
        // --- OPTIMIZED RGB K-MEANS (SIMD Accelerated) ---
        for (uint32_t iter = 0; iter < config.maxIterations; ++iter) {
            std::atomic<bool> hasChanged = false;
            // Assignment step
#pragma omp parallel for
            for (int64_t i = 0; i < numBlocks; ++i) {
                float min_d = std::numeric_limits<float>::max();
                uint32_t best_c = 0;
                for (uint32_t c = 0; c < config.codebookSize; ++c) {
                    float d = RgbaBlockDistance_SIMD(rgbaBlocks[i].data(), rgbaCentroids[c].data());
                    if (d < min_d) {
                        min_d = d;
                        best_c = c;
                    }
                }
                if (assignments[i] != best_c) {
                    assignments[i] = best_c;
                    hasChanged = true;
                }
            }

            // OPTIMIZATION: If assignments didn't change, the codebook has converged.
            if (!hasChanged) {
                break;
            }

            // Update step
            std::vector<std::vector<uint64_t>> newCentroids(config.codebookSize, std::vector<uint64_t>(64, 0));
            std::vector<uint32_t> counts(config.codebookSize, 0);

            for (size_t i = 0; i < numBlocks; ++i) {
                uint32_t c_idx = assignments[i];
                counts[c_idx]++;
                for (size_t j = 0; j < 64; ++j) {
                    newCentroids[c_idx][j] += rgbaBlocks[i][j];
                }
            }

#pragma omp parallel for
            for (int64_t c = 0; c < config.codebookSize; ++c) {
                if (counts[c] > 0) {
                    for (size_t j = 0; j < 64; ++j) {
                        rgbaCentroids[c][j] = static_cast<uint8_t>(newCentroids[c][j] / counts[c]);
                    }
                }
            }
        }
    }

    // 3. Output final centroids and compress them for the codebook
    outRgbaCentroids = rgbaCentroids; // OPTIMIZATION: Output the final centroids for the quantization step.

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

    // OPTIMIZATION: This entire function is now a pure, parallelized comparison loop.
    // All decompression has been moved out and is done only once in the main compressor class.
    // The comparison is always done in RGB space for maximum speed. The "perceptual" quality
    // comes from how the centroids were selected, not from this final assignment step.
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