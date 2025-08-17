#pragma once

#include "vq_bcn_types.h"
#include "bcn_compressor.h"
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>
#include <execution>
#include <numeric>
#include <atomic>
#include <vector>
#include <stdexcept> // For std::runtime_error

using CielabBlock = std::vector<float>;
using OklabFloatBlock = std::vector<float>;

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

struct CompressionConfig {
public:
    uint32_t maxIterations = 32;
    DistanceMetric metric = DistanceMetric::PERCEPTUAL_LAB;

    // --- General Settings ---
    float fastModeSampleRatio = 1.0f;
    uint32_t min_cb_power = 4; // 2^4 = 16 entries at quality=0
    uint32_t max_cb_power = 10; // 2^10 = 1024 entries at quality=1

    CompressionConfig(float quality_level = 0.5f);

    void SetQuality(float quality_level);

    // Quality meter (0.0 = low, 1.0 = high) drives other settings.
    float quality = 0.5f;
    // --- Standard VQ Settings ---
    uint32_t codebookSize;
};

class YUPTEXTURE_API VQEncoder {
private:
    CompressionConfig config;
    BCFormat bcFormat;
    BCnCompressor bcnCompressor;
    std::mt19937 rng;

    // --- Color Space Conversion ---
    void RgbToCielab(const uint8_t* rgb, float* lab) const;
    void CielabToRgb(const float* lab, uint8_t* rgb) const;
    CielabBlock RgbaBlockToCielabBlock(const std::vector<uint8_t>& rgbaBlock) const;
    std::vector<uint8_t> CielabBlockToRgbaBlock(const CielabBlock& labBlock) const;

    // --- sRGB color space math ---
    // --- Distance Functions ---
    float RgbaBlockDistanceSAD_SIMD(const uint8_t* rgbaA, const uint8_t* rgbaB) const;

    // Squared Euclidean distance for CIELAB blocks
    float CielabBlockDistanceSq_SIMD(const CielabBlock& labA, const CielabBlock& labB) const;

    std::vector<uint8_t> CompressSingleBlock(const uint8_t* rgbaBlock, uint8_t channels, int numThreads, float quality, uint8_t alphaThreshold = 128);

    // HDR oklab color space math

    // --- Added luminance weighting to the distance function ---
    float OklabFloatBlockDistanceSq_SIMD(const OklabFloatBlock& labA, const OklabFloatBlock& labB) const;

    std::vector<uint8_t> CompressSingleBlockHDR(const std::vector<float>& rgbaBlock, int numThreads, float quality);

public:
    VQEncoder(const CompressionConfig& cfg = CompressionConfig());

    void SetFormat(BCFormat format);

    // --- Public API ---
    VQCodebook BuildCodebook(
        const std::vector<std::vector<uint8_t>>& allRgbaBlocks,
        uint8_t channels,
        std::vector<std::vector<uint8_t>>& outRgbaCentroids,
        const CompressionParams& params
    );

    std::vector<uint32_t> QuantizeBlocks(
        const std::vector<std::vector<uint8_t>>& rgbaBlocks,
        const std::vector<std::vector<uint8_t>>& rgbaCentroids,
        const CompressionParams& params
    );

    VQCodebook BuildCodebookHDR(
        const std::vector<std::vector<float>>& rgbaFloatBlocks,
        std::vector<std::vector<float>>& outRgbaCentroids,
        const CompressionParams& params
    );

    std::vector<uint32_t> QuantizeBlocksHDR(
        const std::vector<std::vector<float>>& rgbaFloatBlocks,
        const std::vector<std::vector<float>>& rgbaCentroids,
        const CompressionParams& params
    );
};