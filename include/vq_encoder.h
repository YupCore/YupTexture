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

    // --- Color Space Conversion (Originals) ---
    void RgbToCielab(const uint8_t* rgb, float* lab) const;
    void CielabToRgb(const float* lab, uint8_t* rgb) const;

    // --- Color Space Conversion Helpers (Refactored for dynamic channels) ---
    void PixelToCielab(const uint8_t* pixel, uint8_t channelCount, float* lab) const;
    void CielabToPixel(const float* lab, uint8_t channelCount, uint8_t* pixel) const;
    CielabBlock PixelBlockToCielabBlock(const std::vector<uint8_t>& pixelBlock, uint8_t channelCount) const;
    std::vector<uint8_t> CielabBlockToPixelBlock(const CielabBlock& labBlock, uint8_t channelCount) const;

    // --- Distance Functions ---
    float BlockDistanceSAD(const uint8_t* a, const uint8_t* b, uint8_t channelCount) const;
    float CielabBlockDistanceSq_SIMD(const CielabBlock& labA, const CielabBlock& labB) const;

    std::vector<uint8_t> CompressSingleBlock(const uint8_t* pixelBlock, uint8_t channels, int numThreads, float quality);

    // --- HDR ---
    float OklabFloatBlockDistanceSq_SIMD(const OklabFloatBlock& labA, const OklabFloatBlock& labB) const;
    std::vector<uint8_t> CompressSingleBlockHDR(const std::vector<float>& pixelBlock, uint8_t channels, int numThreads, float quality);

public:
    VQEncoder(const CompressionConfig& cfg = CompressionConfig());
    void SetFormat(BCFormat format);

    // --- Public API ---
    VQCodebook BuildCodebook(
        const std::vector<std::vector<uint8_t>>& allPixelBlocks,
        uint8_t channels,
        std::vector<std::vector<uint8_t>>& outPixelCentroids,
        const CompressionParams& params
    );

    std::vector<uint32_t> QuantizeBlocks(
        const std::vector<std::vector<uint8_t>>& pixelBlocks,
        const std::vector<std::vector<uint8_t>>& pixelCentroids,
        const CompressionParams& params
    );

    VQCodebook BuildCodebookHDR(
        const std::vector<std::vector<float>>& allPixelFloatBlocks,
        uint8_t channels,
        std::vector<std::vector<float>>& outPixelFloatCentroids,
        const CompressionParams& params
    );

    std::vector<uint32_t> QuantizeBlocksHDR(
        const std::vector<std::vector<float>>& pixelFloatBlocks,
        uint8_t channels,
        const std::vector<std::vector<float>>& pixelFloatCentroids,
        const CompressionParams& params
    );
};