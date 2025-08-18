#pragma once

#include "vq_bcn_types.h"
#include "bcn_compressor.h"
#include <colorm.h>
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>
#include <execution>
#include <numeric>
#include <atomic>
#include <vector>
#include <stdexcept>

// OklabBlock will be our standard internal format for processing a 4x4 block of pixels.
// Each pixel is represented by 4 floats (L, a, b, Alpha), so a block is 16 * 4 = 64 floats.
using OklabBlock = std::vector<float>;
// A separate type for HDR, though the underlying type is the same.
using OklabFloatBlock = std::vector<float>;

struct CompressionConfig {
public:
    uint32_t maxIterations = 32;
    // Metric enum updated for clarity.
    DistanceMetric metric = DistanceMetric::PERCEPTUAL_OKLAB;

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

class VQEncoder {
private:
    CompressionConfig config;
    BCFormat bcFormat;
    BCnCompressor bcnCompressor;
    std::mt19937 rng;

    // --- Color Space Conversion Helpers (Refactored for Oklab and dynamic channels) ---
    void PixelToOklab(const uint8_t* pixel, uint8_t channelCount, float* oklab) const;
    void OklabToPixel(const float* oklab, uint8_t channelCount, uint8_t* pixel) const;
    OklabBlock PixelBlockToOklabBlock(const std::vector<uint8_t>& pixelBlock, uint8_t channelCount) const;
    std::vector<uint8_t> OklabBlockToPixelBlock(const OklabBlock& oklabBlock, uint8_t channelCount) const;

    // --- HDR Color Space Conversion Helpers ---
    void RgbaFloatToOklab(const float* pixel, uint8_t channelCount, float* oklab) const;
    void OklabToRgbaFloat(const float* oklab, uint8_t channelCount, float* pixel) const;
    OklabFloatBlock RgbaFloatBlockToOklabBlock(const std::vector<float>& pixelBlock, uint8_t channelCount) const;
    std::vector<float> OklabBlockToRgbaFloatBlock(const OklabFloatBlock& oklabBlock, uint8_t channelCount) const;

    // --- Distance Functions ---
    float BlockDistanceSAD(const uint8_t* a, const uint8_t* b, uint8_t channelCount) const;
    float OklabBlockDistanceSq_SIMD(const OklabBlock& oklabA, const OklabBlock& oklabB) const;
    float OklabFloatBlockDistanceSq_SIMD(const OklabFloatBlock& labA, const OklabFloatBlock& labB) const;

    // --- Block Compression ---
    std::vector<uint8_t> CompressSingleBlock(const uint8_t* pixelBlock, uint8_t channels, const CompressionParams& params);
    std::vector<uint8_t> CompressSingleBlockHDR(const std::vector<float>& pixelBlock, uint8_t channels, const CompressionParams& params);

public:
    VQEncoder(const CompressionConfig& cfg = CompressionConfig());
    void SetFormat(BCFormat format);

    // --- Public API (LDR) ---
    VQCodebook BuildCodebook(
        const std::vector<std::vector<uint8_t>>& allPixelBlocks,
        uint8_t channels,
        std::vector<std::vector<uint8_t>>& outPixelCentroids,
        const CompressionParams& params
    );

    std::vector<uint32_t> QuantizeBlocks(
        const std::vector<std::vector<uint8_t>>& pixelBlocks,
        uint8_t channels,
        const std::vector<std::vector<uint8_t>>& pixelCentroids,
        const CompressionParams& params
    );

    // --- Public API (HDR) ---
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