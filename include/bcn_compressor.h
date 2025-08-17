#pragma once

#include "vq_bcn_types.h"
#include <memory>
#include <vector>

class YUPTEXTURE_API BCnCompressor {
private:
    int32_t GetCMPFormat(BCFormat format);
public:
    // --- LDR Compression ---
    std::vector<uint8_t> CompressRGBA(
        const uint8_t* rgbaData,
        uint32_t width,
        uint32_t height,
        uint8_t channels,
        BCFormat format,
        int numThreads,
        float quality = 1.0f,
        uint8_t alphaThreshold = 128
    );

    // --- HDR Compression ---
    std::vector<uint8_t> CompressHDR(
        const float* rgbaData,
        uint32_t width,
        uint32_t height,
        BCFormat format,
        int numThreads,
        float quality = 1.0f
    );

    // --- LDR Decompression ---
    std::vector<uint8_t> DecompressToRGBA(
        const uint8_t* bcData,
        uint32_t width,
        uint32_t height,
        BCFormat format
    );

    // --- HDR Decompression ---
    std::vector<float> DecompressToRGBAF(
        const uint8_t* bcData,
        uint32_t width,
        uint32_t height,
        BCFormat format
    );
};