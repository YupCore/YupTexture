#pragma once

#include "vq_bcn_types.h"
#include <memory>
#include <vector>

class YUPTEXTURE_API BCnCompressor {
private:
    // Gets the Compressonator format for the compressed texture
    int32_t GetCMPFormat(BCFormat format);
    // Gets the Compressonator format for the source (uncompressed) texture
    int32_t GetSourceCMPFormat(uint32_t channelCount, bool isFloat);

public:
    // --- LDR Compression ---
    // Compresses LDR image data (e.g., from a PNG or JPG) into a BCn format.
    std::vector<uint8_t> Compress(
        const uint8_t* inData,
        uint32_t width,
        uint32_t height,
        uint32_t channelCount,
        BCFormat format,
        int numThreads,
        float quality = 1.0f,
        uint8_t alphaThreshold = 128,
        bool flipRGB = false
    );

    // --- HDR Compression ---
    // Compresses HDR image data (float) into a BCn format (typically BC6H).
    std::vector<uint8_t> CompressHDR(
        const float* inData,
        uint32_t width,
        uint32_t height,
        uint32_t channelCount,
        BCFormat format,
        int numThreads,
        float quality = 1.0f
    );

    // --- LDR Decompression ---
    // Decompresses a BCn texture to LDR image data.
    std::vector<uint8_t> Decompress(
        const uint8_t* bcData,
        uint32_t width,
        uint32_t height,
        uint32_t channelCount,
        BCFormat format
    );

    // --- HDR Decompression ---
    // Decompresses a BCn texture (typically BC6H) to HDR (float) image data.
    std::vector<float> DecompressHDR(
        const uint8_t* bcData,
        uint32_t width,
        uint32_t height,
        uint32_t channelCount,
        BCFormat format
    );
};