#pragma once

#include "bcn_compressor.h"
#include <thread>
#include <stdexcept>
#include <atomic>
#include <cstring>
#include <string>
#include <vector>
#include <memory>

class YUPTEXTURE_API VQBCnCompressor {
private:
    BCnCompressor bcnCompressor;

    struct ZstdContext {
        void* cctx;
        void* dctx;
        ZstdContext();
        ~ZstdContext();
    };

    std::unique_ptr<ZstdContext> zstdCtx;
    void* cdict = nullptr;
    void* ddict = nullptr;

    std::vector<uint8_t> compressWithZstd(const std::vector<uint8_t>& payload, int level, int numThreads, bool enableLdm);

public:
    VQBCnCompressor();
    ~VQBCnCompressor();

    void LoadDictionary(const uint8_t* dictData, size_t dictSize);

    // Main compression function for LDR textures
    std::vector<uint8_t>& Compress(
        const uint8_t* inData, // Renamed from rgbaData
        uint32_t width,
        uint32_t height,
        uint8_t channels,
        const CompressionParams& params
    );

    // A distinct method for HDR textures
    std::vector<uint8_t>& CompressHDR(
        const float* inData, // Renamed from rgbaData
        uint32_t width,
        uint32_t height,
        uint8_t channels, // Added channels parameter
        const CompressionParams& params
    );

    std::vector<uint8_t> DecompressToBCn(const std::vector<uint8_t>& compressedData, TextureInfo& outInfo, int numThreads = 16);

    // Decompresses to the original channel count stored in the texture info.
    std::vector<uint8_t> Decompress(const std::vector<uint8_t>& compressedData, TextureInfo& outInfo);

    // Decompresses HDR to the original channel count stored in the texture info.
    std::vector<float> DecompressHDR(const std::vector<uint8_t>& compressedData, TextureInfo& outInfo);
};