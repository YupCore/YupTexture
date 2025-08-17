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

    // --- Dictionary pointers ---
    void* cdict = nullptr;
    void* ddict = nullptr;

    // Helper to compress a payload with ZSTD
    std::vector<uint8_t> compressWithZstd(const std::vector<uint8_t>& payload, int level, int numThreads, bool enableLdm);

public:
    VQBCnCompressor();

    // --- Destructor to free dictionaries ---
    ~VQBCnCompressor();

    // --- Method to load a pre-trained dictionary ---
    void LoadDictionary(const uint8_t* dictData, size_t dictSize);

    // Main compression function for LDR textures
    CompressedTexture Compress(
        const uint8_t* rgbaData,
        uint32_t width,
        uint32_t height,
        uint8_t channels,
        const CompressionParams& params
    );

    // A distinct method for HDR textures
    CompressedTexture CompressHDR(
        const float* rgbaData,
        uint32_t width,
        uint32_t height,
        const CompressionParams& params
    );

    std::vector<uint8_t> DecompressToBCn(const CompressedTexture& compressed, int numThreads = 16);

    std::vector<uint8_t> DecompressToRGBA(const CompressedTexture& compressed);

    std::vector<float> DecompressToRGBAF(const CompressedTexture& compressed);
};