#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <array>
#include <string>
#include <fstream>
#include <cassert>
#include <cstring>

#if defined(_WIN32) || defined(_WIN64)
#if defined(YUPTEXTURE_EXPORT)
#define YUPTEXTURE_API __declspec(dllexport)
#else
#define YUPTEXTURE_API __declspec(dllimport)
#endif
#else
// GCC/Clang: default visibility for shared libs; empty for static.
#if __GNUC__ >= 4
#define YUPTEXTURE_API __attribute__((visibility("default")))
#else
#define YUPTEXTURE_API
#endif
#endif

enum class BCFormat {
    Unknown = 0,
    BC1 = 1,
    BC2,
    BC3,
    BC4,
    BC5,
    BC6H,
    BC7
};

struct BCBlockSize {
    static constexpr size_t BC1 = 8;
    static constexpr size_t BC2 = 16;
    static constexpr size_t BC3 = 16;
    static constexpr size_t BC4 = 8;
    static constexpr size_t BC5 = 16;
    static constexpr size_t BC6H = 16;
    static constexpr size_t BC7 = 16;

    static size_t GetSize(BCFormat format) {
        switch (format) {
        case BCFormat::BC1: return BC1;
        case BCFormat::BC2: return BC2;
        case BCFormat::BC3: return BC3;
        case BCFormat::BC4: return BC4;
        case BCFormat::BC5: return BC5;
        case BCFormat::BC6H: return BC6H;
        case BCFormat::BC7: return BC7;
        default: return 16; // Should not happen
        }
    }
};

// --- Added IS_HDR flag ---
// Flags to indicate which compression steps were used.
enum CompressionFlags : uint32_t {
    COMPRESSION_FLAGS_DEFAULT = 0,
    COMPRESSION_FLAGS_VQ_BYPASSED = 1 << 0, // VQ was skipped, payload is raw BCn data.
    COMPRESSION_FLAGS_ZSTD_BYPASSED = 1 << 1, // ZSTD was skipped, payload is not zstd-compressed.
    COMPRESSION_FLAGS_IS_HDR = 1 << 2,      // The source texture was HDR (float data).
    COMPRESSION_FLAGS_USES_PQ = 1 << 3
};

struct TextureInfo {
    uint32_t width;
    uint32_t height;
    BCFormat format;
    uint8_t originalChannelCount; // ADDED: Store the original channel count of the source image.
    uint32_t storedCodebookEntries;
    uint32_t compressionFlags;

    TextureInfo() :
        width(0),
        height(0),
        format(BCFormat::BC1),
        originalChannelCount(4), // Default to 4, but will be overwritten on compression.
        storedCodebookEntries(0),
        compressionFlags(COMPRESSION_FLAGS_DEFAULT)
    {
    }

    size_t GetBlocksX() const { return (width + 3) / 4; }
    size_t GetBlocksY() const { return (height + 3) / 4; }
    size_t GetTotalBlocks() const { return GetBlocksX() * GetBlocksY(); }
};

struct VQCodebook {
    std::vector<std::vector<uint8_t>> entries;
    uint32_t blockSize;
    uint32_t codebookSize;

    VQCodebook() : blockSize(0), codebookSize(0) {}
    VQCodebook(uint32_t bSize, uint32_t cbSize)
        : blockSize(bSize), codebookSize(cbSize) {
    }
};

enum class DistanceMetric {
    SAD_SIMD,       // Fastest: SAD on RGB values, accelerated with AVX2.
    PERCEPTUAL_OKLAB  // High Quality: Euclidean distance in OKLAB color space.
};

struct CompressionParams {
    BCFormat bcFormat = BCFormat::BC7;
    float bcQuality = 1.0f;
    int zstdLevel = 3;
    int numThreads = 16; // default to 16 threads
    uint8_t alphaThreshold = 128;
	bool useVQ = true; // Vector quantization enabled by default. NOTE: VQ is a very destructive compression method, and should only be used when size is the main concern.
    bool useZstd = true;

    // --- VQ Settings ---
    float vq_FastModeSampleRatio = 1.0f;
    float quality = 0.5f;
    DistanceMetric vq_Metric = DistanceMetric::PERCEPTUAL_OKLAB;
    uint32_t vq_min_cb_power = 4; // 2^4 = 16 entries at quality=0
    uint32_t vq_max_cb_power = 10; // 2^10 = 1024 entries at quality=1
    uint32_t vq_maxIterations = 32;
};