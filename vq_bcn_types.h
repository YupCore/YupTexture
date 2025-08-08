#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <array>

enum class BCFormat {
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

// --- MODIFIED: Added IS_HDR flag ---
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
    uint32_t storedCodebookEntries;
    uint32_t compressionFlags;

    TextureInfo() :
        width(0),
        height(0),
        format(BCFormat::BC1),
        storedCodebookEntries(0),
        compressionFlags(COMPRESSION_FLAGS_DEFAULT)
    { }

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

struct CompressedTexture {
    TextureInfo info;
    // These are only used during compression and are not part of the final file format.
    VQCodebook codebook;
    std::vector<uint32_t> indices;
    // This holds the final data to be written to a file.
    std::vector<uint8_t> compressedData;

    size_t GetUncompressedSize() const {
        return info.GetTotalBlocks() * BCBlockSize::GetSize(info.format);
    }
};

enum class DistanceMetric {
    RGB_SIMD,       // Fastest: SAD on RGB values, accelerated with AVX2.
    PERCEPTUAL_LAB  // High Quality: Euclidean distance in CIELAB color space.
};

struct CompressionParams {
    BCFormat bcFormat = BCFormat::BC7;
    float bcQuality = 1.0f;
    int zstdLevel = 3;
    int numThreads = 16; // default to 16 threads
    uint8_t alphaThreshold = 128;
	bool useVQ = true; // Vector quantization enabled by default. NOTE: VQ is a very destructive compression method, and should be used only when size is the only concern.
    bool useZstd = true;

    // --- VQ Settings ---
    float vq_FastModeSampleRatio = 1.0f;
    float quality = 0.5f;
    DistanceMetric vq_Metric = DistanceMetric::PERCEPTUAL_LAB;
    uint32_t vq_min_cb_power = 4; // 2^4 = 16 entries at quality=0
    uint32_t vq_max_cb_power = 10; // 2^10 = 1024 entries at quality=1
    uint32_t vq_maxIterations = 32;
};