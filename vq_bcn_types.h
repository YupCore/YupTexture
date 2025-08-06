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
        default: return 16;
        }
    }
};

// ADDED: Flags to indicate which compression steps were used.
enum CompressionFlags : uint32_t {
    COMPRESSION_FLAGS_DEFAULT = 0,
    COMPRESSION_FLAGS_VQ_BYPASSED = 1 << 0, // VQ was skipped, payload is raw BCn data.
    COMPRESSION_FLAGS_ZSTD_BYPASSED = 1 << 1, // ZSTD was skipped, payload is not zstd-compressed.
};

struct TextureInfo {
    uint32_t width;
    uint32_t height;
    uint32_t mipLevels;
    BCFormat format;
    uint32_t storedCodebookEntries;
    uint32_t compressionFlags; // MODIFIED: Bitfield using CompressionFlags

    // ADDED: Default constructor for safety.
    TextureInfo() : width(0), height(0), mipLevels(0), format(BCFormat::BC1), storedCodebookEntries(0), compressionFlags(COMPRESSION_FLAGS_DEFAULT) {}

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
    VQCodebook codebook;
    std::vector<uint32_t> indices;  // Index per block
    std::vector<uint8_t> compressedData;  // Can be zstd compressed, or raw data if zstd is bypassed.

    size_t GetUncompressedSize() const {
        return info.GetTotalBlocks() * BCBlockSize::GetSize(info.format);
    }
};