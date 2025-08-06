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

struct TextureInfo {
    uint32_t width;
    uint32_t height;
    uint32_t mipLevels;
    BCFormat format;
    uint32_t storedCodebookEntries; // <-- ADDED: Needed for parsing the zstd stream

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
    std::vector<uint8_t> compressedData;  // zstd compressed

    size_t GetUncompressedSize() const {
        return info.GetTotalBlocks() * BCBlockSize::GetSize(info.format);
    }
};