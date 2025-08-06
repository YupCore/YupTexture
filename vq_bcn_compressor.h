#pragma once

#include "vq_bcn_types.h"
#include "vq_encoder.h"
#include "bcn_compressor.h"
#include <zstd.h>
#include <thread>
#include <stdexcept> // For std::runtime_error
#include <atomic>

class VQBCnCompressor {
private:
    BCnCompressor bcnCompressor;

    struct ZstdContext {
        ZSTD_CCtx* cctx;
        ZSTD_DCtx* dctx;

        ZstdContext() {
            cctx = ZSTD_createCCtx();
            dctx = ZSTD_createDCtx();
            if (!cctx || !dctx) {
                throw std::runtime_error("Failed to create ZSTD context");
            }
        }

        ~ZstdContext() {
            ZSTD_freeCCtx(cctx);
            ZSTD_freeDCtx(dctx);
        }
    };

    std::unique_ptr<ZstdContext> zstdCtx;

public:
    struct CompressionParams {
        BCFormat bcFormat = BCFormat::BC7;
        float bcQuality = 1.0f;
        uint32_t vqCodebookSize = 256;
        VQEncoder::DistanceMetric vqMetric = VQEncoder::DistanceMetric::PERCEPTUAL_LAB;
        int zstdLevel = 3;
        bool useMultithreading = true;
    };

    VQBCnCompressor() : zstdCtx(std::make_unique<ZstdContext>()) {}

    CompressedTexture Compress(
        const uint8_t* rgbaData,
        uint32_t width,
        uint32_t height,
        const CompressionParams& params
    ) {
        // --- 1. Configure Encoders ---
        VQEncoder::Config vqConfig;
        vqConfig.codebookSize = params.vqCodebookSize;
        vqConfig.metric = params.vqMetric;
        VQEncoder vqEncoder(vqConfig);
        vqEncoder.SetFormat(params.bcFormat);

        CompressedTexture result;
        result.info.width = width;
        result.info.height = height;
        result.info.format = params.bcFormat;
        result.info.mipLevels = 1;

        // --- 2. Initial BCn Compression ---
        auto bcData = bcnCompressor.CompressRGBA(
            rgbaData, width, height, params.bcFormat, params.bcQuality
        );
        if (bcData.empty()) {
            throw std::runtime_error("BCn compression failed");
        }
        const size_t blockSize = BCBlockSize::GetSize(params.bcFormat);
        const size_t numBlocks = bcData.size() / blockSize;

        // --- OPTIMIZATION: The New, Efficient VQ Pipeline ---

        // 3. Decompress all blocks to RGBA *ONCE*
        std::vector<std::vector<uint8_t>> rgbaBlocks(numBlocks);
#pragma omp parallel for
        for (int64_t i = 0; i < numBlocks; ++i) {
            rgbaBlocks[i] = bcnCompressor.DecompressToRGBA(&bcData[i * blockSize], 4, 4, params.bcFormat);
        }

        // 4. Build the codebook. This function performs K-means and returns the compressed
        //    codebook entries, while also outputting the final uncompressed RGBA centroids.
        std::vector<std::vector<uint8_t>> rgbaCentroids;
        result.codebook = vqEncoder.BuildCodebook(rgbaBlocks, rgbaCentroids);

        // 5. Quantize the blocks. This is now a pure comparison function using the pre-decompressed
        //    blocks and the final centroids from the build step.
        result.indices = vqEncoder.QuantizeBlocks(rgbaBlocks, rgbaCentroids);

        // --- 6. Final ZSTD Compression ---
        result.info.storedCodebookEntries = result.codebook.entries.size();
        size_t codebookDataSize = result.info.storedCodebookEntries * blockSize;
        size_t indicesDataSize = result.indices.size() * sizeof(uint32_t);
        std::vector<uint8_t> dataToCompress(codebookDataSize + indicesDataSize);

        size_t offset = 0;
        for (const auto& entry : result.codebook.entries) {
            std::memcpy(dataToCompress.data() + offset, entry.data(), blockSize);
            offset += blockSize;
        }
        std::memcpy(dataToCompress.data() + offset, result.indices.data(), indicesDataSize);

        size_t compBound = ZSTD_compressBound(dataToCompress.size());
        result.compressedData.resize(compBound);

        if (params.useMultithreading) {
            ZSTD_CCtx_setParameter(zstdCtx->cctx, ZSTD_c_nbWorkers, std::thread::hardware_concurrency());
        }
        ZSTD_CCtx_setParameter(zstdCtx->cctx, ZSTD_c_compressionLevel, params.zstdLevel);

        size_t compressedSize = ZSTD_compress2(
            zstdCtx->cctx,
            result.compressedData.data(), compBound,
            dataToCompress.data(), dataToCompress.size()
        );

        if (ZSTD_isError(compressedSize)) {
            throw std::runtime_error("Zstd compression failed: " + std::string(ZSTD_getErrorName(compressedSize)));
        }

        result.compressedData.resize(compressedSize);
        result.codebook.entries.clear();
        result.indices.clear();

        return result;
    }

    std::vector<uint8_t> DecompressToRGBA(const CompressedTexture& compressed) {
        if (compressed.compressedData.empty()) {
            throw std::runtime_error("Compressed data stream is empty. Cannot decompress.");
        }

        // 1. Decompress the ZSTD stream to get the payload (codebook + indices)
        size_t decompressedSize = ZSTD_getFrameContentSize(compressed.compressedData.data(), compressed.compressedData.size());
        if (decompressedSize == ZSTD_CONTENTSIZE_ERROR || decompressedSize == ZSTD_CONTENTSIZE_UNKNOWN) {
            // Fallback for when content size isn't stored in the frame header
            decompressedSize = compressed.info.GetTotalBlocks() * sizeof(uint32_t) + compressed.info.storedCodebookEntries * BCBlockSize::GetSize(compressed.info.format) * 2; // Estimate
        }

        std::vector<uint8_t> uncompressedPayload;
        size_t dSizeResult = ZSTD_CONTENTSIZE_UNKNOWN;

        // Loop to handle cases where the initial buffer is too small
        for (int i = 0; i < 3; ++i) { // Try up to 3 times with increasing buffer size
            uncompressedPayload.resize(decompressedSize);
            dSizeResult = ZSTD_decompressDCtx(
                zstdCtx->dctx,
                uncompressedPayload.data(), uncompressedPayload.size(),
                compressed.compressedData.data(), compressed.compressedData.size()
            );
            if (!ZSTD_isError(dSizeResult)) {
                if (dSizeResult <= decompressedSize) break; // Success
            }
            else {
                throw std::runtime_error("Zstd decompression failed: " + std::string(ZSTD_getErrorName(dSizeResult)));
            }
            if (ZSTD_getErrorCode(dSizeResult) == ZSTD_error_dstSize_tooSmall) {
                decompressedSize *= 2; // Double buffer size and retry
            }
        }

        if (ZSTD_isError(dSizeResult)) {
            throw std::runtime_error("Zstd decompression failed after retries: " + std::string(ZSTD_getErrorName(dSizeResult)));
        }
        uncompressedPayload.resize(dSizeResult);

        // 2. Parse the uncompressed payload into a local codebook and indices
        const size_t blockSize = BCBlockSize::GetSize(compressed.info.format);
        const uint32_t numCodebookEntries = compressed.info.storedCodebookEntries;
        const size_t codebookDataSize = numCodebookEntries * blockSize;
        const size_t totalBlocks = compressed.info.GetTotalBlocks();
        const size_t indicesDataSize = totalBlocks * sizeof(uint32_t);

        if (uncompressedPayload.size() != codebookDataSize + indicesDataSize) {
            throw std::runtime_error("Decompressed data size mismatch.");
        }

        VQCodebook localCodebook(blockSize, numCodebookEntries);
        localCodebook.entries.resize(numCodebookEntries);

        size_t offset = 0;
        for (uint32_t i = 0; i < numCodebookEntries; ++i) {
            localCodebook.entries[i].resize(blockSize);
            std::memcpy(localCodebook.entries[i].data(), uncompressedPayload.data() + offset, blockSize);
            offset += blockSize;
        }

        std::vector<uint32_t> localIndices(totalBlocks);
        std::memcpy(localIndices.data(), uncompressedPayload.data() + offset, indicesDataSize);

        // 3. Reconstruct the BCn data from the parsed codebook and indices
        std::vector<uint8_t> bcData(totalBlocks * blockSize);

#pragma omp parallel for
        for (int64_t i = 0; i < totalBlocks; ++i) {
            uint32_t idx = localIndices[i];
            if (idx >= localCodebook.entries.size()) {
                // Handle potential error case of an invalid index gracefully
                std::fill_n(bcData.data() + i * blockSize, blockSize, 0);
                continue;
            }
            std::memcpy(bcData.data() + i * blockSize, localCodebook.entries[idx].data(), blockSize);
        }

        // 4. Decompress the reconstructed BCn data to final RGBA
        return bcnCompressor.DecompressToRGBA(
            bcData.data(), compressed.info.width, compressed.info.height, compressed.info.format
        );
    }
};