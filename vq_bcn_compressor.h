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
        float vqFastModeSampleRatio = 1.0f;
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

    // NEW FUNCTION: Decompresses directly to BCn format for GPU consumption.
    std::vector<uint8_t> DecompressToBCn(const CompressedTexture& compressed) {
        if (compressed.compressedData.empty()) {
            throw std::runtime_error("Compressed data stream is empty. Cannot decompress.");
        }

        // 1. Decompress the ZSTD stream to get the payload (codebook + indices)
        size_t decompressedSize = ZSTD_getFrameContentSize(compressed.compressedData.data(), compressed.compressedData.size());
        if (decompressedSize == ZSTD_CONTENTSIZE_ERROR || decompressedSize == ZSTD_CONTENTSIZE_UNKNOWN) {
            throw std::runtime_error("Failed to get ZSTD decompressed size. The frame may be malformed or not contain the content size.");
        }

        std::vector<uint8_t> uncompressedPayload(decompressedSize);
        size_t const dSize = ZSTD_decompressDCtx(
            zstdCtx->dctx,
            uncompressedPayload.data(), uncompressedPayload.size(),
            compressed.compressedData.data(), compressed.compressedData.size()
        );

        if (ZSTD_isError(dSize) || dSize != decompressedSize) {
            throw std::runtime_error("Zstd decompression failed: " + std::string(ZSTD_getErrorName(dSize)));
        }

        // 2. Parse the uncompressed payload into a local codebook and indices
        const size_t blockSize = BCBlockSize::GetSize(compressed.info.format);
        const uint32_t numCodebookEntries = compressed.info.storedCodebookEntries;
        const size_t codebookDataSize = numCodebookEntries * blockSize;
        const size_t totalBlocks = compressed.info.GetTotalBlocks();
        const size_t indicesDataSize = totalBlocks * sizeof(uint32_t);

        if (decompressedSize != codebookDataSize + indicesDataSize) {
            throw std::runtime_error("Decompressed data size mismatch.");
        }

        // Use a pointer to avoid copying the codebook data
        const uint8_t* codebookDataPtr = uncompressedPayload.data();
        const uint32_t* indicesDataPtr = reinterpret_cast<const uint32_t*>(uncompressedPayload.data() + codebookDataSize);

        // 3. Reconstruct the BCn data from the parsed codebook and indices
        std::vector<uint8_t> bcData(totalBlocks * blockSize);
#pragma omp parallel for
        for (int64_t i = 0; i < totalBlocks; ++i) {
            uint32_t idx = indicesDataPtr[i];
            // No need for index validation if we trust the compressor, but it's safer for production
            // if (idx >= numCodebookEntries) { continue; }
            std::memcpy(bcData.data() + i * blockSize, codebookDataPtr + idx * blockSize, blockSize);
        }

        return bcData;
    }

    std::vector<uint8_t> DecompressToRGBA(const CompressedTexture& compressed) {
        // This function now chains the BCn decompression with a final conversion to RGBA.
        auto bcData = DecompressToBCn(compressed);

        // Decompress the reconstructed BCn data to final RGBA
        return bcnCompressor.DecompressToRGBA(
            bcData.data(), compressed.info.width, compressed.info.height, compressed.info.format
        );
    }
};