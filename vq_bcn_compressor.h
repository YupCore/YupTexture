#pragma once

#include "vq_bcn_types.h"
#include "vq_encoder.h"
#include "bcn_compressor.h"
#include <zstd.h>
#include <thread>
#include <stdexcept> // For std::runtime_error
#include <atomic>
#include <cstring> // For memcpy
#include <string>

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
		// Internal quality setting for BCn compression
        float bcQuality = 1.0f;
		// Should be set to 128 for grayscale textures, 256 for normals and 512 for albedo textures(recommended)
        uint32_t vqCodebookSize = 256;
        VQEncoder::DistanceMetric vqMetric = VQEncoder::DistanceMetric::PERCEPTUAL_LAB;
		// Should be set in range 1-10, higher than that will cause massive slowdown
        int zstdLevel = 3;
        bool useMultithreading = true;
		// 1.0f means use all samples, 0.5f means use half of the samples, etc
        float vqFastModeSampleRatio = 1.0f;

		uint8_t alphaThreshold = 128; // Threshold for alpha channel, default is 128

        // Do not compress with VQ
        bool bypassVQ = false;
		// Do not compress with ZSTD
        bool bypassZstd = false;
    };

    VQBCnCompressor() : zstdCtx(std::make_unique<ZstdContext>()) {}

    CompressedTexture Compress(
        const uint8_t* rgbaData,
        uint32_t width,
        uint32_t height,
		uint8_t channels,
        const CompressionParams& params
    ) {
        CompressedTexture result;
        result.info.width = width;
        result.info.height = height;
        result.info.format = params.bcFormat;
        result.info.compressionFlags = COMPRESSION_FLAGS_DEFAULT; // Start with default

        // --- 1. Initial BCn Compression ---
        auto bcData = bcnCompressor.CompressRGBA(
            rgbaData, width, height, channels, params.bcFormat, params.bcQuality
        );
        if (bcData.empty()) {
            throw std::runtime_error("BCn compression failed");
        }
        const size_t blockSize = BCBlockSize::GetSize(params.bcFormat);

        // --- 2. Handle VQ Bypass ---
        if (params.bypassVQ) {
            result.info.compressionFlags |= COMPRESSION_FLAGS_VQ_BYPASSED;
            result.info.storedCodebookEntries = 0; // No codebook

            if (params.bypassZstd) {
                // VQ bypassed, ZSTD bypassed: store raw BCn data
                result.info.compressionFlags |= COMPRESSION_FLAGS_ZSTD_BYPASSED;
                result.compressedData = std::move(bcData);
            }
            else {
                // VQ bypassed, ZSTD enabled: compress raw BCn data with zstd
                size_t compBound = ZSTD_compressBound(bcData.size());
                result.compressedData.resize(compBound);
                if (params.useMultithreading) {
                    ZSTD_CCtx_setParameter(zstdCtx->cctx, ZSTD_c_nbWorkers, std::thread::hardware_concurrency());
                }
                ZSTD_CCtx_setParameter(zstdCtx->cctx, ZSTD_c_compressionLevel, params.zstdLevel);
                size_t compressedSize = ZSTD_compress2(
                    zstdCtx->cctx,
                    result.compressedData.data(), compBound,
                    bcData.data(), bcData.size()
                );
                if (ZSTD_isError(compressedSize)) {
                    throw std::runtime_error("Zstd compression of raw BCn data failed: " + std::string(ZSTD_getErrorName(compressedSize)));
                }
                result.compressedData.resize(compressedSize);
            }
            return result;
        }

        // --- 3. Standard VQ Path ---
        VQEncoder::Config vqConfig;
        vqConfig.codebookSize = params.vqCodebookSize;
        vqConfig.metric = params.vqMetric;
        vqConfig.fastModeSampleRatio = params.vqFastModeSampleRatio;
        VQEncoder vqEncoder(vqConfig);
        vqEncoder.SetFormat(params.bcFormat);

        const size_t numBlocks = bcData.size() / blockSize;
        size_t numGroupedBlocks;
        std::vector<std::vector<uint8_t>> rgbaBlocks(numBlocks);
#pragma omp parallel for
        for (int64_t i = 0; i < numBlocks; ++i) {
            rgbaBlocks[i] = bcnCompressor.DecompressToRGBA(&bcData[i * blockSize], 4, 4, params.bcFormat); // This is needed to correctly compress alpha
        }

        std::vector<std::vector<uint8_t>> rgbaCentroids;
        result.codebook = vqEncoder.BuildCodebook(rgbaBlocks, channels, rgbaCentroids, params.alphaThreshold);
        result.indices = vqEncoder.QuantizeBlocks(rgbaBlocks, rgbaCentroids);

        // --- 4. Final Data Aggregation & Optional ZSTD Compression ---
        result.info.storedCodebookEntries = result.codebook.entries.size();
        size_t codebookDataSize = result.info.storedCodebookEntries * blockSize;
        size_t indicesDataSize = result.indices.size() * sizeof(uint32_t);
        std::vector<uint8_t> payloadData(codebookDataSize + indicesDataSize);

        size_t offset = 0;
        for (const auto& entry : result.codebook.entries) {
            std::memcpy(payloadData.data() + offset, entry.data(), blockSize);
            offset += blockSize;
        }
        std::memcpy(payloadData.data() + offset, result.indices.data(), indicesDataSize);

        // Clear temporary data
        result.codebook.entries.clear();
        result.indices.clear();

        if (params.bypassZstd) {
            // VQ enabled, ZSTD bypassed: store raw codebook+indices payload
            result.info.compressionFlags |= COMPRESSION_FLAGS_ZSTD_BYPASSED;
            result.compressedData = std::move(payloadData);
        }
        else {
            // VQ enabled, ZSTD enabled: compress payload (original path)
            size_t compBound = ZSTD_compressBound(payloadData.size());
            result.compressedData.resize(compBound);
            if (params.useMultithreading) {
                ZSTD_CCtx_setParameter(zstdCtx->cctx, ZSTD_c_nbWorkers, std::thread::hardware_concurrency());
            }
            ZSTD_CCtx_setParameter(zstdCtx->cctx, ZSTD_c_compressionLevel, params.zstdLevel);
            size_t compressedSize = ZSTD_compress2(
                zstdCtx->cctx,
                result.compressedData.data(), compBound,
                payloadData.data(), payloadData.size()
            );
            if (ZSTD_isError(compressedSize)) {
                throw std::runtime_error("Zstd compression failed: " + std::string(ZSTD_getErrorName(compressedSize)));
            }
            result.compressedData.resize(compressedSize);
        }

        return result;
    }

    std::vector<uint8_t> DecompressToBCn(const CompressedTexture& compressed) {
        if (compressed.compressedData.empty()) {
            throw std::runtime_error("Compressed data stream is empty. Cannot decompress.");
        }

        // --- 1. Handle ZSTD Bypass ---
        std::vector<uint8_t> payload;
        if (compressed.info.compressionFlags & COMPRESSION_FLAGS_ZSTD_BYPASSED) {
            // Data is not zstd-compressed, the payload is the raw data.
            payload = compressed.compressedData;
        }
        else {
            // Decompress the ZSTD stream to get the payload
            size_t decompressedSize = ZSTD_getFrameContentSize(compressed.compressedData.data(), compressed.compressedData.size());
            if (decompressedSize == ZSTD_CONTENTSIZE_ERROR || decompressedSize == ZSTD_CONTENTSIZE_UNKNOWN) {
                throw std::runtime_error("Failed to get ZSTD decompressed size. The frame may be malformed or not contain the content size.");
            }
            payload.resize(decompressedSize);
            size_t const dSize = ZSTD_decompressDCtx(
                zstdCtx->dctx,
                payload.data(), payload.size(),
                compressed.compressedData.data(), compressed.compressedData.size()
            );
            if (ZSTD_isError(dSize) || dSize != decompressedSize) {
                throw std::runtime_error("Zstd decompression failed: " + std::string(ZSTD_getErrorName(dSize)));
            }
        }

        // --- 2. Handle VQ Bypass ---
        if (compressed.info.compressionFlags & COMPRESSION_FLAGS_VQ_BYPASSED) {
            // If VQ was bypassed, the payload is already the final BCn data.
            return payload;
        }

        // --- 3. Standard VQ Reconstruction Path ---
        const size_t blockSize = BCBlockSize::GetSize(compressed.info.format);
        const uint32_t numCodebookEntries = compressed.info.storedCodebookEntries;
        const size_t codebookDataSize = numCodebookEntries * blockSize;
        const size_t totalBlocks = compressed.info.GetTotalBlocks();
        const size_t indicesDataSize = totalBlocks * sizeof(uint32_t);

        if (payload.size() != codebookDataSize + indicesDataSize) {
            throw std::runtime_error("Decompressed data size mismatch. Expected codebook + indices.");
        }

        const uint8_t* codebookDataPtr = payload.data();
        const uint32_t* indicesDataPtr = reinterpret_cast<const uint32_t*>(payload.data() + codebookDataSize);

        std::vector<uint8_t> bcData(totalBlocks * blockSize);
#pragma omp parallel for
        for (int64_t i = 0; i < totalBlocks; ++i) {
            uint32_t idx = indicesDataPtr[i];
            if (idx >= numCodebookEntries) {
                // This should not happen with a valid file, but as a safeguard:
                // You could fill with a default color block or leave it uninitialized.
                // For now, we just proceed, which might copy invalid data if the index is out of bounds.
                // A robust solution would be to check and handle.
                continue;
            }
            std::memcpy(bcData.data() + i * blockSize, codebookDataPtr + idx * blockSize, blockSize);
        }

        return bcData;
    }

    std::vector<uint8_t> DecompressToRGBA(const CompressedTexture& compressed) {
        auto bcData = DecompressToBCn(compressed);
        return bcnCompressor.DecompressToRGBA(
            bcData.data(), compressed.info.width, compressed.info.height, compressed.info.format
        );
    }
};