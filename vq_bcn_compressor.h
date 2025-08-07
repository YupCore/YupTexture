#pragma once

#include "vq_bcn_types.h"
#include "vq_encoder.h"
#include "bcn_compressor.h"
#include <zstd.h>
#include <thread>
#include <stdexcept>
#include <atomic>
#include <cstring>
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

    // Helper to compress a payload with ZSTD
    std::vector<uint8_t> compressWithZstd(const std::vector<uint8_t>& payload, int level, bool useMultithreading) {
        size_t compBound = ZSTD_compressBound(payload.size());
        std::vector<uint8_t> compressedPayload(compBound);
        if (useMultithreading) {
            ZSTD_CCtx_setParameter(zstdCtx->cctx, ZSTD_c_nbWorkers, std::thread::hardware_concurrency());
        }
        ZSTD_CCtx_setParameter(zstdCtx->cctx, ZSTD_c_compressionLevel, level);
        size_t compressedSize = ZSTD_compress2(
            zstdCtx->cctx,
            compressedPayload.data(), compBound,
            payload.data(), payload.size()
        );
        if (ZSTD_isError(compressedSize)) {
            throw std::runtime_error("Zstd compression failed: " + std::string(ZSTD_getErrorName(compressedSize)));
        }
        compressedPayload.resize(compressedSize);
        return compressedPayload;
    }

public:
    struct CompressionParams {
        BCFormat bcFormat = BCFormat::BC7;
        float bcQuality = 1.0f;
        uint32_t vqCodebookSize = 256;
        VQEncoder::DistanceMetric vqMetric = VQEncoder::DistanceMetric::PERCEPTUAL_LAB;
        int zstdLevel = 3;
        bool useMultithreading = true;
        float vqFastModeSampleRatio = 1.0f;
        uint8_t alphaThreshold = 128;
        bool bypassVQ = false;
        bool bypassZstd = false;
    };

    VQBCnCompressor() : zstdCtx(std::make_unique<ZstdContext>()) {}

    // --- MODIFIED: LDR Compression entry point ---
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
        result.info.compressionFlags = COMPRESSION_FLAGS_DEFAULT;

        // --- 1. Initial BCn Compression ---
        auto bcData = bcnCompressor.CompressRGBA(
            rgbaData, width, height, channels, params.bcFormat, params.bcQuality, params.alphaThreshold
        );
        if (bcData.empty()) {
            throw std::runtime_error("BCn compression failed");
        }
        const size_t blockSize = BCBlockSize::GetSize(params.bcFormat);

        // --- 2. Handle VQ Bypass ---
        if (params.bypassVQ) {
            result.info.compressionFlags |= COMPRESSION_FLAGS_VQ_BYPASSED;
            result.info.storedCodebookEntries = 0;

            if (params.bypassZstd) {
                result.info.compressionFlags |= COMPRESSION_FLAGS_ZSTD_BYPASSED;
                result.compressedData = std::move(bcData);
            }
            else {
                result.compressedData = compressWithZstd(bcData, params.zstdLevel, params.useMultithreading);
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
        std::vector<std::vector<uint8_t>> rgbaBlocks(numBlocks);
#pragma omp parallel for
        for (int64_t i = 0; i < numBlocks; ++i) {
            rgbaBlocks[i] = bcnCompressor.DecompressToRGBA(&bcData[i * blockSize], 4, 4, params.bcFormat);
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

        result.codebook.entries.clear();
        result.indices.clear();

        if (params.bypassZstd) {
            result.info.compressionFlags |= COMPRESSION_FLAGS_ZSTD_BYPASSED;
            result.compressedData = std::move(payloadData);
        }
        else {
            result.compressedData = compressWithZstd(payloadData, params.zstdLevel, params.useMultithreading);
        }

        return result;
    }

    // --- ADDED: HDR Compression entry point ---
    CompressedTexture Compress(
        const float* rgbaData,
        uint32_t width,
        uint32_t height,
        const CompressionParams& params
    ) {
        CompressedTexture result;
        result.info.width = width;
        result.info.height = height;
        result.info.format = params.bcFormat; // Should be BC6H
        result.info.compressionFlags = COMPRESSION_FLAGS_IS_HDR | COMPRESSION_FLAGS_VQ_BYPASSED;
        result.info.storedCodebookEntries = 0;

        // --- 1. BCn Compression for HDR ---
        // VQ is always bypassed for HDR
        auto bcData = bcnCompressor.CompressHDR(
            rgbaData, width, height, params.bcFormat, params.bcQuality
        );
        if (bcData.empty()) {
            throw std::runtime_error("HDR BCn compression failed");
        }

        // --- 2. Optional ZSTD Compression ---
        if (params.bypassZstd) {
            result.info.compressionFlags |= COMPRESSION_FLAGS_ZSTD_BYPASSED;
            result.compressedData = std::move(bcData);
        }
        else {
            result.compressedData = compressWithZstd(bcData, params.zstdLevel, params.useMultithreading);
        }

        return result;
    }


    std::vector<uint8_t> DecompressToBCn(const CompressedTexture& compressed) {
        if (compressed.compressedData.empty()) {
            throw std::runtime_error("Compressed data stream is empty. Cannot decompress.");
        }

        std::vector<uint8_t> payload;
        if (compressed.info.compressionFlags & COMPRESSION_FLAGS_ZSTD_BYPASSED) {
            payload = compressed.compressedData;
        }
        else {
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

        if (compressed.info.compressionFlags & COMPRESSION_FLAGS_VQ_BYPASSED) {
            return payload;
        }

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
                continue;
            }
            std::memcpy(bcData.data() + i * blockSize, codebookDataPtr + idx * blockSize, blockSize);
        }

        return bcData;
    }

    // --- MODIFIED: DecompressToRGBA now dispatches based on HDR flag ---
    std::vector<uint8_t> DecompressToRGBA(const CompressedTexture& compressed) {
        if (compressed.info.compressionFlags & COMPRESSION_FLAGS_IS_HDR) {
            throw std::runtime_error("Cannot decompress HDR texture to 8-bit RGBA. Use DecompressToRGBAF instead.");
        }
        auto bcData = DecompressToBCn(compressed);
        return bcnCompressor.DecompressToRGBA(
            bcData.data(), compressed.info.width, compressed.info.height, compressed.info.format
        );
    }

    // --- ADDED: DecompressToRGBAF for HDR textures ---
    std::vector<float> DecompressToRGBAF(const CompressedTexture& compressed) {
        if (!(compressed.info.compressionFlags & COMPRESSION_FLAGS_IS_HDR)) {
            throw std::runtime_error("Cannot decompress LDR texture to float RGBA. Use DecompressToRGBA instead.");
        }
        auto bcData = DecompressToBCn(compressed);
        return bcnCompressor.DecompressToRGBAF(
            bcData.data(), compressed.info.width, compressed.info.height, compressed.info.format
        );
    }
};