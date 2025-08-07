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
#include <vector>
#include <memory>

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

    // --- ADDED: Dictionary pointers ---
    ZSTD_CDict* cdict = nullptr;
    ZSTD_DDict* ddict = nullptr;

    // Helper to compress a payload with ZSTD
    // --- MODIFIED: Now supports Long-Distance Matching and uses dictionaries ---
    std::vector<uint8_t> compressWithZstd(const std::vector<uint8_t>& payload, int level, int numThreads, bool enableLdm) {
        size_t compBound = ZSTD_compressBound(payload.size());
        std::vector<uint8_t> compressedPayload(compBound);

        ZSTD_CCtx_setParameter(zstdCtx->cctx, ZSTD_c_nbWorkers, numThreads);
        ZSTD_CCtx_setParameter(zstdCtx->cctx, ZSTD_c_compressionLevel, level);
        // --- ADDED: Enable/disable Long-Distance Matching ---
        ZSTD_CCtx_setParameter(zstdCtx->cctx, ZSTD_c_enableLongDistanceMatching, enableLdm ? 1 : 0);

        size_t compressedSize;
        // --- ADDED: Use dictionary if available ---
        if (cdict) {
            compressedSize = ZSTD_compress_usingCDict(
                zstdCtx->cctx,
                compressedPayload.data(), compBound,
                payload.data(), payload.size(),
                cdict
            );
        }
        else {
            compressedSize = ZSTD_compress2(
                zstdCtx->cctx,
                compressedPayload.data(), compBound,
                payload.data(), payload.size()
            );
        }

        if (ZSTD_isError(compressedSize)) {
            throw std::runtime_error("Zstd compression failed: " + std::string(ZSTD_getErrorName(compressedSize)));
        }
        compressedPayload.resize(compressedSize);
        return compressedPayload;
    }

public:
    VQBCnCompressor() : zstdCtx(std::make_unique<ZstdContext>()) {}

    // --- ADDED: Destructor to free dictionaries ---
    ~VQBCnCompressor() {
        ZSTD_freeCDict(cdict);
        ZSTD_freeDDict(ddict);
    }

    // --- ADDED: Method to load a pre-trained dictionary ---
    void LoadDictionary(const uint8_t* dictData, size_t dictSize) {
        // Free any existing dictionaries
        ZSTD_freeCDict(cdict);
        ZSTD_freeDDict(ddict);

        cdict = ZSTD_createCDict(dictData, dictSize, 1); // Using default compression level 1 for dict creation
        if (!cdict) {
            throw std::runtime_error("Failed to create ZSTD compression dictionary");
        }

        ddict = ZSTD_createDDict(dictData, dictSize);
        if (!ddict) {
            throw std::runtime_error("Failed to create ZSTD decompression dictionary");
        }
    }

	// Main compression function for LDR textures
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

        // --- ADDED: Check for large texture to enable LDM ---
        bool enableLdm = false;
        if(width >= 4000 || height >= 4000) { // Enable LDM for large textures
			enableLdm = true;
		}

        auto bcData = bcnCompressor.CompressRGBA(
            rgbaData, width, height, channels, params.bcFormat,
            params.numThreads, params.bcQuality, params.alphaThreshold
        );
        if (bcData.empty()) {
            throw std::runtime_error("BCn compression failed");
        }
        const size_t blockSize = BCBlockSize::GetSize(params.bcFormat);

        // --- Handle VQ Bypass ---
        if (params.bypassVQ) {
            result.info.compressionFlags |= COMPRESSION_FLAGS_VQ_BYPASSED;
            result.info.storedCodebookEntries = 0;

            if (params.bypassZstd) {
                result.info.compressionFlags |= COMPRESSION_FLAGS_ZSTD_BYPASSED;
                result.compressedData = std::move(bcData);
            }
            else {
                result.compressedData = compressWithZstd(bcData, params.zstdLevel, params.numThreads, enableLdm);
            }
            return result;
        }

        // --- Standard VQ Path ---
        VQEncoder::Config vqConfig;
        vqConfig.metric = params.vq_Metric;
        vqConfig.fastModeSampleRatio = params.vq_FastModeSampleRatio;
		vqConfig.maxIterations = params.vq_maxIterations;
		vqConfig.SetQuality(params.quality);
		vqConfig.min_cb_power = params.vq_min_cb_power;
		vqConfig.max_cb_power = params.vq_max_cb_power;

        VQEncoder vqEncoder(vqConfig);
        vqEncoder.SetFormat(params.bcFormat);

        const size_t numBlocks = bcData.size() / blockSize;
        std::vector<std::vector<uint8_t>> rgbaBlocks(numBlocks);
#pragma omp parallel for num_threads(params.numThreads)
        for (int64_t i = 0; i < numBlocks; ++i) {
            rgbaBlocks[i] = bcnCompressor.DecompressToRGBA(&bcData[i * blockSize], 4, 4, params.bcFormat);
        }

        std::vector<std::vector<uint8_t>> rgbaCentroids;
        result.codebook = vqEncoder.BuildCodebook(rgbaBlocks, channels, rgbaCentroids, params);
        result.indices = vqEncoder.QuantizeBlocks(rgbaBlocks, rgbaCentroids, params);

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
            result.compressedData = compressWithZstd(payloadData, params.zstdLevel, params.numThreads, enableLdm);
        }

        return result;
    }

	// A distinct method for HDR textures
    CompressedTexture CompressHDR(
        const float* rgbaData,
        uint32_t width,
        uint32_t height,
        const CompressionParams& params
    ) {
        if (params.bcFormat != BCFormat::BC6H) {
            throw std::runtime_error("HDR compression only supports BC6H format.");
        }

        // --- Handle VQ Bypass ---
        if (params.bypassVQ) {
            CompressedTexture result;
            result.info.width = width;
            result.info.height = height;
            result.info.format = params.bcFormat;
            result.info.compressionFlags = COMPRESSION_FLAGS_IS_HDR | COMPRESSION_FLAGS_VQ_BYPASSED;
            result.info.storedCodebookEntries = 0;

            bool enableLdm = (width >= 4000 || height >= 4000);

            auto bcData = bcnCompressor.CompressHDR(rgbaData, width, height, params.bcFormat, params.numThreads, params.bcQuality);
            if (bcData.empty()) {
                throw std::runtime_error("HDR BCn compression failed");
            }
            if (params.bypassZstd) {
                result.info.compressionFlags |= COMPRESSION_FLAGS_ZSTD_BYPASSED;
                result.compressedData = std::move(bcData);
            }
            else {
                result.compressedData = compressWithZstd(bcData, params.zstdLevel, params.numThreads, enableLdm);
            }
            return result;
        }

        // --- HDR VQ Path Setup ---
        CompressedTexture result;
        result.info.width = width;
        result.info.height = height;
        result.info.format = params.bcFormat;
        result.info.compressionFlags = COMPRESSION_FLAGS_IS_HDR;

        VQEncoder::Config vqConfig;
        vqConfig.SetQuality(params.quality);
        vqConfig.min_cb_power = params.vq_min_cb_power;
        vqConfig.max_cb_power = params.vq_max_cb_power;
        vqConfig.maxIterations = params.vq_maxIterations;
        vqConfig.fastModeSampleRatio = params.vq_FastModeSampleRatio;

        VQEncoder vqEncoder(vqConfig);
        vqEncoder.SetFormat(params.bcFormat);

        // 1. Chunk input float data into 4x4 blocks
        const size_t numBlocksX = (width + 3) / 4;
        const size_t numBlocksY = (height + 3) / 4;
        const size_t numBlocks = numBlocksX * numBlocksY;
        std::vector<std::vector<float>> rgbaFloatBlocks(numBlocks);

#pragma omp parallel for num_threads(params.numThreads)
        for (int64_t i = 0; i < numBlocks; ++i) {
            rgbaFloatBlocks[i].resize(16 * 4);
            size_t blockX = i % numBlocksX;
            size_t blockY = i / numBlocksX;
            size_t startX = blockX * 4;
            size_t startY = blockY * 4;

            for (size_t y = 0; y < 4; ++y) {
                for (size_t x = 0; x < 4; ++x) {
                    size_t pX = std::min(startX + x, (size_t)width - 1);
                    size_t pY = std::min(startY + y, (size_t)height - 1);
                    const float* srcPixel = &rgbaData[(pY * width + pX) * 4];
                    float* dstPixel = &rgbaFloatBlocks[i][(y * 4 + x) * 4];
                    std::copy(srcPixel, srcPixel + 4, dstPixel);
                }
            }
        }

        std::vector<uint8_t> payloadData;

         // --- STANDARD VQ PATH ---
        std::vector<std::vector<float>> rgbaCentroids;
        result.codebook = vqEncoder.BuildCodebookHDR(rgbaFloatBlocks, rgbaCentroids, params);
        result.indices = vqEncoder.QuantizeBlocksHDR(rgbaFloatBlocks, rgbaCentroids, params);

        result.info.storedCodebookEntries = result.codebook.entries.size();
        const size_t blockSize = BCBlockSize::GetSize(params.bcFormat);
        size_t codebookDataSize = result.info.storedCodebookEntries * blockSize;
        size_t indicesDataSize = result.indices.size() * sizeof(uint32_t);
        payloadData.resize(codebookDataSize + indicesDataSize);

        size_t offset = 0;
        for (const auto& entry : result.codebook.entries) {
            std::memcpy(payloadData.data() + offset, entry.data(), blockSize);
            offset += blockSize;
        }
        std::memcpy(payloadData.data() + offset, result.indices.data(), indicesDataSize);

        result.codebook.entries.clear();
        result.indices.clear();

        // 4. Compress final payload with Zstd
        if (params.bypassZstd) {
            result.info.compressionFlags |= COMPRESSION_FLAGS_ZSTD_BYPASSED;
            result.compressedData = std::move(payloadData);
        }
        else {
            bool enableLdm = (width >= 4000 || height >= 4000);
            result.compressedData = compressWithZstd(payloadData, params.zstdLevel, params.numThreads, enableLdm);
        }

        return result;
    }


    std::vector<uint8_t> DecompressToBCn(const CompressedTexture& compressed, int numThreads = 16) {
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
                // For dictionary compressed data, the size might not be stored in the frame header.
                // We must rely on a sufficiently large buffer and the return value.
                // Let's estimate a size. A 10x ratio is a generous starting point.
                decompressedSize = compressed.compressedData.size() * 10;
            }
            payload.resize(decompressedSize);

            size_t dSize;
            // --- MODIFIED: Use dictionary if available ---
            if (ddict) {
                dSize = ZSTD_decompress_usingDDict(
                    zstdCtx->dctx,
                    payload.data(), payload.size(),
                    compressed.compressedData.data(), compressed.compressedData.size(),
                    ddict
                );
            }
            else {
                dSize = ZSTD_decompressDCtx(
                    zstdCtx->dctx,
                    payload.data(), payload.size(),
                    compressed.compressedData.data(), compressed.compressedData.size()
                );
            }

            if (ZSTD_isError(dSize)) {
                throw std::runtime_error("Zstd decompression failed: " + std::string(ZSTD_getErrorName(dSize)));
            }
            payload.resize(dSize);
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
#pragma omp parallel for num_threads(numThreads)
        for (int64_t i = 0; i < totalBlocks; ++i) {
            uint32_t idx = indicesDataPtr[i];
            if (idx >= numCodebookEntries) {
                continue;
            }
            std::memcpy(bcData.data() + i * blockSize, codebookDataPtr + idx * blockSize, blockSize);
        }

        return bcData;
    }

    std::vector<uint8_t> DecompressToRGBA(const CompressedTexture& compressed) {
        if (compressed.info.compressionFlags & COMPRESSION_FLAGS_IS_HDR) {
            throw std::runtime_error("Cannot decompress HDR texture to 8-bit RGBA. Use DecompressToRGBAF instead.");
        }
        auto bcData = DecompressToBCn(compressed);
        return bcnCompressor.DecompressToRGBA(
            bcData.data(), compressed.info.width, compressed.info.height, compressed.info.format
        );
    }

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