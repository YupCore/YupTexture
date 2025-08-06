#pragma once

#include "vq_bcn_types.h"
#include "vq_encoder.h"
#include "bcn_compressor.h"
#include <zstd.h>
#include <thread>
#include <stdexcept> // For std::runtime_error

class VQBCnCompressor {
private:
    VQEncoder vqEncoder;
    BCnCompressor bcnCompressor;

    struct ZstdContext {
        ZSTD_CCtx* cctx;
        ZSTD_DCtx* dctx;

        ZstdContext() {
            cctx = ZSTD_createCCtx();
            dctx = ZSTD_createDCtx();
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
        // Configure the VQ Encoder based on parameters
        VQEncoder::Config vqConfig;
        vqConfig.codebookSize = params.vqCodebookSize;
        vqConfig.metric = params.vqMetric;
        vqEncoder = VQEncoder(vqConfig);
        vqEncoder.SetFormat(params.bcFormat);

        CompressedTexture result;
        result.info.width = width;
        result.info.height = height;
        result.info.format = params.bcFormat;
        result.info.mipLevels = 1;

        auto bcData = bcnCompressor.CompressRGBA(
            rgbaData, width, height, params.bcFormat, params.bcQuality
        );

        if (bcData.empty()) {
            throw std::runtime_error("BCn compression failed");
        }

        size_t blockSize = BCBlockSize::GetSize(params.bcFormat);
        result.codebook = vqEncoder.BuildCodebook(bcData, blockSize);
        result.indices = vqEncoder.QuantizeBlocks(bcData, result.codebook);

        size_t codebookDataSize = result.codebook.codebookSize * blockSize;
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
        return result;
    }

    std::vector<uint8_t> DecompressToRGBA(const CompressedTexture& compressed) {
        size_t blockSize = BCBlockSize::GetSize(compressed.info.format);
        size_t totalBlocks = compressed.info.GetTotalBlocks();
        if (totalBlocks != compressed.indices.size()) {
            throw std::runtime_error("Mismatch between texture dimensions and number of indices.");
        }
        std::vector<uint8_t> bcData(totalBlocks * blockSize);

#pragma omp parallel for
        for (int64_t i = 0; i < totalBlocks; ++i) {
            uint32_t idx = compressed.indices[i];
            if (idx >= compressed.codebook.entries.size()) {
                std::fill_n(bcData.data() + i * blockSize, blockSize, 0);
                continue;
            }
            std::memcpy(bcData.data() + i * blockSize, compressed.codebook.entries[idx].data(), blockSize);
        }

        return bcnCompressor.DecompressToRGBA(
            bcData.data(), compressed.info.width, compressed.info.height, compressed.info.format
        );
    }
};