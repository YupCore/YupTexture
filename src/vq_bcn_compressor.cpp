#include "vq_bcn_compressor.h"
#include "vq_encoder.h"
#include <zstd.h>
#include <iostream>

VQBCnCompressor::ZstdContext::ZstdContext()
{
    cctx = ZSTD_createCCtx();
    dctx = ZSTD_createDCtx();
    if (!cctx || !dctx) {
        throw std::runtime_error("Failed to create ZSTD context");
    }
}

VQBCnCompressor::ZstdContext::~ZstdContext()
{
    ZSTD_freeCCtx((ZSTD_CCtx*)cctx);
    ZSTD_freeDCtx((ZSTD_DCtx*)dctx);
}

std::vector<uint8_t> VQBCnCompressor::compressWithZstd(const std::vector<uint8_t>& payload, int level, int numThreads, bool enableLdm)
{
    size_t compBound = ZSTD_compressBound(payload.size());
    std::vector<uint8_t> compressedPayload(compBound);

    ZSTD_CCtx_setParameter((ZSTD_CCtx*)zstdCtx->cctx, ZSTD_c_nbWorkers, numThreads);
    ZSTD_CCtx_setParameter((ZSTD_CCtx*)zstdCtx->cctx, ZSTD_c_compressionLevel, level);
    // --- Enable/disable Long-Distance Matching ---
    ZSTD_CCtx_setParameter((ZSTD_CCtx*)zstdCtx->cctx, ZSTD_c_enableLongDistanceMatching, enableLdm ? 1 : 0);

    size_t compressedSize;
    // --- Use dictionary if available ---
    if (cdict) {
        compressedSize = ZSTD_compress_usingCDict(
            (ZSTD_CCtx*)zstdCtx->cctx,
            compressedPayload.data(), compBound,
            payload.data(), payload.size(),
            (ZSTD_CDict*)cdict
        );
    }
    else {
        compressedSize = ZSTD_compress2(
            (ZSTD_CCtx*)zstdCtx->cctx,
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

VQBCnCompressor::VQBCnCompressor() : zstdCtx(std::make_unique<ZstdContext>()) {}

VQBCnCompressor::~VQBCnCompressor()
{
    ZSTD_freeCDict((ZSTD_CDict*)cdict);
    ZSTD_freeDDict((ZSTD_DDict*)ddict);
}

void VQBCnCompressor::LoadDictionary(const uint8_t* dictData, size_t dictSize)
{
    // Free any existing dictionaries
    ZSTD_freeCDict((ZSTD_CDict*)cdict);
    ZSTD_freeDDict((ZSTD_DDict*)ddict);

    cdict = ZSTD_createCDict(dictData, dictSize, 1); // Using default compression level 1 for dict creation
    if (!cdict) {
        throw std::runtime_error("Failed to create ZSTD compression dictionary");
    }

    ddict = ZSTD_createDDict(dictData, dictSize);
    if (!ddict) {
        throw std::runtime_error("Failed to create ZSTD decompression dictionary");
    }
}

std::vector<uint8_t> VQBCnCompressor::Compress(const uint8_t* inData, uint32_t width, uint32_t height, uint8_t channels, const CompressionParams& params)
{
    TextureInfo info;
    info.width = width;
    info.height = height;
    info.format = params.bcFormat;
    info.originalChannelCount = channels; // Store original channel count
    info.compressionFlags = COMPRESSION_FLAGS_DEFAULT;

    bool enableLdm = (width >= 4000 || height >= 4000);

    bool flipRGB = !params.useVQ;

    std::cout << "Flip rgb?: " << (flipRGB ? "true" : "false") << std::endl;

    // Use the generic 'Compress' which handles all channel counts
    auto bcData = bcnCompressor.Compress(
        inData, width, height, channels, params.bcFormat,
        params.numThreads, params.bcQuality, params.alphaThreshold, flipRGB
    );
    if (bcData.empty()) {
        throw std::runtime_error("BCn compression failed");
    }

    if (!params.useVQ)
        info.compressionFlags |= COMPRESSION_FLAGS_VQ_BYPASSED;
    if (!params.useZstd)
        info.compressionFlags |= COMPRESSION_FLAGS_ZSTD_BYPASSED;

    if (info.width < 32 || info.height < 32) // bypass zstd and vq for VERY small textures(smaller than 32 x 32)
    {
        info.compressionFlags |= COMPRESSION_FLAGS_VQ_BYPASSED;
        info.compressionFlags |= COMPRESSION_FLAGS_ZSTD_BYPASSED;
    }

    std::vector<uint8_t> finalPayload;

    // --- Handle VQ Bypass ---
    if (info.compressionFlags & COMPRESSION_FLAGS_VQ_BYPASSED) {
        info.storedCodebookEntries = 0;

        if (info.compressionFlags & COMPRESSION_FLAGS_ZSTD_BYPASSED) {
            finalPayload = std::move(bcData);
        }
        else {
            finalPayload = compressWithZstd(bcData, params.zstdLevel, params.numThreads, enableLdm);
        }
    }
    else {
        // --- Standard VQ Path ---
        CompressionConfig vqConfig;
        vqConfig.metric = params.vq_Metric;
        vqConfig.fastModeSampleRatio = params.vq_FastModeSampleRatio;
        vqConfig.maxIterations = params.vq_maxIterations;
        vqConfig.SetQuality(params.quality);
        vqConfig.min_cb_power = params.vq_min_cb_power;
        vqConfig.max_cb_power = params.vq_max_cb_power;

        VQEncoder vqEncoder(vqConfig);
        vqEncoder.SetFormat(params.bcFormat);

        const size_t blockSize = BCBlockSize::GetSize(params.bcFormat);
        const size_t numBlocks = bcData.size() / blockSize;

        std::vector<std::vector<uint8_t>> pixelBlocks(numBlocks);
#pragma omp parallel for num_threads(params.numThreads)
        for (int64_t i = 0; i < numBlocks; ++i) {
            pixelBlocks[i] = bcnCompressor.Decompress(&bcData[i * blockSize], 4, 4, channels, params.bcFormat);
        }

        std::vector<std::vector<uint8_t>> pixelCentroids;
        VQCodebook codebook = vqEncoder.BuildCodebook(pixelBlocks, channels, pixelCentroids, params);
        std::vector<uint32_t> indices = vqEncoder.QuantizeBlocks(pixelBlocks, channels, pixelCentroids, params);

        info.storedCodebookEntries = codebook.entries.size();
        size_t codebookDataSize = info.storedCodebookEntries * blockSize;
        size_t indicesDataSize = indices.size() * sizeof(uint32_t);
        std::vector<uint8_t> payloadData(codebookDataSize + indicesDataSize);

        size_t offset = 0;
        for (const auto& entry : codebook.entries) {
            std::memcpy(payloadData.data() + offset, entry.data(), blockSize);
            offset += blockSize;
        }
        std::memcpy(payloadData.data() + offset, indices.data(), indicesDataSize);

        if (info.compressionFlags & COMPRESSION_FLAGS_ZSTD_BYPASSED) {
            finalPayload = std::move(payloadData);
        }
        else {
            finalPayload = compressWithZstd(payloadData, params.zstdLevel, params.numThreads, enableLdm);
        }
    }

    // --- Final serialization ---
    std::vector<uint8_t> outputBuffer(sizeof(TextureInfo) + finalPayload.size());
    std::memcpy(outputBuffer.data(), &info, sizeof(TextureInfo));
    std::memcpy(outputBuffer.data() + sizeof(TextureInfo), finalPayload.data(), finalPayload.size());

    return outputBuffer;
}

std::vector<uint8_t> VQBCnCompressor::CompressHDR(const float* inData, uint32_t width, uint32_t height, uint8_t channels, const CompressionParams& params)
{
    TextureInfo info;
    info.width = width;
    info.height = height;
    info.format = params.bcFormat;
    info.originalChannelCount = channels;
    info.compressionFlags = COMPRESSION_FLAGS_IS_HDR;

    if (!params.useVQ)
        info.compressionFlags |= COMPRESSION_FLAGS_VQ_BYPASSED;
    if (!params.useZstd)
        info.compressionFlags |= COMPRESSION_FLAGS_ZSTD_BYPASSED;

    if (info.width < 32 || info.height < 32) // bypass zstd and vq for VERY small textures(smaller than 32 x 32)
    {
        info.compressionFlags |= COMPRESSION_FLAGS_VQ_BYPASSED;
        info.compressionFlags |= COMPRESSION_FLAGS_ZSTD_BYPASSED;
    }

    std::vector<uint8_t> finalPayload;
    bool enableLdm = (width >= 4000 || height >= 4000);

    // --- Handle VQ Bypass ---
    if (info.compressionFlags & COMPRESSION_FLAGS_VQ_BYPASSED) {
        info.storedCodebookEntries = 0;

        auto bcData = bcnCompressor.CompressHDR(inData, width, height, channels, params.bcFormat, params.numThreads, params.bcQuality);
        if (bcData.empty()) {
            throw std::runtime_error("HDR BCn compression failed");
        }
        if (info.compressionFlags & COMPRESSION_FLAGS_ZSTD_BYPASSED) {
            finalPayload = std::move(bcData);
        }
        else {
            finalPayload = compressWithZstd(bcData, params.zstdLevel, params.numThreads, enableLdm);
        }
    }
    else {
        // --- HDR VQ Path ---
        CompressionConfig vqConfig;
        vqConfig.SetQuality(params.quality);
        vqConfig.min_cb_power = params.vq_min_cb_power;
        vqConfig.max_cb_power = params.vq_max_cb_power;
        vqConfig.maxIterations = params.vq_maxIterations;
        vqConfig.fastModeSampleRatio = params.vq_FastModeSampleRatio;

        VQEncoder vqEncoder(vqConfig);
        vqEncoder.SetFormat(params.bcFormat);

        const size_t numBlocksX = (width + 3) / 4;
        const size_t numBlocksY = (height + 3) / 4;
        const size_t numBlocks = numBlocksX * numBlocksY;
        std::vector<std::vector<float>> pixelFloatBlocks(numBlocks);

#pragma omp parallel for num_threads(params.numThreads)
        for (int64_t i = 0; i < numBlocks; ++i) {
            pixelFloatBlocks[i].resize(16 * channels);
            size_t blockX = i % numBlocksX;
            size_t blockY = i / numBlocksX;
            size_t startX = blockX * 4;
            size_t startY = blockY * 4;

            for (size_t y = 0; y < 4; ++y) {
                for (size_t x = 0; x < 4; ++x) {
                    size_t pX = std::min(startX + x, (size_t)width - 1);
                    size_t pY = std::min(startY + y, (size_t)height - 1);
                    const float* srcPixel = &inData[(pY * width + pX) * channels];
                    float* dstPixel = &pixelFloatBlocks[i][(y * 4 + x) * channels];
                    std::copy(srcPixel, srcPixel + channels, dstPixel);
                }
            }
        }

        std::vector<std::vector<float>> pixelFloatCentroids;
        VQCodebook codebook = vqEncoder.BuildCodebookHDR(pixelFloatBlocks, channels, pixelFloatCentroids, params);
        std::vector<uint32_t> indices = vqEncoder.QuantizeBlocksHDR(pixelFloatBlocks, channels, pixelFloatCentroids, params);

        info.storedCodebookEntries = codebook.entries.size();
        const size_t blockSize = BCBlockSize::GetSize(params.bcFormat);
        size_t codebookDataSize = info.storedCodebookEntries * blockSize;
        size_t indicesDataSize = indices.size() * sizeof(uint32_t);

        std::vector<uint8_t> payloadData(codebookDataSize + indicesDataSize);
        size_t offset = 0;
        for (const auto& entry : codebook.entries) {
            std::memcpy(payloadData.data() + offset, entry.data(), blockSize);
            offset += blockSize;
        }
        std::memcpy(payloadData.data() + offset, indices.data(), indicesDataSize);

        if (info.compressionFlags & COMPRESSION_FLAGS_ZSTD_BYPASSED) {
            finalPayload = std::move(payloadData);
        }
        else {
            finalPayload = compressWithZstd(payloadData, params.zstdLevel, params.numThreads, enableLdm);
        }
    }

    // --- Final serialization ---
    std::vector<uint8_t> outputBuffer(sizeof(TextureInfo) + finalPayload.size());
    std::memcpy(outputBuffer.data(), &info, sizeof(TextureInfo));
    std::memcpy(outputBuffer.data() + sizeof(TextureInfo), finalPayload.data(), finalPayload.size());

    return outputBuffer;
}

std::vector<uint8_t> VQBCnCompressor::DecompressToBCn(const std::vector<uint8_t>& compressedData, TextureInfo& outInfo, int numThreads)
{
    if (compressedData.size() < sizeof(TextureInfo)) {
        throw std::runtime_error("Compressed data stream is too small to contain a header. Cannot decompress.");
    }

    // --- Deserialization ---
    std::memcpy(&outInfo, compressedData.data(), sizeof(TextureInfo));
    const uint8_t* payloadPtr = compressedData.data() + sizeof(TextureInfo);
    const size_t payloadSize = compressedData.size() - sizeof(TextureInfo);


    std::vector<uint8_t> payload;
    if (outInfo.compressionFlags & COMPRESSION_FLAGS_ZSTD_BYPASSED) {
        payload.assign(payloadPtr, payloadPtr + payloadSize);
    }
    else {
        size_t decompressedSize = ZSTD_getFrameContentSize(payloadPtr, payloadSize);
        if (decompressedSize == ZSTD_CONTENTSIZE_ERROR || decompressedSize == ZSTD_CONTENTSIZE_UNKNOWN) {
            decompressedSize = payloadSize * 10; // Generous starting point
        }
        payload.resize(decompressedSize);

        size_t dSize;
        if (ddict) {
            dSize = ZSTD_decompress_usingDDict(
                (ZSTD_DCtx*)zstdCtx->dctx,
                payload.data(), payload.size(),
                payloadPtr, payloadSize,
                (ZSTD_DDict*)ddict
            );
        }
        else {
            dSize = ZSTD_decompressDCtx(
                (ZSTD_DCtx*)zstdCtx->dctx,
                payload.data(), payload.size(),
                payloadPtr, payloadSize
            );
        }

        if (ZSTD_isError(dSize)) {
            throw std::runtime_error("Zstd decompression failed: " + std::string(ZSTD_getErrorName(dSize)));
        }
        payload.resize(dSize);
    }

    if (outInfo.compressionFlags & COMPRESSION_FLAGS_VQ_BYPASSED) {
        return payload;
    }

    const size_t blockSize = BCBlockSize::GetSize(outInfo.format);
    const uint32_t numCodebookEntries = outInfo.storedCodebookEntries;
    const size_t codebookDataSize = numCodebookEntries * blockSize;
    const size_t totalBlocks = outInfo.GetTotalBlocks();
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

std::vector<uint8_t> VQBCnCompressor::Decompress(const std::vector<uint8_t>& compressedData, TextureInfo& outInfo)
{
    auto bcData = DecompressToBCn(compressedData, outInfo);

    if (outInfo.compressionFlags & COMPRESSION_FLAGS_IS_HDR) {
        throw std::runtime_error("Cannot decompress HDR texture to 8-bit data. Use DecompressHDR instead.");
    }

    auto rawData = bcnCompressor.Decompress(
        bcData.data(), outInfo.width, outInfo.height, outInfo.originalChannelCount, outInfo.format
    );

    return rawData;
}

std::vector<float> VQBCnCompressor::DecompressHDR(const std::vector<uint8_t>& compressedData, TextureInfo& outInfo)
{
    auto bcData = DecompressToBCn(compressedData, outInfo);

    if (!(outInfo.compressionFlags & COMPRESSION_FLAGS_IS_HDR)) {
        throw std::runtime_error("Cannot decompress LDR texture to float data. Use Decompress instead.");
    }

    auto rawFloatData = bcnCompressor.DecompressHDR(
        bcData.data(), outInfo.width, outInfo.height, outInfo.originalChannelCount, outInfo.format
    );

    return rawFloatData;
}