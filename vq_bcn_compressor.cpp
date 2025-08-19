#include "vq_bcn_compressor.h"
#include "vq_encoder.h"
#include <zstd.h>

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

std::vector<uint8_t>& VQBCnCompressor::Compress(const uint8_t* inData, uint32_t width, uint32_t height, uint8_t channels, const CompressionParams& params)
{
    TextureInfo info;
    info.width = width;
    info.height = height;
    info.format = params.bcFormat;
    info.originalChannelCount = channels; // Store original channel count
    info.compressionFlags = COMPRESSION_FLAGS_DEFAULT;

    bool enableLdm = (width >= 4000 || height >= 4000);

    // Use the generic 'Compress' which handles all channel counts
    auto bcData = bcnCompressor.Compress(
        inData, width, height, channels, params.bcFormat,
        params.numThreads, params.bcQuality
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

    std::vector<uint8_t> outputBuffer;

    // --- Handle VQ Bypass ---
    if (info.compressionFlags & COMPRESSION_FLAGS_VQ_BYPASSED) {
        info.storedCodebookEntries = 0;

        if (info.compressionFlags & COMPRESSION_FLAGS_ZSTD_BYPASSED) {
            outputBuffer.push_back(bcData);
            result.compressedData = std::move(bcData);
        }
        else {
            result.compressedData = compressWithZstd(bcData, params.zstdLevel, params.numThreads, enableLdm);
        }
        return result;
    }

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
    // Decompress blocks back to their original channel count, not forced RGBA
    std::vector<std::vector<uint8_t>> pixelBlocks(numBlocks);
#pragma omp parallel for num_threads(params.numThreads)
    for (int64_t i = 0; i < numBlocks; ++i) {
        pixelBlocks[i] = bcnCompressor.Decompress(&bcData[i * blockSize], 4, 4, channels, params.bcFormat);
    }

    std::vector<std::vector<uint8_t>> pixelCentroids;
    result.codebook = vqEncoder.BuildCodebook(pixelBlocks, channels, pixelCentroids, params);
    result.indices = vqEncoder.QuantizeBlocks(pixelBlocks, channels, pixelCentroids, params);

    info.storedCodebookEntries = result.codebook.entries.size();
    size_t codebookDataSize = info.storedCodebookEntries * blockSize;
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

    if (info.compressionFlags & COMPRESSION_FLAGS_ZSTD_BYPASSED) {
        result.compressedData = std::move(payloadData);
    }
    else {
        result.compressedData = compressWithZstd(payloadData, params.zstdLevel, params.numThreads, enableLdm);
    }

    return result;
}

std::vector<uint8_t>& VQBCnCompressor::CompressHDR(const float* inData, uint32_t width, uint32_t height, uint8_t channels, const CompressionParams& params)
{
    TextureInfo info;
    info.compressionFlags = COMPRESSION_FLAGS_DEFAULT;
    if (!params.useVQ)
        info.compressionFlags |= COMPRESSION_FLAGS_VQ_BYPASSED;
    if (!params.useZstd)
        info.compressionFlags |= COMPRESSION_FLAGS_ZSTD_BYPASSED;

    if (info.width < 32 || info.height < 32) // bypass zstd and vq for VERY small textures(smaller than 32 x 32)
    {
        info.compressionFlags |= COMPRESSION_FLAGS_VQ_BYPASSED;
        info.compressionFlags |= COMPRESSION_FLAGS_ZSTD_BYPASSED;
    }

    // --- Handle VQ Bypass ---
    if (info.compressionFlags & COMPRESSION_FLAGS_VQ_BYPASSED) {
        info.width = width;
        info.height = height;
        info.format = params.bcFormat;
        info.originalChannelCount = channels;
        info.compressionFlags |= COMPRESSION_FLAGS_IS_HDR;
        info.storedCodebookEntries = 0;

        bool enableLdm = (width >= 4000 || height >= 4000);

        auto bcData = bcnCompressor.CompressHDR(inData, width, height, channels, params.bcFormat, params.numThreads, params.bcQuality);
        if (bcData.empty()) {
            throw std::runtime_error("HDR BCn compression failed");
        }
        if (info.compressionFlags & COMPRESSION_FLAGS_ZSTD_BYPASSED) {
            result.compressedData = std::move(bcData);
        }
        else {
            result.compressedData = compressWithZstd(bcData, params.zstdLevel, params.numThreads, enableLdm);
        }
        return result;
    }

    // --- HDR VQ Path Setup ---
    info.width = width;
    info.height = height;
    info.format = params.bcFormat;
    info.originalChannelCount = channels;
    info.compressionFlags = COMPRESSION_FLAGS_IS_HDR;

    CompressionConfig vqConfig;
    vqConfig.SetQuality(params.quality);
    vqConfig.min_cb_power = params.vq_min_cb_power;
    vqConfig.max_cb_power = params.vq_max_cb_power;
    vqConfig.maxIterations = params.vq_maxIterations;
    vqConfig.fastModeSampleRatio = params.vq_FastModeSampleRatio;

    VQEncoder vqEncoder(vqConfig);
    vqEncoder.SetFormat(params.bcFormat);

    // Chunk input float data into 4x4 blocks respecting channel count
    const size_t numBlocksX = (width + 3) / 4;
    const size_t numBlocksY = (height + 3) / 4;
    const size_t numBlocks = numBlocksX * numBlocksY;
    std::vector<std::vector<float>> pixelFloatBlocks(numBlocks);

#pragma omp parallel for num_threads(params.numThreads)
    for (int64_t i = 0; i < numBlocks; ++i) {
        pixelFloatBlocks[i].resize(16 * channels); // Use dynamic channel count
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

    std::vector<uint8_t> payloadData;
    std::vector<std::vector<float>> pixelFloatCentroids;
    result.codebook = vqEncoder.BuildCodebookHDR(pixelFloatBlocks, channels, pixelFloatCentroids, params);
    result.indices = vqEncoder.QuantizeBlocksHDR(pixelFloatBlocks, channels, pixelFloatCentroids, params);

    info.storedCodebookEntries = result.codebook.entries.size();
    const size_t blockSize = BCBlockSize::GetSize(params.bcFormat);
    size_t codebookDataSize = info.storedCodebookEntries * blockSize;
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

    if (info.compressionFlags & COMPRESSION_FLAGS_ZSTD_BYPASSED) {
        result.compressedData = std::move(payloadData);
    }
    else {
        bool enableLdm = (width >= 4000 || height >= 4000);
        result.compressedData = compressWithZstd(payloadData, params.zstdLevel, params.numThreads, enableLdm);
    }

    return result;
}

std::vector<uint8_t> VQBCnCompressor::DecompressToBCn(const std::vector<uint8_t>& compressedData, TextureInfo& outInfo, int numThreads)
{
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
        // --- Use dictionary if available ---
        if (ddict) {
            dSize = ZSTD_decompress_usingDDict(
                (ZSTD_DCtx*)zstdCtx->dctx,
                payload.data(), payload.size(),
                compressed.compressedData.data(), compressed.compressedData.size(),
                (ZSTD_DDict*)ddict
            );
        }
        else {
            dSize = ZSTD_decompressDCtx(
                (ZSTD_DCtx*)zstdCtx->dctx,
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

std::vector<uint8_t> VQBCnCompressor::Decompress(const std::vector<uint8_t>& compressedData, TextureInfo& outInfo)
{
    if (compressed.info.compressionFlags & COMPRESSION_FLAGS_IS_HDR) {
        throw std::runtime_error("Cannot decompress HDR texture to 8-bit data. Use DecompressHDR instead.");
    }
    auto bcData = DecompressToBCn(compressed);
    // Decompress to the original channel count stored in the file info
    auto rawData = bcnCompressor.Decompress(
        bcData.data(), compressed.info.width, compressed.info.height, compressed.info.originalChannelCount, compressed.info.format
    );

    return rawData;
}

std::vector<float> VQBCnCompressor::DecompressHDR(const std::vector<uint8_t>& compressedData, TextureInfo& outInfo)
{
    if (!(compressed.info.compressionFlags & COMPRESSION_FLAGS_IS_HDR)) {
        throw std::runtime_error("Cannot decompress LDR texture to float data. Use Decompress instead.");
    }
    auto bcData = DecompressToBCn(compressed);
    // Decompress to the original channel count stored in the file info
    auto rawFloatData = bcnCompressor.DecompressHDR(
        bcData.data(), compressed.info.width, compressed.info.height, compressed.info.originalChannelCount, compressed.info.format
    );

    return rawFloatData;
}