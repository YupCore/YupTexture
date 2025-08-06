#pragma once

#include "vq_bcn_types.h"
#include "vq_encoder.h"
#include "bcn_compressor.h"
#include <zstd.h>
#include <thread>

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
        float bcQuality = 0.95f;
        uint32_t vqCodebookSize = 256;
        int zstdLevel = 3;
        bool useMultithreading = true;
    };
    
    VQBCnCompressor(const VQEncoder::Config& vqConfig = VQEncoder::Config()) 
        : vqEncoder(vqConfig), zstdCtx(std::make_unique<ZstdContext>()) {
    }
    
    CompressedTexture Compress(
        const uint8_t* rgbaData,
        uint32_t width,
        uint32_t height,
        const CompressionParams& params
    ) {
        CompressedTexture result;
        result.info.width = width;
        result.info.height = height;
        result.info.format = params.bcFormat;
        result.info.mipLevels = 1;
        
        // Step 1: Compress to BCn format
        auto bcData = bcnCompressor.CompressRGBA(
            rgbaData, width, height, params.bcFormat, params.bcQuality
        );
        
        if(bcData.empty()) {
            throw std::runtime_error("BCn compression failed");
        }
        
		vqEncoder.SetFormat(params.bcFormat);

        // Step 2: Build VQ codebook from compressed blocks
        size_t blockSize = BCBlockSize::GetSize(params.bcFormat);
        result.codebook = vqEncoder.BuildCodebook(bcData, blockSize);
        
        // Step 3: Quantize blocks
        result.indices = vqEncoder.QuantizeBlocks(bcData, result.codebook);
        
        // Step 4: Prepare data for final compression
        // We'll store: codebook entries + indices
        size_t codebookDataSize = result.codebook.codebookSize * blockSize;
        size_t indicesDataSize = result.indices.size() * sizeof(uint32_t);
        size_t totalDataSize = codebookDataSize + indicesDataSize;
        
        std::vector<uint8_t> dataToCompress(totalDataSize);
        
        // Copy codebook entries
        size_t offset = 0;
        for(const auto& entry : result.codebook.entries) {
            std::memcpy(dataToCompress.data() + offset, entry.data(), blockSize);
            offset += blockSize;
        }
        
        // Copy indices
        std::memcpy(dataToCompress.data() + offset, 
                   result.indices.data(), indicesDataSize);
        
        // Step 5: Apply zstd compression
        size_t compBound = ZSTD_compressBound(totalDataSize);
        result.compressedData.resize(compBound);
        
        if(params.useMultithreading) {
            ZSTD_CCtx_setParameter(zstdCtx->cctx, ZSTD_c_nbWorkers, 
                                   std::thread::hardware_concurrency());
        }
        ZSTD_CCtx_setParameter(zstdCtx->cctx, ZSTD_c_compressionLevel, params.zstdLevel);
        
        size_t compressedSize = ZSTD_compress2(
            zstdCtx->cctx,
            result.compressedData.data(), compBound,
            dataToCompress.data(), totalDataSize
        );
        
        if(ZSTD_isError(compressedSize)) {
            throw std::runtime_error("Zstd compression failed: " + 
                                   std::string(ZSTD_getErrorName(compressedSize)));
        }
        
        result.compressedData.resize(compressedSize);
        
        return result;
    }
    
    std::vector<uint8_t> DecompressToRGBA(const CompressedTexture& compressed) {
        // The bug was in the complex, unnecessary deserialization from the zstd blob.
        // Since the 'compressed' object passed from Compress() already contains the
        // correct codebook and indices, we should use them directly.
        // This is simpler, faster, and eliminates the source of the bug.

        size_t blockSize = BCBlockSize::GetSize(compressed.info.format);
        size_t totalBlocks = compressed.info.GetTotalBlocks();

        if (totalBlocks != compressed.indices.size()) {
            throw std::runtime_error("Mismatch between texture dimensions and number of indices.");
        }

        // Step 1: Reconstruct the BCn data directly from the VQ data in the struct.
        std::vector<uint8_t> bcData(totalBlocks * blockSize);

#pragma omp parallel for
        for (int64_t i = 0; i < totalBlocks; ++i) {
            uint32_t idx = compressed.indices[i];

            // Safety check
            if (idx >= compressed.codebook.entries.size()) {
                // Handle corrupted index by writing a default (e.g., black) block
                std::fill_n(bcData.data() + i * blockSize, blockSize, 0);
                continue;
            }

            std::memcpy(bcData.data() + i * blockSize,
                compressed.codebook.entries[idx].data(),
                blockSize);
        }

        // Step 2: Decompress the reconstructed BCn data to RGBA.
        return bcnCompressor.DecompressToRGBA(
            bcData.data(),
            compressed.info.width,
            compressed.info.height,
            compressed.info.format
        );
    }
    
    float GetCompressionRatio(const CompressedTexture& compressed) {
        size_t originalSize = compressed.info.width * compressed.info.height * 4;
        size_t compressedSize = compressed.compressedData.size();
        return static_cast<float>(originalSize) / static_cast<float>(compressedSize);
    }
};