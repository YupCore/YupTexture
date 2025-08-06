#pragma once

#include "vq_bcn_types.h"
#include <Compressonator.h>
#include <memory>

class BCnCompressor {
private:
    CMP_FORMAT GetCMPFormat(BCFormat format) {
        switch(format) {
            case BCFormat::BC1: return CMP_FORMAT_BC1;
            case BCFormat::BC2: return CMP_FORMAT_BC2;
            case BCFormat::BC3: return CMP_FORMAT_BC3;
            case BCFormat::BC4: return CMP_FORMAT_BC4;
            case BCFormat::BC5: return CMP_FORMAT_BC5;
            case BCFormat::BC6H: return CMP_FORMAT_BC6H;
            case BCFormat::BC7: return CMP_FORMAT_BC7;
            default: return CMP_FORMAT_BC1;
        }
    }
    
public:
    std::vector<uint8_t> CompressRGBA(
        const uint8_t* rgbaData,
        uint32_t width,
        uint32_t height,
        BCFormat format,
        float quality = 1.0f
    ) {
        CMP_Texture srcTexture = {};
        srcTexture.dwSize = sizeof(CMP_Texture);
        srcTexture.dwWidth = width;
        srcTexture.dwHeight = height;
        srcTexture.dwPitch = width * 4;
        srcTexture.format = CMP_FORMAT_RGBA_8888;
        srcTexture.dwDataSize = width * height * 4;
        srcTexture.pData = const_cast<uint8_t*>(rgbaData);
        
        CMP_Texture destTexture = {};
        destTexture.dwSize = sizeof(CMP_Texture);
        destTexture.dwWidth = width;
        destTexture.dwHeight = height;
        destTexture.format = GetCMPFormat(format);
        
        size_t blockSize = BCBlockSize::GetSize(format);
        size_t blocksX = (width + 3) / 4;
        size_t blocksY = (height + 3) / 4;
        size_t compressedSize = blocksX * blocksY * blockSize;
        
        std::vector<uint8_t> compressedData(compressedSize);
        destTexture.dwDataSize = compressedSize;
        destTexture.pData = compressedData.data();
        
        CMP_CompressOptions options = {};
        options.dwSize = sizeof(options);
        options.fquality = quality;
        options.dwnumThreads = 0;  // Use all threads
        
        CMP_ERROR error = CMP_ConvertTexture(&srcTexture, &destTexture, &options, nullptr);
        if(error != CMP_OK) {
            return {};
        }
        
        return compressedData;
    }
    
    std::vector<uint8_t> DecompressToRGBA(
        const uint8_t* bcData,
        uint32_t width,
        uint32_t height,
        BCFormat format
    ) {
        CMP_Texture srcTexture = {};
        srcTexture.dwSize = sizeof(CMP_Texture);
        srcTexture.dwWidth = width;
        srcTexture.dwHeight = height;
        srcTexture.format = GetCMPFormat(format);
        
        size_t blockSize = BCBlockSize::GetSize(format);
        size_t blocksX = (width + 3) / 4;
        size_t blocksY = (height + 3) / 4;
        srcTexture.dwDataSize = blocksX * blocksY * blockSize;
        srcTexture.pData = const_cast<uint8_t*>(bcData);
        
        CMP_Texture destTexture = {};
        destTexture.dwSize = sizeof(CMP_Texture);
        destTexture.dwWidth = width;
        destTexture.dwHeight = height;
        destTexture.format = CMP_FORMAT_RGBA_8888;
        
        std::vector<uint8_t> rgbaData(width * height * 4);
        destTexture.dwDataSize = rgbaData.size();
        destTexture.pData = rgbaData.data();
        
        CMP_ERROR error = CMP_ConvertTexture(&srcTexture, &destTexture, nullptr, nullptr);
        if(error != CMP_OK) {
            return {};
        }
        
        return rgbaData;
    }
};