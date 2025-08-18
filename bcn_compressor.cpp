#include "bcn_compressor.h"
#include <compressonator.h>
#include <stdexcept>

// Maps the internal BCFormat enum to the Compressonator CMP_FORMAT enum.
int32_t BCnCompressor::GetCMPFormat(BCFormat format)
{
    switch (format) {
    case BCFormat::BC1: return CMP_FORMAT_BC1;
    case BCFormat::BC2: return CMP_FORMAT_BC2;
    case BCFormat::BC3: return CMP_FORMAT_BC3;
    case BCFormat::BC4: return CMP_FORMAT_BC4;
    case BCFormat::BC5: return CMP_FORMAT_BC5;
    case BCFormat::BC6H: return CMP_FORMAT_BC6H;
    case BCFormat::BC7: return CMP_FORMAT_BC7;
    default:
        // Fallback or error for unknown formats
        return CMP_FORMAT_Unknown;
    }
}

// Determines the correct source/destination CMP_FORMAT based on channel count and data type.
CMP_FORMAT BCnCompressor::GetSourceCMPFormat(uint32_t channelCount, bool isFloat)
{
    if (isFloat) {
        switch (channelCount) {
        case 1: return CMP_FORMAT_R_32F;
        case 2: return CMP_FORMAT_RG_32F;
        case 3: return CMP_FORMAT_RGB_32F;
        case 4: return CMP_FORMAT_RGBA_32F;
        default: return CMP_FORMAT_Unknown;
        }
    }
    else {
        switch (channelCount) {
        case 1: return CMP_FORMAT_R_8;
        case 2: return CMP_FORMAT_RG_8;
        case 3: return CMP_FORMAT_RGB_888;
        case 4: return CMP_FORMAT_RGBA_8888;
        default: return CMP_FORMAT_Unknown;
        }
    }
}


std::vector<uint8_t> BCnCompressor::Compress(const uint8_t* inData, uint32_t width, uint32_t height, uint32_t channelCount, BCFormat format, int numThreads, float quality)
{
    CMP_Texture srcTexture = {};
    srcTexture.dwSize = sizeof(CMP_Texture);
    srcTexture.dwWidth = width;
    srcTexture.dwHeight = height;
    srcTexture.dwPitch = width * channelCount;
    srcTexture.format = GetSourceCMPFormat(channelCount, false);
    srcTexture.dwDataSize = width * height * channelCount;
    srcTexture.pData = const_cast<uint8_t*>(inData);

    if (srcTexture.format == CMP_FORMAT_Unknown) {
        // Handle unsupported channel count
        return {};
    }

    CMP_Texture destTexture = {};
    destTexture.dwSize = sizeof(CMP_Texture);
    destTexture.dwWidth = width;
    destTexture.dwHeight = height;
    destTexture.format = static_cast<CMP_FORMAT>(GetCMPFormat(format));

    size_t blockSize = BCBlockSize::GetSize(format);
    size_t blocksX = (width + 3) / 4;
    size_t blocksY = (height + 3) / 4;
    size_t compressedSize = blocksX * blocksY * blockSize;

    std::vector<uint8_t> compressedData(compressedSize);
    destTexture.dwDataSize = static_cast<CMP_DWORD>(compressedSize);
    destTexture.pData = compressedData.data();

    CMP_CompressOptions options = {};
    options.dwSize = sizeof(options);
    options.fquality = quality;
    options.dwnumThreads = numThreads;

    CMP_ERROR error = CMP_ConvertTexture(&srcTexture, &destTexture, &options, nullptr);
    if (error != CMP_OK) {
        return {};
    }

    return compressedData;
}

std::vector<uint8_t> BCnCompressor::CompressHDR(const float* inData, uint32_t width, uint32_t height, uint32_t channelCount, BCFormat format, int numThreads, float quality)
{
    CMP_Texture srcTexture = {};
    srcTexture.dwSize = sizeof(CMP_Texture);
    srcTexture.dwWidth = width;
    srcTexture.dwHeight = height;
    srcTexture.dwPitch = width * channelCount * sizeof(float);
    srcTexture.format = GetSourceCMPFormat(channelCount, true);
    srcTexture.dwDataSize = width * height * channelCount * sizeof(float);
    srcTexture.pData = (CMP_BYTE*)inData;

    if (srcTexture.format == CMP_FORMAT_Unknown) {
        // Handle unsupported channel count
        return {};
    }

    CMP_Texture destTexture = {};
    destTexture.dwSize = sizeof(CMP_Texture);
    destTexture.dwWidth = width;
    destTexture.dwHeight = height;
    destTexture.format = static_cast<CMP_FORMAT>(GetCMPFormat(format)); // Should be BC6H for HDR

    size_t blockSize = BCBlockSize::GetSize(format);
    size_t blocksX = (width + 3) / 4;
    size_t blocksY = (height + 3) / 4;
    size_t compressedSize = blocksX * blocksY * blockSize;

    std::vector<uint8_t> compressedData(compressedSize);
    destTexture.dwDataSize = static_cast<CMP_DWORD>(compressedSize);
    destTexture.pData = compressedData.data();

    CMP_CompressOptions options = {};
    options.dwSize = sizeof(options);
    options.fquality = quality;
    options.dwnumThreads = numThreads;

    CMP_ERROR error = CMP_ConvertTexture(&srcTexture, &destTexture, &options, nullptr);
    if (error != CMP_OK) {
        return {};
    }

    return compressedData;
}

std::vector<uint8_t> BCnCompressor::Decompress(const uint8_t* bcData, uint32_t width, uint32_t height, uint32_t channelCount, BCFormat format)
{
    CMP_Texture srcTexture = {};
    srcTexture.dwSize = sizeof(CMP_Texture);
    srcTexture.dwWidth = width;
    srcTexture.dwHeight = height;
    srcTexture.format = static_cast<CMP_FORMAT>(GetCMPFormat(format));

    size_t blockSize = BCBlockSize::GetSize(format);
    size_t blocksX = (width + 3) / 4;
    size_t blocksY = (height + 3) / 4;
    srcTexture.dwDataSize = static_cast<CMP_DWORD>(blocksX * blocksY * blockSize);
    srcTexture.pData = const_cast<uint8_t*>(bcData);

    CMP_Texture destTexture = {};
    destTexture.dwSize = sizeof(CMP_Texture);
    destTexture.dwWidth = width;
    destTexture.dwHeight = height;
    destTexture.format = GetSourceCMPFormat(channelCount, false);
    destTexture.dwPitch = width * channelCount;

    if (destTexture.format == CMP_FORMAT_Unknown) {
        // Handle unsupported channel count
        return {};
    }

    std::vector<uint8_t> outData(width * height * channelCount);
    destTexture.dwDataSize = static_cast<CMP_DWORD>(outData.size());
    destTexture.pData = outData.data();

    CMP_ERROR error = CMP_ConvertTexture(&srcTexture, &destTexture, nullptr, nullptr);
    if (error != CMP_OK) {
        return {};
    }

    return outData;
}

std::vector<float> BCnCompressor::DecompressHDR(const uint8_t* bcData, uint32_t width, uint32_t height, uint32_t channelCount, BCFormat format)
{
    CMP_Texture srcTexture = {};
    srcTexture.dwSize = sizeof(CMP_Texture);
    srcTexture.dwWidth = width;
    srcTexture.dwHeight = height;
    srcTexture.format = static_cast<CMP_FORMAT>(GetCMPFormat(format));

    size_t blockSize = BCBlockSize::GetSize(format);
    size_t blocksX = (width + 3) / 4;
    size_t blocksY = (height + 3) / 4;
    srcTexture.dwDataSize = static_cast<CMP_DWORD>(blocksX * blocksY * blockSize);
    srcTexture.pData = const_cast<uint8_t*>(bcData);

    CMP_Texture destTexture = {};
    destTexture.dwSize = sizeof(CMP_Texture);
    destTexture.dwWidth = width;
    destTexture.dwHeight = height;
    destTexture.format = GetSourceCMPFormat(channelCount, true);
    destTexture.dwPitch = width * channelCount * sizeof(float);

    if (destTexture.format == CMP_FORMAT_Unknown) {
        // Handle unsupported channel count
        return {};
    }

    std::vector<float> outData(width * height * channelCount);
    destTexture.dwDataSize = static_cast<CMP_DWORD>(outData.size() * sizeof(float));
    destTexture.pData = (CMP_BYTE*)outData.data();

    CMP_ERROR error = CMP_ConvertTexture(&srcTexture, &destTexture, nullptr, nullptr);
    if (error != CMP_OK) {
        return {};
    }

    return outData;
}