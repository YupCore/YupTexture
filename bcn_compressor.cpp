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
int32_t BCnCompressor::GetSourceCMPFormat(uint32_t channelCount, bool isFloat)
{
    if (isFloat) {
        switch (channelCount) {
        case 1: return CMP_FORMAT_R_32F;
        case 2: return CMP_FORMAT_RG_32F;
        case 3: return CMP_FORMAT_RGBE_32F; // Sadly, Compressonator does not support RGB_32F raw, so we need to passs it as RGBE if want pure RGB
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

std::vector<uint8_t> BCnCompressor::Compress(const uint8_t* inData, uint32_t width, uint32_t height, uint32_t channelCount, BCFormat format, int numThreads, float quality, uint8_t alphaThreshold)
{
    CMP_Texture srcTexture = {};
    srcTexture.dwSize = sizeof(CMP_Texture);
    srcTexture.dwWidth = width;
    srcTexture.dwHeight = height;
    srcTexture.dwPitch = width * channelCount;
    srcTexture.format = (CMP_FORMAT)GetSourceCMPFormat(channelCount, false);
    srcTexture.dwDataSize = width * height * channelCount;
    srcTexture.pData = const_cast<uint8_t*>(inData);

    if (srcTexture.format == CMP_FORMAT_Unknown) {
        // Handle unsupported channel count
        return {};
    }

    //==========================================================================
    // if the source format is RGB_888 swizzle it to BGR_888, because yes, it just works
    //==========================================================================
    if (srcTexture.format == CMP_FORMAT_RGB_888)
    {
        unsigned char red;
        for (CMP_DWORD i = 0; i < srcTexture.dwDataSize; i += 3)
        {
            red = srcTexture.pData[i];
            srcTexture.pData[i] = srcTexture.pData[i + 2];
            srcTexture.pData[i + 2] = red;
        }
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
    options.bUseAdaptiveWeighting = true;
    options.bUseRefinementSteps = true;
    options.doDeltaEncodeBRLG = true;
    options.doPreconditionBRLG = true;
    options.doSwizzleBRLG = true;

    if (channelCount == 4 && format == BCFormat::BC1)
    {
        options.bDXT1UseAlpha = true; // enable punch through alpha
        options.nAlphaThreshold = alphaThreshold;
    }

    CMP_ERROR error = CMP_ConvertTexture(&srcTexture, &destTexture, &options, nullptr);
    if (error != CMP_OK) {
        return {};
    }

    return compressedData;
}


std::vector<uint8_t> BCnCompressor::CompressHDR(const float* inData, uint32_t width, uint32_t height, uint32_t channelCount, BCFormat format, int numThreads, float quality)
{
    // --- START OF FIX ---
    // Temporary buffer for 4-channel data if the source is 3-channel.
    std::vector<float> tempData;
    const float* dataPtr = inData;
    uint32_t sourceChannelCount = channelCount;

    // BC6H compression via Compressonator expects a 4-channel (RGBA) float source.
    // If we have a 3-channel (RGB) source, we must convert it first.
    if (channelCount == 3 && (format == BCFormat::BC6H || format == BCFormat::BC7)) {
        // We are now providing 4 channels to Compressonator.
        sourceChannelCount = 4;

        // Allocate space for the new 4-channel data.
        tempData.resize((size_t)width * height * 4);

        // Copy RGB data and append an alpha of 1.0 for each pixel.
#pragma omp parallel for num_threads(numThreads)
        for (int64_t i = 0; i < (int64_t)width * height; ++i) {
            tempData[i * 4 + 0] = inData[i * 3 + 0]; // R
            tempData[i * 4 + 1] = inData[i * 3 + 1]; // G
            tempData[i * 4 + 2] = inData[i * 3 + 2]; // B
            tempData[i * 4 + 3] = 1.0f;              // A
        }
        // Point to the start of our new 4-channel data buffer.
        dataPtr = tempData.data();
    }

    CMP_Texture srcTexture = {};
    srcTexture.dwSize = sizeof(CMP_Texture);
    srcTexture.dwWidth = width;
    srcTexture.dwHeight = height;
    // Use the potentially updated sourceChannelCount for pitch and data size calculation.
    srcTexture.dwPitch = width * sourceChannelCount * sizeof(float);
    srcTexture.format = (CMP_FORMAT)GetSourceCMPFormat(sourceChannelCount, true);
    srcTexture.dwDataSize = width * height * sourceChannelCount * sizeof(float);
    // Use the dataPtr which points to either the original or the converted data.
    srcTexture.pData = (CMP_BYTE*)dataPtr;

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
    destTexture.format = (CMP_FORMAT)GetSourceCMPFormat(channelCount, false);
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

    // Compressonator must decompress BC6H to a 4-channel RGBA float buffer.
    // We will handle the conversion to the user's requested channel count afterward.
    const uint32_t decompChannelCount = 4;

    CMP_Texture destTexture = {};
    destTexture.dwSize = sizeof(CMP_Texture);
    destTexture.dwWidth = width;
    destTexture.dwHeight = height;
    destTexture.format = (CMP_FORMAT)GetSourceCMPFormat(decompChannelCount, true); // Always decompress to RGBA_32F
    destTexture.dwPitch = width * decompChannelCount * sizeof(float);

    if (destTexture.format == CMP_FORMAT_Unknown) {
        // This should not happen with a hardcoded channel count of 4.
        return {};
    }

    // Decompress into a temporary 4-channel buffer.
    std::vector<float> tempOutData(width * height * decompChannelCount);
    destTexture.dwDataSize = static_cast<CMP_DWORD>(tempOutData.size() * sizeof(float));
    destTexture.pData = (CMP_BYTE*)tempOutData.data();

    CMP_ERROR error = CMP_ConvertTexture(&srcTexture, &destTexture, nullptr, nullptr);
    if (error != CMP_OK) {
        return {};
    }

    // If the requested channel count was already 4, we're done.
    if (channelCount == decompChannelCount) {
        return tempOutData;
    }

    // Otherwise, we need to strip the extra channels to match the request.
    std::vector<float> finalOutData(width * height * channelCount);
#pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)width * height; ++i) {
        for (uint32_t c = 0; c < channelCount; ++c) {
            // Copy only the channels that were originally requested (e.g., R, G, B).
            finalOutData[i * channelCount + c] = tempOutData[i * decompChannelCount + c];
        }
    }

    return finalOutData;
}