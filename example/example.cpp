#include "vq_bcn_compressor.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <string>
#include <atomic>
#include <algorithm>
#include <regex>
#include <variant> // For holding either uint8_t or float data

namespace fs = std::filesystem;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// --- Image class to support LDR (uint8_t) and HDR (float) data ---
class Image {
public:
    int width = 0;
    int height = 0;
    int channels = 0;
    bool isHDR = false;
    // Holds either LDR (vector<uint8_t>) or HDR (vector<float>) pixel data.
    std::variant<std::vector<uint8_t>, std::vector<float>> data;

    bool Load(const std::string& filename) {
        fs::path path(filename);
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });

        if (ext == ".hdr" || ext == ".exr") {
            isHDR = true;
            // Pass 0 as desired_channels to load the image with its original channel count.
            float* pixels = stbi_loadf(filename.c_str(), &width, &height, &channels, 0);
            if (!pixels) {
                std::cerr << "Failed to load HDR image: " << stbi_failure_reason() << std::endl;
                return false;
            }
            // Calculate data size using the actual number of channels loaded.
            size_t dataSize = (size_t)width * height * channels;
            std::vector<float> float_data(pixels, pixels + dataSize);
            data = std::move(float_data);
            stbi_image_free(pixels);
        }
        else {
            isHDR = false;
            // Pass 0 as desired_channels to load the image with its original channel count.
            uint8_t* pixels = stbi_load(filename.c_str(), &width, &height, &channels, 0);
            if (!pixels) {
                std::cerr << "Failed to load LDR image: " << stbi_failure_reason() << std::endl;
                return false;
            }
            // Calculate data size using the actual number of channels loaded.
            size_t dataSize = (size_t)width * height * channels;
            std::vector<uint8_t> byte_data(pixels, pixels + dataSize);
            data = std::move(byte_data);
            stbi_image_free(pixels);
        }

        // Log the number of channels.
        std::cout << "Loaded " << filename << " (" << width << "x" << height
            << ", " << channels << " channels" << ", type: " << (isHDR ? "HDR" : "LDR") << ")" << std::endl;
        return true;
    }

    bool Save(const std::string& filename) {
        fs::path path(filename);
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        int result = 0;

        if (isHDR) {
            if (ext == ".hdr") {
                // Use the image's actual channel count when saving.
                result = stbi_write_hdr(filename.c_str(), width, height, channels, std::get<std::vector<float>>(data).data());
            }
            else {
                std::cerr << "Saving HDR data to non-HDR format (" << ext << ") is not supported. Please use .hdr." << std::endl;
                return false;
            }
        }
        else {
            const auto& byte_data = std::get<std::vector<uint8_t>>(data);
            // Use the image's actual channel count for all write operations.
            if (ext == ".png") result = stbi_write_png(filename.c_str(), width, height, channels, byte_data.data(), width * channels);
            else if (ext == ".jpg" || ext == ".jpeg") result = stbi_write_jpg(filename.c_str(), width, height, channels, byte_data.data(), 95);
            else if (ext == ".bmp") result = stbi_write_bmp(filename.c_str(), width, height, channels, byte_data.data());
            else if (ext == ".tga") result = stbi_write_tga(filename.c_str(), width, height, channels, byte_data.data());
            else { std::cerr << "Unsupported LDR format for saving: " << ext << std::endl; return false; }
        }

        if (result == 0) { std::cerr << "Failed to save image: " << filename << std::endl; return false; }
        std::cout << "Saved " << filename << std::endl;
        return true;
    }
};

enum MyTextureType {
    Albedo,
    Normal,
    AO,
    Bump,
    Displacement,
    Gloss,
    Roughness,
    Specular,
    HDR,
    Unknown
};

struct ImageStats {
    double mean[4] = { 0.0, 0.0, 0.0, 0.0 };
    double variance[4] = { 0.0, 0.0, 0.0, 0.0 };
    bool isGrayscale = true;
    float blueDominance = 0.0f;
    float redDominance = 0.0f;
};

// This function now accepts the channel count and processes pixels accordingly.
ImageStats computeImageStats(const unsigned char* data, int width, int height, int channels) {
    ImageStats stats;
    long pixelCount = (long)width * height;
    if (pixelCount == 0 || channels == 0) return stats;

    std::vector<double> channelSums(channels, 0.0);
    long blueHighCount = 0;
    long redHighCount = 0;

    // Determine if grayscale. Considered grayscale for classification if fewer than 3 channels.
    if (channels < 3) {
        stats.isGrayscale = true;
    }
    else {
        stats.isGrayscale = true; // Assume grayscale until a color difference is found
        for (long i = 0; i < pixelCount; ++i) {
            const unsigned char* p = data + (size_t)i * channels;
            if (std::abs(p[0] - p[1]) > 10 || std::abs(p[1] - p[2]) > 10) {
                stats.isGrayscale = false;
                break;
            }
        }
    }

    // Sum up pixel values for mean calculation and count dominant colors
    for (long i = 0; i < pixelCount; ++i) {
        const unsigned char* p = data + (size_t)i * channels;
        for (int c = 0; c < channels; ++c) {
            channelSums[c] += p[c];
        }
        if (channels >= 3 && p[2] > 200) blueHighCount++; // Blue is channel 2
        if (channels >= 1 && p[0] > 200) redHighCount++;  // Red is channel 0
    }

    int channelsToCalc = std::min(channels, 4); // The stats struct holds up to 4 channels

    // Calculate mean for each channel
    for (int c = 0; c < channelsToCalc; ++c) {
        stats.mean[c] = channelSums[c] / pixelCount;
    }

    // Calculate variance for each channel
    for (long i = 0; i < pixelCount; ++i) {
        const unsigned char* p = data + (size_t)i * channels;
        for (int c = 0; c < channelsToCalc; ++c) {
            double diff = p[c] - stats.mean[c];
            stats.variance[c] += diff * diff;
        }
    }
    for (int c = 0; c < channelsToCalc; ++c) {
        stats.variance[c] /= pixelCount;
    }

    // Calculate color dominance percentages
    stats.blueDominance = (channels >= 3) ? static_cast<float>(blueHighCount) / pixelCount : 0.0f;
    stats.redDominance = (channels >= 1) ? static_cast<float>(redHighCount) / pixelCount : 0.0f;

    return stats;
}

MyTextureType classifyTextureByStats(const ImageStats& stats) {
    if (stats.isGrayscale && stats.variance[0] < 50.0 && stats.variance[1] < 50.0) {
        if (stats.mean[0] > 128.0f && stats.variance[0] < 30.0) return MyTextureType::AO;
        if (stats.redDominance > 0.7f && stats.mean[0] > 150.0f) return MyTextureType::Bump;
        return MyTextureType::Roughness;
    }
    if (stats.blueDominance > 0.8f && stats.mean[2] > 128.0f && stats.variance[2] < 500.0) return MyTextureType::Normal;
    if (!stats.isGrayscale) return MyTextureType::Albedo;
    return MyTextureType::Unknown;
}

MyTextureType classifyTextureByFilename(const std::string& filename) {
    std::string fname = filename;
    std::transform(fname.begin(), fname.end(), fname.begin(), [](unsigned char c) { return std::tolower(c); });
    struct Keyword { std::string name; MyTextureType type; };
    static const std::vector<Keyword> keywords = {
        {"basecolor", Albedo}, {"albedo", Albedo}, {"diffuse", Albedo},
        {"normal", Normal}, {"ao", AO}, {"ambientocclusion", AO},
        {"bump", Bump}, {"displacement", Displacement}, {"gloss", Gloss},
        {"roughness", Roughness}, {"specular", Specular}
    };
    for (const auto& kw : keywords) {
        if (fname.find(kw.name) != std::string::npos) return kw.type;
    }
    return MyTextureType::Unknown;
}

// --- Main processing function now handles LDR/HDR ---
void ProcessImage(const std::filesystem::path& filePath, VQBCnCompressor& compressor) {
    std::cout << "\n--- Processing: " << filePath.filename().string() << " ---\n";
    Image image;
    if (!image.Load(filePath.string())) return;

    MyTextureType type = MyTextureType::Unknown;
    if (image.isHDR) {
        type = MyTextureType::HDR;
    }
    else {
        type = classifyTextureByFilename(filePath.filename().string());
        if (type == MyTextureType::Unknown) {
            // Pass the actual channel count to the stats computation function.
            ImageStats stats = computeImageStats(std::get<std::vector<uint8_t>>(image.data).data(), image.width, image.height, image.channels);
            type = classifyTextureByStats(stats);
        }
    }

    CompressionParams params;
    params.bcQuality = 1.0f;
    params.zstdLevel = 20;
    params.numThreads = 16;
    params.useVQ = true;
    params.useZstd = true;

    switch (type) {
    case HDR:
        std::cout << "Texture Type: HDR (Using BC6H with VQ)\n";
        params.bcFormat = BCFormat::BC6H;
        // --- Enable VQ for HDR and set params ---
        params.bcQuality = 0.25f; // Use a lower quality for HDR to set reasonable compression time
        params.quality = 1.0f; // Use a high quality for HDR VQ
        params.vq_min_cb_power = 6;  // 64 entries
        params.vq_max_cb_power = 12; // 4096 entriess
        params.vq_FastModeSampleRatio = 1.0f;
        params.vq_maxIterations = 64;       // Allow more iterations for K-Means to converge
        break;
    case Albedo:
        params.bcFormat = BCFormat::BC1;
        std::cout << "Texture Type: Albedo using BC1\n";
        params.alphaThreshold = 1; // Use smallest alpha threshold for BC1 compression
        params.quality = 0.8f;
        params.vq_Metric = DistanceMetric::PERCEPTUAL_OKLAB;
        break;
    case Normal:
        std::cout << "Texture Type: Normal (Using BC5)\n";
        params.bcFormat = BCFormat::BC5;
        params.quality = 0.8f;
        params.vq_Metric = DistanceMetric::SAD_SIMD;
        break;
    case AO:
    case Bump:
    case Displacement:
    case Gloss:
    case Roughness:
    case Specular:
        std::cout << "Texture Type: Grayscale/Mask (Using BC4)\n";
        params.bcFormat = BCFormat::BC4;
        params.quality = 0.5f;
        params.vq_Metric = DistanceMetric::SAD_SIMD;
        break;
    default:
        std::cout << "Texture Type: Unknown (Defaulting to BC7)\n";
        params.bcFormat = BCFormat::BC7;
        params.quality = 0.8f;
        params.vq_Metric = DistanceMetric::SAD_SIMD;
        break;
    }

    std::string suffix = "_bc" + std::to_string(static_cast<int>(params.bcFormat));
    if (params.useVQ) {
        suffix += (params.vq_Metric == DistanceMetric::PERCEPTUAL_OKLAB ? "_lab" : "_rgb");
    }

    std::cout << "Compression: BC" << static_cast<int>(params.bcFormat)
        << ", Use VQ: " << (params.useVQ ? "Yes" : "No") <<
        ", Use ZSTD: " << (params.useZstd ? "Yes" : "No") << std::endl;

    try {
        std::string out_name_bin = "output/" + filePath.stem().string() + suffix + ".yupt2";
        {
            auto start_compress = std::chrono::high_resolution_clock::now();
            std::vector<uint8_t> compressed;
            // Call the correct Compress overload based on whether the image is HDR
            if (image.isHDR) {
                compressed = compressor.CompressHDR(std::get<std::vector<float>>(image.data).data(), image.width, image.height, image.channels, params);
            }
            else {
                compressed = compressor.Compress(std::get<std::vector<uint8_t>>(image.data).data(), image.width, image.height, image.channels, params);
            }
            auto end_compress = std::chrono::high_resolution_clock::now();
            std::cout << "Compression finished in " << std::fixed << std::setprecision(2)
                << std::chrono::duration<double>(end_compress - start_compress).count() << "s.\n";

            std::ofstream out;
            out.open(out_name_bin, std::ios::binary);
            out.write(reinterpret_cast<char*>(compressed.data()), compressed.size());
            out.close();
            std::cout << "Saved compressed file: " << out_name_bin << std::endl;
        }

        // --- Decompression and Verification ---
        std::vector<uint8_t> loadedTexture;
        std::ifstream in;
        in.open(out_name_bin, std::ios::binary | std::ios::ate);
        loadedTexture.resize(in.tellg());
        in.seekg(0, std::ios::beg);
        in.read(reinterpret_cast<char*>(loadedTexture.data()), loadedTexture.size());
        in.close();

        auto start_decompress_bcn = std::chrono::high_resolution_clock::now();
        TextureInfo outInfo;
        auto bcData = compressor.DecompressToBCn(loadedTexture, outInfo);
        auto end_decompress_bcn = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_decompress_bcn = end_decompress_bcn - start_decompress_bcn;
        std::cout << "Decompression to BCn (GPU-ready) finished in " << std::fixed << std::setprecision(4) << diff_decompress_bcn.count() << " seconds.\n";

        auto start_decompress = std::chrono::high_resolution_clock::now();
        Image outputImage;
        outputImage.width = outInfo.width;
        outputImage.height = outInfo.height;
        outputImage.channels = outInfo.originalChannelCount;
        outputImage.isHDR = outInfo.compressionFlags & COMPRESSION_FLAGS_IS_HDR;

        if (image.isHDR) {
            outputImage.data = compressor.DecompressHDR(loadedTexture, outInfo);
        }
        else {
            outputImage.data = compressor.Decompress(loadedTexture, outInfo);
            auto end_decompress = std::chrono::high_resolution_clock::now();
            std::cout << "Decompression to RGB finished in " << std::fixed << std::setprecision(4)
                << std::chrono::duration<double>(end_decompress - start_decompress).count() << "s.\n";

            // Special handling for BC5 normal map visualization
            if (params.bcFormat == BCFormat::BC5 && !image.isHDR && image.channels == 3) {
                std::cout << "Reconstructing Z-channel for BC5 normal map visualization.\n";

                auto& src = std::get<std::vector<uint8_t>>(outputImage.data);
                const int srcStride = outputImage.channels; // 2 if RG, 3 if RGB, 4 if RGBA

                std::vector<uint8_t> rgb_data(outputImage.width * outputImage.height * 3);

#pragma omp parallel for
                for (int64_t i = 0; i < static_cast<int64_t>(outputImage.width) * outputImage.height; ++i) {
                    const int64_t s = i * srcStride;
                    const int64_t d = i * 3;

                    const uint8_t r8 = src[s + 0];
                    const uint8_t g8 = src[s + 1];

                    const float x = (r8 / 255.0f) * 2.0f - 1.0f;
                    const float y = (g8 / 255.0f) * 2.0f - 1.0f;
                    const float z2 = 1.0f - x * x - y * y;
                    const float z = (z2 > 0.0f) ? std::sqrt(z2) : 0.0f;

                    rgb_data[d + 0] = r8;
                    rgb_data[d + 1] = g8;
                    rgb_data[d + 2] = static_cast<uint8_t>(std::clamp(z * 0.5f + 0.5f, 0.0f, 1.0f) * 255.0f);
                }

                outputImage.data = std::move(rgb_data);
            }
        }

        std::string out_ext = image.isHDR ? ".hdr" : ".png";
        outputImage.Save("output/" + filePath.stem().string() + suffix + out_ext);

    }
    catch (const std::exception& e) {
        std::cerr << "An error occurred during processing: " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    try {
        VQBCnCompressor compressor;
        fs::create_directory("output");
        std::string test_dir = "test_assets";
        if (!fs::exists(test_dir)) {
            std::cerr << "Error: Test directory '" << test_dir << "' not found." << std::endl; return 1;
        }
        for (const auto& file : fs::directory_iterator(test_dir)) {
            std::string ext = file.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tga" || ext == ".hdr" || ext == ".exr") {
                ProcessImage(file.path(), compressor);
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "A critical error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}