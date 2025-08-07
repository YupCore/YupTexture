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

namespace fs = std::filesystem;

#define __STDC_LIB_EXT1__
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Image helper class using stb_image
class Image {
public:
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> data;  // Always stored as RGBA

    bool Load(const std::string& filename) {
        int w, h, c;
        uint8_t* pixels = stbi_load(filename.c_str(), &w, &h, &c, 4);
        if (!pixels) {
            std::cerr << "Failed to load image: " << stbi_failure_reason() << std::endl;
            return false;
        }
        width = w; height = h; channels = c;
        size_t dataSize = (size_t)width * height * 4;
        data.assign(pixels, pixels + dataSize);
        stbi_image_free(pixels);
        std::cout << "Loaded " << filename << " (" << width << "x" << height
            << ", original channels: " << c << ")" << std::endl;
        return true;
    }

    bool Save(const std::string& filename) {
        std::filesystem::path path(filename);
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        int result = 0;
        if (ext == ".png") result = stbi_write_png(filename.c_str(), width, height, 4, data.data(), width * 4);
        else if (ext == ".jpg" || ext == ".jpeg") result = stbi_write_jpg(filename.c_str(), width, height, 4, data.data(), 95);
        else if (ext == ".bmp") result = stbi_write_bmp(filename.c_str(), width, height, 4, data.data());
        else if (ext == ".tga") result = stbi_write_tga(filename.c_str(), width, height, 4, data.data());
        else { std::cerr << "Unsupported format: " << ext << std::endl; return false; }
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
    Unknown
};

struct ImageStats {
    double mean[4] = { 0.0, 0.0, 0.0, 0.0 };
    double variance[4] = { 0.0, 0.0, 0.0, 0.0 };
    bool isGrayscale = true;
    float blueDominance = 0.0f;
    float redDominance = 0.0f;
};

ImageStats computeImageStats(const unsigned char* data, int width, int height, int channels) {
    ImageStats stats;
    long pixelCount = (long)width * height;
    if (pixelCount == 0) return stats;
    std::vector<double> channelSums(channels, 0.0);
    long blueHighCount = 0;
    long redHighCount = 0;
    for (int i = 0; i < pixelCount; ++i) {
        if (channels >= 3 && (std::abs(data[i * 4 + 0] - data[i * 4 + 1]) > 10 || std::abs(data[i * 4 + 1] - data[i * 4 + 2]) > 10)) {
            stats.isGrayscale = false;
        }
        for (int c = 0; c < channels; ++c) channelSums[c] += data[i * channels + c];
        if (channels >= 3 && data[i * channels + 2] > 200) blueHighCount++;
        if (channels >= 3 && data[i * channels + 0] > 200) redHighCount++;
    }
    for (int c = 0; c < channels; ++c) stats.mean[c] = channelSums[c] / pixelCount;
    for (int i = 0; i < pixelCount; ++i) {
        for (int c = 0; c < channels; ++c) {
            double diff = data[i * channels + c] - stats.mean[c];
            stats.variance[c] += diff * diff;
        }
    }
    for (int c = 0; c < channels; ++c) stats.variance[c] /= pixelCount;
    stats.blueDominance = static_cast<float>(blueHighCount) / pixelCount;
    stats.redDominance = static_cast<float>(redHighCount) / pixelCount;
    return stats;
}

MyTextureType classifyTextureByStats(const ImageStats& stats, int channels) {
    if (channels >= 3 && stats.isGrayscale && stats.variance[0] < 50.0 && stats.variance[1] < 50.0) {
        if (stats.mean[0] > 128.0f && stats.variance[0] < 30.0) return MyTextureType::AO;
        if (stats.redDominance > 0.7f && stats.mean[0] > 150.0f) return MyTextureType::Bump;
        return MyTextureType::Roughness; // Fallback for grayscale
    }
    if (channels >= 3 && stats.blueDominance > 0.8f && stats.mean[2] > 128.0f && stats.variance[2] < 500.0) return MyTextureType::Normal;
    if (channels >= 3 && !stats.isGrayscale) return MyTextureType::Albedo;
    return MyTextureType::Unknown;
}

MyTextureType classifyTextureByFilename(const std::string& filename) {
    std::string fname = filename;
    std::transform(fname.begin(), fname.end(), fname.begin(), [](unsigned char c) { return std::tolower(c); });

    // Define texture keywords and their corresponding types
    struct TextureKeyword {
        std::string keyword;
        MyTextureType type;
    };
    static const std::vector<TextureKeyword> textureKeywords = {
        {"basecolor", MyTextureType::Albedo},
        {"albedo", MyTextureType::Albedo},
        {"diffuse", MyTextureType::Albedo},
        {"normal", MyTextureType::Normal},
        {"ao", MyTextureType::AO},
        {"ambientocclusion", MyTextureType::AO},
        {"bump", MyTextureType::Bump},
        {"displacement", MyTextureType::Displacement},
        {"gloss", MyTextureType::Gloss},
        {"roughness", MyTextureType::Roughness},
        {"specular", MyTextureType::Specular}
    };

    // Regex to match keywords with delimiters (e.g., "_normal.", "-normal_", etc.)
    for (const auto& kw : textureKeywords) {
        std::regex pattern(std::string("[-_.]") + kw.keyword + "[-_.]");
        if (std::regex_search(fname, pattern)) {
            return kw.type;
        }
    }

    return MyTextureType::Unknown;
}

void ProcessImage(const std::filesystem::path& filePath, VQBCnCompressor& compressor) {
    std::cout << "\n--- Processing: " << filePath.filename().string() << " ---\n";
    Image image;
    if (!image.Load(filePath.string())) return;

    // First, try to classify by filename
    MyTextureType type = classifyTextureByFilename(filePath.filename().string());
    bool useStats = (type == MyTextureType::Unknown);

    // If filename classification fails, fall back to statistical analysis
    ImageStats stats;
    if (useStats) {
        stats = computeImageStats(image.data.data(), image.width, image.height, image.channels);
        type = classifyTextureByStats(stats, image.channels);
    }

    VQBCnCompressor::CompressionParams params;
    std::string suffix;

    params.bcQuality = 1.0f;
    params.zstdLevel = 10;
    params.useMultithreading = true;
    params.bypassVQ = false;
    params.bypassZstd = false;

    switch (type) {
    case Albedo:
        std::cout << "Texture Type: Albedo (Using BC1 for color)\n";
        params.bcFormat = BCFormat::BC1;
        if (image.channels == 4)
        {
			params.bypassVQ = true; // Bypass VQ for alpha channel
        }
        else
        {
            params.bypassVQ = false;
            params.vqCodebookSize = 512;
            params.vqMetric = VQEncoder::DistanceMetric::PERCEPTUAL_LAB;
        }
        break;
    case Normal:
        std::cout << "Texture Type: Normal (Using BC5 for two-channel data)\n";
        params.bcFormat = BCFormat::BC5;
        params.vqCodebookSize = 256;
        params.vqMetric = VQEncoder::DistanceMetric::RGB_SIMD;
        break;
    case AO:
        std::cout << "Texture Type: Ambient Occlusion (Using BC4 for single-channel data)\n";
        params.bcFormat = BCFormat::BC4;
        params.vqCodebookSize = 128;
        params.vqMetric = VQEncoder::DistanceMetric::RGB_SIMD;
        break;
    case Bump:
        std::cout << "Texture Type: Bump (Using BC4 for single-channel data)\n";
        params.bcFormat = BCFormat::BC4;
        params.vqCodebookSize = 128;
        params.vqMetric = VQEncoder::DistanceMetric::RGB_SIMD;
        break;
    case Displacement:
        std::cout << "Texture Type: Displacement (Using BC4 for single-channel data)\n";
        params.bcFormat = BCFormat::BC4;
        params.vqCodebookSize = 128;
        params.vqMetric = VQEncoder::DistanceMetric::RGB_SIMD;
        break;
    case Gloss:
        std::cout << "Texture Type: Gloss (Using BC4 for single-channel data)\n";
        params.bcFormat = BCFormat::BC4;
        params.vqCodebookSize = 128;
        params.vqMetric = VQEncoder::DistanceMetric::RGB_SIMD;
        break;
    case Roughness:
        std::cout << "Texture Type: Roughness (Using BC4 for single-channel data)\n";
        params.bcFormat = BCFormat::BC4;
        params.vqCodebookSize = 128;
        params.vqMetric = VQEncoder::DistanceMetric::RGB_SIMD;
        break;
    case Specular:
        std::cout << "Texture Type: Specular (Using BC4 for single-channel data)\n";
        params.bcFormat = BCFormat::BC4;
        params.vqCodebookSize = 128;
        params.vqMetric = VQEncoder::DistanceMetric::RGB_SIMD;
        break;
    default:
        std::cout << "Texture Type: Unknown (Defaulting to BC1)\n";
        params.bcFormat = BCFormat::BC1;
        params.vqCodebookSize = 256;
        params.vqMetric = VQEncoder::DistanceMetric::RGB_SIMD;
        break;
    }

    // Build suffix based on params
    suffix = "_bc" + std::to_string(static_cast<int>(params.bcFormat));
    if (params.bypassVQ) {
        suffix += "_noVQ";
    }
    else {
        suffix += (params.vqMetric == VQEncoder::DistanceMetric::PERCEPTUAL_LAB ? "_lab" : "_rgb");
    }
    if (params.bypassZstd) {
        suffix += "_noZSTD";
    }

    std::cout << "Compression Settings: BC" << static_cast<int>(params.bcFormat)
        << ", VQ Bypass: " << (params.bypassVQ ? "Yes" : "No")
        << ", ZSTD Bypass: " << (params.bypassZstd ? "Yes" : "No") << std::endl;
    if (!params.bypassVQ) {
        std::cout << "VQ Settings: Codebook Size=" << params.vqCodebookSize << ", Metric="
            << (params.vqMetric == VQEncoder::DistanceMetric::PERCEPTUAL_LAB ? "PERCEPTUAL_LAB" : "RGB_SIMD") << std::endl;
    }

    try {
        auto start_compress = std::chrono::high_resolution_clock::now();
        auto compressed = compressor.Compress(image.data.data(), image.width, image.height, image.channels, params);
        auto end_compress = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_compress = end_compress - start_compress;
        std::cout << "Compression finished in " << std::fixed << std::setprecision(2) << diff_compress.count() << " seconds.\n";

        std::string out_name_bin = "output/" + filePath.stem().string() + suffix + ".yupt2";
        std::ofstream outFile(out_name_bin, std::ios::binary);
        if (!outFile) throw std::runtime_error("Failed to open " + out_name_bin + " for writing.");
        outFile.write(reinterpret_cast<const char*>(&compressed.info), sizeof(TextureInfo));
        outFile.write(reinterpret_cast<const char*>(compressed.compressedData.data()), compressed.compressedData.size());
        outFile.close();
        std::cout << "Saved compressed file: " << out_name_bin << std::endl;

        std::ifstream inFile(out_name_bin, std::ios::binary);
        if (!inFile) throw std::runtime_error("Failed to open " + out_name_bin + " for reading.");
        CompressedTexture loadedTexture;
        inFile.read(reinterpret_cast<char*>(&loadedTexture.info), sizeof(TextureInfo));
        inFile.seekg(0, std::ios::end);
        size_t fileDataSize = static_cast<size_t>(inFile.tellg()) - sizeof(TextureInfo);
        loadedTexture.compressedData.resize(fileDataSize);
        inFile.seekg(sizeof(TextureInfo), std::ios::beg);
        inFile.read(reinterpret_cast<char*>(loadedTexture.compressedData.data()), fileDataSize);
        inFile.close();

        auto start_decompress_bcn = std::chrono::high_resolution_clock::now();
        auto bcData = compressor.DecompressToBCn(loadedTexture);
        auto end_decompress_bcn = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_decompress_bcn = end_decompress_bcn - start_decompress_bcn;
        std::cout << "Decompression to BCn (GPU-ready) finished in " << std::fixed << std::setprecision(4) << diff_decompress_bcn.count() << " seconds.\n";

        BCnCompressor bcn_decoder;
        auto decompressed = bcn_decoder.DecompressToRGBA(bcData.data(), loadedTexture.info.width, loadedTexture.info.height, loadedTexture.info.format);

        if (params.bcFormat == BCFormat::BC5) {
            std::cout << "Reconstructing Z-channel for BC5 normal map visualization...\n";
            for (size_t i = 0; i < decompressed.size(); i += 4) {
                float x = (decompressed[i + 0] / 255.0f) * 2.0f - 1.0f;
                float y = (decompressed[i + 1] / 255.0f) * 2.0f - 1.0f;
                float z_squared = 1.0f - x * x - y * y;
                float z = (z_squared > 0.0f) ? sqrt(z_squared) : 0.0f;
                decompressed[i + 2] = static_cast<uint8_t>((z * 0.5f + 0.5f) * 255.0f);
                decompressed[i + 3] = 255;
            }
        }
        Image output;
        output.width = image.width; output.height = image.height; output.channels = 4;
        output.data = std::move(decompressed);
        output.Save("output/" + filePath.stem().string() + suffix + ".png");
    }
    catch (const std::exception& e) {
        std::cerr << "An error occurred during processing: " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    try {
        VQBCnCompressor compressor;
        std::filesystem::create_directory("output");
        std::string test_dir = "test_texture_set";
        if (!fs::exists(test_dir)) {
            std::cerr << "Error: Test directory '" << test_dir << "' not found." << std::endl; return 1;
        }
        for (const auto& file : std::filesystem::directory_iterator(test_dir)) {
            std::string ext = file.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tga") {
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