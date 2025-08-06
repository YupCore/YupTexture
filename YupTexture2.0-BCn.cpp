#include "vq_bcn_compressor.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <string>
#include <atomic>

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
        // Force load as RGBA
        uint8_t* pixels = stbi_load(filename.c_str(), &w, &h, &c, 4);
        if (!pixels) {
            std::cerr << "Failed to load image: " << stbi_failure_reason() << std::endl;
            return false;
        }

        width = w;
        height = h;
        channels = 4;  // We forced RGBA

        size_t dataSize = width * height * 4;
        data.resize(dataSize);
        std::memcpy(data.data(), pixels, dataSize);

        stbi_image_free(pixels);

        std::cout << "Loaded " << filename << " (" << width << "x" << height
            << ", original channels: " << c << ")" << std::endl;
        return true;
    }

    bool Save(const std::string& filename) {
        std::filesystem::path path(filename);
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        int result = 0;
        if (ext == ".png") {
            result = stbi_write_png(filename.c_str(), width, height, 4,
                data.data(), width * 4);
        }
        else if (ext == ".jpg" || ext == ".jpeg") {
            result = stbi_write_jpg(filename.c_str(), width, height, 4,
                data.data(), 95);  // 95% quality
        }
        else if (ext == ".bmp") {
            result = stbi_write_bmp(filename.c_str(), width, height, 4,
                data.data());
        }
        else if (ext == ".tga") {
            result = stbi_write_tga(filename.c_str(), width, height, 4,
                data.data());
        }
        else {
            std::cerr << "Unsupported format: " << ext << std::endl;
            return false;
        }

        if (result == 0) {
            std::cerr << "Failed to save image: " << filename << std::endl;
            return false;
        }

        std::cout << "Saved " << filename << std::endl;
        return true;
    }
};

// Enum for texture types
enum SimpleTextureType {
    Albedo,
    Normal,
    Grayscale, // AO, Roughness, Specular, etc.
    Unknown
};

// Structure to hold image statistics
struct ImageStats {
    double mean[4] = { 0.0, 0.0, 0.0, 0.0 };
    double variance[4] = { 0.0, 0.0, 0.0, 0.0 };
    bool isGrayscale = true;
    float blueDominance = 0.0f; // For normal map detection
};

ImageStats computeImageStats(const unsigned char* data, int width, int height, int channels) {
    ImageStats stats;
    long pixelCount = width * height;
    if (pixelCount == 0) return stats;

    std::vector<double> channelSums(channels, 0.0);
    std::vector<double> channelSumsSq(channels, 0.0);
    long blueHighCount = 0;

    // First pass: compute means and check grayscale
    for (int i = 0; i < pixelCount; ++i) {
        bool pixelIsGrayscale = true;
        if (channels >= 3) {
            if (std::abs(data[i * channels + 0] - data[i * channels + 1]) > 10 || std::abs(data[i * channels + 1] - data[i * channels + 2]) > 10) {
                pixelIsGrayscale = false;
            }
        }
        if (!pixelIsGrayscale) stats.isGrayscale = false;

        for (int c = 0; c < channels; ++c) {
            channelSums[c] += data[i * channels + c];
        }
        if (channels >= 3 && data[i * channels + 2] > 200) { // Blue channel > ~0.8
            blueHighCount++;
        }
    }

    for (int c = 0; c < channels; ++c) {
        stats.mean[c] = channelSums[c] / pixelCount;
    }

    // Second pass: compute variance
    for (int i = 0; i < pixelCount; ++i) {
        for (int c = 0; c < channels; ++c) {
            double diff = data[i * channels + c] - stats.mean[c];
            stats.variance[c] += diff * diff;
        }
    }
    for (int c = 0; c < channels; ++c) {
        stats.variance[c] = stats.variance[c] / pixelCount;
    }

    stats.blueDominance = static_cast<float>(blueHighCount) / pixelCount;

    return stats;
}

SimpleTextureType classifyTexture(const ImageStats& stats, int channels) {
    // Grayscale texture (AO, Roughness, etc.)
    // Check if variance is low on color channels
    if (channels >= 3 && stats.isGrayscale && stats.variance[0] < 50.0 && stats.variance[1] < 50.0) {
        return SimpleTextureType::Grayscale;
    }

    // Normal map: high blue channel, low variance in blue
    if (channels >= 3 && stats.blueDominance > 0.8f && stats.mean[2] > 128.0f && stats.variance[2] < 500.0) {
        return SimpleTextureType::Normal;
    }

    // Albedo: not grayscale, not a normal map
    if (channels >= 3 && !stats.isGrayscale) {
        return SimpleTextureType::Albedo;
    }

    return SimpleTextureType::Unknown;
}

void ProcessImage(const std::filesystem::path& filePath, VQBCnCompressor& compressor) {
    std::cout << "\n--- Processing: " << filePath.filename().string() << " ---\n";
    Image image;
    if (!image.Load(filePath.string())) return;

    ImageStats stats = computeImageStats(image.data.data(), image.width, image.height, image.channels);
    SimpleTextureType type = classifyTexture(stats, image.channels);
    VQBCnCompressor::CompressionParams params;
    std::string suffix;

    // --- CONFIGURE COMPRESSION BASED ON TEXTURE TYPE ---
    params.bcQuality = 1.0f; // Always use highest quality for the initial BCn compression
    params.zstdLevel = 20;   // High ZSTD level for maximum final compression
    params.vqMetric = VQEncoder::DistanceMetric::PERCEPTUAL_LAB; // Use highest quality metric by default

    // OPTIMIZATION: Adaptive codebook sizing based on texture type.
    // Simple textures don't need a large codebook, saving space and speeding up compression.
    // Complex textures get a larger codebook for better quality.
    switch (type) {
    case Albedo:
        std::cout << "Texture Type: Albedo (Using BC1 for color)\n";
        params.bcFormat = BCFormat::BC1;
        params.vqCodebookSize = 512; // High detail, large codebook
        suffix = "_bc1_lab";
        break;
    case Normal:
        std::cout << "Texture Type: Normal (Using BC5 for two-channel data)\n";
        params.bcFormat = BCFormat::BC5;
        params.vqCodebookSize = 256; // Medium detail
        suffix = "_bc5_lab";
        break;
    case Grayscale:
        std::cout << "Texture Type: Grayscale (Using BC4 for single-channel data)\n";
        params.bcFormat = BCFormat::BC4;
        params.vqCodebookSize = 128; // Low detail, small codebook
        suffix = "_bc4_lab";
        break;
    default:
        std::cout << "Texture Type: Unknown (Defaulting to BC1, medium codebook)\n";
        params.bcFormat = BCFormat::BC1;
        params.vqCodebookSize = 256;
        suffix = "_bc1_lab_unknown";
        break;
    }
    std::cout << "Adaptive Codebook Size: " << params.vqCodebookSize << std::endl;


    try {
        // --- COMPRESS and SAVE ---
        auto start_compress = std::chrono::high_resolution_clock::now();
        auto compressed = compressor.Compress(image.data.data(), image.width, image.height, params);
        auto end_compress = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_compress = end_compress - start_compress;
        std::cout << "Compression finished in " << std::fixed << std::setprecision(2) << diff_compress.count() << " seconds.\n";

        std::string out_name_bin = "output/" + filePath.stem().string() + suffix + ".yupt2";
        std::ofstream outFile(out_name_bin, std::ios::binary);
        if (!outFile) {
            throw std::runtime_error("Failed to open " + out_name_bin + " for writing.");
        }
        outFile.write(reinterpret_cast<const char*>(&compressed.info), sizeof(TextureInfo));
        outFile.write(reinterpret_cast<const char*>(compressed.compressedData.data()), compressed.compressedData.size());
        outFile.close();
        std::cout << "Saved compressed file: " << out_name_bin << std::endl;

        // --- LOAD and DECOMPRESS (to verify correctness of the full pipeline) ---
        std::ifstream inFile(out_name_bin, std::ios::binary);
        if (!inFile) {
            throw std::runtime_error("Failed to open " + out_name_bin + " for reading.");
        }

        CompressedTexture loadedTexture;
        inFile.read(reinterpret_cast<char*>(&loadedTexture.info), sizeof(TextureInfo));
        inFile.seekg(0, std::ios::end);
        size_t zstdDataSize = static_cast<size_t>(inFile.tellg()) - sizeof(TextureInfo);
        loadedTexture.compressedData.resize(zstdDataSize);
        inFile.seekg(sizeof(TextureInfo), std::ios::beg);
        inFile.read(reinterpret_cast<char*>(loadedTexture.compressedData.data()), zstdDataSize);
        inFile.close();

        auto start_decompress = std::chrono::high_resolution_clock::now();
        auto decompressed = compressor.DecompressToRGBA(loadedTexture);
        auto end_decompress = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_decompress = end_decompress - start_decompress;
        std::cout << "Decompression finished in " << std::fixed << std::setprecision(2) << diff_decompress.count() << " seconds.\n";


        if (params.bcFormat == BCFormat::BC5) {
            std::cout << "Reconstructing Z-channel for BC5 normal map visualization...\n";
            for (size_t i = 0; i < decompressed.size(); i += 4) {
                float x = (decompressed[i + 0] / 255.0f) * 2.0f - 1.0f; // Red channel is X
                float y = (decompressed[i + 1] / 255.0f) * 2.0f - 1.0f; // Green channel is Y
                float z_squared = 1.0f - x * x - y * y;
                float z = (z_squared > 0.0f) ? sqrt(z_squared) : 0.0f;
                decompressed[i + 2] = static_cast<uint8_t>((z * 0.5f + 0.5f) * 255.0f); // Reconstruct Z into Blue
                decompressed[i + 3] = 255; // Full alpha
            }
        }

        Image output;
        output.width = image.width;
        output.height = image.height;
        output.channels = 4;
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
            std::cerr << "Error: Test directory '" << test_dir << "' not found." << std::endl;
            return 1;
        }

        for (const auto& file : std::filesystem::directory_iterator(test_dir)) {
            std::string ext = file.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
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