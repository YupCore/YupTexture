#include "vq_bcn_compressor.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <string>

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
    float mean[4]; // R, G, B, A
    float variance[4];
    bool isGrayscale;
    float blueDominance; // For normal map detection
};

ImageStats computeImageStats(const unsigned char* data, int width, int height, int channels) {
    ImageStats stats = { {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, true, 0.0f };
    std::vector<float> channelSums(channels, 0.0f);
    std::vector<float> channelSumsSq(channels, 0.0f);
    long pixelCount = width * height;
    long blueHighCount = 0;

    // First pass: compute means and check grayscale
    for (int i = 0; i < pixelCount; ++i) {
        for (int c = 0; c < channels; ++c) {
            float value = data[i * channels + c] / 255.0f;
            channelSums[c] += value;
            channelSumsSq[c] += value * value;
            if (c == 2 && value > 0.5f) { // Blue channel > 128
                blueHighCount++;
            }
        }
        // Check if pixel is grayscale (R ~= G ~= B)
        if (channels >= 3) {
            float r = data[i * channels + 0] / 255.0f;
            float g = data[i * channels + 1] / 255.0f;
            float b = data[i * channels + 2] / 255.0f;
            if (std::abs(r - g) > 0.05f || std::abs(g - b) > 0.05f || std::abs(r - b) > 0.05f) {
                stats.isGrayscale = false;
            }
        }
    }

    // Compute means
    for (int c = 0; c < channels; ++c) {
        stats.mean[c] = channelSums[c] / pixelCount;
    }

    // Second pass: compute variance
    for (int i = 0; i < pixelCount; ++i) {
        for (int c = 0; c < channels; ++c) {
            float value = data[i * channels + c] / 255.0f;
            float diff = value - stats.mean[c];
            stats.variance[c] += diff * diff;
        }
    }
    for (int c = 0; c < channels; ++c) {
        stats.variance[c] = stats.variance[c] / pixelCount;
    }

    // Compute blue dominance (for normal maps)
    stats.blueDominance = static_cast<float>(blueHighCount) / pixelCount;

    return stats;
}

SimpleTextureType classifyTexture(const ImageStats& stats, int channels) {
    // Grayscale texture (AO, Roughness, Specular, etc.)
    if (stats.isGrayscale && channels >= 3) {
        return SimpleTextureType::Grayscale;
    }

    // Normal map: high blue channel, low variance in blue, RGB forms normalized vectors
    if (channels >= 3 && stats.blueDominance > 0.8f && stats.mean[2] > 0.5f && stats.variance[2] < 0.05f) {
        return SimpleTextureType::Normal;
    }

    // Albedo: high variance, diverse colors
    if (channels >= 3 && stats.variance[0] > 0.01f && stats.variance[1] > 0.01f && stats.variance[2] > 0.01f && !stats.isGrayscale) {
        return SimpleTextureType::Albedo;
    }

    return SimpleTextureType::Unknown;
}

void ProcessImage(const std::filesystem::path& filePath, VQBCnCompressor& compressor) {
    std::cout << "\n--- Processing: " << filePath.filename() << " ---\n";
    Image image;
    if (!image.Load(filePath.string())) return;

    ImageStats stats = computeImageStats(image.data.data(), image.width, image.height, image.channels);
    SimpleTextureType type = classifyTexture(stats, image.channels);
    VQBCnCompressor::CompressionParams params;
    std::string suffix;

    // --- CONFIGURE COMPRESSION BASED ON TEXTURE TYPE ---
    params.vqCodebookSize = 512;
    params.bcQuality = 1.0f;
    params.zstdLevel = 20;
    // USE THE HIGHEST QUALITY METRIC BY DEFAULT
    params.vqMetric = VQEncoder::DistanceMetric::PERCEPTUAL_LAB;

    switch (type) {
    case Albedo:
        std::cout << "Texture Type: Albedo (Using BC1 for color)\n";
        params.bcFormat = BCFormat::BC1;
        suffix = "_bc1_lab";
        break;
    case Normal:
        std::cout << "Texture Type: Normal (Using BC5 for two-channel data)\n";
        params.bcFormat = BCFormat::BC5;
        suffix = "_bc5_lab";
        break;
    case Grayscale:
        std::cout << "Texture Type: Grayscale (Using BC4 for single-channel data)\n";
        params.bcFormat = BCFormat::BC4;
        suffix = "_bc4_lab";
        break;
    default:
        std::cout << "Texture Type: Unknown (Defaulting to BC1)\n";
        params.bcFormat = BCFormat::BC1;
        suffix = "_bc1_lab_unknown";
        break;
    }

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
        // Write the header (TextureInfo) and then the compressed data
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
        // Read header
        inFile.read(reinterpret_cast<char*>(&loadedTexture.info), sizeof(TextureInfo));
        // Read the rest of the file into the compressed data buffer
        inFile.seekg(0, std::ios::end);
        size_t zstdDataSize = static_cast<size_t>(inFile.tellg()) - sizeof(TextureInfo);
        loadedTexture.compressedData.resize(zstdDataSize);
        inFile.seekg(sizeof(TextureInfo), std::ios::beg);
        inFile.read(reinterpret_cast<char*>(loadedTexture.compressedData.data()), zstdDataSize);
        inFile.close();

        auto start_decompress = std::chrono::high_resolution_clock::now();
        auto decompressed = compressor.DecompressToRGBA(loadedTexture); // Use the newly loaded texture
        auto end_decompress = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_decompress = end_decompress - start_decompress;
        std::cout << "Decompression finished in " << std::fixed << std::setprecision(2) << diff_decompress.count() << " seconds.\n";


        if (params.bcFormat == BCFormat::BC5) {
            std::cout << "Reconstructing Z-channel for BC5 normal map visualization...\n";

            // Iterate over each pixel in the decompressed RGBA data
            for (size_t i = 0; i < decompressed.size(); i += 4) {
                float x = (decompressed[i + 0] / 255.0f) * 2.0f - 1.0f;
                float y = (decompressed[i + 1] / 255.0f) * 2.0f - 1.0f;
                float z_squared = 1.0f - x * x - y * y;
                float z = (z_squared > 0.0f) ? sqrt(z_squared) : 0.0f;
                decompressed[i + 2] = static_cast<uint8_t>(z * 255.0f); // Store Z in Blue
                decompressed[i + 3] = 255; // Ensure Alpha is at full opacity
            }
        }

        Image output;
        output.width = image.width;
        output.height = image.height;
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

        for (const auto& file : std::filesystem::directory_iterator("test_texture_set")) {
            std::string ext = file.path().extension().string();
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