#include "vq_bcn_compressor.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <string>

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

int main(int argc, char** argv) {
    try {

        VQEncoder::Config vqConfig;
        vqConfig.codebookSize = 512;
        VQBCnCompressor compressor(vqConfig);

        for(auto file : std::filesystem::directory_iterator("test_texture_set")) {
            if (file.path().extension() == ".png" || file.path().extension() == ".jpg" ||
                file.path().extension() == ".jpeg" || file.path().extension() == ".bmp" ||
                file.path().extension() == ".tga") {
                std::cout << "Found image: " << file.path().filename() << std::endl;

                Image image;
				image.Load(file.path().string());

                ImageStats stats = computeImageStats(image.data.data(), image.width, image.height, image.channels);
                SimpleTextureType type = classifyTexture(stats, image.channels);

                std::cout << "Image Stats:\n";
                std::cout << "Mean (R,G,B,A): " << stats.mean[0] << ", " << stats.mean[1] << ", " << stats.mean[2];
                if (image.channels == 4) std::cout << ", " << stats.mean[3];
                std::cout << "\nVariance (R,G,B,A): " << stats.variance[0] << ", " << stats.variance[1] << ", " << stats.variance[2];
                if (image.channels == 4) std::cout << ", " << stats.variance[3];
                std::cout << "\nBlue Dominance: " << stats.blueDominance << "\n";
                std::cout << "Is Grayscale: " << (stats.isGrayscale ? "Yes" : "No") << "\n";

                switch (type) {
                case SimpleTextureType::Unknown:
                case SimpleTextureType::Albedo:
                    std::cout << "Texture Type: Albedo\n";
                    {
                        VQBCnCompressor::CompressionParams params;
                        params.bcFormat = BCFormat::BC1;
                        params.vqCodebookSize = 512;
                        params.bcQuality = 1.0f;

                        auto compressed = compressor.Compress(image.data.data(),
                            image.width, image.height, params);
                        std::ofstream outFile("output\\" + file.path().stem().string() + "_bc1.bin", std::ios::binary);
                        outFile.write(reinterpret_cast<const char*>(compressed.compressedData.data()),
                            compressed.compressedData.size());
                        outFile.close();
                        auto decompressed = compressor.DecompressToRGBA(compressed);

                        Image output;
                        output.width = image.width;
                        output.height = image.height;
                        output.data = std::move(decompressed);
                        output.Save("output\\" + file.path().stem().string() + "_bc1.png");
                    }
                    break;
                case SimpleTextureType::Normal:
                    std::cout << "Texture Type: Normal\n";
                    {
                        VQBCnCompressor::CompressionParams params;
                        params.bcFormat = BCFormat::BC5;
                        params.vqCodebookSize = 512;
                        params.bcQuality = 1.0f;

                        auto compressed = compressor.Compress(image.data.data(),
                            image.width, image.height, params);
                        std::ofstream outFile("output\\" + file.path().stem().string() + "_bc5.bin", std::ios::binary);
                        outFile.write(reinterpret_cast<const char*>(compressed.compressedData.data()),
                            compressed.compressedData.size());
                        outFile.close();
                        auto decompressed = compressor.DecompressToRGBA(compressed);

                        Image output;
                        output.width = image.width;
                        output.height = image.height;
                        output.data = std::move(decompressed);
                        output.Save("output\\" + file.path().stem().string() + "_bc5.png");
                    }
                    break;
                case SimpleTextureType::Grayscale:
                    std::cout << "Texture Type: Grayscale (AO/Roughness/Specular/etc.)\n";
                    {
                        VQBCnCompressor::CompressionParams params;
                        params.bcFormat = BCFormat::BC4;
                        params.vqCodebookSize = 512;
                        params.bcQuality = 1.0f;

                        auto compressed = compressor.Compress(image.data.data(),
                            image.width, image.height, params);
                        std::ofstream outFile("output\\" + file.path().stem().string() + "_bc4.bin", std::ios::binary);
                        outFile.write(reinterpret_cast<const char*>(compressed.compressedData.data()),
                            compressed.compressedData.size());
                        outFile.close();
                        auto decompressed = compressor.DecompressToRGBA(compressed);

                        Image output;
                        output.width = image.width;
                        output.height = image.height;
                        output.data = std::move(decompressed);
                        output.Save("output\\" + file.path().stem().string() + "_bc4.bin");
                    }
                    break;
                }
            }
		}
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}