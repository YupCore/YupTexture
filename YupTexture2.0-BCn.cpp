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

#define __STDC_LIB_EXT1__
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// --- MODIFIED: Image class to support LDR (uint8_t) and HDR (float) data ---
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
            float* pixels = stbi_loadf(filename.c_str(), &width, &height, &channels, 4);
            if (!pixels) {
                std::cerr << "Failed to load HDR image: " << stbi_failure_reason() << std::endl;
                return false;
            }
            size_t dataSize = (size_t)width * height * 4;
            std::vector<float> float_data(pixels, pixels + dataSize);
            data = std::move(float_data);
            stbi_image_free(pixels);
        }
        else {
            isHDR = false;
            uint8_t* pixels = stbi_load(filename.c_str(), &width, &height, &channels, 4);
            if (!pixels) {
                std::cerr << "Failed to load LDR image: " << stbi_failure_reason() << std::endl;
                return false;
            }
            size_t dataSize = (size_t)width * height * 4;
            std::vector<uint8_t> byte_data(pixels, pixels + dataSize);
            data = std::move(byte_data);
            stbi_image_free(pixels);
        }

        std::cout << "Loaded " << filename << " (" << width << "x" << height
            << ", type: " << (isHDR ? "HDR" : "LDR") << ")" << std::endl;
        return true;
    }

    bool Save(const std::string& filename) {
        fs::path path(filename);
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        int result = 0;

        if (isHDR) {
            if (ext == ".hdr") {
                result = stbi_write_hdr(filename.c_str(), width, height, 4, std::get<std::vector<float>>(data).data());
            }
            else {
                std::cerr << "Saving HDR data to non-HDR format (" << ext << ") is not supported. Please use .hdr." << std::endl;
                return false;
            }
        }
        else {
            const auto& byte_data = std::get<std::vector<uint8_t>>(data);
            if (ext == ".png") result = stbi_write_png(filename.c_str(), width, height, 4, byte_data.data(), width * 4);
            else if (ext == ".jpg" || ext == ".jpeg") result = stbi_write_jpg(filename.c_str(), width, height, 4, byte_data.data(), 95);
            else if (ext == ".bmp") result = stbi_write_bmp(filename.c_str(), width, height, 4, byte_data.data());
            else if (ext == ".tga") result = stbi_write_tga(filename.c_str(), width, height, 4, byte_data.data());
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

ImageStats computeImageStats(const unsigned char* data, int width, int height) {
    ImageStats stats;
    long pixelCount = (long)width * height;
    if (pixelCount == 0) return stats;
    std::vector<double> channelSums(4, 0.0);
    long blueHighCount = 0;
    long redHighCount = 0;
    for (int i = 0; i < pixelCount; ++i) {
        if (std::abs(data[i * 4 + 0] - data[i * 4 + 1]) > 10 || std::abs(data[i * 4 + 1] - data[i * 4 + 2]) > 10) {
            stats.isGrayscale = false;
        }
        for (int c = 0; c < 3; ++c) channelSums[c] += data[i * 4 + c];
        if (data[i * 4 + 2] > 200) blueHighCount++;
        if (data[i * 4 + 0] > 200) redHighCount++;
    }
    for (int c = 0; c < 3; ++c) stats.mean[c] = channelSums[c] / pixelCount;
    for (int i = 0; i < pixelCount; ++i) {
        for (int c = 0; c < 3; ++c) {
            double diff = data[i * 4 + c] - stats.mean[c];
            stats.variance[c] += diff * diff;
        }
    }
    for (int c = 0; c < 3; ++c) stats.variance[c] /= pixelCount;
    stats.blueDominance = static_cast<float>(blueHighCount) / pixelCount;
    stats.redDominance = static_cast<float>(redHighCount) / pixelCount;
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

// --- MODIFIED: Main processing function now handles LDR/HDR ---
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
            ImageStats stats = computeImageStats(std::get<std::vector<uint8_t>>(image.data).data(), image.width, image.height);
            type = classifyTextureByStats(stats);
        }
    }

    CompressionParams params;
    params.bcQuality = 1.0f;
    params.zstdLevel = 10;
    params.numThreads = 16;
    params.bypassVQ = false;
    params.bypassZstd = false;

    switch (type) {
    case HDR:
        std::cout << "Texture Type: HDR (Using BC6H with VQ)\n";
        params.bcFormat = BCFormat::BC6H;
        // --- MODIFIED: Enable VQ for HDR and set params ---
		params.bcQuality = 0.25f; // Use a lower quality for HDR to set reasonable compression time
        params.quality = 1.0f; // Use a high quality for HDR VQ
        // VQ quality for HDR might need different tuning than for LDR
        params.vq_min_cb_power = 6;  // 64 entries
        params.vq_max_cb_power = 12; // 4096 entriess
        params.vq_FastModeSampleRatio = 0.5f;
        break;
    case Albedo:
        params.bcFormat = BCFormat::BC1;
        std::cout << "Texture Type: Albedo using BC1\n";
		params.alphaThreshold = 1; // Use smallest alpha threshold for BC1 compression
        params.quality = 0.8f;
        params.vq_Metric = DistanceMetric::PERCEPTUAL_LAB;
        break;
    case Normal:
        std::cout << "Texture Type: Normal (Using BC5)\n";
        params.bcFormat = BCFormat::BC5;
        params.quality = 0.8f;
        params.vq_Metric = DistanceMetric::RGB_SIMD;
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
        params.vq_Metric = DistanceMetric::RGB_SIMD;
        break;
    default:
        std::cout << "Texture Type: Unknown (Defaulting to BC7)\n";
        params.bcFormat = BCFormat::BC7;
        params.quality = 0.8f;
        params.vq_Metric = DistanceMetric::RGB_SIMD;
        break;
    }

    std::string suffix = "_bc" + std::to_string(static_cast<int>(params.bcFormat));
    if (!params.bypassVQ) {
        suffix += (params.vq_Metric == DistanceMetric::PERCEPTUAL_LAB ? "_lab" : "_rgb");
    }

    std::cout << "Compression: BC" << static_cast<int>(params.bcFormat)
        << ", VQ Bypass: " << (params.bypassVQ ? "Yes" : "No") <<
        ", ZSTD Bypass: " << (params.bypassZstd ? "Yes" : "No") << std::endl;

    try {
        std::string out_name_bin = "output/" + filePath.stem().string() + suffix + ".yupt2";
        {
            auto start_compress = std::chrono::high_resolution_clock::now();
            CompressedTexture compressed;
            // Call the correct Compress overload based on whether the image is HDR
            if (image.isHDR) {
                compressed = compressor.CompressHDR(std::get<std::vector<float>>(image.data).data(), image.width, image.height, params);
            }
            else {
                compressed = compressor.Compress(std::get<std::vector<uint8_t>>(image.data).data(), image.width, image.height, image.channels, params);
            }
            auto end_compress = std::chrono::high_resolution_clock::now();
            std::cout << "Compression finished in " << std::fixed << std::setprecision(2)
                << std::chrono::duration<double>(end_compress - start_compress).count() << "s.\n";

            std::ofstream outFile(out_name_bin, std::ios::binary);
            outFile.write(reinterpret_cast<const char*>(&compressed.info), sizeof(TextureInfo));
            outFile.write(reinterpret_cast<const char*>(compressed.compressedData.data()), compressed.compressedData.size());
            outFile.close();
            std::cout << "Saved compressed file: " << out_name_bin << std::endl;
        }

        // --- Decompression and Verification ---

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

        auto start_decompress = std::chrono::high_resolution_clock::now();
        Image outputImage;
        outputImage.width = image.width;
        outputImage.height = image.height;
        outputImage.channels = 4;
        outputImage.isHDR = image.isHDR;

        if (image.isHDR) {
            outputImage.data = compressor.DecompressToRGBAF(loadedTexture);
        }
        else {
            outputImage.data = compressor.DecompressToRGBA(loadedTexture);
            auto end_decompress = std::chrono::high_resolution_clock::now();
            std::cout << "Decompression to RGBA finished in " << std::fixed << std::setprecision(4)
                << std::chrono::duration<double>(end_decompress - start_decompress).count() << "s.\n";

            // Special handling for BC5 normal map visualization
            if (params.bcFormat == BCFormat::BC5) {
                std::cout << "Reconstructing Z-channel for BC5 normal map visualization...\n";
                auto& decompressed_data = std::get<std::vector<uint8_t>>(outputImage.data);
                for (size_t i = 0; i < decompressed_data.size(); i += 4) {
                    float x = (decompressed_data[i + 0] / 255.0f) * 2.0f - 1.0f;
                    float y = (decompressed_data[i + 1] / 255.0f) * 2.0f - 1.0f;
                    float z_squared = 1.0f - x * x - y * y;
                    float z = (z_squared > 0.0f) ? sqrt(z_squared) : 0.0f;
                    decompressed_data[i + 2] = static_cast<uint8_t>((z * 0.5f + 0.5f) * 255.0f);
                    decompressed_data[i + 3] = 255;
                }
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
        std::string test_dir = "test_texture_set";
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