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
#include <variant> // for holding either uint8_t or float data

namespace fs = std::filesystem;

#define __stdc_lib_ext1__
#define stb_image_implementation
#include "stb_image.h"
#define stb_image_write_implementation
#include "stb_image_write.h"

class image {
public:
    int width = 0;
    int height = 0;
    int channels = 0;
    bool ishdr = false;
    // holds either ldr (vector<uint8_t>) or hdr (vector<float>) pixel data.
    std::variant<std::vector<uint8_t>, std::vector<float>> data;

    bool load(const std::string& filename) {
        fs::path path(filename);
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });

        if (ext == ".hdr" || ext == ".exr") {
            ishdr = true;
            float* pixels = stbi_loadf(filename.c_str(), &width, &height, &channels, 4); // this doesnt even load hdr data fully, pathetic
            if (!pixels) {
                std::cerr << "failed to load hdr image: " << stbi_failure_reason() << std::endl;
                return false;
            }
            size_t datasize = (size_t)width * height * 4;
            std::vector<float> float_data(pixels, pixels + datasize);
            data = std::move(float_data);
            stbi_image_free(pixels);
        }
        else {
            ishdr = false;
            uint8_t* pixels = stbi_load(filename.c_str(), &width, &height, &channels, 4);
            if (!pixels) {
                std::cerr << "failed to load ldr image: " << stbi_failure_reason() << std::endl;
                return false;
            }
            size_t datasize = (size_t)width * height * 4;
            std::vector<uint8_t> byte_data(pixels, pixels + datasize);
            data = std::move(byte_data);
            stbi_image_free(pixels);
        }

        std::cout << "loaded " << filename << " (" << width << "x" << height
            << ", type: " << (ishdr ? "hdr" : "ldr") << ")" << std::endl;
        return true;
    }

    bool save(const std::string& filename) {
        fs::path path(filename);
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        int result = 0;

        if (ishdr) {
            if (ext == ".hdr") {
                result = stbi_write_hdr(filename.c_str(), width, height, 4, std::get<std::vector<float>>(data).data());
            }
            else {
                std::cerr << "saving hdr data to non-hdr format (" << ext << ") is not supported. please use .hdr." << std::endl;
                return false;
            }
        }
        else {
            const auto& byte_data = std::get<std::vector<uint8_t>>(data);
            if (ext == ".png") result = stbi_write_png(filename.c_str(), width, height, 4, byte_data.data(), width * 4);
            else if (ext == ".jpg" || ext == ".jpeg") result = stbi_write_jpg(filename.c_str(), width, height, 4, byte_data.data(), 95);
            else if (ext == ".bmp") result = stbi_write_bmp(filename.c_str(), width, height, 4, byte_data.data());
            else if (ext == ".tga") result = stbi_write_tga(filename.c_str(), width, height, 4, byte_data.data());
            else { std::cerr << "unsupported ldr format for saving: " << ext << std::endl; return false; }
        }

        if (result == 0) { std::cerr << "failed to save image: " << filename << std::endl; return false; }
        std::cout << "saved " << filename << std::endl;
        return true;
    }
};

enum mytexturetype {
    albedo,
    normal,
    ao,
    bump,
    displacement,
    gloss,
    roughness,
    specular,
    hdr,
    unknown
};

struct imagestats {
    double mean[4] = { 0.0, 0.0, 0.0, 0.0 };
    double variance[4] = { 0.0, 0.0, 0.0, 0.0 };
    bool isgrayscale = true;
    float bluedominance = 0.0f;
    float reddominance = 0.0f;
};

imagestats computeimagestats(const unsigned char* data, int width, int height) {
    imagestats stats;
    long pixelcount = (long)width * height;
    if (pixelcount == 0) return stats;
    std::vector<double> channelsums(4, 0.0);
    long bluehighcount = 0;
    long redhighcount = 0;
    for (int i = 0; i < pixelcount; ++i) {
        if (std::abs(data[i * 4 + 0] - data[i * 4 + 1]) > 10 || std::abs(data[i * 4 + 1] - data[i * 4 + 2]) > 10) {
            stats.isgrayscale = false;
        }
        for (int c = 0; c < 3; ++c) channelsums[c] += data[i * 4 + c];
        if (data[i * 4 + 2] > 200) bluehighcount++;
        if (data[i * 4 + 0] > 200) redhighcount++;
    }
    for (int c = 0; c < 3; ++c) stats.mean[c] = channelsums[c] / pixelcount;
    for (int i = 0; i < pixelcount; ++i) {
        for (int c = 0; c < 3; ++c) {
            double diff = data[i * 4 + c] - stats.mean[c];
            stats.variance[c] += diff * diff;
        }
    }
    for (int c = 0; c < 3; ++c) stats.variance[c] /= pixelcount;
    stats.bluedominance = static_cast<float>(bluehighcount) / pixelcount;
    stats.reddominance = static_cast<float>(redhighcount) / pixelcount;
    return stats;
}

mytexturetype classifytexturebystats(const imagestats& stats) {
    if (stats.isgrayscale && stats.variance[0] < 50.0 && stats.variance[1] < 50.0) {
        if (stats.mean[0] > 128.0f && stats.variance[0] < 30.0) return mytexturetype::ao;
        if (stats.reddominance > 0.7f && stats.mean[0] > 150.0f) return mytexturetype::bump;
        return mytexturetype::roughness;
    }
    if (stats.bluedominance > 0.8f && stats.mean[2] > 128.0f && stats.variance[2] < 500.0) return mytexturetype::normal;
    if (!stats.isgrayscale) return mytexturetype::albedo;
    return mytexturetype::unknown;
}

mytexturetype classifytexturebyfilename(const std::string& filename) {
    std::string fname = filename;
    std::transform(fname.begin(), fname.end(), fname.begin(), [](unsigned char c) { return std::tolower(c); });
    struct keyword { std::string name; mytexturetype type; };
    static const std::vector<keyword> keywords = {
        {"basecolor", albedo}, {"albedo", albedo}, {"diffuse", albedo},
        {"normal", normal}, {"ao", ao}, {"ambientocclusion", ao},
        {"bump", bump}, {"displacement", displacement}, {"gloss", gloss},
        {"roughness", roughness}, {"specular", specular}
    };
    for (const auto& kw : keywords) {
        if (fname.find(kw.name) != std::string::npos) return kw.type;
    }
    return mytexturetype::unknown;
}

void processimage(const std::filesystem::path& filepath, vqbcncompressor& compressor) {
    std::cout << "\n--- processing: " << filepath.filename().string() << " ---\n";
    image image;
    if (!image.load(filepath.string())) return;

    mytexturetype type = mytexturetype::unknown;
    if (image.ishdr) {
        type = mytexturetype::hdr;
    }
    else {
        type = classifytexturebyfilename(filepath.filename().string());
        if (type == mytexturetype::unknown) {
            imagestats stats = computeimagestats(std::get<std::vector<uint8_t>>(image.data).data(), image.width, image.height);
            type = classifytexturebystats(stats);
        }
    }

    compressionparams params;
    params.bcquality = 1.0f;
    params.zstdlevel = 16;
    params.numthreads = 16;
    params.usevq = true;
    params.usezstd = true;

    switch (type) {
    case hdr:
        std::cout << "texture type: hdr (using bc6h with vq)\n";
        params.bcformat = bcformat::bc6h;
		params.bcquality = 0.15f; // use a lower quality for hdr to set reasonable compression time
        params.quality = 1.0f; // use a high quality for hdr vq
        params.vq_min_cb_power = 6;  // 64 entries
        params.vq_max_cb_power = 12; // 4096 entriess
        params.vq_fastmodesampleratio = 0.9f;
        params.vq_maxiterations = 48;         // allow more iterations for k-means to converge
        break;
    case albedo:
        params.bcformat = bcformat::bc1;
        std::cout << "texture type: albedo using bc1\n";
		params.alphathreshold = 1; // use smallest alpha threshold for bc1 compression
        params.quality = 0.8f;
        params.vq_metric = distancemetric::perceptual_lab;
        break;
    case normal:
        std::cout << "texture type: normal (using bc5)\n";
        params.bcformat = bcformat::bc5;
        params.quality = 0.8f;
        params.vq_metric = distancemetric::rgb_simd;
        break;
    case ao:
    case bump:
    case displacement:
    case gloss:
    case roughness:
    case specular:
        std::cout << "texture type: grayscale/mask (using bc4)\n";
        params.bcformat = bcformat::bc4;
        params.quality = 0.5f;
        params.vq_metric = distancemetric::rgb_simd;
        break;
    default:
        std::cout << "texture type: unknown (defaulting to bc7)\n";
        params.bcformat = bcformat::bc7;
        params.quality = 0.8f;
        params.vq_metric = distancemetric::rgb_simd;
        break;
    }

    std::string suffix = "_bc" + std::to_string(static_cast<int>(params.bcformat));
    if (params.usevq) {
        suffix += (params.vq_metric == distancemetric::perceptual_lab ? "_lab" : "_rgb");
    }

    std::cout << "compression: bc" << static_cast<int>(params.bcformat)
        << ", use vq: " << (params.usevq ? "yes" : "no") <<
        ", use zstd: " << (params.usezstd ? "yes" : "no") << std::endl;

    try {
        std::string out_name_bin = "output/" + filepath.stem().string() + suffix + ".yupt2";
        {
            auto start_compress = std::chrono::high_resolution_clock::now();
            compressedtexture compressed;
            // call the correct compress overload based on whether the image is hdr
            if (image.ishdr) {
                compressed = compressor.compresshdr(std::get<std::vector<float>>(image.data).data(), image.width, image.height, params);
            }
            else {
                compressed = compressor.compress(std::get<std::vector<uint8_t>>(image.data).data(), image.width, image.height, image.channels, params);
            }
            auto end_compress = std::chrono::high_resolution_clock::now();
            std::cout << "compression finished in " << std::fixed << std::setprecision(2)
                << std::chrono::duration<double>(end_compress - start_compress).count() << "s.\n";

            std::ofstream outfile(out_name_bin, std::ios::binary);
            outfile.write(reinterpret_cast<const char*>(&compressed.info), sizeof(textureinfo));
            outfile.write(reinterpret_cast<const char*>(compressed.compresseddata.data()), compressed.compresseddata.size());
            outfile.close();
            std::cout << "saved compressed file: " << out_name_bin << std::endl;
        }

        // --- decompression and verification ---

        std::ifstream infile(out_name_bin, std::ios::binary);
        if (!infile) throw std::runtime_error("failed to open " + out_name_bin + " for reading.");
        compressedtexture loadedtexture;
        infile.read(reinterpret_cast<char*>(&loadedtexture.info), sizeof(textureinfo));
        infile.seekg(0, std::ios::end);
        size_t filedatasize = static_cast<size_t>(infile.tellg()) - sizeof(textureinfo);
        loadedtexture.compresseddata.resize(filedatasize);
        infile.seekg(sizeof(textureinfo), std::ios::beg);
        infile.read(reinterpret_cast<char*>(loadedtexture.compresseddata.data()), filedatasize);
        infile.close();

        auto start_decompress_bcn = std::chrono::high_resolution_clock::now();
        auto bcdata = compressor.decompresstobcn(loadedtexture);
        auto end_decompress_bcn = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_decompress_bcn = end_decompress_bcn - start_decompress_bcn;
        std::cout << "decompression to bcn (gpu-ready) finished in " << std::fixed << std::setprecision(4) << diff_decompress_bcn.count() << " seconds.\n";

        auto start_decompress = std::chrono::high_resolution_clock::now();
        image outputimage;
        outputimage.width = image.width;
        outputimage.height = image.height;
        outputimage.channels = 4;
        outputimage.ishdr = image.ishdr;

        if (image.ishdr) {
            outputimage.data = compressor.decompresstorgbaf(loadedtexture);
        }
        else {
            outputimage.data = compressor.decompresstorgba(loadedtexture);
            auto end_decompress = std::chrono::high_resolution_clock::now();
            std::cout << "decompression to rgba finished in " << std::fixed << std::setprecision(4)
                << std::chrono::duration<double>(end_decompress - start_decompress).count() << "s.\n";

            // special handling for bc5 normal map visualization
            if (params.bcformat == bcformat::bc5) {
                std::cout << "reconstructing z-channel for bc5 normal map visualization...\n";
                auto& decompressed_data = std::get<std::vector<uint8_t>>(outputimage.data);
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

        std::string out_ext = image.ishdr ? ".hdr" : ".png";
        outputimage.save("output/" + filepath.stem().string() + suffix + out_ext);

    }
    catch (const std::exception& e) {
        std::cerr << "an error occurred during processing: " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    try {
        vqbcncompressor compressor;
        fs::create_directory("output");
        std::string test_dir = "test_assets";
        if (!fs::exists(test_dir)) {
            std::cerr << "error: test directory '" << test_dir << "' not found." << std::endl; return 1;
        }
        for (const auto& file : fs::directory_iterator(test_dir)) {
            std::string ext = file.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tga" || ext == ".hdr" || ext == ".exr") {
                processimage(file.path(), compressor);
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "a critical error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}