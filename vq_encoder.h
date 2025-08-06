// vq_encoder.h (Fully Revised)

#pragma once

#include "vq_bcn_types.h"
#include "bcn_compressor.h" // <-- ADDED DEPENDENCY
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>
#include <execution>

class VQEncoder {
public:
    struct Config {
        uint32_t codebookSize = 256;
        uint32_t maxIterations = 20; // Reduced for performance
        float convergenceThreshold = 1.0f; // Loosened for performance
        bool useKMeansPlusPlus = true;
    };

private:
    Config config;
    BCFormat bcFormat;
    BCnCompressor bcnCompressor; // We need this for on-the-fly conversions
    std::mt19937 rng;

    // Calculates distance between two 4x4 RGBA blocks (64 bytes each)
    float RgbaBlockDistance(const uint8_t* rgbaA, const uint8_t* rgbaB) {
        float dist = 0.0f;
        for (size_t i = 0; i < 16 * 4; ++i) {
            float diff = static_cast<float>(rgbaA[i]) - static_cast<float>(rgbaB[i]);
            dist += diff * diff;
        }
        return dist; // Using squared Euclidean distance is faster and sufficient for comparison
    }

    // Decompresses a single BCn block to a 4x4 RGBA buffer
    std::vector<uint8_t> DecompressSingleBlock(const uint8_t* bcBlock) {
        return bcnCompressor.DecompressToRGBA(bcBlock, 4, 4, bcFormat);
    }

    // Compresses a single 4x4 RGBA block to a BCn block
    std::vector<uint8_t> CompressSingleBlock(const uint8_t* rgbaBlock) {
        return bcnCompressor.CompressRGBA(rgbaBlock, 4, 4, bcFormat, 1.0f);
    }

public:
    VQEncoder(const Config& cfg = Config(), BCFormat format = BCFormat::BC7)
        : config(cfg), bcFormat(format), rng(std::random_device{}()) {
    }

    void SetFormat(BCFormat format) {
        bcFormat = format;
	}

    VQCodebook BuildCodebook(const std::vector<uint8_t>& bcBlocks, size_t blockSize) {
        size_t numBlocks = bcBlocks.size() / blockSize;
        if (numBlocks < config.codebookSize) {
            config.codebookSize = numBlocks;
        }

        VQCodebook codebook(blockSize, config.codebookSize);
        codebook.entries.resize(config.codebookSize);

        // --- K-MEANS LOGIC NOW OPERATES ON DECOMPRESSED RGBA DATA ---

        // Step 1: Decompress all input blocks to RGBA for processing
        std::vector<std::vector<uint8_t>> rgbaBlocks(numBlocks);
#pragma omp parallel for
        for (int64_t i = 0; i < numBlocks; ++i) {
            rgbaBlocks[i] = DecompressSingleBlock(&bcBlocks[i * blockSize]);
        }

        // Step 2: Initialize codebook (e.g., random sampling)
        std::vector<size_t> initialIndices(numBlocks);
        std::iota(initialIndices.begin(), initialIndices.end(), 0);
        std::shuffle(initialIndices.begin(), initialIndices.end(), rng);
        for (uint32_t i = 0; i < config.codebookSize; ++i) {
            codebook.entries[i].assign(
                bcBlocks.begin() + initialIndices[i] * blockSize,
                bcBlocks.begin() + (initialIndices[i] + 1) * blockSize
            );
        }

        // Step 3: K-means iterations
        std::vector<uint32_t> assignments(numBlocks);
        std::vector<std::vector<uint8_t>> codebookRgba(config.codebookSize);

        for (uint32_t iter = 0; iter < config.maxIterations; ++iter) {
            // Decompress current codebook for distance checks
            for (uint32_t i = 0; i < config.codebookSize; ++i) {
                codebookRgba[i] = DecompressSingleBlock(codebook.entries[i].data());
            }

            // Assignment step (in RGBA space)
#pragma omp parallel for
            for (int64_t i = 0; i < numBlocks; ++i) {
                float minDist = std::numeric_limits<float>::max();
                uint32_t bestIdx = 0;
                for (uint32_t j = 0; j < config.codebookSize; ++j) {
                    float dist = RgbaBlockDistance(rgbaBlocks[i].data(), codebookRgba[j].data());
                    if (dist < minDist) {
                        minDist = dist;
                        bestIdx = j;
                    }
                }
                assignments[i] = bestIdx;
            }

            // Update step (averaging in RGBA space, then re-compressing)
            std::vector<std::vector<float>> centroids(config.codebookSize, std::vector<float>(16 * 4, 0.0f));
            std::vector<uint32_t> counts(config.codebookSize, 0);

            for (size_t i = 0; i < numBlocks; ++i) {
                uint32_t idx = assignments[i];
                counts[idx]++;
                for (size_t j = 0; j < 16 * 4; ++j) {
                    centroids[idx][j] += static_cast<float>(rgbaBlocks[i][j]);
                }
            }

#pragma omp parallel for
            for (int64_t i = 0; i < config.codebookSize; ++i) {
                if (counts[i] > 0) {
                    std::vector<uint8_t> avgRgbaBlock(16 * 4);
                    for (size_t j = 0; j < 16 * 4; ++j) {
                        float avgValue = centroids[i][j] / counts[i];
                        avgRgbaBlock[j] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, avgValue)));
                    }
                    // Re-compress the new average RGBA block to get the new BCn codebook entry
                    codebook.entries[i] = CompressSingleBlock(avgRgbaBlock.data());
                }
            }
        }
        return codebook;
    }

    std::vector<uint32_t> QuantizeBlocks(const std::vector<uint8_t>& bcBlocks, const VQCodebook& codebook) {
        size_t numBlocks = bcBlocks.size() / codebook.blockSize;
        std::vector<uint32_t> indices(numBlocks);

        // Decompress codebook once for faster comparisons
        std::vector<std::vector<uint8_t>> codebookRgba(codebook.codebookSize);
        for (uint32_t i = 0; i < codebook.codebookSize; ++i) {
            codebookRgba[i] = DecompressSingleBlock(codebook.entries[i].data());
        }

#pragma omp parallel for
        for (int64_t i = 0; i < numBlocks; ++i) {
            auto rgbaBlock = DecompressSingleBlock(&bcBlocks[i * codebook.blockSize]);
            float minDist = std::numeric_limits<float>::max();
            uint32_t bestIdx = 0;
            for (uint32_t j = 0; j < codebook.codebookSize; ++j) {
                float dist = RgbaBlockDistance(rgbaBlock.data(), codebookRgba[j].data());
                if (dist < minDist) {
                    minDist = dist;
                    bestIdx = j;
                }
            }
            indices[i] = bestIdx;
        }
        return indices;
    }
};