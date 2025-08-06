// vq_encoder.h (Optimized)

#pragma once

#include "vq_bcn_types.h"
#include "bcn_compressor.h"
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>
#include <execution>
#include <numeric> // Required for std::iota

class VQEncoder {
public:
    struct Config {
        uint32_t codebookSize = 256;
        uint32_t maxIterations = 20;
        float convergenceThreshold = 1.0f;
        bool useKMeansPlusPlus = true; // This is now a more meaningful option
    };

private:
    Config config;
    BCFormat bcFormat;
    BCnCompressor bcnCompressor;
    std::mt19937 rng;

    // Calculates squared Euclidean distance between two 4x4 RGBA blocks (64 bytes each)
    float RgbaBlockDistance(const uint8_t* rgbaA, const uint8_t* rgbaB) {
        float dist = 0.0f;
        // This is a candidate for SIMD optimization for further performance gains
        for (size_t i = 0; i < 16 * 4; ++i) {
            float diff = static_cast<float>(rgbaA[i]) - static_cast<float>(rgbaB[i]);
            dist += diff * diff;
        }
        return dist;
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
            config.codebookSize = numBlocks > 0 ? numBlocks : 1;
        }

        // OPTIMIZATION 1: Decompress all blocks ONCE at the beginning.
        std::vector<std::vector<uint8_t>> rgbaBlocks(numBlocks);
#pragma omp parallel for
        for (int64_t i = 0; i < numBlocks; ++i) {
            rgbaBlocks[i] = DecompressSingleBlock(&bcBlocks[i * blockSize]);
        }

        // OPTIMIZATION 2: The k-means algorithm will now operate on RGBA centroids.
        std::vector<std::vector<uint8_t>> rgbaCentroids(config.codebookSize);

        // --- Initialize Centroids (Random Sampling) ---
        std::vector<size_t> initialIndices(numBlocks);
        std::iota(initialIndices.begin(), initialIndices.end(), 0);
        std::shuffle(initialIndices.begin(), initialIndices.end(), rng);
        for (uint32_t i = 0; i < config.codebookSize; ++i) {
            rgbaCentroids[i] = rgbaBlocks[initialIndices[i]];
        }

        // --- K-means Iterations ---
        std::vector<uint32_t> assignments(numBlocks);

        for (uint32_t iter = 0; iter < config.maxIterations; ++iter) {
            // Assignment step: Assign each block to the nearest RGBA centroid.
            // No per-iteration decompression is needed here.
#pragma omp parallel for
            for (int64_t i = 0; i < numBlocks; ++i) {
                float minDist = std::numeric_limits<float>::max();
                uint32_t bestIdx = 0;
                for (uint32_t j = 0; j < config.codebookSize; ++j) {
                    float dist = RgbaBlockDistance(rgbaBlocks[i].data(), rgbaCentroids[j].data());
                    if (dist < minDist) {
                        minDist = dist;
                        bestIdx = j;
                    }
                }
                assignments[i] = bestIdx;
            }

            // Update step: Recalculate centroids by averaging in RGBA space.
            // No per-iteration re-compression is needed here.
            std::vector<std::vector<float>> newCentroids(config.codebookSize, std::vector<float>(16 * 4, 0.0f));
            std::vector<uint32_t> counts(config.codebookSize, 0);

            for (size_t i = 0; i < numBlocks; ++i) {
                uint32_t idx = assignments[i];
                counts[idx]++;
                for (size_t j = 0; j < 16 * 4; ++j) {
                    newCentroids[idx][j] += static_cast<float>(rgbaBlocks[i][j]);
                }
            }

#pragma omp parallel for
            for (int64_t i = 0; i < config.codebookSize; ++i) {
                if (counts[i] > 0) {
                    for (size_t j = 0; j < 16 * 4; ++j) {
                        float avgValue = newCentroids[i][j] / counts[i];
                        rgbaCentroids[i][j] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, avgValue)));
                    }
                }
                else {
                    // If a centroid has no points, re-initialize it to a random block
                    // to prevent it from becoming empty.
                    rgbaCentroids[i] = rgbaBlocks[initialIndices[(iter + i) % numBlocks]];
                }
            }
        }

        // OPTIMIZATION 3: Compress the final RGBA centroids to BCn format ONCE at the end.
        VQCodebook finalCodebook(blockSize, config.codebookSize);
        finalCodebook.entries.resize(config.codebookSize);
#pragma omp parallel for
        for (int64_t i = 0; i < config.codebookSize; ++i) {
            finalCodebook.entries[i] = CompressSingleBlock(rgbaCentroids[i].data());
        }

        return finalCodebook;
    }

    // QuantizeBlocks remains the same, as it was already efficient.
    std::vector<uint32_t> QuantizeBlocks(const std::vector<uint8_t>& bcBlocks, const VQCodebook& codebook) {
        size_t numBlocks = bcBlocks.size() / codebook.blockSize;
        std::vector<uint32_t> indices(numBlocks);

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