#include "vq_encoder.h"
#include <immintrin.h> // For SIMD intrinsics (AVX2)

// =================================================================================================
// Configuration
// =================================================================================================

CompressionConfig::CompressionConfig(float quality_level) {
    SetQuality(quality_level);
}

void CompressionConfig::SetQuality(float quality_level) {
    quality = std::clamp(quality_level, 0.0f, 1.0f);

    // Map quality non-linearly to the power of the codebook size
    uint32_t power = min_cb_power + static_cast<uint32_t>(roundf(quality * (max_cb_power - min_cb_power)));
    codebookSize = 1 << power; // 2^power
}

// =================================================================================================
// VQEncoder Implementation
// =================================================================================================

VQEncoder::VQEncoder(const CompressionConfig& cfg)
    : config(cfg), rng(std::random_device{}()), bcFormat(BCFormat::Unknown) {
}

void VQEncoder::SetFormat(BCFormat format) {
    bcFormat = format;
}

// =================================================================================================
// Color Space Conversion (LDR)
// =================================================================================================

// Converts a single LDR pixel (any channel count) to our standard 4-component Oklab format (L, a, b, A).
void VQEncoder::PixelToOklab(const uint8_t* pixel, uint8_t channelCount, float* oklab) const {
    colorm::Rgb rgb;
    float alpha = 1.0f;
    switch (channelCount) {
    case 1: // Grayscale
        rgb.setRed8(pixel[0]).setGreen8(pixel[0]).setBlue8(pixel[0]);
        break;
    case 2: // Grayscale + Alpha
        rgb.setRed8(pixel[0]).setGreen8(pixel[0]).setBlue8(pixel[0]);
        alpha = pixel[1] / 255.0f;
        break;
    case 3: // RGB
        rgb.setRed8(pixel[0]).setGreen8(pixel[1]).setBlue8(pixel[2]);
        break;
    case 4: // RGBA
        rgb.setRed8(pixel[0]).setGreen8(pixel[1]).setBlue8(pixel[2]);
        alpha = pixel[3] / 255.0f;
        break;
    default: // Should not happen
        oklab[0] = oklab[1] = oklab[2] = oklab[3] = 0;
        return;
    }
    colorm::Oklab oklab_color(rgb);
    oklab[0] = oklab_color.lightness();
    oklab[1] = oklab_color.a();
    oklab[2] = oklab_color.b();
    oklab[3] = alpha;
}

// Converts a single 4-component Oklab pixel back to a target LDR pixel format.
void VQEncoder::OklabToPixel(const float* oklab, uint8_t channelCount, uint8_t* pixel) const {
    colorm::Oklab oklab_color(oklab[0], oklab[1], oklab[2]);
    colorm::Rgb rgb(oklab_color);
    rgb.clip(); // Clip to ensure the color is within the sRGB gamut.

    const uint8_t r = rgb.red8();
    const uint8_t g = rgb.green8();
    const uint8_t b = rgb.blue8();
    const uint8_t a = static_cast<uint8_t>(std::clamp(oklab[3] * 255.0f, 0.0f, 255.0f));

    switch (channelCount) {
    case 1: // Grayscale (BT.709 Luminance)
        pixel[0] = static_cast<uint8_t>(r * 0.2126f + g * 0.7152f + b * 0.0722f);
        break;
    case 2: // Grayscale + Alpha
        pixel[0] = static_cast<uint8_t>(r * 0.2126f + g * 0.7152f + b * 0.0722f);
        pixel[1] = a;
        break;
    case 3: // RGB
        pixel[0] = r; pixel[1] = g; pixel[2] = b;
        break;
    case 4: // RGBA
        pixel[0] = r; pixel[1] = g; pixel[2] = b;
        pixel[3] = a;
        break;
    }
}

// Converts a 4x4 block of LDR pixels to a 64-float OklabBlock.
OklabBlock VQEncoder::PixelBlockToOklabBlock(const std::vector<uint8_t>& pixelBlock, uint8_t channelCount) const {
    OklabBlock oklabBlock(16 * 4);
    for (size_t i = 0; i < 16; ++i) {
        PixelToOklab(&pixelBlock[i * channelCount], channelCount, &oklabBlock[i * 4]);
    }
    return oklabBlock;
}

// Converts a 64-float OklabBlock back to a 4x4 block of LDR pixels.
std::vector<uint8_t> VQEncoder::OklabBlockToPixelBlock(const OklabBlock& oklabBlock, uint8_t channelCount) const {
    std::vector<uint8_t> pixelBlock(16 * channelCount);
    for (size_t i = 0; i < 16; ++i) {
        OklabToPixel(&oklabBlock[i * 4], channelCount, &pixelBlock[i * channelCount]);
    }
    return pixelBlock;
}

// =================================================================================================
// Color Space Conversion (HDR)
// =================================================================================================

// Converts a single HDR float pixel to our standard 4-component Oklab format.
void VQEncoder::RgbaFloatToOklab(const float* pixel, uint8_t channelCount, float* oklab) const {
    // NOTE: The original ACES tonemapping has been removed to unify the color conversion pipeline.
    // For HDR, we use colorm::Rgb which operates on a 0.0 to 1.0 float range.
    colorm::Rgb rgb;
    float alpha = 1.0f;
    switch (channelCount) {
    case 1:
        rgb.setRed(pixel[0]).setGreen(pixel[0]).setBlue(pixel[0]);
        break;
    case 2:
        rgb.setRed(pixel[0]).setGreen(pixel[0]).setBlue(pixel[0]);
        alpha = pixel[1];
        break;
    case 3:
        rgb.setRed(pixel[0]).setGreen(pixel[1]).setBlue(pixel[2]);
        break;
    case 4:
        rgb.setRed(pixel[0]).setGreen(pixel[1]).setBlue(pixel[2]);
        alpha = pixel[3];
        break;
    default:
        oklab[0] = oklab[1] = oklab[2] = oklab[3] = 0;
        return;
    }
    colorm::Oklab oklab_color(rgb);
    oklab[0] = oklab_color.lightness();
    oklab[1] = oklab_color.a();
    oklab[2] = oklab_color.b();
    oklab[3] = alpha;
}

// Converts a single 4-component Oklab pixel back to a target HDR float pixel format.
void VQEncoder::OklabToRgbaFloat(const float* oklab, uint8_t channelCount, float* pixel) const {
    colorm::Oklab oklab_color(oklab[0], oklab[1], oklab[2]);
    colorm::Rgb rgb(oklab_color);
    rgb.clip();

    const float r = rgb.red();
    const float g = rgb.green();
    const float b = rgb.blue();
    const float a = oklab[3];

    switch (channelCount) {
    case 1:
        pixel[0] = r * 0.2126f + g * 0.7152f + b * 0.0722f;
        break;
    case 2:
        pixel[0] = r * 0.2126f + g * 0.7152f + b * 0.0722f;
        pixel[1] = a;
        break;
    case 3:
        pixel[0] = r; pixel[1] = g; pixel[2] = b;
        break;
    case 4:
        pixel[0] = r; pixel[1] = g; pixel[2] = b;
        pixel[3] = a;
        break;
    }
}

// Converts a 4x4 block of HDR float pixels to a 64-float OklabFloatBlock.
OklabFloatBlock VQEncoder::RgbaFloatBlockToOklabBlock(const std::vector<float>& pixelBlock, uint8_t channelCount) const {
    OklabFloatBlock oklabBlock(16 * 4);
    for (size_t i = 0; i < 16; ++i) {
        RgbaFloatToOklab(&pixelBlock[i * channelCount], channelCount, &oklabBlock[i * 4]);
    }
    return oklabBlock;
}

// Converts a 64-float OklabFloatBlock back to a 4x4 block of HDR float pixels.
std::vector<float> VQEncoder::OklabBlockToRgbaFloatBlock(const OklabFloatBlock& oklabBlock, uint8_t channelCount) const {
    std::vector<float> pixelBlock(16 * channelCount);
    for (size_t i = 0; i < 16; ++i) {
        OklabToRgbaFloat(&oklabBlock[i * 4], channelCount, &pixelBlock[i * channelCount]);
    }
    return pixelBlock;
}

// =================================================================================================
// Distance Functions
// =================================================================================================

// Calculates Sum of Absolute Differences (SAD) for a 4x4 block. Works for any channel count.
float VQEncoder::BlockDistanceSAD(const uint8_t* a, const uint8_t* b, uint8_t channelCount) const {
    const size_t blockSizeBytes = 16 * channelCount;
    uint32_t sad = 0;
    for (size_t i = 0; i < blockSizeBytes; ++i) {
        sad += std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
    }
    return static_cast<float>(sad);
}

// Calculates the squared Euclidean distance between two Oklab blocks using AVX2 intrinsics.
// This is used for both LDR and HDR, as they share the same internal OklabBlock format.
float VQEncoder::OklabBlockDistanceSq_SIMD(const OklabBlock& oklabA, const OklabBlock& oklabB) const {
    __m256 sum_sq_diff = _mm256_setzero_ps();

    // The internal layout for each pixel is [L, a, b, A]. We give the L channel
    // a higher weight as it's perceptually the most important component.
    // The _mm256_set_ps instruction takes arguments in reverse order.
    // For two pixels [L0,a0,b0,A0, L1,a1,b1,A1], the weights are [wA1,wb1,wa1,wL1, wA0,wb0,wa0,wL0].
    const __m256 weight = _mm256_set_ps(1.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 1.0f, 2.0f);

    // Process 16 pixels (64 floats) in 8-float chunks (2 pixels at a time).
    for (size_t i = 0; i < 64; i += 8) {
        __m256 a = _mm256_loadu_ps(&oklabA[i]);
        __m256 b = _mm256_loadu_ps(&oklabB[i]);
        __m256 diff = _mm256_sub_ps(a, b);
        __m256 weighted_diff = _mm256_mul_ps(diff, weight);
        sum_sq_diff = _mm256_fmadd_ps(weighted_diff, weighted_diff, sum_sq_diff);
    }

    // Horizontal sum of the 8 floats in the accumulator to get the final result.
    __m128 lo_half = _mm256_castps256_ps128(sum_sq_diff);
    __m128 hi_half = _mm256_extractf128_ps(sum_sq_diff, 1);
    __m128 sum_128 = _mm_add_ps(lo_half, hi_half);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    return _mm_cvtss_f32(sum_128);
}

// Alias for HDR for clarity, calls the same implementation.
float VQEncoder::OklabFloatBlockDistanceSq_SIMD(const OklabFloatBlock& labA, const OklabFloatBlock& labB) const {
    return OklabBlockDistanceSq_SIMD(labA, labB);
}

// =================================================================================================
// Block Compression
// =================================================================================================

std::vector<uint8_t> VQEncoder::CompressSingleBlock(const uint8_t* pixelBlock, uint8_t channels, const CompressionParams& params) {
    return bcnCompressor.Compress(pixelBlock, 4, 4, channels, bcFormat, params.numThreads, params.bcQuality);
}

std::vector<uint8_t> VQEncoder::CompressSingleBlockHDR(const std::vector<float>& pixelBlock, uint8_t channels, const CompressionParams& params) {
    return bcnCompressor.CompressHDR(pixelBlock.data(), 4, 4, channels, bcFormat, params.numThreads, params.bcQuality);
}

// =================================================================================================
// Codebook Building (LDR)
// =================================================================================================

VQCodebook VQEncoder::BuildCodebook(const std::vector<std::vector<uint8_t>>& allPixelBlocks, uint8_t channels, std::vector<std::vector<uint8_t>>& outPixelCentroids, const CompressionParams& params) {
    // --- Sampling ---
    std::vector<const std::vector<uint8_t>*> sampledBlocksPtrs;
    if (config.fastModeSampleRatio < 1.0f && config.fastModeSampleRatio > 0.0f) {
        size_t numToSample = static_cast<size_t>(allPixelBlocks.size() * config.fastModeSampleRatio);
        numToSample = std::max(static_cast<size_t>(config.codebookSize), numToSample);
        numToSample = std::min(numToSample, allPixelBlocks.size());
        sampledBlocksPtrs.reserve(numToSample);

        std::vector<size_t> indices(allPixelBlocks.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t i = 0; i < numToSample; ++i) {
            sampledBlocksPtrs.push_back(&allPixelBlocks[indices[i]]);
        }
    }
    else {
        sampledBlocksPtrs.reserve(allPixelBlocks.size());
        for (const auto& block : allPixelBlocks) {
            sampledBlocksPtrs.push_back(&block);
        }
    }
    const auto& blocksToProcess = sampledBlocksPtrs;
    size_t numBlocks = blocksToProcess.size();

    // --- K-Means++ Initialization (always use fast SAD for initialization) ---
    std::vector<std::vector<uint8_t>> pixelCentroids(config.codebookSize);
    std::vector<float> minDistSq(numBlocks, std::numeric_limits<float>::max());
    std::uniform_int_distribution<size_t> distrib(0, numBlocks - 1);
    pixelCentroids[0] = *blocksToProcess[distrib(rng)];
    for (uint32_t i = 1; i < config.codebookSize; ++i) {
        double current_sum = 0.0;
#pragma omp parallel for num_threads(params.numThreads) reduction(+:current_sum)
        for (int64_t j = 0; j < numBlocks; ++j) {
            float d = BlockDistanceSAD((*blocksToProcess[j]).data(), pixelCentroids[i - 1].data(), channels);
            minDistSq[j] = std::min(d * d, minDistSq[j]);
            current_sum += minDistSq[j];
        }
        if (current_sum <= 0) {
            for (uint32_t k = i; k < config.codebookSize; ++k) pixelCentroids[k] = pixelCentroids[0];
            break;
        }
        std::uniform_real_distribution<double> p_distrib(0.0, current_sum);
        double p = p_distrib(rng);
        double cumulative_p = 0.0;
        for (size_t j = 0; j < numBlocks; ++j) {
            cumulative_p += minDistSq[j];
            if (cumulative_p >= p) {
                pixelCentroids[i] = *blocksToProcess[j];
                break;
            }
        }
    }

    // --- K-Means Iterations ---
    std::vector<uint32_t> assignments(numBlocks, 0);
    std::vector<float> errors(numBlocks);

    if (config.metric == DistanceMetric::PERCEPTUAL_OKLAB) {
        std::vector<OklabBlock> oklabBlocks(numBlocks);
#pragma omp parallel for num_threads(params.numThreads)
        for (int64_t i = 0; i < numBlocks; ++i) oklabBlocks[i] = PixelBlockToOklabBlock(*blocksToProcess[i], channels);

        std::vector<OklabBlock> oklabCentroids(config.codebookSize);
#pragma omp parallel for num_threads(params.numThreads)
        for (int64_t i = 0; i < config.codebookSize; ++i) oklabCentroids[i] = PixelBlockToOklabBlock(pixelCentroids[i], channels);

        for (uint32_t iter = 0; iter < config.maxIterations; ++iter) {
            std::atomic<bool> hasChanged = false;
#pragma omp parallel for num_threads(params.numThreads)
            for (int64_t i = 0; i < numBlocks; ++i) {
                float min_d = std::numeric_limits<float>::max();
                uint32_t best_c = 0;
                for (uint32_t c = 0; c < config.codebookSize; ++c) {
                    float d = OklabBlockDistanceSq_SIMD(oklabBlocks[i], oklabCentroids[c]);
                    if (d < min_d) { min_d = d; best_c = c; }
                }
                errors[i] = min_d;
                if (assignments[i] != best_c) { assignments[i] = best_c; hasChanged = true; }
            }
            if (!hasChanged && iter > 0) break;

            std::vector<OklabBlock> newCentroids(config.codebookSize, OklabBlock(64, 0.0f));
            std::vector<uint32_t> counts(config.codebookSize, 0);
#pragma omp parallel num_threads(params.numThreads)
            {
                std::vector<OklabBlock> localNewCentroids(config.codebookSize, OklabBlock(64, 0.0f));
                std::vector<uint32_t> localCounts(config.codebookSize, 0);
#pragma omp for nowait
                for (int64_t i = 0; i < numBlocks; ++i) {
                    uint32_t c_idx = assignments[i];
                    localCounts[c_idx]++;
                    for (size_t j = 0; j < 64; ++j) localNewCentroids[c_idx][j] += oklabBlocks[i][j];
                }
#pragma omp critical
                {
                    for (uint32_t c = 0; c < config.codebookSize; ++c) {
                        counts[c] += localCounts[c];
                        for (size_t j = 0; j < 64; ++j) newCentroids[c][j] += localNewCentroids[c][j];
                    }
                }
            }

#pragma omp parallel for num_threads(params.numThreads)
            for (int64_t c = 0; c < config.codebookSize; ++c) {
                if (counts[c] > 0) {
                    float inv_count = 1.0f / counts[c];
                    for (size_t j = 0; j < 64; ++j) oklabCentroids[c][j] = newCentroids[c][j] * inv_count;
                }
                else {
                    size_t worstBlockIdx = std::distance(errors.begin(), std::max_element(std::execution::par, errors.begin(), errors.end()));
                    oklabCentroids[c] = oklabBlocks[worstBlockIdx];
                    errors[worstBlockIdx] = 0.0f;
                }
            }
        }
#pragma omp parallel for num_threads(params.numThreads)
        for (int64_t i = 0; i < config.codebookSize; ++i) pixelCentroids[i] = OklabBlockToPixelBlock(oklabCentroids[i], channels);
    }
    else { // SAD Path
        for (uint32_t iter = 0; iter < config.maxIterations; ++iter) {
            std::atomic<bool> hasChanged = false;
#pragma omp parallel for num_threads(params.numThreads)
            for (int64_t i = 0; i < numBlocks; ++i) {
                float min_d = std::numeric_limits<float>::max();
                uint32_t best_c = 0;
                for (uint32_t c = 0; c < config.codebookSize; ++c) {
                    float d = BlockDistanceSAD((*blocksToProcess[i]).data(), pixelCentroids[c].data(), channels);
                    if (d < min_d) { min_d = d; best_c = c; }
                }
                errors[i] = min_d;
                if (assignments[i] != best_c) { assignments[i] = best_c; hasChanged = true; }
            }
            if (!hasChanged && iter > 0) break;

            const size_t centroid_size = 16 * channels;
            std::vector<std::vector<uint64_t>> newCentroids(config.codebookSize, std::vector<uint64_t>(centroid_size, 0));
            std::vector<uint32_t> counts(config.codebookSize, 0);
#pragma omp parallel num_threads(params.numThreads)
            {
                std::vector<std::vector<uint64_t>> localNewCentroids(config.codebookSize, std::vector<uint64_t>(centroid_size, 0));
                std::vector<uint32_t> localCounts(config.codebookSize, 0);
#pragma omp for nowait
                for (int64_t i = 0; i < numBlocks; ++i) {
                    uint32_t c_idx = assignments[i];
                    localCounts[c_idx]++;
                    for (size_t j = 0; j < centroid_size; ++j) localNewCentroids[c_idx][j] += (*blocksToProcess[i])[j];
                }
#pragma omp critical
                {
                    for (uint32_t c = 0; c < config.codebookSize; ++c) {
                        counts[c] += localCounts[c];
                        for (size_t j = 0; j < centroid_size; ++j) newCentroids[c][j] += localNewCentroids[c][j];
                    }
                }
            }

#pragma omp parallel for num_threads(params.numThreads)
            for (int64_t c = 0; c < config.codebookSize; ++c) {
                if (counts[c] > 0) {
                    for (size_t j = 0; j < centroid_size; ++j) pixelCentroids[c][j] = static_cast<uint8_t>(newCentroids[c][j] / counts[c]);
                }
                else {
                    size_t worstBlockIdx = std::distance(errors.begin(), std::max_element(std::execution::par, errors.begin(), errors.end()));
                    pixelCentroids[c] = *blocksToProcess[worstBlockIdx];
                    errors[worstBlockIdx] = 0.0f;
                }
            }
        }
    }

    // --- Finalize Codebook ---
    outPixelCentroids = pixelCentroids;
    VQCodebook finalCodebook(BCBlockSize::GetSize(bcFormat), config.codebookSize);
    finalCodebook.entries.resize(config.codebookSize);
#pragma omp parallel for num_threads(params.numThreads)
    for (int64_t i = 0; i < config.codebookSize; ++i) {
        finalCodebook.entries[i] = CompressSingleBlock(pixelCentroids[i].data(), channels, params);
    }
    return finalCodebook;
}

// =================================================================================================
// Quantization (LDR)
// =================================================================================================

std::vector<uint32_t> VQEncoder::QuantizeBlocks(const std::vector<std::vector<uint8_t>>& pixelBlocks, uint8_t channels, const std::vector<std::vector<uint8_t>>& pixelCentroids, const CompressionParams& params) {
    size_t numBlocks = pixelBlocks.size();
    if (numBlocks == 0) return {};
    std::vector<uint32_t> indices(numBlocks);
    uint32_t codebookSize = static_cast<uint32_t>(pixelCentroids.size());

    if (config.metric == DistanceMetric::PERCEPTUAL_OKLAB) {
        std::vector<OklabBlock> oklabBlocks(numBlocks);
#pragma omp parallel for num_threads(params.numThreads)
        for (int64_t i = 0; i < numBlocks; ++i) oklabBlocks[i] = PixelBlockToOklabBlock(pixelBlocks[i], channels);

        std::vector<OklabBlock> oklabCentroids(codebookSize);
#pragma omp parallel for num_threads(params.numThreads)
        for (int64_t i = 0; i < codebookSize; ++i) oklabCentroids[i] = PixelBlockToOklabBlock(pixelCentroids[i], channels);

#pragma omp parallel for num_threads(params.numThreads)
        for (int64_t i = 0; i < numBlocks; ++i) {
            float minDist = std::numeric_limits<float>::max();
            uint32_t bestIdx = 0;
            for (uint32_t j = 0; j < codebookSize; ++j) {
                float dist = OklabBlockDistanceSq_SIMD(oklabBlocks[i], oklabCentroids[j]);
                if (dist < minDist) { minDist = dist; bestIdx = j; }
            }
            indices[i] = bestIdx;
        }
    }
    else { // SAD path
#pragma omp parallel for num_threads(params.numThreads)
        for (int64_t i = 0; i < numBlocks; ++i) {
            float minDist = std::numeric_limits<float>::max();
            uint32_t bestIdx = 0;
            for (uint32_t j = 0; j < codebookSize; ++j) {
                float dist = BlockDistanceSAD(pixelBlocks[i].data(), pixelCentroids[j].data(), channels);
                if (dist < minDist) { minDist = dist; bestIdx = j; }
            }
            indices[i] = bestIdx;
        }
    }
    return indices;
}


// =================================================================================================
// Codebook Building (HDR)
// =================================================================================================

VQCodebook VQEncoder::BuildCodebookHDR(const std::vector<std::vector<float>>& allPixelFloatBlocks, uint8_t channels, std::vector<std::vector<float>>& outPixelFloatCentroids, const CompressionParams& params) {
    // --- Sampling ---
    std::vector<const std::vector<float>*> sampledBlocksPtrs;
    if (config.fastModeSampleRatio < 1.0f && config.fastModeSampleRatio > 0.0f) {
        size_t numToSample = static_cast<size_t>(allPixelFloatBlocks.size() * config.fastModeSampleRatio);
        numToSample = std::max(static_cast<size_t>(config.codebookSize), numToSample);
        numToSample = std::min(numToSample, allPixelFloatBlocks.size());
        sampledBlocksPtrs.reserve(numToSample);
        std::vector<size_t> indices(allPixelFloatBlocks.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        for (size_t i = 0; i < numToSample; ++i) {
            sampledBlocksPtrs.push_back(&allPixelFloatBlocks[indices[i]]);
        }
    }
    else {
        sampledBlocksPtrs.reserve(allPixelFloatBlocks.size());
        for (const auto& block : allPixelFloatBlocks) {
            sampledBlocksPtrs.push_back(&block);
        }
    }

    const auto& blocksToProcess = sampledBlocksPtrs;
    size_t numBlocks = blocksToProcess.size();
    if (numBlocks == 0) return {};

    // --- Pre-transformation: Convert all blocks to Oklab once ---
    std::vector<OklabFloatBlock> oklabBlocks(numBlocks);
#pragma omp parallel for num_threads(params.numThreads)
    for (int64_t i = 0; i < numBlocks; ++i) {
        oklabBlocks[i] = RgbaFloatBlockToOklabBlock(*blocksToProcess[i], channels);
    }

    // --- K-Means++ Initialization (on Oklab data) ---
    std::vector<OklabFloatBlock> oklabCentroids(config.codebookSize);
    std::vector<float> minDistSq(numBlocks, std::numeric_limits<float>::max());
    std::uniform_int_distribution<size_t> distrib(0, numBlocks - 1);
    oklabCentroids[0] = oklabBlocks[distrib(rng)];

    for (uint32_t i = 1; i < config.codebookSize; ++i) {
        double current_sum = 0.0;
#pragma omp parallel for num_threads(params.numThreads) reduction(+:current_sum)
        for (int64_t j = 0; j < numBlocks; ++j) {
            float d = OklabFloatBlockDistanceSq_SIMD(oklabBlocks[j], oklabCentroids[i - 1]);
            minDistSq[j] = std::min(d, minDistSq[j]);
            current_sum += minDistSq[j];
        }
        if (current_sum <= 0) {
            for (uint32_t k = i; k < config.codebookSize; ++k) oklabCentroids[k] = oklabCentroids[0];
            break;
        }
        std::uniform_real_distribution<double> p_distrib(0.0, current_sum);
        double p = p_distrib(rng);
        double cumulative_p = 0.0;
        for (size_t j = 0; j < numBlocks; ++j) {
            cumulative_p += minDistSq[j];
            if (cumulative_p >= p) {
                oklabCentroids[i] = oklabBlocks[j];
                break;
            }
        }
    }

    // --- K-Means Iterations ---
    std::vector<uint32_t> assignments(numBlocks, 0);
    std::vector<float> errors(numBlocks);
    for (uint32_t iter = 0; iter < config.maxIterations; ++iter) {
        std::atomic<bool> hasChanged = false;
#pragma omp parallel for num_threads(params.numThreads)
        for (int64_t i = 0; i < numBlocks; ++i) {
            float min_d = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < config.codebookSize; ++c) {
                float d = OklabFloatBlockDistanceSq_SIMD(oklabBlocks[i], oklabCentroids[c]);
                if (d < min_d) { min_d = d; best_c = c; }
            }
            errors[i] = min_d;
            if (assignments[i] != best_c) { assignments[i] = best_c; hasChanged = true; }
        }
        if (!hasChanged && iter > 1) break;

        std::vector<OklabFloatBlock> newCentroids(config.codebookSize, OklabFloatBlock(64, 0.0f));
        std::vector<uint32_t> counts(config.codebookSize, 0);
#pragma omp parallel num_threads(params.numThreads)
        {
            std::vector<OklabFloatBlock> localNewCentroids(config.codebookSize, OklabFloatBlock(64, 0.0f));
            std::vector<uint32_t> localCounts(config.codebookSize, 0);
#pragma omp for nowait
            for (int64_t i = 0; i < numBlocks; ++i) {
                uint32_t c_idx = assignments[i];
                localCounts[c_idx]++;
                for (size_t j = 0; j < 64; ++j) localNewCentroids[c_idx][j] += oklabBlocks[i][j];
            }
#pragma omp critical
            {
                for (uint32_t c = 0; c < config.codebookSize; ++c) {
                    counts[c] += localCounts[c];
                    for (size_t j = 0; j < 64; ++j) newCentroids[c][j] += localNewCentroids[c][j];
                }
            }
        }
#pragma omp parallel for num_threads(params.numThreads)
        for (int64_t c = 0; c < config.codebookSize; ++c) {
            if (counts[c] > 0) {
                float inv_count = 1.0f / counts[c];
                for (size_t j = 0; j < 64; ++j) oklabCentroids[c][j] = newCentroids[c][j] * inv_count;
            }
            else {
                size_t worstBlockIdx = std::distance(errors.begin(), std::max_element(std::execution::par, errors.begin(), errors.end()));
                oklabCentroids[c] = oklabBlocks[worstBlockIdx];
                errors[worstBlockIdx] = 0.0f;
            }
        }
    }

    // --- Finalize Codebook ---
    outPixelFloatCentroids.resize(config.codebookSize);
#pragma omp parallel for num_threads(params.numThreads)
    for (int64_t i = 0; i < config.codebookSize; ++i) {
        outPixelFloatCentroids[i] = OklabBlockToRgbaFloatBlock(oklabCentroids[i], channels);
    }

    VQCodebook finalCodebook(BCBlockSize::GetSize(bcFormat), config.codebookSize);
    finalCodebook.entries.resize(config.codebookSize);
#pragma omp parallel for num_threads(params.numThreads)
    for (int64_t i = 0; i < config.codebookSize; ++i) {
        finalCodebook.entries[i] = CompressSingleBlockHDR(outPixelFloatCentroids[i], channels, params);
    }
    return finalCodebook;
}


// =================================================================================================
// Quantization (HDR)
// =================================================================================================

std::vector<uint32_t> VQEncoder::QuantizeBlocksHDR(const std::vector<std::vector<float>>& pixelFloatBlocks, uint8_t channels, const std::vector<std::vector<float>>& pixelFloatCentroids, const CompressionParams& params) {
    size_t numBlocks = pixelFloatBlocks.size();
    if (numBlocks == 0) return {};
    std::vector<uint32_t> indices(numBlocks);
    uint32_t codebookSize = static_cast<uint32_t>(pixelFloatCentroids.size());

    // Convert all final centroids to Oklab space once.
    std::vector<OklabFloatBlock> oklabCentroids(codebookSize);
#pragma omp parallel for num_threads(params.numThreads)
    for (int64_t i = 0; i < codebookSize; ++i) {
        oklabCentroids[i] = RgbaFloatBlockToOklabBlock(pixelFloatCentroids[i], channels);
    }

    // Find best index for each block, converting to Oklab space as we go.
#pragma omp parallel for num_threads(params.numThreads)
    for (int64_t i = 0; i < numBlocks; ++i) {
        OklabFloatBlock oklabBlock = RgbaFloatBlockToOklabBlock(pixelFloatBlocks[i], channels);
        float minDist = std::numeric_limits<float>::max();
        uint32_t bestIdx = 0;
        for (uint32_t j = 0; j < codebookSize; ++j) {
            float dist = OklabFloatBlockDistanceSq_SIMD(oklabBlock, oklabCentroids[j]);
            if (dist < minDist) {
                minDist = dist;
                bestIdx = j;
            }
        }
        indices[i] = bestIdx;
    }
    return indices;
}