#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

// A block is 16 pixels (4x4) of 4 floats (L, a, b, alpha)
using OklabBlock = std::vector<float>;

namespace Oklab
{
    // --- Forward Transformation: Linear sRGB -> Oklab ---

    inline void srgb_to_lms(const float* rgb, float* lms) {
        lms[0] = 0.4122214708f * rgb[0] + 0.5363325363f * rgb[1] + 0.0514459929f * rgb[2];
        lms[1] = 0.2119034982f * rgb[0] + 0.6806995451f * rgb[1] + 0.1073969566f * rgb[2];
        lms[2] = 0.0883024619f * rgb[0] + 0.2817188376f * rgb[1] + 0.6299787005f * rgb[2]; // fixed
    }

    inline void lms_to_oklab(const float* lms, float* lab) {
        float l = cbrtf(lms[0]);
        float m = cbrtf(lms[1]);
        float s = cbrtf(lms[2]);

        lab[0] = 0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s;
        lab[1] = 1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s;
        lab[2] = 0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s;
    }

    // --- Inverse Transformation: Oklab -> Linear sRGB ---

    inline void oklab_to_lms(const float* lab, float* lms) {
        float l = lab[0] + 0.3963377774f * lab[1] + 0.2158037573f * lab[2];
        float m = lab[0] - 0.1055613458f * lab[1] - 0.0638541728f * lab[2];
        float s = lab[0] - 0.0894841775f * lab[1] - 1.2914855480f * lab[2];

        lms[0] = l * l * l;
        lms[1] = m * m * m;
        lms[2] = s * s * s;
    }

    inline void lms_to_srgb(const float* lms, float* rgb) {
        rgb[0] = +4.0767416621f * lms[0] - 3.3077115913f * lms[1] + 0.2309699292f * lms[2];
        rgb[1] = -1.2681437731f * lms[0] + 2.6093323231f * lms[1] - 0.3411344290f * lms[2];
        rgb[2] = -0.0041119885f * lms[0] - 0.7034763098f * lms[1] + 1.7075952153f * lms[2];
    }

    // --- Block Conversions ---

    inline OklabBlock RgbaFloatBlockToOklabBlock(const std::vector<float>& rgbaBlock) {
        OklabBlock labBlock(16 * 4);
        float lms[3];
        for (size_t i = 0; i < 16; ++i) {
            srgb_to_lms(&rgbaBlock[i * 4], lms);
            lms_to_oklab(lms, &labBlock[i * 4]);
            labBlock[i * 4 + 3] = rgbaBlock[i * 4 + 3]; // Alpha passthrough
        }
        return labBlock;
    }

    inline std::vector<float> OklabBlockToRgbaFloatBlock(const OklabBlock& labBlock, bool clampForLDR = false) {
        std::vector<float> rgbaBlock(16 * 4);
        float lms[3];
        for (size_t i = 0; i < 16; ++i) {
            oklab_to_lms(&labBlock[i * 4], lms);
            lms_to_srgb(lms, &rgbaBlock[i * 4]);
            rgbaBlock[i * 4 + 3] = labBlock[i * 4 + 3];
            if (clampForLDR) {
                rgbaBlock[i * 4 + 0] = std::max(0.0f, rgbaBlock[i * 4 + 0]);
                rgbaBlock[i * 4 + 1] = std::max(0.0f, rgbaBlock[i * 4 + 1]);
                rgbaBlock[i * 4 + 2] = std::max(0.0f, rgbaBlock[i * 4 + 2]);
            }
        }
        return rgbaBlock;
    }

    // --- Perceptual Transforms for Lightness channel ---
    inline float L_to_perceptual(float L) {
        return log2f(1.0f + L);
    }

    inline float perceptual_to_L(float pL) {
        return exp2f(pL) - 1.0f;
    }
}