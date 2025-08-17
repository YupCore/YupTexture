#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <array>

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif

using OklabFloatBlock = std::vector<float>;

namespace Oklab {

    // -----------------------------
    // Numeric helpers / runtime CPU feature detection
    // -----------------------------

    static inline bool cpu_has_avx2() {
#if defined(__GNUC__) || defined(__clang__)
        unsigned int eax, ebx, ecx, edx;
        if (!__get_cpuid_max(0, nullptr)) return false;
        if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
        return (ebx & (1u << 5)) != 0; // AVX2 bit in leaf 7.ebx bit 5
#elif defined(_MSC_VER)
        int info[4];
        __cpuid(info, 0);
        int nIds = info[0];
        if (nIds >= 7) {
            __cpuidex(info, 7, 0);
            return (info[1] & (1 << 5)) != 0;
        }
        return false;
#else
        return false;
#endif
    }

    static inline bool hasFMA3() {
        uint32_t eax, ebx, ecx, edx;

#ifdef _WIN32
        int cpuInfo[4];
        __cpuid(cpuInfo, 1);
        ecx = cpuInfo[2];
#else
        __cpuid(1, eax, ebx, ecx, edx);
#endif

        // FMA3 is indicated by bit 12 in ECX
        return (ecx & (1 << 12)) != 0;
    }

    static inline bool cpu_has_FMA3_cached() {
        static bool checked = false;
        static bool has = false;
        if (!checked) {
            has = hasFMA3();
            checked = true;
        }
        return has;
    }

    static inline float clamp01(float x) { return std::min(1.0f, std::max(0.0f, x)); }

    // -----------------------------
    // RGB tables
    // -----------------------------
    // Table size: tradeoff between memory and accuracy. 4096 -> 4KB table of uint8_t.
    static constexpr int LINEAR_TO_SRGB_LUT_SIZE = 4096;

    // lazily-initialized table (uint8 entries 0..255)
    static inline const std::array<uint8_t, LINEAR_TO_SRGB_LUT_SIZE>& linear_to_srgb8_table() {
        static const std::array<uint8_t, LINEAR_TO_SRGB_LUT_SIZE> tbl = []() {
            std::array<uint8_t, LINEAR_TO_SRGB_LUT_SIZE> t{};
            const int N = LINEAR_TO_SRGB_LUT_SIZE;
            for (int i = 0; i < N; ++i) {
                float v = static_cast<float>(i) / static_cast<float>(N - 1); // in [0,1]
                float s;
                if (v <= 0.0031308f) {
                    s = v * 12.92f;
                }
                else {
                    s = 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
                }
                // clamp and quantize
                float q = std::min(1.0f, std::max(0.0f, s));
                t[i] = static_cast<uint8_t>(q * 255.0f + 0.5f);
            }
            return t;
            }();
        return tbl;
    }

    // lazy initialized once
    static inline const std::array<float, 256>& srgb8_to_linear_table() {
        static const std::array<float, 256> tbl = []() {
            std::array<float, 256> t{};
            for (int i = 0; i < 256; ++i) {
                // reuse your scalar helper
                float val = static_cast<float>(i) / 255.0f;
                if (val <= 0.04045f) t[i] = val / 12.92f;
                else t[i] = std::pow((val + 0.055f) / 1.055f, 2.4f);
            }
            return t;
            }();
        return tbl;
    }

    // -----------------------------
    // Matrices / constants (from Björn Ottosson reference)
    // -----------------------------

    static constexpr float SRGB_TO_LMS[9] = {
        0.4122214708f, 0.5363325363f, 0.0514459929f,
        0.2119034982f, 0.6806995451f, 0.1073969566f,
        0.0883024619f, 0.2817186075f, 0.6299787005f
    };

    static constexpr float LMS_TO_SRGB[9] = {
         4.0767416621f, -3.3077115913f,  0.2309699292f,
        -1.2681437731f,  2.6093323231f, -0.3411344290f,
        -0.0041119885f, -0.7034763098f,  1.7075952153f
    };

    static constexpr float LMS_TO_OKLAB[9] = {
        0.2104542553f,  0.7936177850f, -0.0040720468f,
        1.9779984951f, -2.4285922050f,  0.4505937099f,
        0.0259040371f,  0.7827717662f, -0.8086757660f
    };

    static constexpr float OKLAB_TO_LMS[9] = {
        1.0f,  0.3963377774f,  0.2158037573f,
        1.0f, -0.1055613458f, -0.0638541728f,
        1.0f, -0.0894841775f, -1.2914855480f
    };

    // ACES RRT+ODT rational fit coefficients (hoisted)
    static inline __m256 v_aces_a;
    static inline __m256 v_aces_b;
    static inline __m256 v_aces_c;
    static inline __m256 v_aces_d;
    static inline __m256 v_aces_e;

    // Hoisted matrix entries as __m256
    static inline __m256 v_s2l_0, v_s2l_1, v_s2l_2, v_s2l_3, v_s2l_4, v_s2l_5, v_s2l_6, v_s2l_7, v_s2l_8;
    static inline __m256 v_l2o_0, v_l2o_1, v_l2o_2, v_l2o_3, v_l2o_4, v_l2o_5, v_l2o_6, v_l2o_7, v_l2o_8;
    static inline __m256 v_o2l_0, v_o2l_1, v_o2l_2, v_o2l_3, v_o2l_4, v_o2l_5, v_o2l_6, v_o2l_7, v_o2l_8;
    static inline __m256 v_l2s_0, v_l2s_1, v_l2s_2, v_l2s_3, v_l2s_4, v_l2s_5, v_l2s_6, v_l2s_7, v_l2s_8;

    static inline void initAVX()
    {
        // ACES
        v_aces_a = _mm256_set1_ps(0.0245786f);
        v_aces_b = _mm256_set1_ps(0.000090537f);
        v_aces_c = _mm256_set1_ps(0.983729f);
        v_aces_d = _mm256_set1_ps(0.4329510f);
        v_aces_e = _mm256_set1_ps(0.238081f);

        // SRGB_TO_LMS
        v_s2l_0 = _mm256_set1_ps(SRGB_TO_LMS[0]);
        v_s2l_1 = _mm256_set1_ps(SRGB_TO_LMS[1]);
        v_s2l_2 = _mm256_set1_ps(SRGB_TO_LMS[2]);
        v_s2l_3 = _mm256_set1_ps(SRGB_TO_LMS[3]);
        v_s2l_4 = _mm256_set1_ps(SRGB_TO_LMS[4]);
        v_s2l_5 = _mm256_set1_ps(SRGB_TO_LMS[5]);
        v_s2l_6 = _mm256_set1_ps(SRGB_TO_LMS[6]);
        v_s2l_7 = _mm256_set1_ps(SRGB_TO_LMS[7]);
        v_s2l_8 = _mm256_set1_ps(SRGB_TO_LMS[8]);

        // LMS_TO_OKLAB
        v_l2o_0 = _mm256_set1_ps(LMS_TO_OKLAB[0]);
        v_l2o_1 = _mm256_set1_ps(LMS_TO_OKLAB[1]);
        v_l2o_2 = _mm256_set1_ps(LMS_TO_OKLAB[2]);
        v_l2o_3 = _mm256_set1_ps(LMS_TO_OKLAB[3]);
        v_l2o_4 = _mm256_set1_ps(LMS_TO_OKLAB[4]);
        v_l2o_5 = _mm256_set1_ps(LMS_TO_OKLAB[5]);
        v_l2o_6 = _mm256_set1_ps(LMS_TO_OKLAB[6]);
        v_l2o_7 = _mm256_set1_ps(LMS_TO_OKLAB[7]);
        v_l2o_8 = _mm256_set1_ps(LMS_TO_OKLAB[8]);

        // OKLAB_TO_LMS
        v_o2l_0 = _mm256_set1_ps(OKLAB_TO_LMS[0]);
        v_o2l_1 = _mm256_set1_ps(OKLAB_TO_LMS[1]);
        v_o2l_2 = _mm256_set1_ps(OKLAB_TO_LMS[2]);
        v_o2l_3 = _mm256_set1_ps(OKLAB_TO_LMS[3]);
        v_o2l_4 = _mm256_set1_ps(OKLAB_TO_LMS[4]);
        v_o2l_5 = _mm256_set1_ps(OKLAB_TO_LMS[5]);
        v_o2l_6 = _mm256_set1_ps(OKLAB_TO_LMS[6]);
        v_o2l_7 = _mm256_set1_ps(OKLAB_TO_LMS[7]);
        v_o2l_8 = _mm256_set1_ps(OKLAB_TO_LMS[8]);

        // LMS_TO_SRGB
        v_l2s_0 = _mm256_set1_ps(LMS_TO_SRGB[0]);
        v_l2s_1 = _mm256_set1_ps(LMS_TO_SRGB[1]);
        v_l2s_2 = _mm256_set1_ps(LMS_TO_SRGB[2]);
        v_l2s_3 = _mm256_set1_ps(LMS_TO_SRGB[3]);
        v_l2s_4 = _mm256_set1_ps(LMS_TO_SRGB[4]);
        v_l2s_5 = _mm256_set1_ps(LMS_TO_SRGB[5]);
        v_l2s_6 = _mm256_set1_ps(LMS_TO_SRGB[6]);
        v_l2s_7 = _mm256_set1_ps(LMS_TO_SRGB[7]);
        v_l2s_8 = _mm256_set1_ps(LMS_TO_SRGB[8]);
    }

    static inline bool cpu_has_avx2_cached() {
        static bool checked = false;
        static bool has = false;
        if (!checked) {
            has = cpu_has_avx2();
            checked = true;
            if (has)
                initAVX(); // Only initialize AVX if we detected it, prevents crashes
        }
        return has;
    }

    // -----------------------------
    // Fast reciprocal refine + selectable high-precision fallback
    // Define OKLAB_FORCE_HIGH_PRECISION to use exact division over rcp+refine
    // -----------------------------

    static inline __m256 rcp_refine(__m256 x) {
        // initial approx
        __m256 r = _mm256_rcp_ps(x);

        // Newton iteration 1: r = r * (2 - x*r)
        r = _mm256_mul_ps(r, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(x, r)));

        // Newton iteration 2: same formula — small cost, much better accuracy
        r = _mm256_mul_ps(r, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(x, r)));

        return r;
    }

    static inline __m256 inv_ps(__m256 x) {
#ifdef OKLAB_FORCE_HIGH_PRECISION
        return _mm256_div_ps(_mm256_set1_ps(1.0f), x);
#else
        return rcp_refine(x);
#endif
    }

    // -----------------------------
    // ACES RRT+ODT rational fit (vectorized) with hoisted constants
    // uses inv_ps for faster reciprocal (can be overridden for precision)
    // -----------------------------

    static inline __m256 aces_rttod_fit_ps(__m256 x) {
        __m256 xx = _mm256_mul_ps(x, x);
        __m256 num = _mm256_add_ps(_mm256_add_ps(xx, _mm256_mul_ps(v_aces_a, x)), v_aces_b);
        __m256 den = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_aces_c, xx), _mm256_mul_ps(v_aces_d, x)), v_aces_e);
        __m256 inv_den = inv_ps(den);
        return _mm256_mul_ps(num, inv_den);
    }

    static inline __m256 aces_rttod_fit_derivative_ps(__m256 x) {
        __m256 xx = _mm256_mul_ps(x, x);
        __m256 num = _mm256_add_ps(_mm256_add_ps(xx, _mm256_mul_ps(v_aces_a, x)), v_aces_b); // numerator
        __m256 den = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_aces_c, xx), _mm256_mul_ps(v_aces_d, x)), v_aces_e); // denominator
        __m256 num_p = _mm256_add_ps(_mm256_add_ps(x, x), v_aces_a);
        __m256 den_p = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(v_aces_c, x)), v_aces_d);
        __m256 top = _mm256_sub_ps(_mm256_mul_ps(num_p, den), _mm256_mul_ps(num, den_p));
        __m256 den2 = _mm256_mul_ps(den, den);
        __m256 eps = _mm256_set1_ps(1e-12f);
        den2 = _mm256_add_ps(den2, eps);
        __m256 inv_den2 = inv_ps(den2);
        return _mm256_mul_ps(top, inv_den2);
    }

    // -----------------------------
    // Improved vectorized inverse using Newton-Raphson with better initial guess,
    // derivative clamping and faster reciprocal using rcp+refine
    // -----------------------------

    static inline __m256 aces_rttod_inverse_ps(__m256 y) {
        const __m256 zero = _mm256_set1_ps(0.0f);
        const __m256 eps_small = _mm256_set1_ps(1e-7f); // slightly larger to avoid stagnation

        // target = max(0, y)
        __m256 target = _mm256_max_ps(y, zero);

        // Hoist ACES scalars as floats (for building quadratic)
        const float a = 0.0245786f;
        const float b = 0.000090537f;
        const float c = 0.983729f;
        const float d = 0.4329510f;
        const float e = 0.238081f;

        // Build quadratic coefficients A*(x^2) + B*x + C = 0 where A = (1 - c*y), B = (a - d*y), C = (b - e*y)
        __m256 A = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(_mm256_set1_ps(c), target));
        __m256 B = _mm256_sub_ps(_mm256_set1_ps(a), _mm256_mul_ps(_mm256_set1_ps(d), target));
        __m256 C = _mm256_sub_ps(_mm256_set1_ps(b), _mm256_mul_ps(_mm256_set1_ps(e), target));

        // discriminant D = B^2 - 4*A*C
        __m256 B2 = _mm256_mul_ps(B, B);
        __m256 AC4 = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(4.0f), A), C);
        __m256 D = _mm256_sub_ps(B2, AC4);

        // clamp D >= 0
        __m256 Dpos = _mm256_max_ps(D, _mm256_set1_ps(0.0f));
        __m256 sqrtD = _mm256_sqrt_ps(Dpos);

        // x_quadratic = (-B + sqrtD) / (2*A) as a candidate. When |A| is small, fallback to -C/B.
        __m256 denom = _mm256_mul_ps(_mm256_set1_ps(2.0f), A);

        // smallA mask
        __m256 absA = _mm256_and_ps(A, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
        __m256 mask_smallA = _mm256_cmp_ps(absA, _mm256_set1_ps(1e-6f), _CMP_LT_OQ);

        // compute candidate via quadratic safely
        __m256 negB = _mm256_sub_ps(_mm256_set1_ps(0.0f), B);
        __m256 safe_denom = _mm256_blendv_ps(denom, _mm256_set1_ps(1.0f), mask_smallA);
        __m256 inv_denom = inv_ps(safe_denom);
        __m256 x_quad = _mm256_mul_ps(_mm256_add_ps(negB, sqrtD), inv_denom);

        // fallback x_lin = -C / B, handle small B as well
        __m256 mask_smallB = _mm256_cmp_ps(_mm256_and_ps(B, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF))), _mm256_set1_ps(1e-8f), _CMP_LT_OQ);
        __m256 invB = inv_ps(B);
        invB = _mm256_blendv_ps(invB, _mm256_set1_ps(0.0f), mask_smallB);
        __m256 x_lin = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(0.0f), C), invB);

        // choose initial guess: if A is small -> x_lin else x_quad. If B small -> use target as fallback.
        __m256 init_x = _mm256_blendv_ps(x_quad, x_lin, mask_smallA);
        init_x = _mm256_blendv_ps(init_x, target, mask_smallB);

        // Ensure initial guess non-negative
        __m256 x = _mm256_max_ps(init_x, zero);

        // Pre-hoisted constants
        const __m256 two_c = _mm256_mul_ps(_mm256_set1_ps(2.0f), v_aces_c);
        const __m256 den_eps = _mm256_set1_ps(1e-12f);
        const __m256 one = _mm256_set1_ps(1.0f);

        // Newton iterations with derivative clamping and reciprocal via rcp+refine
        for (int iter = 0; iter < 4; ++iter) {
            __m256 xx = _mm256_mul_ps(x, x);
            __m256 num = _mm256_add_ps(_mm256_add_ps(xx, _mm256_mul_ps(v_aces_a, x)), v_aces_b);
            __m256 den = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_aces_c, xx), _mm256_mul_ps(v_aces_d, x)), v_aces_e);

            // f = num/den - target  => use inv_ps(den)
            __m256 inv_den = inv_ps(den);
            __m256 f = _mm256_sub_ps(_mm256_mul_ps(num, inv_den), target);

            // derivative fp
            __m256 num_p = _mm256_add_ps(_mm256_add_ps(x, x), v_aces_a);
            __m256 den_p = _mm256_add_ps(_mm256_mul_ps(two_c, x), v_aces_d);
            __m256 den2 = _mm256_mul_ps(den, den);
            __m256 top = _mm256_sub_ps(_mm256_mul_ps(num_p, den), _mm256_mul_ps(num, den_p));
            __m256 fp = _mm256_mul_ps(top, inv_ps(_mm256_add_ps(den2, den_eps)));

            // abs(fp) and mask
            __m256 abs_fp = _mm256_and_ps(fp, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
            __m256 mask_small = _mm256_cmp_ps(abs_fp, eps_small, _CMP_LT_OQ); // true if |fp| < eps_small

            // clamp fp to sign(fp)*eps_small where it's too small
            __m256 mask_neg = _mm256_cmp_ps(fp, zero, _CMP_LT_OQ);
            __m256 eps_pos = eps_small;
            __m256 eps_neg = _mm256_sub_ps(_mm256_set1_ps(0.0f), eps_small);
            __m256 eps_signed = _mm256_blendv_ps(eps_pos, eps_neg, mask_neg); // eps with same sign as fp
            __m256 fp_clamped = _mm256_blendv_ps(fp, eps_signed, mask_small);

            // compute inv_fp via rcp + 1 Newton refine
            __m256 inv_fp = rcp_refine(fp_clamped);

            // dx = f * inv_fp
            __m256 dx = _mm256_mul_ps(f, inv_fp);

            // x -= dx
            x = _mm256_sub_ps(x, dx);
            x = _mm256_max_ps(x, zero);
        }

        return x;
    }

    // -----------------------------
    // SIMD cube-root approx (use corrected version from earlier patch), with inv_ps usage
    // -----------------------------

    static inline __m256 avx2_cbrt_approx_ps(__m256 x) {
        const __m256i sign_mask_i = _mm256_set1_epi32(0x80000000);
        const __m256i exp_mask_i = _mm256_set1_epi32(0x7F800000);
        const __m256i mant_mask_i = _mm256_set1_epi32(0x007FFFFF);
        const __m256i exp_bias_i = _mm256_set1_epi32(127);

        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 third = _mm256_set1_ps(1.0f / 3.0f);
        const __m256 zero = _mm256_setzero_ps();

        __m256i xi = _mm256_castps_si256(x);

        // Extract sign, abs
        __m256i signbits_i = _mm256_and_si256(xi, sign_mask_i);
        __m256 abs_x = _mm256_andnot_ps(_mm256_castsi256_ps(sign_mask_i), x);

        // zero mask
        __m256 mask_zero = _mm256_cmp_ps(abs_x, zero, _CMP_EQ_OQ);

        // Handle INF/NAN
        __m256i biased_exp_bits = _mm256_and_si256(xi, exp_mask_i);
        __m256i is_inf_nan_i = _mm256_cmpeq_epi32(biased_exp_bits, exp_mask_i);
        __m256 mask_inf_nan = _mm256_castsi256_ps(is_inf_nan_i);

        // Decompose exponent and mantissa (normalized float)
        __m256i e_i = _mm256_sub_epi32(_mm256_srli_epi32(biased_exp_bits, 23), exp_bias_i); // exponent (biased->unbiased)
        __m256 one_bits = _mm256_castsi256_ps(_mm256_slli_epi32(exp_bias_i, 23));
        __m256 m = _mm256_or_ps(_mm256_and_ps(abs_x, _mm256_castsi256_ps(mant_mask_i)), one_bits);

        // compute k = floor(e/3)
        __m256 e_f = _mm256_cvtepi32_ps(e_i);
        __m256 k_f = _mm256_floor_ps(_mm256_mul_ps(e_f, third));
        __m256i k_i = _mm256_cvtps_epi32(k_f);

        // remainder r = e - 3*k
        __m256i three = _mm256_set1_epi32(3);
        __m256i r_i = _mm256_sub_epi32(e_i, _mm256_mullo_epi32(k_i, three));

        // scale for remainder (1, 2 or 3)
        const __m256 scale_r1 = _mm256_set1_ps(1.2599210498948732f); // 2^(1/3)
        const __m256 scale_r2 = _mm256_set1_ps(1.587401051f); // 2^(2/3)
        __m256 mask_r1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(r_i, _mm256_set1_epi32(1)));
        __m256 mask_r2 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(r_i, _mm256_set1_epi32(2)));
        __m256 scale_r = _mm256_blendv_ps(one, scale_r1, mask_r1);
        scale_r = _mm256_blendv_ps(scale_r, scale_r2, mask_r2);

        // build scaled exponent part
        __m256i new_exp_biased = _mm256_add_epi32(k_i, exp_bias_i);
        __m256 scale_k = _mm256_castsi256_ps(_mm256_slli_epi32(new_exp_biased, 23));

        // initial approximation for mantissa using cheap polynomial (same idea you used)
        const __m256 C0 = _mm256_set1_ps(0.46946158f);
        const __m256 C1 = _mm256_set1_ps(0.59449863f);
        const __m256 C2 = _mm256_set1_ps(-0.06404994f);

        __m256 y = _mm256_add_ps(C0, _mm256_mul_ps(C1, m));
        y = _mm256_add_ps(y, _mm256_mul_ps(C2, _mm256_mul_ps(m, m)));

        // normalize initial mantissa for cube root and combine scaling factors
        y = _mm256_mul_ps(y, scale_r);
        y = _mm256_mul_ps(y, scale_k);

        // Newton-Raphson refinement for cube root:
        // y_{n+1} = (2*y_n + x / (y_n*y_n)) / 3
        // Do 2 iterations (sufficient for float accuracy)
        for (int iter = 0; iter < 2; ++iter) {
            __m256 y2 = _mm256_mul_ps(y, y);
            __m256 inv_y2 = inv_ps(y2); // uses refined reciprocal (improved)
            __m256 x_over_y2 = _mm256_mul_ps(abs_x, inv_y2);
            __m256 two_y = _mm256_add_ps(y, y);
            y = _mm256_mul_ps(third, _mm256_add_ps(two_y, x_over_y2));
        }

        // restore sign for negative inputs (though cube root of negative is negative)
        __m256 result = _mm256_or_ps(y, _mm256_castsi256_ps(signbits_i));

        // Handle INF/NAN and zeros properly
        result = _mm256_blendv_ps(result, x, mask_inf_nan); // keep NaN/Inf as-is
        result = _mm256_blendv_ps(result, zero, mask_zero); // cbrt(0) = 0

        return result;
    }

    // -----------------------------
    // Scalar helpers (fallbacks)
    // -----------------------------

    static inline float aces_rttod_fit_scalar(float x) {
        const float a = 0.0245786f;
        const float b = 0.000090537f;
        const float c = 0.983729f;
        const float d = 0.4329510f;
        const float e = 0.238081f;
        float num = x * x + a * x + b;
        float den = c * x * x + d * x + e;
        return num / den;
    }

    static inline float aces_rttod_inverse_scalar(float y) {
        float target = std::max(0.0f, y);
        // improved initial guess via quadratic solve (scalar version)
        const float a = 0.0245786f;
        const float b = 0.000090537f;
        const float c = 0.983729f;
        const float d = 0.4329510f;
        const float e = 0.238081f;
        float A = 1.0f - c * target;
        float B = a - d * target;
        float C = b - e * target;
        float x = target;
        // quadratic discriminant
        float D = B * B - 4.0f * A * C;
        if (D >= 0.0f && std::fabs(A) > 1e-6f) {
            float sqrtD = std::sqrt(D);
            x = (-B + sqrtD) / (2.0f * A);
        }
        else if (std::fabs(B) > 1e-8f) {
            x = -C / B;
        }
        else {
            x = target; // fallback
        }
        x = std::max(0.0f, x);

        // Newton iterations with derivative clamping
        for (int i = 0; i < 4; ++i) {
            float xx = x * x;
            float num = xx + a * x + b;
            float den = c * xx + d * x + e;
            float f = num / den - target;
            float num_p = 2.0f * x + a;
            float den_p = 2.0f * c * x + d;
            float top = num_p * den - num * den_p;
            float den2 = den * den + 1e-12f;
            float fp = top / den2;
            float eps_small = 1e-7f;
            if (std::fabs(fp) < eps_small) fp = (fp < 0.0f) ? -eps_small : eps_small;
            float inv_fp = 1.0f / fp;
            float dx = f * inv_fp;
            x -= dx;
            if (x < 0.0f) x = 0.0f;
        }
        return x;
    }

    // -----------------------------
    // Per-pixel scalar conversions
    // -----------------------------

    static inline void mat3_mul_vec3(const float M[9], const float v[3], float out[3]) {
        out[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2];
        out[1] = M[3] * v[0] + M[4] * v[1] + M[5] * v[2];
        out[2] = M[6] * v[0] + M[7] * v[1] + M[8] * v[2];
    }

    static inline void srgb_to_lms_scalar(const float rgb[3], float lms[3]) { mat3_mul_vec3(SRGB_TO_LMS, rgb, lms); }
    static inline void lms_to_oklab_scalar(const float lms[3], float lab[3]) {
        float L = std::cbrt(lms[0]);
        float M = std::cbrt(lms[1]);
        float S = std::cbrt(lms[2]);
        float t[3] = { L, M, S };
        mat3_mul_vec3(LMS_TO_OKLAB, t, lab);
    }
    static inline void oklab_to_lms_scalar(const float lab[3], float lms[3]) {
        float t[3]; mat3_mul_vec3(OKLAB_TO_LMS, lab, t);
        lms[0] = t[0] * t[0] * t[0]; lms[1] = t[1] * t[1] * t[1]; lms[2] = t[2] * t[2] * t[2];
    }
    static inline void lms_to_srgb_scalar(const float lms[3], float rgb[3]) { mat3_mul_vec3(LMS_TO_SRGB, lms, rgb); }

    // -----------------------------
    // AVX2 block conversions (unchanged structure but using hoisted constants & fma where possible)
    // -----------------------------

    inline OklabFloatBlock FloatBlockToOklabFloatBlock_AVX2(const std::vector<float>& dataBlock, bool rgba, bool applyACESTonemap = false) {
        OklabFloatBlock labBlock;
        if (rgba) {
            if (dataBlock.size() < 16 * 4) return OklabFloatBlock();
        }
        else {
            if (dataBlock.size() < 16 * 3) return OklabFloatBlock();
        }
        labBlock.resize(rgba ? 16 * 4 : 16 * 3);

        for (int i = 0; i < 16; i += 8) {
            alignas(32) float r_arr[8], g_arr[8], b_arr[8];
            alignas(32) float a_arr_stack[8];
            for (int j = 0; j < 8; ++j) {
                int in_idx = rgba ? (i + j) * 4 : (i + j) * 3;
                r_arr[j] = dataBlock[in_idx + 0];
                g_arr[j] = dataBlock[in_idx + 1];
                b_arr[j] = dataBlock[in_idx + 2];
                if (rgba) a_arr_stack[j] = dataBlock[in_idx + 3];
            }

            __m256 vr = _mm256_load_ps(r_arr);
            __m256 vg = _mm256_load_ps(g_arr);
            __m256 vb = _mm256_load_ps(b_arr);

            if (applyACESTonemap) {
                vr = aces_rttod_fit_ps(vr);
                vg = aces_rttod_fit_ps(vg);
                vb = aces_rttod_fit_ps(vb);
            }

            // Matrix multiply sRGB->LMS using hoisted consts and FMA when available
            __m256 l_part;
            __m256 m_part;
            __m256 s_part;
            if (cpu_has_FMA3_cached())
            {
                l_part = _mm256_fmadd_ps(v_s2l_0, vr, _mm256_fmadd_ps(v_s2l_1, vg, _mm256_mul_ps(v_s2l_2, vb)));
                m_part = _mm256_fmadd_ps(v_s2l_3, vr, _mm256_fmadd_ps(v_s2l_4, vg, _mm256_mul_ps(v_s2l_5, vb)));
                s_part = _mm256_fmadd_ps(v_s2l_6, vr, _mm256_fmadd_ps(v_s2l_7, vg, _mm256_mul_ps(v_s2l_8, vb)));
            }
            else
            {
                l_part = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_s2l_0, vr), _mm256_mul_ps(v_s2l_1, vg)), _mm256_mul_ps(v_s2l_2, vb));
                m_part = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_s2l_3, vr), _mm256_mul_ps(v_s2l_4, vg)), _mm256_mul_ps(v_s2l_5, vb));
                s_part = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_s2l_6, vr), _mm256_mul_ps(v_s2l_7, vg)), _mm256_mul_ps(v_s2l_8, vb));
            }

            __m256 Lc = avx2_cbrt_approx_ps(l_part);
            __m256 Mc = avx2_cbrt_approx_ps(m_part);
            __m256 Sc = avx2_cbrt_approx_ps(s_part);

            __m256 lab0;
            __m256 lab1;
            __m256 lab2;
            if (cpu_has_FMA3_cached())
            {
                lab0 = _mm256_fmadd_ps(v_l2o_0, Lc, _mm256_fmadd_ps(v_l2o_1, Mc, _mm256_mul_ps(v_l2o_2, Sc)));
                lab1 = _mm256_fmadd_ps(v_l2o_3, Lc, _mm256_fmadd_ps(v_l2o_4, Mc, _mm256_mul_ps(v_l2o_5, Sc)));
                lab2 = _mm256_fmadd_ps(v_l2o_6, Lc, _mm256_fmadd_ps(v_l2o_7, Mc, _mm256_mul_ps(v_l2o_8, Sc)));
            }
            else
            {
                lab0 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_l2o_0, Lc), _mm256_mul_ps(v_l2o_1, Mc)), _mm256_mul_ps(v_l2o_2, Sc));
                lab1 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_l2o_3, Lc), _mm256_mul_ps(v_l2o_4, Mc)), _mm256_mul_ps(v_l2o_5, Sc));
                lab2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_l2o_6, Lc), _mm256_mul_ps(v_l2o_7, Mc)), _mm256_mul_ps(v_l2o_8, Sc));
            }

            alignas(32) float out0[8], out1[8], out2[8];
            _mm256_store_ps(out0, lab0);
            _mm256_store_ps(out1, lab1);
            _mm256_store_ps(out2, lab2);

            for (int j = 0; j < 8; ++j) {
                const int num_channels = rgba ? 4 : 3;
                int outIdx = (i + j) * num_channels;
                labBlock[outIdx + 0] = out0[j];
                labBlock[outIdx + 1] = out1[j];
                labBlock[outIdx + 2] = out2[j];
                if (rgba) labBlock[outIdx + 3] = a_arr_stack[j];
            }
        }
        return labBlock;
    }

    inline std::vector<float> OklabFloatBlockToFloatBlock_AVX2(const OklabFloatBlock& labBlock, bool rgba, bool clampForLDR = false, bool applyACESInverse = false) {
        std::vector<float> floatBlock;
        if (rgba) {
            if (labBlock.size() < 16 * 4) return std::vector<float>();
        }
        else {
            if (labBlock.size() < 16 * 3) return std::vector<float>();
        }
        floatBlock.resize(rgba ? 16 * 4 : 16 * 3);

        for (int i = 0; i < 16; i += 8) {
            alignas(32) float l_arr[8], a_arr[8], b_arr[8];
            alignas(32) float alpha_arr_stack[8];
            for (int j = 0; j < 8; ++j) {
                const int num_channels = rgba ? 4 : 3;
                int idx = (i + j) * num_channels;
                l_arr[j] = labBlock[idx + 0];
                a_arr[j] = labBlock[idx + 1];
                b_arr[j] = labBlock[idx + 2];
                if (rgba) alpha_arr_stack[j] = labBlock[idx + 3];
            }

            __m256 vL = _mm256_load_ps(l_arr);
            __m256 vA = _mm256_load_ps(a_arr);
            __m256 vB = _mm256_load_ps(b_arr);

            __m256 t0;
            __m256 t1;
            __m256 t2;

            if (cpu_has_FMA3_cached())
            {
                t0 = _mm256_fmadd_ps(v_o2l_0, vL, _mm256_fmadd_ps(v_o2l_1, vA, _mm256_mul_ps(v_o2l_2, vB)));
                t1 = _mm256_fmadd_ps(v_o2l_3, vL, _mm256_fmadd_ps(v_o2l_4, vA, _mm256_mul_ps(v_o2l_5, vB)));
                t2 = _mm256_fmadd_ps(v_o2l_6, vL, _mm256_fmadd_ps(v_o2l_7, vA, _mm256_mul_ps(v_o2l_8, vB)));
            }
            else
            {
                t0 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_o2l_0, vL), _mm256_mul_ps(v_o2l_1, vA)), _mm256_mul_ps(v_o2l_2, vB));
                t1 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_o2l_3, vL), _mm256_mul_ps(v_o2l_4, vA)), _mm256_mul_ps(v_o2l_5, vB));
                t2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_o2l_6, vL), _mm256_mul_ps(v_o2l_7, vA)), _mm256_mul_ps(v_o2l_8, vB));
            }

            __m256 lms0 = _mm256_mul_ps(t0, _mm256_mul_ps(t0, t0));
            __m256 lms1 = _mm256_mul_ps(t1, _mm256_mul_ps(t1, t1));
            __m256 lms2 = _mm256_mul_ps(t2, _mm256_mul_ps(t2, t2));

            __m256 r;
            __m256 g;
            __m256 b;

            if (cpu_has_FMA3_cached())
            {
                r = _mm256_fmadd_ps(v_l2s_0, lms0, _mm256_fmadd_ps(v_l2s_1, lms1, _mm256_mul_ps(v_l2s_2, lms2)));
                g = _mm256_fmadd_ps(v_l2s_3, lms0, _mm256_fmadd_ps(v_l2s_4, lms1, _mm256_mul_ps(v_l2s_5, lms2)));
                b = _mm256_fmadd_ps(v_l2s_6, lms0, _mm256_fmadd_ps(v_l2s_7, lms1, _mm256_mul_ps(v_l2s_8, lms2)));
            }
            else
            {
                r = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_l2s_0, lms0), _mm256_mul_ps(v_l2s_1, lms1)), _mm256_mul_ps(v_l2s_2, lms2));
                g = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_l2s_3, lms0), _mm256_mul_ps(v_l2s_4, lms1)), _mm256_mul_ps(v_l2s_5, lms2));
                b = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v_l2s_6, lms0), _mm256_mul_ps(v_l2s_7, lms1)), _mm256_mul_ps(v_l2s_8, lms2));
            }

            if (applyACESInverse) {
                r = aces_rttod_inverse_ps(r);
                g = aces_rttod_inverse_ps(g);
                b = aces_rttod_inverse_ps(b);
            }

            if (clampForLDR) {
                const __m256 zerov = _mm256_set1_ps(0.0f);
                const __m256 onev = _mm256_set1_ps(1.0f);
                r = _mm256_min_ps(onev, _mm256_max_ps(zerov, r));
                g = _mm256_min_ps(onev, _mm256_max_ps(zerov, g));
                b = _mm256_min_ps(onev, _mm256_max_ps(zerov, b));
            }

            alignas(32) float r_out[8], g_out[8], b_out[8];
            _mm256_store_ps(r_out, r);
            _mm256_store_ps(g_out, g);
            _mm256_store_ps(b_out, b);

            for (int j = 0; j < 8; ++j) {
                if (rgba) {
                    int outIdx = (i + j) * 4;
                    floatBlock[outIdx + 0] = r_out[j];
                    floatBlock[outIdx + 1] = g_out[j];
                    floatBlock[outIdx + 2] = b_out[j];
                    floatBlock[outIdx + 3] = alpha_arr_stack[j];
                }
                else {
                    int outIdx = (i + j) * 3;
                    floatBlock[outIdx + 0] = r_out[j];
                    floatBlock[outIdx + 1] = g_out[j];
                    floatBlock[outIdx + 2] = b_out[j];
                }
            }
        }
        return floatBlock;
    }

    // -----------------------------
    // Simple scalar block converters (runtime fallback)
    // -----------------------------

    inline OklabFloatBlock FloatBlockToOklabFloatBlock_Scalar(const std::vector<float>& dataBlock, bool rgba, bool applyACESTonemap = false) {
        OklabFloatBlock labBlock;
        labBlock.resize(rgba ? 16 * 4 : 16 * 3);
        for (int i = 0; i < 16; ++i) {
            int in_idx = rgba ? i * 4 : i * 3;
            float rgb[3] = { dataBlock[in_idx + 0], dataBlock[in_idx + 1], dataBlock[in_idx + 2] };
            if (applyACESTonemap) {
                rgb[0] = aces_rttod_fit_scalar(rgb[0]);
                rgb[1] = aces_rttod_fit_scalar(rgb[1]);
                rgb[2] = aces_rttod_fit_scalar(rgb[2]);
            }
            float lms[3]; srgb_to_lms_scalar(rgb, lms);
            float lab[3]; lms_to_oklab_scalar(lms, lab);
            int out_idx = rgba ? i * 4 : i * 3;
            labBlock[out_idx + 0] = lab[0]; labBlock[out_idx + 1] = lab[1]; labBlock[out_idx + 2] = lab[2];
            if (rgba) labBlock[out_idx + 3] = dataBlock[in_idx + 3];
        }
        return labBlock;
    }

    inline std::vector<float> OklabFloatBlockToFloatBlock_Scalar(const OklabFloatBlock& labBlock, bool rgba, bool clampForLDR = false, bool applyACESInverse = false) {
        std::vector<float> floatBlock;
        floatBlock.resize(rgba ? 16 * 4 : 16 * 3);
        for (int i = 0; i < 16; ++i) {
            int idx = rgba ? i * 4 : i * 3;
            float lab[3] = { labBlock[idx + 0], labBlock[idx + 1], labBlock[idx + 2] };
            float lms[3]; oklab_to_lms_scalar(lab, lms);
            float rgb[3]; lms_to_srgb_scalar(lms, rgb);
            if (applyACESInverse) {
                rgb[0] = aces_rttod_inverse_scalar(rgb[0]);
                rgb[1] = aces_rttod_inverse_scalar(rgb[1]);
                rgb[2] = aces_rttod_inverse_scalar(rgb[2]);
            }
            if (clampForLDR) { rgb[0] = clamp01(rgb[0]); rgb[1] = clamp01(rgb[1]); rgb[2] = clamp01(rgb[2]); }
            if (rgba) {
                floatBlock[idx + 0] = rgb[0]; floatBlock[idx + 1] = rgb[1]; floatBlock[idx + 2] = rgb[2]; floatBlock[idx + 3] = labBlock[idx + 3];
            }
            else {
                floatBlock[idx + 0] = rgb[0]; floatBlock[idx + 1] = rgb[1]; floatBlock[idx + 2] = rgb[2];
            }
        }
        return floatBlock;
    }

    // -----------------------------
    // Public compatibility wrappers: runtime pick AVX2 or scalar
    // -----------------------------

    inline OklabFloatBlock FloatBlockToOklabFloatBlock(const std::vector<float>& block, bool rgba, bool applyACESTonemap = false) {
        if (cpu_has_avx2_cached()) return FloatBlockToOklabFloatBlock_AVX2(block, rgba, applyACESTonemap);
        return FloatBlockToOklabFloatBlock_Scalar(block, rgba, applyACESTonemap);
    }

    inline std::vector<float> OklabFloatBlockToFloatBlock(const OklabFloatBlock& labBlock, bool rgba, bool clampForLDR = false, bool applyACESInverse = false) {
        if (cpu_has_avx2_cached()) return OklabFloatBlockToFloatBlock_AVX2(labBlock, rgba, clampForLDR, applyACESInverse);
        return OklabFloatBlockToFloatBlock_Scalar(labBlock, rgba, clampForLDR, applyACESInverse);
    }

    inline OklabFloatBlock RgbaFloatBlockToOklabFloatBlock(const std::vector<float>& rgbaBlock, bool applyACESTonemap = false) {
        if (rgbaBlock.size() != 16 * 4) return OklabFloatBlock();
        return FloatBlockToOklabFloatBlock(rgbaBlock, true, applyACESTonemap);
    }

    inline std::vector<float> OklabFloatBlockToRgbaFloatBlock(const OklabFloatBlock& labBlock, bool clampForLDR = false, bool applyACESInverse = false) {
        if (labBlock.size() != 16 * 4) return std::vector<float>();
        return OklabFloatBlockToFloatBlock(labBlock, true, clampForLDR, applyACESInverse);
    }

    inline OklabFloatBlock RgbFloatBlockToOklabFloatBlock(const std::vector<float>& rgbBlock, bool applyACESTonemap = false) {
        if (rgbBlock.size() != 16 * 3) return OklabFloatBlock();
        return FloatBlockToOklabFloatBlock(rgbBlock, false, applyACESTonemap);
    }

    inline std::vector<float> OklabFloatBlockToRgbFloatBlock(const OklabFloatBlock& labBlock, bool clampForLDR = false, bool applyACESInverse = false) {
        if (labBlock.size() != 16 * 3) return std::vector<float>();
        return OklabFloatBlockToFloatBlock(labBlock, false, clampForLDR, applyACESInverse);
    }

    // =========================================================================
    // NEW WRAPPERS FOR 8-BIT SRGB DATA
    // =========================================================================

    // Helper to convert a single 8-bit sRGB component to a linear float.
    static inline float srgb_to_linear(uint8_t c_srgb) {
        float val = static_cast<float>(c_srgb) / 255.0f;
        if (val <= 0.04045f) {
            return val / 12.92f;
        }
        return std::pow((val + 0.055f) / 1.055f, 2.4f);
    }

    // Helper to convert a single linear float component to an 8-bit sRGB value.
    static inline uint8_t linear_to_srgb(float c_linear) {
        float val;
        if (c_linear <= 0.0031308f) {
            val = c_linear * 12.92f;
        }
        else {
            val = 1.055f * std::pow(c_linear, 1.0f / 2.4f) - 0.055f;
        }
        // Clamp to [0, 1], scale to [0, 255], and round to the nearest integer.
        float clamped_val = std::min(1.0f, std::max(0.0f, val));
        return static_cast<uint8_t>(clamped_val * 255.0f + 0.5f);
    }

    /**
     * @brief Converts a block of 8-bit sRGB data to a block of Oklab floats.
     *
     * This function first decodes the non-linear 8-bit sRGB values to linear floats,
     * then feeds them into the existing FloatBlockToOklabFloatBlock pipeline.
     *
     * @param srgbBlock The input vector of 8-bit sRGB data (16 pixels).
     * @param rgba True if the input data is RGBA, false for RGB.
     * @param applyACESTonemap Whether to apply the ACES filmic tonemapper to the linear data.
     * @return An OklabFloatBlock (std::vector<float>) containing the converted data.
     */
    inline OklabFloatBlock SrgbBlockToOklabFloatBlock_Scalar(const std::vector<uint8_t>& srgbBlock, bool rgba, bool applyACESTonemap = false) {
        const int num_channels = rgba ? 4 : 3;
        if (srgbBlock.size() != 16 * num_channels) return OklabFloatBlock();

        // Convert 8-bit sRGB to linear float
        std::vector<float> linearFloatBlock;
        linearFloatBlock.resize(srgbBlock.size());

        for (int i = 0; i < 16; ++i) {
            int idx = i * num_channels;
            // Convert R, G, B from sRGB to linear
            linearFloatBlock[idx + 0] = srgb_to_linear(srgbBlock[idx + 0]);
            linearFloatBlock[idx + 1] = srgb_to_linear(srgbBlock[idx + 1]);
            linearFloatBlock[idx + 2] = srgb_to_linear(srgbBlock[idx + 2]);

            // Alpha channel is considered linear, so it only needs normalization
            if (rgba) {
                linearFloatBlock[idx + 3] = static_cast<float>(srgbBlock[idx + 3]) / 255.0f;
            }
        }

        // Use the existing pipeline with the converted linear float data
        return FloatBlockToOklabFloatBlock(linearFloatBlock, rgba, applyACESTonemap);
    }

    inline OklabFloatBlock SrgbBlockToOklabFloatBlock_AVX2(const std::vector<uint8_t>& srgbBlock, bool rgba, bool applyACESTonemap = false) {
        const int num_channels = rgba ? 4 : 3;
        if (srgbBlock.size() != 16 * num_channels) return OklabFloatBlock();

        // temporary linear floats; will be fed to FloatBlockToOklabFloatBlock
        std::vector<float> linearFloatBlock(srgbBlock.size());

        const float* table = srgb8_to_linear_table().data();

        // process 8 pixels at a time -> two iterations (0 and 8)
        for (int i = 0; i < 16; i += 8) {
            // gather 8 R bytes, 8 G, 8 B into small arrays (8 bytes)
            alignas(16) uint8_t rb[8], gb[8], bb[8], ab[8];
            for (int j = 0; j < 8; ++j) {
                int idx = (i + j) * num_channels;
                rb[j] = srgbBlock[idx + 0];
                gb[j] = srgbBlock[idx + 1];
                bb[j] = srgbBlock[idx + 2];
                if (rgba) ab[j] = srgbBlock[idx + 3];
            }

            // load 8 bytes into XMM and widen to 8 i32s in a 256 register
            __m128i r_load = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(rb)); // loads 64 bits (8 bytes)
            __m128i g_load = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(gb));
            __m128i b_load = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(bb));

            __m256i r_idx = _mm256_cvtepu8_epi32(r_load); // 8 x i32 indices
            __m256i g_idx = _mm256_cvtepu8_epi32(g_load);
            __m256i b_idx = _mm256_cvtepu8_epi32(b_load);

            // gather table (scale = 4 because table is float)
            __m256 vr = _mm256_i32gather_ps(table, r_idx, 4);
            __m256 vg = _mm256_i32gather_ps(table, g_idx, 4);
            __m256 vb = _mm256_i32gather_ps(table, b_idx, 4);

            // store gathered linear floats to temporary arrays and write them to linearFloatBlock
            alignas(32) float r_f[8], g_f[8], b_f[8];
            _mm256_store_ps(r_f, vr);
            _mm256_store_ps(g_f, vg);
            _mm256_store_ps(b_f, vb);

            for (int j = 0; j < 8; ++j) {
                int out_idx = (i + j) * num_channels;
                linearFloatBlock[out_idx + 0] = r_f[j];
                linearFloatBlock[out_idx + 1] = g_f[j];
                linearFloatBlock[out_idx + 2] = b_f[j];
                if (rgba) linearFloatBlock[out_idx + 3] = static_cast<float>(ab[j]) / 255.0f;
            }
        }

        // now reuse your existing float pipeline (will pick AVX2 float path if available)
        return FloatBlockToOklabFloatBlock(linearFloatBlock, rgba, applyACESTonemap);
    }


    // Fast conversion from a single linear float to 8-bit sRGB using LUT + linear interpolation.
    // Very cheap compared to pow.
    static inline uint8_t linear_to_srgb_lut(float c_linear) {
        // clamp to [0,1]
        float c = c_linear;
        if (c <= 0.0f) return 0;
        if (c >= 1.0f) return 255;

        const auto& tbl = linear_to_srgb8_table();
        const float idxf = c * static_cast<float>(LINEAR_TO_SRGB_LUT_SIZE - 1);
        const int i0 = static_cast<int>(idxf); // trunc toward zero (ok because idxf >= 0)
        const int i1 = (i0 + 1 < LINEAR_TO_SRGB_LUT_SIZE) ? (i0 + 1) : i0;
        const float frac = idxf - static_cast<float>(i0);

        // linear interpolate between byte entries (works well and is cheap)
        const uint8_t a = tbl[i0];
        const uint8_t b = tbl[i1];
        // compute as integer lerp with float fraction
        const float v = static_cast<float>(a) + (static_cast<float>(b) - static_cast<float>(a)) * frac;
        return static_cast<uint8_t>(v + 0.5f);
    }

    // Helper: convert a linear float block (interleaved RGB[A] floats) -> vector<uint8_t> using LUT
    // This is used by the replacement OklabFloatBlockToSrgbBlock below.
    static inline std::vector<uint8_t> convert_linear_floats_to_srgb8(const std::vector<float>& linearFloatBlock, bool rgba) {
        const int num_channels = rgba ? 4 : 3;
        const size_t expected_size = 16 * num_channels;
        if (linearFloatBlock.size() != expected_size) return std::vector<uint8_t>();

        std::vector<uint8_t> out;
        out.resize(expected_size);

        for (int i = 0; i < 16; ++i) {
            int base = i * num_channels;
            // R
            out[base + 0] = linear_to_srgb_lut(linearFloatBlock[base + 0]);
            // G
            out[base + 1] = linear_to_srgb_lut(linearFloatBlock[base + 1]);
            // B
            out[base + 2] = linear_to_srgb_lut(linearFloatBlock[base + 2]);

            if (rgba) {
                // Alpha: clamp and round (alpha is linear in your pipeline)
                float a = linearFloatBlock[base + 3];
                if (a <= 0.0f) out[base + 3] = 0;
                else if (a >= 1.0f) out[base + 3] = 255;
                else out[base + 3] = static_cast<uint8_t>(a * 255.0f + 0.5f);
            }
        }
        return out;
    }

    /**
     * @brief Converts a block of Oklab floats back to a block of 8-bit sRGB data.
     *
     * This function first calls the existing Oklab-to-Float pipeline to get linear
     * RGB floats, then encodes those linear floats into non-linear 8-bit sRGB values.
     *
     * @param labBlock The input OklabFloatBlock.
     * @param rgba True if the output data should be RGBA, false for RGB.
     * @param applyACESInverse Whether to apply the inverse ACES tonemapper before sRGB encoding.
     * @return A std::vector<uint8_t> containing the 8-bit sRGB data.
     */
    inline std::vector<uint8_t> OklabFloatBlockToSrgbBlock(const OklabFloatBlock& labBlock, bool rgba, bool applyACESInverse = false) {
        const int num_channels = rgba ? 4 : 3;
        if (labBlock.size() != 16 * num_channels) return std::vector<uint8_t>();

        // 1) Produce linear floats using existing pipeline (already vectorized if AVX available)
        std::vector<float> linearFloatBlock = OklabFloatBlockToFloatBlock(labBlock, rgba, /*clampForLDR=*/true, applyACESInverse);

        // 2) Fast float->8bit using LUT + interpolation
        return convert_linear_floats_to_srgb8(linearFloatBlock, rgba);
    }

    // =========================================================================
    // CONVENIENCE WRAPPERS for 8-bit RGBA and RGB
    // =========================================================================

    /**
     * @brief Convenience wrapper for converting a 16xRGBA (8-bit) block to Oklab.
     */
    inline OklabFloatBlock Srgba8BlockToOklabFloatBlock(const std::vector<uint8_t>& srgbaBlock, bool applyACESTonemap = false) {
        if (srgbaBlock.size() != 16 * 4) return OklabFloatBlock();
        if (cpu_has_avx2_cached()) return SrgbBlockToOklabFloatBlock_AVX2(srgbaBlock, true, applyACESTonemap);
        return SrgbBlockToOklabFloatBlock_Scalar(srgbaBlock, true, applyACESTonemap);
    }

    /**
     * @brief Convenience wrapper for converting a 16xRGB (8-bit) block to Oklab.
     */
    inline OklabFloatBlock Srgb8BlockToOklabFloatBlock(const std::vector<uint8_t>& srgbBlock, bool applyACESTonemap = false) {
        if (srgbBlock.size() != 16 * 3) return OklabFloatBlock();
        if (cpu_has_avx2_cached()) return SrgbBlockToOklabFloatBlock_AVX2(srgbBlock, false, applyACESTonemap);
        return SrgbBlockToOklabFloatBlock_Scalar(srgbBlock, false, applyACESTonemap);
    }

} // namespace Oklab