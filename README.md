# **TL;DR**  
This is a "get it done quick" vector quantization implementation, not tested very thoroughly, but it gets the job done. I'm nowhere near a texture compression specialist, especially when it comes to working with color space conversions and SIMD operations. Most of the math in this library was written either by Gemini 2.5 Pro or GPT-5, not even going to hide it.  
> Is it efficient?  
  
As much as I could get it to be efficient, yes. It uses modern perceptual Oklab color space for distance calculations, does not waste memory (I think) and is heavily parallelized.  
 
> Can it be improved?  

Most certainly.  
If you're a texture compression enthusiast, and you happen to come across this library and you have time and the desire to change some things in here, please do. I would greatly appreciate that.  
# Overview  
Yup Texture is a texture super-compression library, taking inspiration from BinomialLLC's [Basis Universal](https://github.com/BinomialLLC/basis_universal).  
***
It uses Vector Quantization to quantize BCn blocks using K-Means++ and Mean Distance calculation, output centroids get saved into a compressed fast-to-lookup codebook.  
It can supper-compress any BCn format(including BC6H HDR data, and even preserve most of it correctly) to sizes even smaller than JPGs.  
It has an extensive CompressionParams struct, where you can tweak the compression parameters for speed/quality, and optionally disable VQ entirely(now it will just use ZSTD, which will yield the highest quality yet higher decompression times and bigger size).
 ***
Decompression in this library is blazingly fast, because its as simple as rebuilding the source BCn blocks from the codebook by **one(!)** memcpy operation, which is done in parallel thanks to OpenMP and is extremely fast(around **7-11 ms** for a **4096x4096** texture!)  
This comes with a downside of course: results produced by current implementation have visible quantization errors for smaller texture sizes, and ideally such textures need to be pre-filtered with a simple de-blocking filter either on CPU or in the GPU fragment shader to maintain a desirable look. 
*Which is a fair trade-off* considering it can compress a 4K texture from **12.3MB** down to just **1.3~ MB**, or an 4096x2048 HDR texture from **17.8MB** down to just **208KB**(some luminance is lost, but it's still pretty decent for skyboxes in games for example).
But for high-res textures (**2K, 4K**) compression artifacts are visually negligible on large surfaces.

Thanks for the star [WzrterFX](https://github.com/WzrterFX) : D

Here is an example of using this library:
```cpp
#include "vq_bcn_compressor.h" // Include the compressor
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <vector>

int main() {
    // --- Step 1: Load an image (LDR in this example) ---
    int width, height, channels;
    unsigned char* pixels = stbi_load("input.png", &width, &height, &channels, 0);
    if (!pixels) {
        std::cerr << "Failed to load image!\n";
        return 1;
    }

    // --- Step 2: Setup compressor and parameters ---
    VQBCnCompressor compressor;
    CompressionParams params;
    params.bcFormat = BCFormat::BC1;                 // Use BC1 (good for albedo textures)
    params.quality = 0.8f;                           // VQ quality (0..1)
    params.useVQ = true;                             // Enable Vector Quantization
    params.useZstd = true;                           // Enable Zstd compression
    params.numThreads = 8;                           // Use 8 threads for speed

    // --- Step 3: Compress image ---
    std::vector<uint8_t> compressed = compressor.Compress(
        pixels, width, height, channels, params);

    stbi_image_free(pixels); // free original pixels

    // --- Step 4: Save compressed data to disk ---
    {
        std::ofstream out("output_texture.yupt2", std::ios::binary);
        out.write(reinterpret_cast<char*>(compressed.data()), compressed.size());
    }

    // --- Step 5: Decompress back to raw pixels ---
    TextureInfo outInfo;
    std::vector<uint8_t> decompressed = compressor.Decompress(compressed, outInfo);

    // --- Step 6: Save decompressed image as PNG for verification ---
    stbi_write_png("output.png", outInfo.width, outInfo.height,
        outInfo.originalChannelCount, decompressed.data(),
        outInfo.width * outInfo.originalChannelCount);

    std::cout << "Done! Compressed -> Decompressed -> output.png\n";
    return 0;
}
```