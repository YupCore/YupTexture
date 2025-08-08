# **TL;DR**
I'm nowhere near a texture compression specialist, especially when it comes to working with color space conversions and SIMD operations. Most of the math in this library was written either by Gemini 2.5 Pro or GPT-5, not even going to hide it. 

> Is it efficent?

Not really, as I'm fully aware the double color space is just a waste of space(it uses Cielab for LDR and Oklab for HDR, should use Oklab for both etc). 

> Is it fast to write and iterate with AIs?

 Absolutely, this took me only 3 days to finish and it already handles hi-res LDR and HDR textures(2K>) fine enough for my purposes.
If you're a texture compression enthusiast, and you happen to come across this library and you have time and the desire to change some things in here, please do. I would greatly appreciate that.
*For now it will probably stay like this, unless I get a dose of motivation and finish this myself.*
#  Overview
Yup Texture is a texture suppercompression library, taking inspiration from BinomialLLC's [Basis Universal](https://github.com/BinomialLLC/basis_universal). It uses Vector Quantization to quantize BCn blocks using K-Means++ and Mean Distance calculation, output centroids get saved into a compressed fast to lookup codebook. Decompression is just as simple as rebuilding the source BCn blocks from the codebook by **one(!)** memcpy operation, which is done in parallel thanks to OpenMP and is extremely fast(around **6-8 ms** for a **4096x4096** RGB texture!)
This comes with a downside of course: results produced by current implementation have visibible quantization errors, and ideally textures need to be pre-filtered with a simple deblocking filter either on CPU or in the GPU fragment shader to maintain a desirable look.
Certain settings in the CompressionParams can be tweaked to increase the VQ's efforts and quality, but it will still look blocky.
*Which is a fair trade-off* considering it can compress a 4K texture from **12.3MB** down to just **1.3~ MB**, or an 4096x2048 HDR texture from **17.8MB** down to just **208KB**(some luminance is lost, but it's still pretty decent for skyboxes in games for example)

Here are some code examples of using this library:
```cpp
	// Initialize compression parameters
	#include <vq_bcn_compressor.h>
	
    CompressionParams params;
    params.bcQuality = 1.0f; // internal BCn compressor quality
    params.zstdLevel = 16; // zstd compression level(0 fastest, 22 max)
    params.numThreads = 16; // number of threads used in parallel operations
    params.useVQ = true; // use Vector Quantization?(yes, you can skip it and use just ZSTD for losless results)
    params.useZstd = true; // use zstd?
    // Select the appropriate settings based on texture type
    switch (textureType) {
	 case HDR:
	     std::cout << "Texture Type: HDR (Using BC6H with VQ)\n";
	     params.bcFormat = BCFormat::BC6H;
	     // --- Enable VQ for HDR and set params ---
			params.bcQuality = 0.25f; // Use a lower quality for HDR to set reasonable compression time
	     params.quality = 0.9f; // Use a high quality for HDR VQ
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
	 CompressedTexture compressed;
	 // Call the correct Compress overload based on whether the image is HDR
	 if (image.isHDR) {
	     compressed = compressor.CompressHDR(std::get<std::vector<float>>(image.data).data(), image.width, image.height, params);
	 }
	 else {
	     compressed = compressor.Compress(std::get<std::vector<uint8_t>>(image.data).data(), image.width, image.height, image.channels, params);
	 }
	 std::cout << "Compression finished in " << std::fixed << std::setprecision(2)
	     << std::chrono::duration<double>(end_compress - start_compress).count() << "s.\n";

	 // Decompression sample:
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

     auto bcData = compressor.DecompressToBCn(loadedTexture);
     std::cout << "Decompression to BCn (GPU-ready) finished in " << std::fixed << std::setprecision(4) << diff_decompress_bcn.count() << " seconds.\n";

     Image outputImage;
     outputImage.width = image.width;
     outputImage.height = image.height;
     outputImage.channels = 4;
     outputImage.isHDR = image.isHDR;

     if (image.isHDR) {
         outputImage.data = compressor.DecompressToRGBAF(loadedTexture, true);
     }
     else {
         outputImage.data = compressor.DecompressToRGBA(loadedTexture, true);
         std::cout << "Decompression to RGBA finished in " << std::fixed << std::setprecision(4)
             << std::chrono::duration<double>(end_decompress - start_decompress).count() << "s.\n";
     }
      
```
Still, this library have lots of improvements and TODO's so I'm going to list them here:

 - [ ] Implement uniform color space for both LDR and HDR data(Oklab is the best one for this)
 - [ ] Implement the handling of mean distances for LDR and HDR in Oklab color space and use correct formulas for HDR pre-processing(correct log-luminance implementation for compressing the dynamic)
 - [ ] Improve K-means++ initializations and remove redundant pre-calculations

I'll finish this later...
