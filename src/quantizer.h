#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <string>

struct QuantizedTensor {
    std::vector<int8_t> data;   // quantized weights [-128, 127]
    float scale;                 // scale factor for dequantization
    float zero_point;            // zero point (0 for symmetric)
};

struct QuantizationStats {
    size_t fp32_bytes;
    size_t fp16_bytes;
    size_t int8_bytes;
    size_t int4_bytes;
};

class Quantizer {
public:
    // Quantize FP32 weights to INT8 (symmetric quantization)
    QuantizedTensor quantize(const std::vector<float>& weights);
    
    // Dequantize INT8 back to FP32
    std::vector<float> dequantize(const QuantizedTensor& tensor);
    
    // INT8 matrix-vector multiply (quantized inference kernel)
    // result = dequant(int8_weights · float_input)
    std::vector<float> int8_matvec(const QuantizedTensor& weights,
                                    const float* input, uint32_t input_size,
                                    uint32_t output_size);
    
    // Compare FP32 vs INT8 model sizes
    QuantizationStats compute_stats(size_t total_fp32_params);
    
    // Print summary
    void print_summary(const QuantizationStats& stats);
};
