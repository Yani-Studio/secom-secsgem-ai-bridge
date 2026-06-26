#include "quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// 대칭 양자화 (Symmetric Quantization): FP32 → INT8
// scale = max(|w|) / 127, q = clamp(round(w / scale), -128, 127)
// ─────────────────────────────────────────────────────────────────────────────
QuantizedTensor Quantizer::quantize(const std::vector<float>& weights) {
    QuantizedTensor tensor;
    tensor.zero_point = 0.0f;  // 대칭 양자화이므로 zero_point = 0

    if (weights.empty()) {
        tensor.scale = 1.0f;
        return tensor;
    }

    // 가중치 절대값의 최대값 탐색
    float abs_max = 0.0f;
    for (const float w : weights) {
        const float a = std::fabs(w);
        if (a > abs_max) abs_max = a;
    }

    // scale 계산 (0-division 방지)
    tensor.scale = (abs_max > 0.0f) ? abs_max / 127.0f
                                     : 1.0f;

    // 양자화 수행
    tensor.data.resize(weights.size());
    const float inv_scale = 1.0f / tensor.scale;

    for (size_t i = 0; i < weights.size(); ++i) {
        float q = std::round(weights[i] * inv_scale);
        // clamp to [-128, 127]
        q = std::max(-128.0f, std::min(127.0f, q));
        tensor.data[i] = static_cast<int8_t>(q);
    }

    return tensor;
}

// ─────────────────────────────────────────────────────────────────────────────
// 역양자화 (Dequantization): INT8 → FP32 근사값
// w_approx = q * scale
// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> Quantizer::dequantize(const QuantizedTensor& tensor) {
    std::vector<float> result(tensor.data.size());

    for (size_t i = 0; i < tensor.data.size(); ++i) {
        result[i] = static_cast<float>(tensor.data[i]) * tensor.scale;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// INT8 행렬-벡터 곱: 양자화된 가중치와 FP32 입력의 내적 후 역양자화
// weights.data 는 row-major (output_size x input_size) 배열로 해석
// result[o] = scale * Σ( int8_weight[o][i] * quantized_input[i] )
// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> Quantizer::int8_matvec(const QuantizedTensor& weights,
                                           const float* input,
                                           uint32_t input_size,
                                           uint32_t output_size) {
    if (weights.data.size() != static_cast<size_t>(output_size) * input_size) {
        throw std::invalid_argument(
            "Weight tensor size mismatch: expected "
            + std::to_string(static_cast<size_t>(output_size) * input_size)
            + ", got " + std::to_string(weights.data.size()));
    }

    // 입력 벡터도 양자화하여 정수 내적 수행 (INT8 × INT8 시뮬레이션)
    float input_abs_max = 0.0f;
    for (uint32_t i = 0; i < input_size; ++i) {
        const float a = std::fabs(input[i]);
        if (a > input_abs_max) input_abs_max = a;
    }
    const float input_scale = (input_abs_max > 0.0f) ? input_abs_max / 127.0f
                                                      : 1.0f;
    const float inv_input_scale = 1.0f / input_scale;

    // 입력 양자화 (스택 할당 가능하면 좋지만 크기가 가변이므로 힙 사용)
    std::vector<int8_t> q_input(input_size);
    for (uint32_t i = 0; i < input_size; ++i) {
        float q = std::round(input[i] * inv_input_scale);
        q = std::max(-128.0f, std::min(127.0f, q));
        q_input[i] = static_cast<int8_t>(q);
    }

    // 정수 내적 수행 후 결합 scale로 역양자화
    const float combined_scale = weights.scale * input_scale;
    std::vector<float> result(output_size);

    for (uint32_t o = 0; o < output_size; ++o) {
        int32_t acc = 0;  // INT32 누산기 (오버플로 방지)
        const size_t row_offset = static_cast<size_t>(o) * input_size;
        for (uint32_t i = 0; i < input_size; ++i) {
            acc += static_cast<int32_t>(weights.data[row_offset + i])
                 * static_cast<int32_t>(q_input[i]);
        }
        result[o] = static_cast<float>(acc) * combined_scale;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// 모델 크기 비교 통계: FP32 vs FP16 vs INT8 vs INT4
// ─────────────────────────────────────────────────────────────────────────────
QuantizationStats Quantizer::compute_stats(size_t total_fp32_params) {
    QuantizationStats stats{};
    stats.fp32_bytes = total_fp32_params * sizeof(float);   // 4 bytes per param
    stats.fp16_bytes = total_fp32_params * 2 + 8;           // 2 bytes per param + overhead
    stats.int8_bytes = total_fp32_params * 1 + 8;           // 1 byte per param + overhead
    stats.int4_bytes = (total_fp32_params + 1) / 2 + 8;     // 0.5 bytes per param + overhead

    return stats;
}

// ─────────────────────────────────────────────────────────────────────────────
// 양자화 결과 요약 출력
// ─────────────────────────────────────────────────────────────────────────────
void Quantizer::print_summary(const QuantizationStats& stats) {
    const float fp32_kb = static_cast<float>(stats.fp32_bytes) / 1024.0f;
    const float fp16_kb = static_cast<float>(stats.fp16_bytes) / 1024.0f;
    const float int8_kb = static_cast<float>(stats.int8_bytes) / 1024.0f;
    const float int4_kb = static_cast<float>(stats.int4_bytes) / 1024.0f;

    std::printf("  ✓ Quantization Sizes:\n");
    std::printf("      - FP32 : %7.1f KB (Baseline)\n", fp32_kb);
    std::printf("      - FP16 : %7.1f KB (%.1f%% reduction)\n", fp16_kb, (1.0f - fp16_kb/fp32_kb)*100.0f);
    std::printf("      - INT8 : %7.1f KB (%.1f%% reduction)\n", int8_kb, (1.0f - int8_kb/fp32_kb)*100.0f);
    std::printf("      - INT4 : %7.1f KB (%.1f%% reduction)\n", int4_kb, (1.0f - int4_kb/fp32_kb)*100.0f);
}
