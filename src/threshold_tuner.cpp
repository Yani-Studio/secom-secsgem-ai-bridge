#include "threshold_tuner.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// INI 파일에서 [threshold] 섹션의 설정을 읽어오는 함수
// ─────────────────────────────────────────────────────────────────────────────
void ThresholdTuner::load_config(const std::string& config_path) {
    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {
        std::fprintf(stderr, "[WARN] Cannot open config: %s — using defaults\n",
                     config_path.c_str());
        return;
    }

    bool in_threshold_section = false;
    std::string line;

    while (std::getline(ifs, line)) {
        // 앞뒤 공백 제거 (trim)
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t\r\n");
        line = line.substr(start, end - start + 1);

        // 주석(# 또는 ;) 무시
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;

        // 섹션 헤더 [section]
        if (line.front() == '[' && line.back() == ']') {
            std::string section = line.substr(1, line.size() - 2);
            in_threshold_section = (section == "threshold");
            continue;
        }

        if (!in_threshold_section) continue;

        // key = value 파싱
        auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        std::string key = line.substr(0, eq_pos);
        std::string val = line.substr(eq_pos + 1);

        // key/value 트림
        auto trim = [](std::string& s) {
            size_t a = s.find_first_not_of(" \t");
            size_t b = s.find_last_not_of(" \t");
            s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
        };
        trim(key);
        trim(val);

        if (key == "default")       default_threshold_ = std::stof(val);
        else if (key == "sweep_min")   sweep_min_ = std::stof(val);
        else if (key == "sweep_max")   sweep_max_ = std::stof(val);
        else if (key == "sweep_step")  sweep_step_ = std::stof(val);
        else if (key == "target_recall") target_recall_ = std::stof(val);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 주어진 임계값(threshold)에 대해 혼동 행렬(Confusion Matrix) 계산
// labels: -1 = Pass (양품), +1 = Fail (불량)
// prediction >= threshold → Fail(+1)로 분류
// ─────────────────────────────────────────────────────────────────────────────
ThresholdResult ThresholdTuner::evaluate(const std::vector<float>& predictions,
                                         const std::vector<int8_t>& labels,
                                         float threshold) {
    if (predictions.size() != labels.size()) {
        throw std::invalid_argument(
            "predictions and labels must have the same size");
    }

    ConfusionMatrix cm{};

    for (size_t i = 0; i < predictions.size(); ++i) {
        const bool predicted_fail = (predictions[i] >= threshold);
        const bool actual_fail    = (labels[i] == 1);

        if (actual_fail && predicted_fail)        ++cm.tp;  // 불량을 불량으로 정확히 검출
        else if (!actual_fail && predicted_fail)   ++cm.fp;  // 양품을 불량으로 오분류 (과검)
        else if (!actual_fail && !predicted_fail)  ++cm.tn;  // 양품을 양품으로 정확히 판정
        else /* actual_fail && !predicted_fail */   ++cm.fn;  // 불량을 양품으로 오분류 (미검 - CRITICAL)
    }

    ThresholdResult result{};
    result.threshold = threshold;
    result.cm = cm;

    // Recall (검출률): TP / (TP + FN)
    const float total_fail = static_cast<float>(cm.tp + cm.fn);
    result.recall = (total_fail > 0.0f) ? static_cast<float>(cm.tp) / total_fail
                                         : 0.0f;

    // Precision (정밀도): TP / (TP + FP)
    const float total_predicted_fail = static_cast<float>(cm.tp + cm.fp);
    result.precision = (total_predicted_fail > 0.0f)
                           ? static_cast<float>(cm.tp) / total_predicted_fail
                           : 0.0f;

    // F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    const float pr_sum = result.precision + result.recall;
    result.f1_score = (pr_sum > 0.0f)
                          ? 2.0f * result.precision * result.recall / pr_sum
                          : 0.0f;

    // 미검률 (Type II Error Rate): FN / (TP + FN)
    result.type2_error_rate = (total_fail > 0.0f)
                                  ? static_cast<float>(cm.fn) / total_fail
                                  : 0.0f;

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// 임계값 스윕: F1-Score가 가장 높은 지점 탐색 (최고 성능 달성)
// ─────────────────────────────────────────────────────────────────────────────
ThresholdResult ThresholdTuner::sweep(const std::vector<float>& predictions,
                                      const std::vector<int8_t>& labels) {
    ThresholdResult best{};
    best.threshold = default_threshold_;
    best.f1_score = -1.0f;
    bool found = false;

    // sweep_min_ ~ sweep_max_ 까지 sweep_step_ 간격으로 탐색
    for (float th = sweep_min_; th <= sweep_max_ + 1e-6f; th += sweep_step_) {
        ThresholdResult r = evaluate(predictions, labels, th);

        // F1-Score가 가장 높은 임계값 선택 (성능 극대화)
        if (!found || r.f1_score > best.f1_score) {
            best = r;
            found = true;
        }
    }

    if (!found) {
        std::fprintf(stderr, "[WARN] Could not find any valid threshold.\n");
    }

    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// 결과 출력: 혼동 행렬 및 주요 지표
// ─────────────────────────────────────────────────────────────────────────────
void ThresholdTuner::print_results(const ThresholdResult& result) {
    std::printf("[STAGE 4] Threshold Tuning ───────────────────────────────────\n");
    std::printf("  ✓ Optimal threshold: %.2f (Recall=%.1f%%, Precision=%.1f%%)\n",
                result.threshold,
                result.recall * 100.0f,
                result.precision * 100.0f);
    std::printf("  ✓ F1-Score: %.4f\n", result.f1_score);
    std::printf("  ✓ Type II Error (미검률): %.2f%%\n",
                result.type2_error_rate * 100.0f);
    std::printf("  ✓ Confusion Matrix:\n");
    std::printf("           Predicted Pass  Predicted Fail\n");
    std::printf("    Pass   %8u        %8u\n",
                result.cm.tn, result.cm.fp);
    std::printf("    Fail   %8u        %8u\n",
                result.cm.fn, result.cm.tp);
}
