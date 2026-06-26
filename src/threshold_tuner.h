#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct ConfusionMatrix {
    uint32_t tp = 0;  // True Positive (correctly identified Fail)
    uint32_t fp = 0;  // False Positive (Pass misclassified as Fail)
    uint32_t tn = 0;  // True Negative (correctly identified Pass)
    uint32_t fn = 0;  // False Negative (Fail misclassified as Pass - CRITICAL)
};

struct ThresholdResult {
    float threshold;
    float recall;
    float precision;
    float f1_score;
    float type2_error_rate;  // 미검률 = FN / (TP + FN)
    ConfusionMatrix cm;
};

class ThresholdTuner {
public:
    // Load threshold config from INI file
    void load_config(const std::string& config_path);
    
    // Evaluate a single threshold
    ThresholdResult evaluate(const std::vector<float>& predictions,
                            const std::vector<int8_t>& labels,
                            float threshold);
    
    // Sweep thresholds to find optimal for target recall
    ThresholdResult sweep(const std::vector<float>& predictions,
                         const std::vector<int8_t>& labels);
    
    // Print results
    void print_results(const ThresholdResult& result);
    
    float default_threshold() const { return default_threshold_; }
    
private:
    float default_threshold_ = 0.20f;
    float sweep_min_ = 0.05f;
    float sweep_max_ = 0.95f;
    float sweep_step_ = 0.01f;
    float target_recall_ = 0.99f;
};
