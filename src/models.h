#pragma once
#include <memory>
#include <array>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <random>
#include <numeric>
#include <algorithm>
#include "domain_enricher.h"

// ─── Utility ────────────────────────────────────────────────────────────────────

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-std::clamp(x, -20.0f, 20.0f)));
}

// ─── Abstract Model Interface ───────────────────────────────────────────────────

class IModel {
public:
    virtual ~IModel() = default;
    virtual float predict(const float* features, uint32_t n) = 0;  // returns probability 0.0~1.0
    virtual const char* name() const = 0;
    virtual size_t param_count() const = 0;

    // For quantization access
    virtual std::vector<float>& weights() = 0;
    virtual std::vector<float>& biases() = 0;
    
    // For loading trained weights
    virtual bool load_weights(std::ifstream& in) = 0;
};

// ─── Tree Ensemble Model (LightGBM, XGBoost, Random Forest) ─────────────────────

class TreeEnsembleModel : public IModel {
public:
    struct TreeNode {
        int32_t feature_idx;     // -1 for leaf
        float threshold;
        float leaf_value;        // only valid for leaf nodes
        int32_t left_child;      // index in nodes array
        int32_t right_child;
    };

    struct Tree {
        std::vector<TreeNode> nodes;
    };

    explicit TreeEnsembleModel(const std::string& name) : name_(name) {}
    
    float predict(const float* features, uint32_t n) override;
    const char* name() const override { return name_.c_str(); }
    size_t param_count() const override;
    std::vector<float>& weights() override;
    std::vector<float>& biases() override;
    size_t tree_count() const { return trees_.size(); }
    bool load_weights(std::ifstream& in) override;

private:
    std::string name_;
    std::vector<Tree> trees_;
    std::vector<float> weights_;  // flattened tree thresholds/values
    std::vector<float> biases_;   // tree base scores
    float base_score_ = 0.0f;
};

// ─── Logistic Regression (Linear Model) ─────────────────────────────────────────

class LinearModel : public IModel {
public:
    explicit LinearModel(const std::string& name) : name_(name) {}

    float predict(const float* features, uint32_t n) override;
    const char* name() const override { return name_.c_str(); }
    size_t param_count() const override;
    std::vector<float>& weights() override { return weights_; }
    std::vector<float>& biases() override { return biases_; }
    bool load_weights(std::ifstream& in) override;

private:
    std::string name_;
    std::vector<float> weights_;
    std::vector<float> biases_; // contains a single bias value
};

// ─── Ensemble Engine — Soft Voting ──────────────────────────────────────────────

class EnsembleEngine {
public:
    void initialize(const float* feature_stats, uint32_t n_features,
                    const float* fail_means, const float* fail_stds,
                    const float* pass_means);
                    
    bool load_weights(const std::string& bin_path);

    float predict(const float* features, uint32_t n);

    // Access individual models for quantization
    IModel* get_model(size_t idx);
    size_t model_count() const { return models_.size(); }

    void print_summary();

private:
    std::vector<std::unique_ptr<IModel>> models_;
    std::vector<float> ensemble_weights_;
};
