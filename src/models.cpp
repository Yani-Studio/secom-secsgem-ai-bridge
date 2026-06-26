#include "models.h"
#include <cstdio>
#include <cassert>
#include <fstream>
#include <iostream>

// ═════════════════════════════════════════════════════════════════════════════════
//  Tree Ensemble Model (LightGBM, XGBoost, Random Forest)
// ═════════════════════════════════════════════════════════════════════════════════

float TreeEnsembleModel::predict(const float* features, uint32_t n) {
    float sum = base_score_;

    for (auto& tree : trees_) {
        int node_idx = 0;
        while (tree.nodes[node_idx].feature_idx != -1) {
            auto& node = tree.nodes[node_idx];
            uint32_t fidx = static_cast<uint32_t>(node.feature_idx);
            if (fidx < n && features[fidx] <= node.threshold) {
                node_idx = node.left_child;
            } else {
                node_idx = node.right_child;
            }
        }
        sum += tree.nodes[node_idx].leaf_value;
    }

    return sigmoid(sum);
}

size_t TreeEnsembleModel::param_count() const {
    size_t count = 0;
    for (const auto& tree : trees_) {
        count += tree.nodes.size() * 5;
    }
    return count + 1;
}

bool TreeEnsembleModel::load_weights(std::ifstream& in) {
    uint32_t num_trees;
    in.read(reinterpret_cast<char*>(&num_trees), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&base_score_), sizeof(float));

    trees_.resize(num_trees);
    for (uint32_t i = 0; i < num_trees; ++i) {
        uint32_t num_nodes;
        in.read(reinterpret_cast<char*>(&num_nodes), sizeof(uint32_t));
        trees_[i].nodes.resize(num_nodes);
        in.read(reinterpret_cast<char*>(trees_[i].nodes.data()), num_nodes * sizeof(TreeNode));
    }

    // Initialize quantization weights and biases
    weights_.clear();
    for (const auto& tree : trees_) {
        for (const auto& node : tree.nodes) {
            if (node.feature_idx != -1) {
                weights_.push_back(node.threshold);
            } else {
                weights_.push_back(node.leaf_value);
            }
        }
    }
    biases_.assign(num_trees > 0 ? num_trees : 1, base_score_ / static_cast<float>(num_trees > 0 ? num_trees : 1));
    return true;
}

std::vector<float>& TreeEnsembleModel::weights() { return weights_; }
std::vector<float>& TreeEnsembleModel::biases()  { return biases_;  }

// ═════════════════════════════════════════════════════════════════════════════════
//  Logistic Regression (Linear Model)
// ═════════════════════════════════════════════════════════════════════════════════

float LinearModel::predict(const float* features, uint32_t n) {
    float sum = biases_.empty() ? 0.0f : biases_[0];
    uint32_t len = std::min(n, static_cast<uint32_t>(weights_.size()));
    for (uint32_t i = 0; i < len; ++i) {
        sum += features[i] * weights_[i];
    }
    return sigmoid(sum);
}

size_t LinearModel::param_count() const {
    return weights_.size() + biases_.size();
}

bool LinearModel::load_weights(std::ifstream& in) {
    uint32_t len;
    in.read(reinterpret_cast<char*>(&len), sizeof(uint32_t));
    weights_.resize(len);
    in.read(reinterpret_cast<char*>(weights_.data()), len * sizeof(float));

    in.read(reinterpret_cast<char*>(&len), sizeof(uint32_t));
    biases_.resize(len);
    if (len > 0) {
        in.read(reinterpret_cast<char*>(biases_.data()), len * sizeof(float));
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════════
//  Ensemble Engine — Soft Voting
// ═════════════════════════════════════════════════════════════════════════════════

void EnsembleEngine::initialize(const float* /*feature_stats*/, uint32_t /*n_features*/,
                                 const float* /*fail_means*/, const float* /*fail_stds*/,
                                 const float* /*pass_means*/) {
    // Only load from weights, no heuristic initialization
}

float EnsembleEngine::predict(const float* features, uint32_t n) {
    float sum = 0.0f;
    float weight_sum = 0.0f;
    for (size_t i = 0; i < models_.size(); ++i) {
        float w = (i < ensemble_weights_.size()) ? ensemble_weights_[i] : 1.0f;
        sum += models_[i]->predict(features, n) * w;
        weight_sum += w;
    }
    return weight_sum > 0.0f ? (sum / weight_sum) : 0.0f;
}

bool EnsembleEngine::load_weights(const std::string& bin_path) {
    std::printf("Entering EnsembleEngine::load_weights(%s)\n", bin_path.c_str());
    std::ifstream in(bin_path, std::ios::binary);
    if (!in.is_open()) {
        std::printf("Failed to open weights file: %s\n", bin_path.c_str());
        return false;
    }
    
    char magic[4];
    in.read(magic, 4);
    if (std::strncmp(magic, "SECM", 4) != 0) {
        std::printf("Invalid magic number in weights file.\n");
        return false;
    }
    
    uint32_t version;
    in.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    
    char marker[4];
    while (in.read(marker, 4)) {
        if (in.gcount() < 4) break;
        std::printf("Loading Model... marker: %.4s\n", marker);
        if (std::strncmp(marker, "LGBM", 4) == 0) {
            auto m = std::make_unique<TreeEnsembleModel>("LightGBM");
            m->load_weights(in);
            models_.push_back(std::move(m));
        } else if (std::strncmp(marker, "XGBO", 4) == 0) {
            auto m = std::make_unique<TreeEnsembleModel>("XGBoost");
            m->load_weights(in);
            models_.push_back(std::move(m));
        } else if (std::strncmp(marker, "RFOR", 4) == 0) {
            auto m = std::make_unique<TreeEnsembleModel>("RandomForest");
            m->load_weights(in);
            models_.push_back(std::move(m));
        } else if (std::strncmp(marker, "LREG", 4) == 0) {
            auto m = std::make_unique<LinearModel>("LogisticRegression");
            m->load_weights(in);
            models_.push_back(std::move(m));
        } else if (std::strncmp(marker, "GBMM", 4) == 0) {
            auto m = std::make_unique<TreeEnsembleModel>("GBM");
            m->load_weights(in);
            models_.push_back(std::move(m));
        } else if (std::strncmp(marker, "ADAB", 4) == 0) {
            auto m = std::make_unique<TreeEnsembleModel>("AdaBoost");
            m->load_weights(in);
            models_.push_back(std::move(m));
        } else if (std::strncmp(marker, "WGHT", 4) == 0) {
            uint32_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(uint32_t));
            ensemble_weights_.resize(len);
            in.read(reinterpret_cast<char*>(ensemble_weights_.data()), len * sizeof(float));
            std::printf("  -> Loaded %u ensemble weights\n", len);
        } else {
            std::printf("Unknown model marker: %.4s\n", marker);
            break;
        }
    }
    
    in.close();
    return true;
}

IModel* EnsembleEngine::get_model(size_t idx) {
    if (idx < models_.size()) return models_[idx].get();
    return nullptr;
}

void EnsembleEngine::print_summary() {
    std::printf("[STAGE 3] Ensemble Model Init ───────────────────────────────\n");
    for (size_t i = 0; i < models_.size(); ++i) {
        float w = (i < ensemble_weights_.size()) ? ensemble_weights_[i] : 1.0f;
        std::printf("  ✓ %-20s : %zu params, weight=%.3f\n", models_[i]->name(), models_[i]->param_count(), w);
    }
    std::printf("  ✓ Ensemble strategy: Weighted Soft Voting\n");
}
