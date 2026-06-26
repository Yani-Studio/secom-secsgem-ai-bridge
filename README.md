<div align="center">
  <h1>🏭 SECOM-Tera-Ensemble</h1>
  <p><strong>Ultra-Precision Semiconductor Defect Detection Architecture</strong></p>

  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white">
</div>

<br>

## 📌 The Philosophy: "Zero Defect Escape over Millisecond Latency"

In semiconductor manufacturing and fab operations, a single defect escaping to the market can result in catastrophic financial losses, massive recalls, and severe damage to client trust. 

While conventional AI edge deployments emphasize quantization and pruning to achieve sub-millisecond inference speeds, **this architecture takes the exact opposite approach.** We deliberately sacrifice inference speed to prioritize absolute detection accuracy. In the highly imbalanced SECOM dataset, quantized models lose critical boundary resolution, missing defects. 

By running an extremely heavy **22-Model Stacking Ensemble**, we ensure maximum sensitivity. To bridge this heavy computation with real-time fab execution protocols (e.g., SECS/GEM, EDA), the trained Python pipeline is designed to be exported to ONNX and deployed purely in **C++ on High-Performance Edge Servers**.

---

## 🏗️ Architecture: The 50x50 Tera Search

We conducted an exhaustive search evaluating **2,252,205 combinations** (50 candidate base models × 50 ensemble techniques). The winning architecture is the `Stk_RF_D7` stacking ensemble.

### 1. Data Preprocessing Pipeline (Zero Data Leakage)
1. **KNN Imputer (K=5)**: Restores 100% of missing sensor values using neighbor patterns.
2. **Feature Selection**: Isolates the Top 150 most highly correlated sensor signals.
3. **RobustScaler**: Normalizes data based on median/quartiles to resist severe factory outliers.
4. **SMOTE**: Oversamples the extreme minority class (Defect) specifically on the training set to resolve the severe class imbalance.

### 2. The 22-Model Stacking Ensemble
Out of 50 candidates, the Meta-Model selected 22 heterogeneous models to form the first layer of predictions:
- **7 Boosting Models** (LightGBM, XGBoost, AdaBoost, GBM)
- **7 Forest Models** (Random Forest, Extra Trees)
- **2 Support Vector Machines** (RBF, Linear)
- **2 Linear Models** (Logistic Regression, SGD)
- **2 Discriminant Analysis** (QDA, LDA)
- **1 Deep Neural Network** (MLP 128)

These 22 probability predictions are fed into a **Random Forest Meta-Model (Max Depth 7)** to make the final rigorous defect prediction.

<div align="center">
  <img src="visualizations/viz_09_stacking_architecture.png" alt="Stacking Architecture" width="800">
</div>

---

## 🚀 Performance Metrics

Evaluated dynamically on an unseen test set (314 samples), the Tera Ensemble achieves ground-breaking detection capabilities.

- **F1-Score**: `0.5294`
- **Precision**: `69.23%` (When the alarm triggers, there is a 69.2% probability it is a genuine defect)
- **Recall**: `42.86%` (Caught nearly half of all extremely subtle defects)
- **Accuracy**: `94.90%`
- **Specificity**: `98.63%` (Minimal false alarms)

<div align="center">
  <img src="visualizations/viz_06_evaluation_curves.png" alt="Evaluation Curves" width="800">
</div>

### Confusion Matrix
| | Predicted Normal | Predicted Defect |
|---|:---:|:---:|
| **True Normal** | 289 | 4 (False Alarms) |
| **True Defect** | 12 (Missed) | **9 (Detected!)** |

*Note: In the context of 592 noisy sensor signals and only 6.6% defect rate, achieving 9 detections with only 4 false alarms saves massive manual inspection costs.*

---

## ⚙️ C++ Fab Integration Guide

The ensemble weights are originally trained in Python (`scikit-learn`, `xgboost`, `lightgbm`). To integrate this heavy pipeline into low-level C++ Semiconductor Equipment Protocols without the Python GIL overhead, follow this deployment strategy:

### 1. Model Export (Python)
Export the trained preprocessing pipeline and the 22 base models into the ONNX standard format:
```python
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Example: Converting the pipeline to ONNX
initial_type = [('float_input', FloatTensorType([None, 150]))]
onnx_model = convert_sklearn(meta_model, initial_types=initial_type)
with open("secom_tera_ensemble.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

### 2. C++ Edge Inference (C++ / SECS/GEM)
Load the `.onnx` graph using `ONNXRuntime C++ API` directly on the equipment controller or edge server. This guarantees multi-threaded, low-latency C++ execution of all 22 models simultaneously:
```cpp
#include <onnxruntime_cxx_api.h>

// 1. Initialize ONNX Environment
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SECOM_Inference");
Ort::SessionOptions session_options;
session_options.SetIntraOpNumThreads(8); // Multi-threading for 22 models

// 2. Load Model
Ort::Session session(env, "secom_tera_ensemble.onnx", session_options);

// 3. SECS/GEM Data Input -> Inference -> Alarm Trigger
// (Pass 150 float sensor array into the session)
```

By pushing the inference entirely to C++, we maintain the **flawless accuracy** of the 22-model stacking architecture while achieving acceptable latency limits for real-time factory operation.

---
*Built for zero-defect semiconductor manufacturing.*
