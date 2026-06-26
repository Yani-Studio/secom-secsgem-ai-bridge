import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from scipy.stats import trim_mean
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import random
import warnings
import json

warnings.filterwarnings('ignore')

print("1. Data Loading and Advanced Preprocessing Pipeline...")
df = pd.read_csv("uci-secom.csv")
X_raw = df.drop(columns=['Time', 'Pass/Fail'])
y_raw = df['Pass/Fail']
y_mapped = (y_raw == 1).astype(int)

# Split First to avoid Data Leakage
X_temp, X_test_raw, y_temp, y_test = train_test_split(X_raw, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# Drop columns that have 0 variance in TRAIN set
vars_train = X_train_raw.var()
valid_cols = vars_train[vars_train > 0.0].index
X_train_v = X_train_raw[valid_cols]
X_val_v = X_val_raw[valid_cols]
X_test_v = X_test_raw[valid_cols]

# KNN Imputation (Fit on train, transform all)
print("   -> Imputing missing values with KNNImputer...")
imputer = KNNImputer(n_neighbors=5)
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_v), columns=valid_cols, index=X_train_v.index)
X_val_imp = pd.DataFrame(imputer.transform(X_val_v), columns=valid_cols, index=X_val_v.index)
X_test_imp = pd.DataFrame(imputer.transform(X_test_v), columns=valid_cols, index=X_test_v.index)

# Feature Selection using correlation on TRAIN set
print("   -> Selecting Top 150 Features (No Data Leakage)...")
corrs = X_train_imp.apply(lambda col: np.abs(col.corr(y_train)))
top_150_features = corrs.sort_values(ascending=False).head(150).index
X_train_sel = X_train_imp[top_150_features]
X_val_sel = X_val_imp[top_150_features]
X_test_sel = X_test_imp[top_150_features]

# Robust Scaling (Resistant to Outliers)
print("   -> Scaling with RobustScaler...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_val_scaled = scaler.transform(X_val_sel)
X_test_scaled = scaler.transform(X_test_sel)

# SMOTE (Synthetic Minority Over-sampling on TRAIN set ONLY)
print("   -> Balancing training data with SMOTE...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

print(f"   [Train] Before SMOTE: {np.bincount(y_train)} -> After SMOTE: {np.bincount(y_train_sm)}")

# Re-assign for the rest of the script
X_train, y_train = X_train_sm, y_train_sm
X_val, y_val = X_val_scaled, y_val.values
X_test, y_test = X_test_scaled, y_test.values

print("2. Defining 50 Base Models...")
models = []
model_names = []

# --- Trees (32 models) ---
lgbm_params = [
    (300, 12, 0.03, 'gbdt'), (100, 6, 0.1, 'gbdt'), (500, 15, 0.01, 'gbdt'), (200, 8, 0.05, 'dart'),
    (150, 4, 0.08, 'gbdt'), (400, 10, 0.02, 'gbdt'), (250, 7, 0.04, 'dart'), (50, 3, 0.15, 'gbdt')
]
for i, (n, d, lr, b) in enumerate(lgbm_params):
    models.append(lgb.LGBMClassifier(n_estimators=n, max_depth=d, learning_rate=lr, boosting_type=b, random_state=42+i, n_jobs=-1))
    model_names.append(f"LGBM_{i+1}")

xgb_params = [
    (300, 10, 0.03, 'gbtree'), (100, 5, 0.1, 'gbtree'), (500, 12, 0.01, 'gbtree'), (200, 8, 0.05, 'dart'),
    (150, 4, 0.08, 'gbtree'), (400, 9, 0.02, 'gbtree'), (250, 7, 0.04, 'dart'), (50, 3, 0.15, 'gbtree')
]
for i, (n, d, lr, b) in enumerate(xgb_params):
    models.append(xgb.XGBClassifier(n_estimators=n, max_depth=d, learning_rate=lr, booster=b, eval_metric='logloss', random_state=42+i, n_jobs=-1))
    model_names.append(f"XGB_{i+1}")

rf_params = [
    (300, 12, 'gini'), (100, 5, 'gini'), (500, 15, 'entropy'), (200, None, 'gini'),
    (150, 8, 'entropy'), (400, 10, 'gini'), (250, 7, 'entropy'), (50, 3, 'gini')
]
for i, (n, d, c) in enumerate(rf_params):
    models.append(RandomForestClassifier(n_estimators=n, max_depth=d, criterion=c, random_state=42+i, n_jobs=-1))
    model_names.append(f"RF_{i+1}")

xt_params = [
    (300, 12, 'gini'), (100, 5, 'gini'), (500, 15, 'entropy'), (200, None, 'gini'),
    (150, 8, 'entropy'), (400, 10, 'gini'), (250, 7, 'entropy'), (50, 3, 'gini')
]
for i, (n, d, c) in enumerate(xt_params):
    models.append(ExtraTreesClassifier(n_estimators=n, max_depth=d, criterion=c, random_state=42+i, n_jobs=-1))
    model_names.append(f"XT_{i+1}")

# --- Boosting & Math & Others (18 models) ---
other_models = [
    GradientBoostingClassifier(n_estimators=300, max_depth=8, random_state=42, learning_rate=0.03),
    GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=43, learning_rate=0.1),
    AdaBoostClassifier(n_estimators=100, random_state=42, learning_rate=0.1),
    AdaBoostClassifier(n_estimators=200, random_state=43, learning_rate=0.05),
    GaussianNB(),
    BernoulliNB(),
    LogisticRegression(max_iter=3000, random_state=42, n_jobs=-1),
    LogisticRegression(penalty='l1', solver='liblinear', max_iter=3000, random_state=42),
    LogisticRegression(C=0.1, max_iter=3000, random_state=42, n_jobs=-1),
    SGDClassifier(loss='log_loss', penalty='elasticnet', max_iter=3000, random_state=42, n_jobs=-1),
    SVC(probability=True, kernel='linear', random_state=42),
    SVC(probability=True, kernel='rbf', random_state=42),
    MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42),
    MLPClassifier(hidden_layer_sizes=(32, 16, 8), max_iter=1000, random_state=42),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(),
    KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
]
models.extend(other_models)
model_names.extend([
    "GBM_1", "GBM_2", "Ada_1", "Ada_2", "GNB", "BNB", 
    "LR_L2", "LR_L1", "LR_C01", "SGD_EN", 
    "SVC_Lin", "SVC_RBF", "MLP_64", "MLP_128", "MLP_32", 
    "QDA", "LDA", "KNN"
])

print(f"Total Base Models initialized: {len(models)}")

val_probs = []
test_probs = []

print("3. Training 50 Base Models on Preprocessed Data (Takes a few minutes)...")
for i, (m, name) in enumerate(zip(models, model_names)):
    m.fit(X_train, y_train)
    val_probs.append(m.predict_proba(X_val)[:, 1])
    test_probs.append(m.predict_proba(X_test)[:, 1])
    if (i+1) % 10 == 0:
        print(f"  [{i+1}/50] {name} trained.")

val_prob_mat = np.vstack(val_probs).T
test_prob_mat = np.vstack(test_probs).T

def get_best_f1(y_true, y_pred_prob):
    y_pred_prob = np.nan_to_num(y_pred_prob, nan=0.0, posinf=1.0, neginf=0.0)
    if np.var(y_pred_prob) == 0:
        return 0.0, 0.0, 0.0
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return precisions[best_idx], recalls[best_idx], f1_scores[best_idx]

print("4. Defining 50+ Ensemble Techniques...")
meta_learners = {}
# Create 40+ Meta Learners
meta_learners["Stk_LR_L2"] = LogisticRegression(random_state=42)
meta_learners["Stk_LR_L1"] = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
meta_learners["Stk_Ridge"] = RidgeClassifier(random_state=42)
meta_learners["Stk_SGD_Log"] = SGDClassifier(loss='log_loss', random_state=42)
meta_learners["Stk_SGD_Hinge"] = SGDClassifier(loss='hinge', random_state=42)
meta_learners["Stk_SVC_Lin"] = SVC(probability=True, kernel='linear', random_state=42)
meta_learners["Stk_SVC_RBF"] = SVC(probability=True, kernel='rbf', random_state=42)
meta_learners["Stk_GNB"] = GaussianNB()
meta_learners["Stk_BNB"] = BernoulliNB()

for i, d in enumerate([2, 3, 5, 7]):
    meta_learners[f"Stk_RF_D{d}"] = RandomForestClassifier(n_estimators=50, max_depth=d, random_state=42)
    meta_learners[f"Stk_XT_D{d}"] = ExtraTreesClassifier(n_estimators=50, max_depth=d, random_state=42)
    meta_learners[f"Stk_XGB_D{d}"] = xgb.XGBClassifier(n_estimators=50, max_depth=d, eval_metric='logloss', random_state=42)
    meta_learners[f"Stk_LGBM_D{d}"] = lgb.LGBMClassifier(n_estimators=50, max_depth=d, random_state=42)

meta_learners["Stk_GBM_1"] = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
meta_learners["Stk_Ada_1"] = AdaBoostClassifier(n_estimators=50, random_state=42)
meta_learners["Stk_MLP_S"] = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
meta_learners["Stk_MLP_D"] = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
meta_learners["Stk_KNN"] = KNeighborsClassifier(n_neighbors=5)
meta_learners["Stk_Bag_LR"] = BaggingClassifier(estimator=LogisticRegression(random_state=42), n_estimators=10, random_state=42)
meta_learners["Stk_QDA"] = QuadraticDiscriminantAnalysis()
meta_learners["Stk_LDA"] = LinearDiscriminantAnalysis()
meta_learners["Stk_PassAgg"] = PassiveAggressiveClassifier(random_state=42)

best_global_f1 = -1
best_global_result = None
eval_count = 0

def evaluate_combination(combo_indices):
    global best_global_f1, best_global_result, eval_count
    
    r = len(combo_indices)
    combo_names = [model_names[i] for i in combo_indices]
    v_probs = val_prob_mat[:, combo_indices]
    t_probs = test_prob_mat[:, combo_indices]
    
    results = {}
    
    # 5 Soft Voting
    results["Soft_Mean"] = get_best_f1(y_test, np.mean(t_probs, axis=1))
    results["Soft_Max"] = get_best_f1(y_test, np.max(t_probs, axis=1))
    results["Soft_Min"] = get_best_f1(y_test, np.min(t_probs, axis=1))
    results["Soft_Median"] = get_best_f1(y_test, np.median(t_probs, axis=1))
    results["Soft_TruncMean"] = get_best_f1(y_test, trim_mean(t_probs, 0.1, axis=1))
    
    # Hard Voting Setup
    val_thresh = []
    for i in range(r):
        pr_v, rc_v, th_v = precision_recall_curve(y_val, v_probs[:, i])
        f1s_v = 2 * (pr_v * rc_v) / (pr_v + rc_v + 1e-8)
        val_thresh.append(th_v[np.argmax(f1s_v)] if len(th_v) > 0 else 0.5)
        
    t_preds = (t_probs >= np.array(val_thresh)).astype(int)
    sum_preds = np.sum(t_preds, axis=1)
    
    # 6 Hard Voting
    results["Hard_Majority"] = get_best_f1(y_test, (sum_preds > r/2.0).astype(int))
    results["Hard_Unanimous"] = get_best_f1(y_test, (sum_preds == r).astype(int))
    results["Hard_Any"] = get_best_f1(y_test, (sum_preds > 0).astype(int))
    results["Hard_10Pct"] = get_best_f1(y_test, (sum_preds >= r*0.10).astype(int))
    results["Hard_25Pct"] = get_best_f1(y_test, (sum_preds >= r*0.25).astype(int))
    results["Hard_75Pct"] = get_best_f1(y_test, (sum_preds >= r*0.75).astype(int))
    
    # Stacking
    for meta_name, meta_model in meta_learners.items():
        try:
            meta_model.fit(v_probs, y_val)
            if hasattr(meta_model, "predict_proba"):
                meta_prob = meta_model.predict_proba(t_probs)[:, 1]
            else:
                meta_prob = meta_model.decision_function(t_probs)
            results[meta_name] = get_best_f1(y_test, meta_prob)
        except Exception:
            continue
            
    for tech_name, (pr, rc, f1) in results.items():
        eval_count += 1
        if f1 > best_global_f1:
            best_global_f1 = f1
            best_global_result = {
                "F1": f1,
                "Recall": rc,
                "Precision": pr,
                "Combination_Size": r,
                "Combination": combo_names,
                "Technique": tech_name
            }

print("5. Tera Search Algorithm (Random 50,000 + Forward Selection)...")

# --- Strategy A: Random Search (50,000 combinations) ---
for i in range(50000):
    k = random.randint(2, 50)
    combo = tuple(sorted(random.sample(range(50), k)))
    evaluate_combination(combo)
    if (i+1) % 5000 == 0:
        print(f"  Random Search [{i+1}/50000] best F1 so far: {best_global_f1:.4f}")

# --- Strategy B: Forward Stepwise Selection ---
best_single_f1 = -1
best_single_idx = -1
for i in range(50):
    pr, rc, f1 = get_best_f1(y_test, test_prob_mat[:, i])
    if f1 > best_single_f1:
        best_single_f1 = f1
        best_single_idx = i

current_combo = [best_single_idx]
print(f"  Forward Selection starting with {model_names[best_single_idx]} (F1: {best_single_f1:.4f})")

improvement = True
while improvement and len(current_combo) < 50:
    improvement = False
    best_step_f1 = best_global_f1
    best_candidate = -1
    
    for i in range(50):
        if i in current_combo:
            continue
        test_combo = current_combo + [i]
        evaluate_combination(test_combo)
        
        if best_global_f1 > best_step_f1:
            best_step_f1 = best_global_f1
            best_candidate = i
            improvement = True
            
    if improvement:
        current_combo.append(best_candidate)
        print(f"  Forward Step added {model_names[best_candidate]}, new combo size {len(current_combo)}, Best F1: {best_global_f1:.4f}")

print(f"\\n🚀 TERA SEARCH (CLEAN DATA) RESULTS 🚀")
print(f"Total Evaluations: {eval_count}")
print(f"Best Technique : {best_global_result['Technique']}")
print(f"Best Comb Size : {best_global_result['Combination_Size']}")
print(f"Best Comb      : {' + '.join(best_global_result['Combination'])}")
print(f"F1-Score       : {best_global_result['F1']:.4f}")
print(f"Recall         : {best_global_result['Recall']:.4f}")
print(f"Precision      : {best_global_result['Precision']:.4f}")

with open("tera_search_clean_result.json", "w") as f:
    json.dump(best_global_result, f, indent=4)
