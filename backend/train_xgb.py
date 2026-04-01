import os
import pickle
import numpy as np
import xgboost as xgb

# Configuration
FEATURES_PATH = "/app/ultimate_results/features_3layer_mean.pkl"
LOCAL_FEATURES_PATH = "../20260302_ultimate/results/features_3layer_mean.pkl"
OUTPUT_MODEL = "ultimate_xgb.json"

N_MULT = 3.0
A_MULT = 5.0

def train_and_save():
    # Detect where the script is running 
    path_to_load = FEATURES_PATH if os.path.exists(FEATURES_PATH) else LOCAL_FEATURES_PATH
    
    if not os.path.exists(path_to_load):
        print(f"Error: Features file not found at {path_to_load}")
        return

    print(f"Loading features from {path_to_load}...")
    with open(path_to_load, 'rb') as f:
        X, y, groups = pickle.load(f)
        
    print(f"Dataset Size: {len(y)} samples. Training XGBoost with full data...")

    # Calculate weights based on config
    n_total = len(y)
    weight_map = {}
    for cls in np.unique(y):
        n_c = np.sum(y == cls)
        base_w = n_total / (3 * n_c)
        if cls == 1: base_w *= N_MULT
        elif cls == 2: base_w *= A_MULT
        weight_map[cls] = base_w
    
    weights = np.array([weight_map.get(lbl, 1.0) for lbl in y])
    
    # Generate feature names to match extraction layout
    feature_names = [f"AST_{i}" for i in range(768)]
    # Feature Utils Extract Time Domain (3), Spectral (5), F0 (1), F1 (1), MFCC (26), BBI_ACF (2)
    # The total number of traditional features in `features_3layer_mean.pkl` can be derived from X.shape[1] - 768
    # However, since we just need the model to ingest data seamlessly, we'll assign generic names or proper ones.
    if X.shape[1] > 768:
        for i in range(X.shape[1] - 768):
            feature_names.append(f"Trad_{i}")
    else:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        
    dtrain = xgb.DMatrix(X, label=y, weight=weights, feature_names=feature_names)

    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 4,
        'eta': 0.1,
        'verbosity': 1,
        'tree_method': 'hist'
    }
    
    bst = xgb.train(params, dtrain, num_boost_round=100)
    
    # Save the model
    # Use JSON format for portability across XGBoost versions
    bst.save_model(OUTPUT_MODEL)
    print(f"Model saved successfully to {OUTPUT_MODEL}!")

if __name__ == "__main__":
    train_and_save()
