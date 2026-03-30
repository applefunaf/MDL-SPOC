import pandas as pd
import numpy as np
import glob
import os
import sys
import argparse
from datetime import datetime
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression

def load_data(yield_csv_path, descriptor_dir='.'):
    print(f"Loading yields from {yield_csv_path}...")
    try:
        yield_df = pd.read_csv(yield_csv_path)
    except FileNotFoundError:
        print("Yield file not found. Please check the path.")
        return None, None, None

    X_list = []
    y_list = []
    names = []

    # 查找所有描述符文件
    descriptor_files = glob.glob(os.path.join(descriptor_dir, '*_descriptor.csv'))
    print(f"Found {len(descriptor_files)} descriptor files.")

    for desc_file in descriptor_files:
        filename = os.path.basename(desc_file)
        
        # 匹配逻辑：检查产率表中的 Mutation 字段是否包含在文件名中
        match_row = None
        for _, row in yield_df.iterrows():
            mut = str(row['Mutation']).strip().lower()
            if not mut: continue
            
            # 简化匹配：如果文件名包含突变名称 (如 's18a' in '7p76_s18a_descriptor.csv')
            # 特殊处理 'wt'
            if f"_{mut}_" in filename or filename.endswith(f"_{mut}_descriptor.csv") or \
               (mut == 'wt' and '_wt_' in filename):
                match_row = row
                break
        
        if match_row is None:
            continue

        try:
            # 读取描述符
            desc_df = pd.read_csv(desc_file)
            if desc_df.empty: continue
            
            # --- 特征工程 ---
            # 聚合原子描述符为分子描述符
            # 针对 radius, esp, distance_to_centroid 计算统计量
            stats = []
            for col in ['radius', 'esp', 'distance_to_centroid']:
                if col in desc_df.columns:
                    col_data = desc_df[col].dropna()
                    if len(col_data) == 0:
                        stats.extend([0, 0, 0, 0])
                    else:
                        stats.extend([
                            col_data.mean(),
                            col_data.std(),
                            col_data.min(),
                            col_data.max()
                        ])
                else:
                    stats.extend([0, 0, 0, 0])
            
            X_list.append(stats)
            y_list.append(match_row['Yield'])
            names.append(match_row['Mutation'])
            
        except Exception as e:
            print(f"Error reading {desc_file}: {e}")

    return np.array(X_list), np.array(y_list), names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PLS Regression Model")
    parser.add_argument("yield_csv", help="Path to the yield CSV file")
    args = parser.parse_args()
    
    current_dir = os.getcwd()
    X, y, sample_names = load_data(args.yield_csv, current_dir)

    if X is not None and len(X) > 0:
        print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")
        
        print("=== Training PLS Regression with Nested LOOCV ===")
        print("Outer Loop: Leave-One-Out Cross-Validation (Evaluation)")
        print("Inner Loop: 3-Fold Grid Search (Hyperparameter Tuning)")
        
        # Determine max components possible
        # Safe upper bound: min(features, N - 1)
        # Note: during inner CV (3-fold), train size is ~2/3 * N.
        # We need to be careful with n_components.
        max_n_comp = min(X.shape[1], max(1, int(X.shape[0] * 0.6))) 
        n_comps_list = [i for i in range(1, max_n_comp + 1)]
        if not n_comps_list:
             n_comps_list = [1]
        
        param_grid = {
            'n_components': n_comps_list
        }
        
        # Inner CV: 3-fold or LOO depending on size? 3-fold is safer for tiny data to avoid overfitting specific points
        pls = PLSRegression()
        grid_search = GridSearchCV(pls, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
        
        # Outer CV: Leave-One-Out
        loo = LeaveOneOut()
        y_pred_pls = cross_val_predict(grid_search, X, y, cv=loo, n_jobs=-1)
        
        rmse_pls = np.sqrt(mean_squared_error(y, y_pred_pls))
        r2_pls = r2_score(y, y_pred_pls)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] PLS LOOCV Result - RMSE: {rmse_pls:.4f}, R2: {r2_pls:.4f}")
        
        # Final fit
        print("Fitting final model on all data...")
        grid_search.fit(X, y)
        print(f"Best parameters (All Data): {grid_search.best_params_}")
        r2_pls = r2_score(y_test, y_pred_pls)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] PLS Result - RMSE: {rmse_pls:.4f}, R2: {r2_pls:.4f}")
        
    else:
        print("No data matched. Please check file naming conventions or paths.")
