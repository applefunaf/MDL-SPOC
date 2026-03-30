import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys

# Import custom MDL
sys.path.append('/home/ljf/ml4e')
from mdl.mdl import MDLRegressionModel
from train_spoc_mdl import parse_pqr, get_centroid, compute_spoc

def analyze_mdl_features():
    print("1. Loading yield data and features...")
    csv_path = '/home/ljf/ml4e/descriptor/5mol_ch3/dara_5mol_ch3_6.5_S_yield.csv'
    pqr_dir = '/home/ljf/ml4e/descriptor/5mol_ch3'
    df_yield = pd.read_csv(csv_path)
    
    X_list = []
    y_list = []
    
    for index, row in df_yield.iterrows():
        mut = row['Mutation']
        yld = row['Yield']
        pqr_path = os.path.join(pqr_dir, f"7p76_{mut}.pqr")
        if os.path.exists(pqr_path):
            df_atoms = parse_pqr(pqr_path)
            center = get_centroid(df_atoms)
            feats = compute_spoc(df_atoms, center, max_radius=30.0, bins=30)
            X_list.append(feats)
            y_list.append(yld)
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Dataset shape: {X.shape}")
    
    # 训练MDL筛选器以获取特征重要性
    from sklearn.linear_model import LassoLarsIC
    
    # 手动提取MDL挑选的频次以定位核心空间
    n_rounds = 100
    feature_counts = np.zeros(X.shape[1])
    
    print(f"2. Running MDL feature selection over {n_rounds} rounds to find stable important regions...")
    for r in range(n_rounds):
        np.random.seed(42 + r)
        X_noisy = X + np.random.randn(*X.shape) * 1e-6
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_noisy)
        
        # 运行MDL (LassoLarsIC)
        mdl = LassoLarsIC(criterion='bic')
        if X_scaled.shape[0] <= X_scaled.shape[1]:
            mdl.noise_variance = np.var(y) * 0.1
        
        mdl.fit(X_scaled, y)
        selected = np.where(mdl.coef_ != 0)[0]
        feature_counts[selected] += 1
        
    print("\n3. Deciphering highly selected SPOC features...")
    # SPOC features 结构: 30 个 bins, 每个 bin 有 6 个特征 (esp, steric, C, N, O, S)
    # 总计 180 维
    feature_names = []
    for i in range(30):
        b_min = i * 1.0 # 因为 max_radius=30, bins=30
        b_max = (i + 1) * 1.0
        feature_names.extend([
            f"[{b_min}-{b_max}A]_ESP", f"[{b_min}-{b_max}A]_Steric", 
            f"[{b_min}-{b_max}A]_C_atoms", f"[{b_min}-{b_max}A]_N_atoms", 
            f"[{b_min}-{b_max}A]_O_atoms", f"[{b_min}-{b_max}A]_S_atoms"
        ])
    
    # 获取被选中的频率 > 0的特征并排序
    important_indices = np.argsort(feature_counts)[::-1]
    
    print("\n[ Top 15 Stable Critical Interaction Zones Identified by MDL ]")
    print("-" * 100)
    
    # 额外为了映射真实的WT PQR残基准备：
    wt_pqr = os.path.join(pqr_dir, "7p76_wt.pqr")
    if os.path.exists(wt_pqr):
        wt_atoms = parse_pqr(wt_pqr)
        wt_center = get_centroid(wt_atoms)
        wt_atoms['dist'] = np.linalg.norm(wt_atoms[['x', 'y', 'z']].values - wt_center, axis=1)
    else:
        wt_atoms = None

    residue_scores = {}

    for idx in important_indices[:15]:
        if feature_counts[idx] > 0:
            feat_name = feature_names[idx]
            print(f"Feature: {feat_name:<25} | Selected {int(feature_counts[idx]):>3}/{n_rounds} times")
            
            # 如果有了 WT 数据，我们反向映射出位点
            if wt_atoms is not None:
                try:
                    # Parse distance bin like "[10.0-11.0A]_O_atoms"
                    dist_part, prop_part = feat_name.split('A]_')
                    d_min_str, d_max_str = dist_part[1:].split('-')
                    d_min, d_max = float(d_min_str), float(d_max_str)
                    
                    # 确定要找的元素类型
                    element_filter = None
                    if prop_part == 'O_atoms': element_filter = 'O'
                    elif prop_part == 'C_atoms': element_filter = 'C'
                    elif prop_part == 'N_atoms': element_filter = 'N'
                    elif prop_part == 'S_atoms': element_filter = 'S'
                    
                    # 在该区间寻找残基
                    sub_atoms = wt_atoms[(wt_atoms['dist'] > d_min) & (wt_atoms['dist'] <= d_max)]
                    if element_filter:
                        sub_atoms = sub_atoms[sub_atoms['element'] == element_filter]
                        
                    # 提取具体的位点集合 (去重)
                    if not sub_atoms.empty:
                        res_list = sub_atoms[['res_name', 'res_id']].drop_duplicates()
                        res_strings = [f"{row['res_name']}{row['res_id']}" for _, row in res_list.iterrows()]
                        print(f"   └─> Related WT Residues: {', '.join(res_strings)}")
                        # 聚集分数：给该区间的每个相关残基增加所对应特征的被选中频次权重
                        for rs in res_strings:
                            residue_scores[rs] = residue_scores.get(rs, 0) + feature_counts[idx]
                    else:
                        print(f"   └─> Related WT Residues: None found in this specific shell/element.")
                except Exception as e:
                    print(f"   └─> Error mapping mapping residue: {e}")
            print("-" * 100)
            
    print("\n[ Mutation Strategy Recommendation ]")
    print("These highlighted residues are strong candidates for targeted combinatorial libraries or site-directed mutagenesis.")
    
    # === 新增代码: 保存位点与特征频率分布的柱状图 ===
    print("\n4. Generating Feature Frequency Distribution Plot...")
    import matplotlib.pyplot as plt
    
    # 为了避免字体重叠，只取 Top 20 制作图表
    top_n = 20
    top_indices = important_indices[:top_n]
    
    # 提取有价值的特征供绘图
    plot_feat_names = []
    plot_freqs = []
    
    for idx in top_indices:
        if feature_counts[idx] > 0:
            plot_feat_names.append(feature_names[idx])
            plot_freqs.append(feature_counts[idx])
            
    # 反转列表，让最高频的排在图表最上面
    plot_feat_names = plot_feat_names[::-1]
    plot_freqs = plot_freqs[::-1]
    
    plt.figure(figsize=(12, 8))
    
    # 设置一个渐变颜色风格的横向柱形图
    colors = plt.cm.viridis(np.linspace(0.4, 0.9, len(plot_feat_names)))
    
    bars = plt.barh(plot_feat_names, plot_freqs, color=colors, edgecolor='black', alpha=0.8)
    
    # 图表基础设置
    plt.xlabel('Selection Frequency (Times selected out of 100 random perturbations)', fontsize=14)
    plt.ylabel('Spatial SPOC Features', fontsize=14)
    plt.title('MDL Stable Critical Features Distribution', fontsize=18, fontweight='bold', pad=20)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 在条形图边上标注数字
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                 f'{int(width)}', 
                 ha='left', va='center', fontsize=12, fontweight='bold')
                 
    plt.tight_layout()
    output_png = '/home/ljf/ml4e/plots/MDL_Feature_Frequency_Distribution.png'
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Plot explicitly saved to: {output_png}")

    # === 新增代码: 保存位点的得分柱状图 ===
    print("\n5. Generating Residue Importance Distribution Plot...")
    if residue_scores:
        # 按照聚合分数降序排列
        sorted_res = sorted(residue_scores.items(), key=lambda x: x[1], reverse=True)
        top_res_to_plot = min(30, len(sorted_res)) # 取前30个画图
        
        plot_res_names = [x[0] for x in sorted_res[:top_res_to_plot]][::-1]
        plot_res_scores = [x[1] for x in sorted_res[:top_res_to_plot]][::-1]
        
        plt.figure(figsize=(10, 12))
        colors_res = plt.cm.magma(np.linspace(0.3, 0.9, len(plot_res_names)))
        bars_res = plt.barh(plot_res_names, plot_res_scores, color=colors_res, edgecolor='black', alpha=0.8)
        
        plt.xlabel('Aggregated MDL Importance Score (Sum of feature frequencies)', fontsize=14)
        plt.ylabel('Wild Type Residues', fontsize=14)
        plt.title('Top 30 Highly Implicated Residues by MDL Model', fontsize=18, fontweight='bold', pad=20)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        for bar in bars_res:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                     f'{int(width)}', 
                     ha='left', va='center', fontsize=11, fontweight='bold')
                     
        plt.tight_layout()
        res_png = '/home/ljf/ml4e/plots/MDL_Residue_Importance_Distribution.png'
        plt.savefig(res_png, dpi=300)
        plt.close()
        print(f"Plot explicitly saved to: {res_png}")
    
if __name__ == "__main__":
    analyze_mdl_features()
