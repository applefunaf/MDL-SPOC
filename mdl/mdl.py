import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.linear_model import LinearRegression, LassoLarsIC, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

class MDLFeatureSelector(BaseEstimator, TransformerMixin):
    """
    基于最小描述长度（Minimum Description Length, MDL）原理的特征选择器。
    用于处理高维描述符，通过平衡模型拟合度（数据描述长度）和模型复杂度（模型描述长度）
    来提取对产率预测最有效的特征子集。
    这里使用近似MDL的BIC（贝叶斯信息准则）评分作为基础实现，结合Lasso的正则化路径。
    """
    def __init__(self, criterion='bic'):
        self.criterion = criterion
        self.selected_indices_ = None
        self.model_ = LassoLarsIC(criterion=self.criterion)

    def fit(self, X, y):
        # 确保输入为numpy数组
        X = np.asarray(X)
        y = np.asarray(y)
        
        # 样本数 < 特征数时，LassoLarsIC无法估计噪声方差，这里自己提供一个基础预估
        noise_variance = None
        if X.shape[0] <= X.shape[1]:
            noise_variance = np.var(y) * 0.1 # 简单预估一个经验噪声方差
            self.model_.noise_variance = noise_variance
            
        # 使用LARS算法探索正则化路径，并根据MDL(BIC)准则定位最佳模型复杂度
        self.model_.fit(X, y)
        
        # 获取非零系数的特征索引
        self.selected_indices_ = np.where(self.model_.coef_ != 0)[0]
        
        if len(self.selected_indices_) == 0:
            # 如果MDL惩罚过大导致没有特征被选中，则保留相关性最高的1个特征（防止全空）
            corrs = np.abs(np.corrcoef(X.T, y)[0, 1:])
            self.selected_indices_ = [np.argmax(corrs)]
            
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.selected_indices_]

class MDLRegressionModel(BaseEstimator, RegressorMixin):
    """
    基于MDL降维特征的回归模型，用于高维描述符情况下的产率预测。
    包含数据标准化、MDL特征选择和最终预测的完整Pipeline。
    """
    def __init__(self, base_estimator=None):
        if base_estimator is None:
            self.base_estimator = Ridge(alpha=1.0)
        else:
            self.base_estimator = base_estimator
            
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mdl_selector', MDLFeatureSelector(criterion='bic')), 
            ('regressor', self.base_estimator)
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)
        
    def get_selected_features_count(self):
        return len(self.pipeline.named_steps['mdl_selector'].selected_indices_)

if __name__ == "__main__":
    # === 示例使用流程 ===
    print("="*50)
    print("基于MDL（最小描述长度）高维描述符选择与产率预测模型")
    print("="*50)
    
    # 1. 模拟高维生化描述符数据 (例如：100个样本，500维描述符)
    np.random.seed(42)
    n_samples, n_features = 100, 500
    X_dummy = np.random.randn(n_samples, n_features)
    
    # 模拟真实产率 (只有其中5个描述符真正起作用)
    true_important_features = [12, 55, 128, 256, 401]
    weights = np.array([5.5, -3.2, 8.1, 2.4, -6.7])
    # 产率 y = w*x + 噪音
    y_dummy = np.dot(X_dummy[:, true_important_features], weights) + np.random.randn(n_samples) * 0.5
    
    print(f"数据维度: {X_dummy.shape}, 真实有效特征数: {len(true_important_features)}")
    
    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42)
    
    # 3. 初始化并训练MDL模型
    mdl_model = MDLRegressionModel()
    mdl_model.fit(X_train, y_train)
    
    # 4. 预测与评估
    y_pred = mdl_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # 获取选择的特征索引
    selector = mdl_model.pipeline.named_steps['mdl_selector']
    
    print("\n[模型评估结果]")
    print(f"MDL保留的描述符数量: {mdl_model.get_selected_features_count()} / {n_features}")
    print(f"实际保留的特征索引: {selector.selected_indices_}")
    print(f"测试集 R2 Score: {r2:.4f}")
    print(f"测试集 RMSE: {rmse:.4f}")
    print("="*50)
    
    # 将最佳测试结果保存到 plots 文件夹
    import os
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        'True_Yield': y_test,
        'Predicted_Yield': y_pred
    })
    
    output_path = os.path.join(output_dir, 'mdl_test_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"测试集预测结果已保存至: {output_path}")

