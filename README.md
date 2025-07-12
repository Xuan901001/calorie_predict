# 🥗 Calorie Expenditure Prediction - Kaggle Playground Series S5E5

本專案為參加 Kaggle Playground S5E5 比賽所建立的完整機器學習流程，目標為預測使用者的熱量消耗量（Calories Burned），結合資料前處理、特徵工程、模型比較與集成學習（Stacking Ensemble），達到更佳的預測表現。

- **完整資料處理流程**：缺失值處理、偏態修正、標準化、特徵擴增
- **多模型比較與調參**：支援 RandomForest、XGBoost、LightGBM、Ridge 等
- **Stacking Ensemble 融合模型**：結合多個基模型，最終以 Ridge 進行融合
- **評估指標 RMSLE**：採用 Root Mean Squared Logarithmic Error 作為主評估指標

## 專案流程簡述

1. **資料前處理**
   - 缺失值檢查與填補
   - 數值標準化（StandardScaler）
   - log1p 轉換修正偏態分布
2. **特徵工程**
   - 建立交互特徵與非線性特徵（如平方、對數等）
   - 類別特徵 one-hot 編碼
3. **模型訓練與比較**
   - 基礎回歸模型（Linear, Ridge, RF, XGB, LGB）
   - GridSearchCV 調參找最佳參數
4. **Stacking Ensemble**
   - 第一層基模型：RandomForest, XGBoost, LightGBM, Ridge
   - 第二層模型：Ridge Regression 融合預測結果

