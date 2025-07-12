import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# 中文字
plt.rcParams['font.family'] = 'Microsoft JhengHei'
# 讀資料
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# 檢查欄位型別與缺失值
print("\nTrain info:")
print(train.info())

print("\nTest info:")
print(test.info())

print("\nTrain 缺失值統計：")
print(train.isnull().sum())

# ===================================
# EDA分析
# 分析目標欄位 Calories 分佈
plt.figure(figsize=(12,5))
sns.histplot(train["Calories"], kde=True, bins=50)
plt.title("Calories 分佈圖（原始）")
plt.xlabel("Calories")
# plt.show()
'''
原始分佈 發現強烈右偏（右長尾），大多數樣本集中在 0–50，少部分超過 200–300。
數值差距大，分佈不均，很容易導致模型預測偏移
預測時容易低估高熱量樣本、導致過擬合常見值
要轉換 => 嚴重偏態，會影響回歸模型穩定性與準確率
'''

# log1p 轉換
plt.figure(figsize=(12,5))
sns.histplot(np.log1p(train["Calories"]), kde=True, bins=50, color='orange')
plt.title("Calories 分佈圖（log1p）")
plt.xlabel("log1p(Calories)")
# plt.show()
print("原始 Skewness：", train['Calories'].skew())
print("log1p Skewness：", np.log1p(train['Calories']).skew())
'''
明顯偏態減少，接近常態分布（鐘形曲線）。右邊略有長尾，但改善效果顯著
解決偏態問題，使資料更符合線性模型與 RMSLE 的假設
分布變得更對稱，較容易建立回歸模型
'''
# # =============================================================
#EDA：變數 vs Calories
#各特徵對 Calories 的關係
#correlation heatmap 只能顯示「線性相關性」的強弱，卻無法揭露「關係的形狀」與「資料分布特性」
#如果變數與 Calories 是非線性關係（例如對數、二次曲線、S型），它的 corr 可能接近 0，但實際上仍有很強的預測力
sns.set(style="whitegrid")
plt.figure(figsize=(12, 10))

# 連續變數 vs Calories
for i, col in enumerate(['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']):
    plt.subplot(2, 3, i+1)
    sns.scatterplot(data=train, x=col, y='Calories', alpha=0.3)
    plt.title(f'{col} vs Calories')
plt.tight_layout()
plt.show()
'''
| 特徵                  | 與 Calories 關係      | 解讀                   
| ----------------------| ---------------------| -------------------- 
| Duration              | 線性強相關            | 運動時間愈久，消耗愈多          
| Heart_Rate           | 正相關                | 心跳率越高，代表運動強度高，消耗卡路里多 
| Body_Temp            | 也有正相關            | 運動時體溫升高與熱量消耗連動       
| Age / Height / Weight | 幾乎沒有明顯相關性     | 可用作參考或交互特徵但影響有限      

'''
# 類別型資料（如 Sex）無法用 corr() 處理
# 觀察 Sex 與 Calories 的關係
plt.figure(figsize=(6, 5))
sns.boxplot(data=train, x='Sex', y='Calories')
plt.title('Sex vs Calories')
#plt.show()

# 相關係數熱力圖
plt.figure(figsize=(10, 6))
sns.heatmap(train.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
#plt.show()
#===================================================
# 相關係數（熱力圖 + log1p 對應）
# 加入 log1p(Calories) 欄位
train['Calories_log'] = np.log1p(train['Calories'])
# 計算與 log1p(Calories) 的相關性
corr_matrix = train.corr(numeric_only=True)
corr_with_logcal = corr_matrix["Calories_log"].sort_values(ascending=False)

print("與 log1p(Calories) 最相關的前 10 欄位：")
print(corr_with_logcal[1:11])  # 跳過 self-correlation

# 繪製條狀圖
plt.figure(figsize=(8, 5))
sns.barplot(x=corr_with_logcal[1:11].values, y=corr_with_logcal[1:11].index)
plt.title("Top 10 Features Correlated with log1p(Calories)")
plt.xlabel("Correlation")
plt.tight_layout()
plt.show()
#===================================
# 資料清理與特徵工程
# 複製資料備份 + Log 轉換目標欄位
df_train = train.copy()
df_test = test.copy()
# 使用 log1p 轉換 Calories，避免偏態影響

df_train['Calories_log'] = np.log1p(df_train['Calories'])

# 類別欄位 One-hot encoding
# Sex 做 One-hot encoding（drop_first=True 避免多重共線性）
df_train = pd.get_dummies(df_train, columns=['Sex'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Sex'], drop_first=True)

#  建立交互特徵（Feature Engineering）
for df in [df_train, df_test]:
    df['Duration*Heart'] = df['Duration'] * df['Heart_Rate']
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Heart_per_Min'] = df['Heart_Rate'] / df['Duration']
    df['Temp_per_Min'] = (df['Body_Temp'] - 36.5) / df['Duration']
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,18,30,50,70,100], labels=[0,1,2,3,4])
    df['BMIGroup'] = pd.cut(df['BMI'], bins=[0,18.5,25,30,40], labels=[0,1,2,3])

df_train = pd.get_dummies(df_train, columns=['AgeGroup', 'BMIGroup'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['AgeGroup', 'BMIGroup'], drop_first=True)
# 特徵列表準備
# 要使用的特徵欄位（不包含 id / 原始 Calories）
features = [col for col in df_train.columns if col not in ['id', 'Calories', 'Calories_log']]
X_train = df_train[features]
y_train = df_train['Calories_log']
X_test = df_test[features]
# 數值標準化（StandardScaler）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#===============================================
# 匯入模型與 RMSLE 評估函數
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error,  make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import xgboost as xgb
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor

# RMSLE Scorer 函數
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)
# 設定模型與交叉驗證流程
# 定義 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.001),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}
print("模型比較RMSLE")
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring=rmsle_scorer)
    print(f"{name:<15} 平均 RMSLE = {-scores.mean():.5f}（± {scores.std():.5f}）")
'''
以上是使用交叉編譯 去尋找表現最好的模型
'''
# XGBoost 調參 GridSearchCV
xgb_model = xgb.XGBRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=rmsle_scorer,
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)
final_model = grid_search.best_estimator_
#===============================================
# 建立第一層模型（基模型）

# LightGBM baseline
lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    random_state=42
)
lgb_model.fit(X_train_scaled, y_train)

# RandomForest baseline
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Ridge baseline
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# 第一層模型（回復為 4 個）
base_models = [
    ("xgb", final_model),
    ("lgb", lgb_model),
    ("rf", rf_model),
    ("ridge", ridge_model)
]

#================================================
# KFold 交叉預測 → 產生 stacked features
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

stacked_train = np.zeros((X_train_scaled.shape[0], len(base_models)))
stacked_test = np.zeros((X_test_scaled.shape[0], len(base_models)))
# 對每個模型做KFold預測
for i, (name, model) in enumerate(base_models):
    test_fold_preds = np.zeros((X_test_scaled.shape[0], n_folds))
    oof_preds = np.zeros(X_train_scaled.shape[0])
    for j, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)
        test_fold_preds[:, j] = model.predict(X_test_scaled)
#儲存每個模型的oof預測與平均test預測
    stacked_train[:, i] = oof_preds
    stacked_test[:, i] = test_fold_preds.mean(axis=1)
#=========================================================
# 第二層模型（Meta-Model）訓練與預測 使用 Ridge
# 將 base model 預測值 + 原始標準化特徵一起輸入 Ridge
meta_X_train = np.hstack([stacked_train, X_train_scaled])
meta_X_test = np.hstack([stacked_test, X_test_scaled])
# 使用Ridge
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X_train, y_train)
#預測log
pred_log_meta = meta_model.predict(meta_X_test)
#還原預設值
pred_calories = np.clip(np.expm1(pred_log_meta), 0, None)

# 輸出檔案
submission = pd.DataFrame({
    'id': df_test['id'],
    'Calories': pred_calories
})
submission.to_csv('submission_stacking.csv', index=False)
print("已輸出融合版 stacking 的 submission_stacking.csv，可上傳 Kaggle！")