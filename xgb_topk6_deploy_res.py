import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# =========================
# 0) 固定随机种子与线程
# =========================
random_seed = 42
np.random.seed(random_seed)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# =========================
# 1) 路径设置
# =========================
BASE_DIR = Path(r"D:\4.毕业论文相关\数据重整-12-31\4月\部署代码-XGB")
BASE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = Path(r"D:\4.毕业论文相关\数据重整-12-31\TRAIN\一些前期结果\train_data.xlsx")

# 直接导出到当前目录，不再放 deploy_resources 子文件夹
OUT_PATH = BASE_DIR / "xgb_topk6_deploy_res.joblib"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TARGET_COL = "EPDSLL"
ID_COL = "id_仅标识"

# =========================
# 2) 固定 Top-K=6 变量
# =========================
selected_vars_top6 = ["EPDSA", "Anxiety", "Insomnia", "GA", "PG", "DBP"]

# 固定最优阈值
best_threshold_top6 = 0.376241386

# =========================
# 3) 变量类型定义
# =========================
float_cols = ["BMI"]

num_cols = [
    "age", "GA", "SBP", "DBP", "HR", "menstrual",
    "EPDSA", "Insomnia", "Capital", "Anxiety"
]

cat_cols = [
    "PG", "parity", "CM", "Abnormity", "Registration", "Occupation", "OS", "COelderly",
    "personality", "Suicidal", "health", "Smoking", "Alcohol", "Caffeine",
    "reactions", "Fear", "COVID19"
]

ord_cols = ["gravidity", "Educational", "HMI", "Social"]

# =========================
# 4) 基础清洗
# =========================
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in (float_cols + num_cols + ord_cols + cat_cols + [TARGET_COL]):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# =========================
# 5) 检查训练数据文件
# =========================
if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"找不到训练数据文件：{TRAIN_PATH}")

print(f"正在读取训练数据：{TRAIN_PATH}")

# =========================
# 6) 读取训练集
# =========================
train_df = pd.read_excel(TRAIN_PATH)
train_df = basic_clean(train_df)

if TARGET_COL not in train_df.columns:
    raise ValueError(f"训练数据中缺少目标变量列：{TARGET_COL}")

train_df = train_df.dropna(subset=[TARGET_COL]).copy()

if train_df.empty:
    raise ValueError("训练数据在删除目标变量缺失值后为空，无法继续训练。")

train_df[TARGET_COL] = train_df[TARGET_COL].astype(int)

drop_cols = [TARGET_COL]
if ID_COL in train_df.columns:
    drop_cols.append(ID_COL)

X_train_df = train_df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
y_train = train_df[TARGET_COL].reset_index(drop=True)

missing_train_vars = [c for c in selected_vars_top6 if c not in X_train_df.columns]
if len(missing_train_vars) > 0:
    raise ValueError(f"train_data 中缺少以下入模变量: {missing_train_vars}")

X_train_sel = X_train_df[selected_vars_top6].copy()

# =========================
# 7) 构建预处理器
# =========================
sel_num = [c for c in (float_cols + num_cols) if c in selected_vars_top6]
sel_cat = [c for c in cat_cols if c in selected_vars_top6]
sel_ord = [c for c in ord_cols if c in selected_vars_top6]

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

ord_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocess_top6 = ColumnTransformer(
    transformers=[
        ("num", num_pipe, sel_num),
        ("cat", cat_pipe, sel_cat),
        ("ord", ord_pipe, sel_ord),
    ],
    remainder="drop"
)

# =========================
# 8) 固定参数最终模型
# =========================
final_top6_pipe = Pipeline(steps=[
    ("prep", preprocess_top6),
    ("clf", XGBClassifier(
        random_state=random_seed,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        learning_rate=0.01,
        max_depth=3,
        n_estimators=170
    ))
])

# =========================
# 9) 拟合最终模型
# =========================
print("开始拟合最终模型...")
final_top6_pipe.fit(X_train_sel, y_train)

# =========================
# 10) 导出部署资源
# =========================
deploy_res = {
    "best_model": final_top6_pipe,
    "youden_threshold": best_threshold_top6,
    "final_top6_vars": selected_vars_top6,
    "model_name": "XGBoost_Final_TopK6_Fixed"
}

joblib.dump(deploy_res, OUT_PATH)

print("=====================================")
print("部署资源已保存：", OUT_PATH)
print("特征顺序：", selected_vars_top6)
print("Youden 阈值：", best_threshold_top6)
print("=====================================")