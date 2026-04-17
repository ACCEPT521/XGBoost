from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="孕晚期抑郁症状预测模型",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "xgb_topk6_deploy_res.joblib"

# -----------------------------
# Constants
# -----------------------------
PG_MAP = {
    0: "计划内",
    1: "计划外"
}

# -----------------------------
# Load model resources
# -----------------------------
@st.cache_resource
def load_deploy_resources(path: Path):
    res = joblib.load(path)

    required = ["best_model", "youden_threshold", "final_top6_vars"]
    missing = [k for k in required if k not in res]
    if missing:
        raise ValueError(f"Deploy resource missing key(s): {missing}")

    return res

# -----------------------------
# Style
# -----------------------------
st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .block-card {
        padding: 1rem 1.1rem;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #f7f9fc;
        border: 1px solid #e8edf5;
        padding: 12px;
        border-radius: 14px;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------
st.title("孕晚期抑郁症状预测模型")
st.caption("基于 XGBoost Final TopK=6 的部署版预测页面")

st.markdown(
    """
    <div class="block-card">
    请输入以下 6 项信息，点击 <b>开始预测</b>，系统将输出孕晚期抑郁阳性（EPDSLL=1）的预测概率，
    并根据训练集确定的 Youden 阈值给出风险提示。
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Model loading
# -----------------------------
with st.expander("模型信息"):
    st.write("应用目录：", str(APP_DIR))
    st.write("模型文件：", str(MODEL_PATH))
    st.write("模型文件是否存在：", MODEL_PATH.exists())

if not MODEL_PATH.exists():
    st.error(f"找不到模型文件：{MODEL_PATH}")
    st.info("请先运行 build_deploy_resources.py 生成 xgb_topk6_deploy_res.joblib，并确保它与 app.py 位于同一目录。")
    st.stop()

try:
    res = load_deploy_resources(MODEL_PATH)
except Exception as e:
    st.error(f"模型文件加载失败：{e}")
    st.stop()

best_model = res["best_model"]
thr_star = float(res["youden_threshold"])
final_top6_vars = res["final_top6_vars"]
model_name = res.get("model_name", "XGBoost_Final_TopK6_Fixed")

expected_vars = ["EPDSA", "Anxiety", "Insomnia", "GA", "PG", "DBP"]
if list(final_top6_vars) != expected_vars:
    st.warning(
        f"当前模型中的特征顺序为：{final_top6_vars}；"
        f"页面预期顺序为：{expected_vars}。请确认训练与部署资源一致。"
    )

with st.expander("部署参数核对"):
    st.write("模型名称：", model_name)
    st.write("特征顺序：", final_top6_vars)
    st.write(f"Youden 阈值：{thr_star:.6f}")

# -----------------------------
# Input area
# -----------------------------
st.subheader("输入变量")

col1, col2 = st.columns(2)

with col1:
    EPDSA = st.number_input(
        "孕早期 EPDS 分数（EPDSA）",
        min_value=0.0,
        value=10.0,
        step=1.0
    )
    Anxiety = st.number_input(
        "妊娠焦虑分数（Anxiety）",
        min_value=0.0,
        value=1.0,
        step=1.0
    )
    Insomnia = st.number_input(
        "睡眠情况分数（Insomnia）",
        min_value=0.0,
        value=1.0,
        step=1.0
    )

with col2:
    GA = st.number_input(
        "孕周（GA）",
        min_value=0.0,
        value=8.0,
        step=1.0,
        format="%.1f"
    )
    DBP = st.number_input(
        "舒张压（DBP）",
        min_value=40.0,
        value=70.0,
        step=1.0
    )
    pg_code = st.selectbox(
        "是否计划怀孕（PG）",
        options=list(PG_MAP.keys()),
        format_func=lambda x: f"{x} - {PG_MAP[x]}",
        index=0
    )

# -----------------------------
# Build input dataframe
# -----------------------------
input_dict = {
    "EPDSA": float(EPDSA),
    "Anxiety": float(Anxiety),
    "Insomnia": float(Insomnia),
    "GA": float(GA),
    "PG": int(pg_code),
    "DBP": float(DBP),
}

try:
    x = pd.DataFrame([input_dict], columns=final_top6_vars)
except Exception as e:
    st.error(f"构造输入数据失败：{e}")
    st.stop()

x["PG"] = x["PG"].astype(int)

# -----------------------------
# Predict
# -----------------------------
st.divider()
predict_btn = st.button("开始预测", type="primary", use_container_width=True)

if predict_btn:
    try:
        proba = float(best_model.predict_proba(x)[0, 1])
    except Exception as e:
        st.error(f"预测失败：{e}")
        st.stop()

    risk_label = "高风险" if proba >= thr_star else "低风险"

    c1, c2 = st.columns(2)
    with c1:
        st.metric("预测阳性概率", f"{proba * 100:.2f}%")
    with c2:
        st.metric("风险判定", risk_label)

    if proba >= thr_star:
        st.error(f"结果提示：预测概率 ≥ Youden 阈值（{thr_star:.6f}），判定为高风险。")
    else:
        st.success(f"结果提示：预测概率 < Youden 阈值（{thr_star:.6f}），判定为低风险。")

    with st.expander("查看传入模型的编码值"):
        st.dataframe(x, use_container_width=True)

st.markdown(
    """
    <div class="small-note">
    本页面仅用于模型部署展示。变量设置与训练代码保持一致：PG 为分类变量，
    EPDSA、Anxiety、Insomnia、GA、DBP 为数值变量。
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("运行方式：streamlit run app.py")