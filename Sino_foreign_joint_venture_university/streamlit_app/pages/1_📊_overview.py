import streamlit as st
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import load_programs, programs_to_df, load_chinese_db, load_foreign_db, DATA_DIR

st.title("📊 中外合作办学总览")

programs = load_programs()
df = programs_to_df(programs)

if df.empty:
    st.error(f"数据加载失败！DATA_DIR={DATA_DIR}, 文件存在: {list(DATA_DIR.glob('*.json')) if DATA_DIR.exists() else 'DIR NOT FOUND'}")
    st.stop()
chinese_db = load_chinese_db()
foreign_db = load_foreign_db()

active_df = df[df["状态"] == "active"]

st.markdown(f"**活跃项目: {len(active_df)}** / 总计: {len(df)}")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("地域分布")
    region_counts = active_df["地区"].value_counts().reset_index()
    region_counts.columns = ["地区", "数量"]
    fig = px.bar(region_counts.head(15), x="地区", y="数量", color="数量",
                 color_continuous_scale="Blues")
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("办学层次分布")
    level_map = {"本科": 0, "硕士": 0, "博士": 0, "专科": 0, "其他": 0}
    for lv in active_df["层次"]:
        lv_str = str(lv)
        if "本科" in lv_str:
            level_map["本科"] += 1
        elif "硕士" in lv_str or "研究生" in lv_str:
            level_map["硕士"] += 1
        elif "博士" in lv_str:
            level_map["博士"] += 1
        elif "专科" in lv_str or "高职" in lv_str:
            level_map["专科"] += 1
        else:
            level_map["其他"] += 1
    level_df = pd.DataFrame(list(level_map.items()), columns=["层次", "数量"])
    level_df = level_df[level_df["数量"] > 0]
    fig = px.pie(level_df, values="数量", names="层次", hole=0.4)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("中方院校层级分布")
    tier_counts = active_df["中方层级"].replace("", "未标注").value_counts().reset_index()
    tier_counts.columns = ["层级", "数量"]
    fig = px.bar(tier_counts, x="层级", y="数量", color="层级")
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("外方院校QS排名分布")
    qs_bins = {"QS 1-50": 0, "QS 51-150": 0, "QS 151-500": 0, "QS 500+": 0, "未上榜/未知": 0}
    for qs in active_df["QS排名"]:
        if pd.isna(qs):
            qs_bins["未上榜/未知"] += 1
        elif qs <= 50:
            qs_bins["QS 1-50"] += 1
        elif qs <= 150:
            qs_bins["QS 51-150"] += 1
        elif qs <= 500:
            qs_bins["QS 151-500"] += 1
        else:
            qs_bins["QS 500+"] += 1
    qs_df = pd.DataFrame(list(qs_bins.items()), columns=["排名区间", "数量"])
    fig = px.bar(qs_df, x="排名区间", y="数量", color="排名区间",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("合作国家 Top 10")
country_counts = active_df["外方国家"].replace("", pd.NA).dropna().value_counts().head(10).reset_index()
country_counts.columns = ["国家", "数量"]
fig = px.bar(country_counts, x="数量", y="国家", orientation="h",
             color="数量", color_continuous_scale="Viridis")
fig.update_layout(showlegend=False, height=350, yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)
