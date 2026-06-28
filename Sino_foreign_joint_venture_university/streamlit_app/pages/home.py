import streamlit as st
import pandas as pd

from utils.data_loader import load_programs, programs_to_df

programs = load_programs()
df = programs_to_df(programs)

st.title("🎓 中外合作办学智能指南")
st.markdown("帮助考生和家长全面了解中外合作办学项目，做出明智选择")

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("项目总数", f"{len(programs)}")
with col2:
    regions = df["地区"].nunique()
    st.metric("覆盖省份", f"{regions}")
with col3:
    countries = df["外方国家"].replace("", pd.NA).dropna().nunique()
    st.metric("合作国家", f"{countries}")
with col4:
    top_count = len(df[df["中方层级"].str.contains("985", na=False)])
    st.metric("985院校项目", f"{top_count}")

st.divider()

st.markdown("""
### 使用指南

- **📊 总览** — 查看中外合办的整体态势和数据分布
- **🔍 项目筛选** — 按地区、层次、专业、学费等条件精准查找
- **📋 项目对比** — 多个项目并排对比，一目了然
- **💬 AI咨询** — 输入你的条件，AI 为你个性化推荐

---

*数据来源：教育部中外合作办学监管信息平台 + 各项目官网招生简章*
""")
