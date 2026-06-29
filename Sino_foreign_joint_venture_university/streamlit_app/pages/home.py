import streamlit as st
import pandas as pd

from utils.data_loader import load_programs, programs_to_df

programs = load_programs()
df = programs_to_df(programs)

st.title("🎓 中外合办选校智能体")
st.caption("JointAdmit Agent — 智能中外合作办学志愿填报助手")

st.divider()

st.markdown("""
#### 什么是中外合作办学？

中外合作办学是经教育部批准，由中国高校与境外高校联合开设的本科或研究生学位项目。
毕业后可同时获得中外双方学位，学历受国家认可。近年来这类项目快速扩张，已成为高考志愿填报中**兼顾国内升学与国际视野**的热门选择。

#### 为什么要关注？

- 🎯 **分数性价比高** — 部分 985/211 院校的中外合办专业录取分比普通批低 20-50 分
- 🌍 **不出国也能拿海外文凭** — 4+0 模式全程在国内就读，毕业获海外大学学位
- 💰 **费用远低于留学** — 学费通常 3-12 万/年，远低于出国留学的总花费
- 📈 **近几年新增项目多** — 很多项目 2020 年后才获批，家长和考生了解有限

#### 本站能帮你什么？

本站汇总了教育部批准的 **{len(programs)} 个中外合作办学项目**，覆盖全国多个省份，
并通过 AI 深度分析补充了学费、录取分数、师资、住宿、海外交换等官方平台没有的关键信息。

> **已收录地区**：北京、上海、天津、重庆、江苏、浙江、广东、山东、福建、湖北、湖南、四川、辽宁、黑龙江、吉林、安徽、江西、山西（共 18 个省市）。
> 其他省份数据正在陆续采集中，暂未覆盖的地区不代表没有中外合作办学项目。
""")

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
- **💬 AI咨询** — 输入你的条件（分数、专业、预算、地区偏好），AI 为你个性化推荐
- **👤 个人设置** — 管理你的账号信息

---

*数据来源：教育部中外合作办学监管信息平台 + 各项目官网招生简章*
*适用场景：2026 年高考志愿填报参考。数据持续更新中，以各校官方最新招生简章为准。*
""")
