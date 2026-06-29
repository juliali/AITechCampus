import streamlit as st
import pandas as pd
import json

from utils.data_loader import load_programs, programs_to_df
from utils.filters import filter_programs

st.title("🔍 项目筛选")

programs = load_programs()
df = programs_to_df(programs)

# Sidebar filters
st.sidebar.header("筛选条件")

regions = st.sidebar.multiselect("地区", sorted(df["地区"].unique()), default=[])

level_options = ["本科", "硕士", "博士"]
levels = st.sidebar.multiselect("办学层次", level_options, default=[])

tier_options = sorted(df["中方层级"].replace("", None).dropna().unique())
chinese_tiers = st.sidebar.multiselect("中方院校层级", tier_options, default=[])

qs_max = st.sidebar.slider("外方QS排名上限", 0, 1000, 0, step=50,
                           help="0 表示不限制")

keyword = st.sidebar.text_input("专业关键词", placeholder="如：计算机、金融、医学")

overseas_only = st.sidebar.checkbox("仅看有海外交换的项目")
active_only = st.sidebar.checkbox("仅看在招项目", value=True)

# Apply filters
filters = {}
if regions:
    filters["regions"] = regions
if levels:
    filters["levels"] = levels
if chinese_tiers:
    filters["chinese_tiers"] = chinese_tiers
if qs_max > 0:
    filters["qs_max"] = qs_max
if keyword:
    filters["keyword"] = keyword
if overseas_only:
    filters["overseas_only"] = True
if active_only:
    filters["active_only"] = True

filtered = filter_programs(df, filters)
st.markdown(f"**找到 {len(filtered)} 个项目**")

# Display results
display_cols = ["项目名称", "中方院校", "外方院校", "地区", "层次", "专业", "学费/年", "中方层级", "QS排名", "海外交换"]
st.dataframe(
    filtered[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=400,
)

# Detail expander
st.divider()

if len(filtered) > 0:
    selected_name = st.selectbox(
        "选择项目查看详情",
        filtered["项目名称"].tolist(),
    )

    if selected_name:
        prog = next((p for p in programs if p.get("project_name", p.get("name", "")) == selected_name), None)
        if prog:
            with st.expander("📋 项目详情", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### 基本信息")
                    st.write(f"**中方院校:** {prog.get('chinese_partner', '')}")
                    st.write(f"**外方院校:** {prog.get('foreign_partner', '')}")
                    st.write(f"**办学层次:** {prog.get('level_type', '')}")
                    st.write(f"**专业:** {prog.get('major', '')}")
                    st.write(f"**学制:** {prog.get('duration', '')}")
                    st.write(f"**招生方式:** {prog.get('admission_method', '')}")

                    st.markdown("##### 学费与住宿")
                    st.write(f"**年学费:** {prog.get('tuition_per_year') or '未知'}")
                    st.write(f"**总学费:** {prog.get('tuition_total') or '未知'}")
                    st.write(f"**录取分数:** {prog.get('admission_score') or '未知'}")
                    st.write(f"**住宿:** {prog.get('housing_conditions') or '未知'}")
                    st.write(f"**授课地点:** {prog.get('campus_location') or prog.get('address', '未知')}")

                with col2:
                    st.markdown("##### 师资与教学")
                    st.write(f"**外方授课:** {prog.get('foreign_faculty_arrangement') or '未知'}")
                    st.write(f"**师资构成:** {prog.get('faculty_breakdown') or '未知'}")

                    st.markdown("##### 海外交流与升学")
                    st.write(f"**海外交换:** {prog.get('overseas_exchange') or '未知'}")
                    st.write(f"**交换时长:** {prog.get('overseas_duration') or '未知'}")
                    st.write(f"**交换目的地:** {prog.get('overseas_destination') or '未知'}")
                    st.write(f"**硕士直升:** {prog.get('masters_pathway') or '未知'}")

                    st.markdown("##### 就业与实习")
                    st.write(f"**实习安排:** {prog.get('internship_details') or '未知'}")
                    st.write(f"**就业指导:** {prog.get('career_guidance_details') or '未知'}")

                # University profiles
                cp = prog.get("chinese_university_profile")
                fp = prog.get("foreign_university_profile")
                if cp or fp:
                    st.markdown("##### 院校画像")
                    c1, c2 = st.columns(2)
                    with c1:
                        if cp and isinstance(cp, dict):
                            st.write(f"**中方层级:** {cp.get('tier', '')}")
                            st.write(f"**类型:** {cp.get('type', '')}")
                    with c2:
                        if fp and isinstance(fp, dict):
                            st.write(f"**QS排名:** {fp.get('qs_2025_rank', '未上榜')}")
                            st.write(f"**国家:** {fp.get('country', '')}")
                            st.write(f"**优势学科:** {', '.join(fp.get('strengths', []))}")
                            if fp.get("alliance"):
                                st.write(f"**联盟:** {fp['alliance']}")
