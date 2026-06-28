import streamlit as st
import pandas as pd

from utils.data_loader import load_programs, programs_to_df

st.title("📋 项目对比")

programs = load_programs()
df = programs_to_df(programs)

names = df["项目名称"].tolist()
selected = st.multiselect("选择 2-4 个项目进行对比", names, max_selections=4)

if len(selected) >= 2:
    compare_programs = [p for p in programs if p.get("project_name", p.get("name", "")) in selected]

    compare_fields = [
        ("中方院校", "chinese_partner"),
        ("外方院校", "foreign_partner"),
        ("办学层次", "level_type"),
        ("专业", "major"),
        ("学制", "duration"),
        ("年学费", "tuition_per_year"),
        ("总学费", "tuition_total"),
        ("录取分数", "admission_score"),
        ("授课地点", "campus_location"),
        ("外方授课形式", "foreign_faculty_arrangement"),
        ("海外交换", "overseas_exchange"),
        ("交换时长", "overseas_duration"),
        ("交换目的地", "overseas_destination"),
        ("硕士直升", "masters_pathway"),
        ("实习安排", "internship_details"),
        ("就业指导", "career_guidance_details"),
    ]

    # Build comparison table
    table_data = {}
    for prog in compare_programs:
        name = prog.get("project_name", prog.get("name", ""))
        short_name = name[:25] + "..." if len(name) > 25 else name
        col_data = {}
        for label, field in compare_fields:
            val = prog.get(field)
            if val is True:
                val = "是"
            elif val is False:
                val = "否"
            elif val is None:
                val = "-"
            col_data[label] = str(val)[:100]

        # Add university tier info
        cp = prog.get("chinese_university_profile", {}) or {}
        fp = prog.get("foreign_university_profile", {}) or {}
        col_data["中方层级"] = cp.get("tier", "-") if isinstance(cp, dict) else "-"
        col_data["QS排名"] = str(fp.get("qs_2025_rank", "-")) if isinstance(fp, dict) else "-"
        col_data["外方国家"] = fp.get("country", "-") if isinstance(fp, dict) else "-"

        table_data[short_name] = col_data

    compare_df = pd.DataFrame(table_data)
    st.dataframe(compare_df, use_container_width=True, height=600)

elif len(selected) == 1:
    st.info("请至少选择 2 个项目进行对比")
else:
    st.info("从上方搜索框选择 2-4 个项目开始对比")
    st.markdown("**提示:** 可以先去「🔍 项目筛选」页找到感兴趣的项目名称，再来这里对比")
