import json
from pathlib import Path
import streamlit as st
import pandas as pd

# Resolve DATA_DIR: try parent of streamlit_app first, then streamlit_app/data
_THIS_FILE = Path(__file__).resolve()
_APP_DIR = _THIS_FILE.parent.parent
_CANDIDATE_DIRS = [
    _APP_DIR.parent / "data",
    _APP_DIR / "data",
]
DATA_DIR = next((d for d in _CANDIDATE_DIRS if (d / "programs.json").exists() or (d / "programs_enriched.json").exists()), _CANDIDATE_DIRS[0])


@st.cache_data(ttl=3600)
def load_programs():
    """加载项目数据（取数量最多的版本）"""
    candidates = []
    for name in ["programs_enriched.json", "programs.json"]:
        path = DATA_DIR / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                candidates.append(data)
    if not candidates:
        return []
    return max(candidates, key=len)


@st.cache_data(ttl=3600)
def load_chinese_db():
    path = DATA_DIR / "chinese_universities.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600)
def load_foreign_db():
    path = DATA_DIR / "foreign_universities.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def programs_to_df(programs):
    """将项目列表转为 DataFrame，提取关键列"""
    rows = []
    for p in programs:
        cp = p.get("chinese_university_profile", {}) or {}
        fp = p.get("foreign_university_profile", {}) or {}
        qs = fp.get("qs_2025_rank") if isinstance(fp, dict) else None

        rows.append({
            "项目名称": p.get("project_name", p.get("name", "")),
            "中方院校": p.get("chinese_partner", ""),
            "外方院校": p.get("foreign_partner", ""),
            "地区": p.get("region", ""),
            "层次": p.get("level_type", ""),
            "专业": p.get("major", ""),
            "学制": p.get("duration", ""),
            "学费/年": p.get("tuition_per_year", ""),
            "中方层级": cp.get("tier", "") if isinstance(cp, dict) else "",
            "QS排名": qs if isinstance(qs, int) else None,
            "外方国家": fp.get("country", "") if isinstance(fp, dict) else "",
            "海外交换": "是" if p.get("overseas_exchange") else ("否" if p.get("overseas_exchange") is False else ""),
            "状态": p.get("status", ""),
            "detail_url": p.get("detail_url", ""),
        })
    return pd.DataFrame(rows)
