import streamlit as st
import importlib
import importlib.util
import sys
from pathlib import Path

_APP_DIR = Path(__file__).parent


def _ensure_utils():
    """Register the local utils package for Python 3.14 compatibility."""
    utils_dir = _APP_DIR / "utils"
    if "utils" in sys.modules and hasattr(sys.modules["utils"], "__file__"):
        if sys.modules["utils"].__file__ and str(utils_dir) in sys.modules["utils"].__file__:
            return
    spec = importlib.util.spec_from_file_location(
        "utils", utils_dir / "__init__.py",
        submodule_search_locations=[str(utils_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["utils"] = mod
    spec.loader.exec_module(mod)


_ensure_utils()

from utils.db import init_db
from utils.auth import restore_session

st.set_page_config(
    page_title="中外合办选校智能体 | JointAdmit Agent",
    page_icon="🎓",
    layout="wide",
)

init_db()
restore_session()

if st.session_state.get("authenticated"):
    pages = [
        st.Page("pages/home.py", title="首页", icon="🎓", default=True),
        st.Page("pages/1_📊_overview.py", title="总览", icon="📊"),
        st.Page("pages/2_🔍_filter.py", title="项目筛选", icon="🔍"),
        st.Page("pages/3_📋_compare.py", title="项目对比", icon="📋"),
        st.Page("pages/4_💬_ai_consult.py", title="AI咨询", icon="💬"),
        st.Page("pages/7_👤_profile.py", title="个人设置", icon="👤"),
    ]
    if st.session_state.get("is_admin"):
        pages.append(st.Page("pages/5_🔐_admin.py", title="管理面板", icon="🔐"))
        pages.append(st.Page("pages/6_📈_dashboard.py", title="使用统计", icon="📈"))

    display_name = st.session_state.get("nickname") or st.session_state.get("user_email", "")
    st.sidebar.markdown(f"👤 {display_name}")
    if st.sidebar.button("退出登录"):
        from utils.auth import logout
        logout()
        st.rerun()
else:
    pages = [
        st.Page("pages/login.py", title="登录", icon="🔑", default=True),
    ]

nav = st.navigation(pages)
nav.run()
