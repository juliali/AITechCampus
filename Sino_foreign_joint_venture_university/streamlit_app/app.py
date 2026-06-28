import streamlit as st
import os
import sys
from pathlib import Path

# Change working directory so Python can find local 'utils' package
os.chdir(Path(__file__).parent)
if "" not in sys.path:
    sys.path.insert(0, "")

from utils.db import init_db
from utils.auth import restore_session

st.set_page_config(
    page_title="中外合作办学指南",
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
    ]
    if st.session_state.get("is_admin"):
        pages.append(st.Page("pages/5_🔐_admin.py", title="管理面板", icon="🔐"))
        pages.append(st.Page("pages/6_📈_dashboard.py", title="使用统计", icon="📈"))

    st.sidebar.markdown(f"👤 {st.session_state.get('user_email', '')}")
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
