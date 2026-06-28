import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.auth import (
    get_all_users, get_pending_requests,
    approve_request, deny_request, toggle_ai_access, delete_user,
)
from utils.llm_client import get_available_backends, get_admin_selected_backend, set_admin_backend

st.title("🔐 管理面板")

tab1, tab2, tab3 = st.tabs(["待审批请求", "用户管理", "AI模型设置"])

with tab1:
    requests = get_pending_requests()
    if not requests:
        st.info("暂无待审批请求")
    else:
        for req in requests:
            with st.container(border=True):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{req['email']}**")
                    if req["reason"]:
                        st.caption(f"理由：{req['reason']}")
                    st.caption(f"申请时间：{req['created_at']}")
                with col2:
                    if st.button("批准", key=f"approve_{req['id']}"):
                        approve_request(req["id"], st.session_state.user_id)
                        st.rerun()
                with col3:
                    if st.button("拒绝", key=f"deny_{req['id']}"):
                        deny_request(req["id"], st.session_state.user_id)
                        st.rerun()

with tab2:
    users = get_all_users()
    st.markdown(f"共 **{len(users)}** 个注册用户")
    for user in users:
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                badge = " 👑管理员" if user["is_admin"] else ""
                ai_badge = " ✅AI" if user["ai_access_approved"] else ""
                st.markdown(f"**{user['email']}**{badge}{ai_badge}")
                st.caption(f"注册时间：{user['created_at']}")
            with col2:
                if not user["is_admin"]:
                    label = "撤销AI权限" if user["ai_access_approved"] else "授予AI权限"
                    if st.button(label, key=f"toggle_{user['id']}"):
                        toggle_ai_access(user["id"], bool(user["ai_access_approved"]))
                        st.rerun()
            with col3:
                if not user["is_admin"]:
                    if st.button("删除", key=f"del_{user['id']}"):
                        delete_user(user["id"])
                        st.rerun()

with tab3:
    st.subheader("AI模型选择")
    st.markdown("选择所有用户使用的默认AI模型后端")
    backends = get_available_backends()
    if backends:
        current = get_admin_selected_backend()
        idx = backends.index(current) if current in backends else 0
        selected = st.selectbox("当前使用的AI模型", backends, index=idx)
        if st.button("保存设置"):
            set_admin_backend(selected)
            st.success(f"已设置为：{selected}")
    else:
        st.warning("未配置任何AI后端，请先在 secrets.toml 中添加 API Key")
