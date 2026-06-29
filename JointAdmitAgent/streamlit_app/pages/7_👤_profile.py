import streamlit as st

from utils.auth import check_login, get_user_profile, update_nickname, change_password

check_login()

st.title("👤 个人设置")

user = get_user_profile(st.session_state.user_id)

st.markdown("### 账户信息")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**邮箱**: {user['email']}")
    st.markdown(f"**注册时间**: {user['created_at']}")
with col2:
    if user["is_admin"]:
        st.markdown("**角色**: 🔑 管理员")
    else:
        st.markdown("**角色**: 普通用户")
    if user["ai_access_approved"]:
        st.markdown("**AI咨询**: ✅ 已开通")
    else:
        st.markdown("**AI咨询**: ❌ 未开通")

st.divider()

st.markdown("### 修改昵称")
with st.form("nickname_form"):
    new_nickname = st.text_input(
        "昵称",
        value=st.session_state.get("nickname", ""),
        placeholder="输入你的昵称",
        max_chars=20,
    )
    submitted = st.form_submit_button("保存昵称", use_container_width=True)
    if submitted:
        update_nickname(st.session_state.user_id, new_nickname)
        st.success("昵称已更新")
        st.rerun()

st.divider()

st.markdown("### 修改密码")
with st.form("password_form"):
    old_pwd = st.text_input("当前密码", type="password")
    new_pwd = st.text_input("新密码", type="password")
    new_pwd2 = st.text_input("确认新密码", type="password")
    submitted = st.form_submit_button("修改密码", use_container_width=True)
    if submitted:
        if new_pwd != new_pwd2:
            st.error("两次输入的新密码不一致")
        elif not old_pwd:
            st.error("请输入当前密码")
        else:
            ok, msg = change_password(st.session_state.user_id, old_pwd, new_pwd)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
