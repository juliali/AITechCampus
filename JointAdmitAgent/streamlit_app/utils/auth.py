import hashlib
import re
import secrets
import streamlit as st
from streamlit.components.v1 import html as st_html
from utils.db import get_connection, init_db
from utils.logger import log_action

SALT = "sino_foreign_auth_2024"
_COOKIE_NAME = "sfju_session_token"
_COOKIE_MAX_AGE = 7 * 24 * 3600


def _set_cookie(token: str):
    st_html(
        f'<script>document.cookie="{_COOKIE_NAME}={token}; path=/; max-age={_COOKIE_MAX_AGE}; SameSite=Lax";</script>',
        height=0,
    )


def _delete_cookie():
    st_html(
        f'<script>document.cookie="{_COOKIE_NAME}=; path=/; max-age=0";</script>',
        height=0,
    )


def _get_cookie_token() -> str | None:
    try:
        return st.context.cookies.get(_COOKIE_NAME)
    except Exception:
        return None


def hash_password(password: str) -> str:
    return hashlib.sha256(f"{SALT}{password}".encode()).hexdigest()


def _create_session(user_id: int) -> str:
    token = secrets.token_hex(32)
    conn = get_connection()
    conn.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, user_id))
    conn.commit()
    conn.close()
    return token


def _set_session_state(user):
    st.session_state.authenticated = True
    st.session_state.user_email = user["email"]
    st.session_state.user_id = user["id"]
    st.session_state.is_admin = bool(user["is_admin"])
    st.session_state.ai_access = bool(user["ai_access_approved"])
    st.session_state.nickname = user["nickname"] or ""


def restore_session():
    """Try to restore session from cookie, or session_state."""
    if st.session_state.get("authenticated"):
        return
    token = _get_cookie_token() or st.session_state.get("_session_token")
    if not token:
        return
    conn = get_connection()
    row = conn.execute("""
        SELECT u.* FROM sessions s JOIN users u ON s.user_id = u.id
        WHERE s.token = ?
    """, (token,)).fetchone()
    conn.close()
    if row:
        _set_session_state(row)
        st.session_state["_session_token"] = token


def register_user(email: str, password: str) -> tuple:
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "请输入有效的邮箱地址"
    if len(password) < 6:
        return False, "密码至少6位"
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email.lower().strip(), hash_password(password)),
        )
        conn.commit()
        return True, "注册成功，请登录"
    except Exception:
        return False, "该邮箱已被注册"
    finally:
        conn.close()


def login_user(email: str, password: str) -> tuple:
    conn = get_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
    ).fetchone()
    conn.close()
    if not user or user["password_hash"] != hash_password(password):
        return False, "邮箱或密码错误"
    _set_session_state(user)
    token = _create_session(user["id"])
    st.session_state["_session_token"] = token
    _set_cookie(token)
    log_action(user["id"], "login")
    return True, "登录成功"


def logout():
    token = _get_cookie_token() or st.session_state.get("_session_token")
    if token:
        conn = get_connection()
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
    _delete_cookie()
    for key in ["authenticated", "user_email", "user_id", "is_admin", "ai_access", "nickname", "_session_token"]:
        st.session_state.pop(key, None)


def show_auth_form():
    st.title("🎓 中外合办选校智能体")
    st.markdown("请登录或注册后使用本系统")
    tab_login, tab_register = st.tabs(["登录", "注册"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("邮箱", key="login_email")
            password = st.text_input("密码", type="password", key="login_pwd")
            submitted = st.form_submit_button("登录", use_container_width=True)
            if submitted:
                ok, msg = login_user(email, password)
                if ok:
                    st.rerun()
                else:
                    st.error(msg)

    with tab_register:
        with st.form("register_form"):
            email = st.text_input("邮箱", key="reg_email")
            password = st.text_input("密码", type="password", key="reg_pwd")
            password2 = st.text_input("确认密码", type="password", key="reg_pwd2")
            submitted = st.form_submit_button("注册", use_container_width=True)
            if submitted:
                if password != password2:
                    st.error("两次密码不一致")
                else:
                    ok, msg = register_user(email, password)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)


def check_login():
    init_db()
    if not st.session_state.get("authenticated"):
        show_auth_form()
        st.stop()


def check_ai_access():
    check_login()
    if st.session_state.get("user_api_key"):
        st.session_state.ai_access = True
        return
    conn = get_connection()
    user = conn.execute(
        "SELECT ai_access_approved, is_admin FROM users WHERE id = ?",
        (st.session_state.user_id,),
    ).fetchone()
    conn.close()
    if user and (user["ai_access_approved"] or user["is_admin"]):
        st.session_state.ai_access = True
        return
    st.session_state.ai_access = False
    _show_ai_request_form()
    st.stop()


def _show_ai_request_form():
    conn = get_connection()
    pending = conn.execute(
        "SELECT * FROM access_requests WHERE user_id = ? AND status = 'pending'",
        (st.session_state.user_id,),
    ).fetchone()
    conn.close()

    st.warning("AI咨询功能需要管理员审批或提供自己的API Key")

    tab_key, tab_apply = st.tabs(["输入API Key", "申请管理员审批"])

    with tab_key:
        st.markdown("输入您自己的智谱GLM API Key，即可直接使用AI咨询功能。")
        st.caption("获取方式：访问 [bigmodel.cn](https://bigmodel.cn) 注册并创建API Key")
        with st.form("user_api_key_form"):
            key = st.text_input("GLM API Key", type="password", placeholder="xxxxxxxx.xxxxxxxx")
            submitted = st.form_submit_button("保存并使用", use_container_width=True)
            if submitted:
                if key and len(key) > 10:
                    st.session_state.user_api_key = key
                    st.success("API Key 已保存，即将进入AI咨询")
                    st.rerun()
                else:
                    st.error("请输入有效的API Key")

    with tab_apply:
        if pending:
            st.info("您的申请已提交，请等待管理员审批")
        else:
            with st.form("ai_request_form"):
                reason = st.text_area("申请理由（选填）", placeholder="请简要说明您的使用需求")
                submitted = st.form_submit_button("申请AI咨询权限")
                if submitted:
                    conn = get_connection()
                    conn.execute(
                        "INSERT INTO access_requests (user_id, reason) VALUES (?, ?)",
                        (st.session_state.user_id, reason or None),
                    )
                    conn.commit()
                    conn.close()
                    st.success("申请已提交，请等待管理员审批")
                    st.rerun()


# --- Profile functions ---


def get_user_profile(user_id: int):
    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return user


def update_nickname(user_id: int, nickname: str):
    conn = get_connection()
    conn.execute("UPDATE users SET nickname = ? WHERE id = ?", (nickname.strip(), user_id))
    conn.commit()
    conn.close()
    st.session_state.nickname = nickname.strip()


def change_password(user_id: int, old_password: str, new_password: str) -> tuple:
    conn = get_connection()
    user = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user or user["password_hash"] != hash_password(old_password):
        conn.close()
        return False, "当前密码错误"
    if len(new_password) < 6:
        conn.close()
        return False, "新密码至少6位"
    conn.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (hash_password(new_password), user_id),
    )
    conn.commit()
    conn.close()
    return True, "密码修改成功"


# --- Admin functions ---


def get_all_users():
    conn = get_connection()
    users = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return users


def get_pending_requests():
    conn = get_connection()
    rows = conn.execute("""
        SELECT r.*, u.email FROM access_requests r
        JOIN users u ON r.user_id = u.id
        WHERE r.status = 'pending'
        ORDER BY r.created_at
    """).fetchall()
    conn.close()
    return rows


def approve_request(request_id: int, admin_id: int):
    conn = get_connection()
    req = conn.execute("SELECT user_id FROM access_requests WHERE id = ?", (request_id,)).fetchone()
    if req:
        conn.execute(
            "UPDATE access_requests SET status='approved', reviewed_by=?, reviewed_at=datetime('now') WHERE id=?",
            (admin_id, request_id),
        )
        conn.execute("UPDATE users SET ai_access_approved=1 WHERE id=?", (req["user_id"],))
        conn.commit()
    conn.close()


def deny_request(request_id: int, admin_id: int):
    conn = get_connection()
    conn.execute(
        "UPDATE access_requests SET status='denied', reviewed_by=?, reviewed_at=datetime('now') WHERE id=?",
        (admin_id, request_id),
    )
    conn.commit()
    conn.close()


def toggle_ai_access(user_id: int, current_value: bool):
    conn = get_connection()
    conn.execute(
        "UPDATE users SET ai_access_approved=? WHERE id=?",
        (0 if current_value else 1, user_id),
    )
    conn.commit()
    conn.close()


def delete_user(user_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM access_requests WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
