import streamlit as st
import pandas as pd


from utils.logger import get_user_logs, get_ai_conversations, get_usage_stats
from utils.db import get_connection

st.title("📈 用户使用统计")

tab1, tab2, tab3 = st.tabs(["使用概览", "活动日志", "AI问答详情"])

with tab1:
    stats = get_usage_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("注册用户数", stats["total_users"])
    with col2:
        st.metric("今日登录", stats["today_logins"])
    with col3:
        st.metric("AI提问总数", stats["total_questions"])
    with col4:
        st.metric("7日活跃用户", stats["active_users_7d"])

    st.subheader("最近7天活动趋势")
    if stats["daily_activity"]:
        chart_data = pd.DataFrame(
            [{"日期": row["day"], "操作次数": row["count"]} for row in stats["daily_activity"]]
        )
        st.bar_chart(chart_data.set_index("日期"))
    else:
        st.info("暂无活动数据")

with tab2:
    st.subheader("用户活动日志")

    col1, col2 = st.columns(2)
    with col1:
        conn = get_connection()
        users = conn.execute("SELECT id, email FROM users ORDER BY email").fetchall()
        conn.close()
        user_options = {"全部用户": None}
        for u in users:
            user_options[u["email"]] = u["id"]
        selected_user = st.selectbox("筛选用户", options=list(user_options.keys()), key="log_user_filter")
    with col2:
        action_options = {"全部操作": None, "登录": "login", "AI提问": "ai_question", "AI回答": "ai_answer"}
        selected_action = st.selectbox("筛选操作", options=list(action_options.keys()), key="log_action_filter")

    logs = get_user_logs(
        limit=500,
        user_id=user_options[selected_user],
        action=action_options[selected_action],
    )

    if logs:
        log_data = []
        for log in logs:
            detail_preview = (log["detail"][:80] + "...") if log["detail"] and len(log["detail"]) > 80 else (log["detail"] or "")
            log_data.append({
                "时间": log["created_at"],
                "用户": log["email"],
                "操作": log["action"],
                "详情": detail_preview,
            })
        st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
    else:
        st.info("暂无日志记录")

with tab3:
    st.subheader("AI问答详情")

    conn = get_connection()
    ai_users = conn.execute("""
        SELECT DISTINCT u.id, u.email FROM user_logs l
        JOIN users u ON l.user_id = u.id
        WHERE l.action = 'ai_question'
        ORDER BY u.email
    """).fetchall()
    conn.close()

    ai_user_options = {"全部用户": None}
    for u in ai_users:
        ai_user_options[u["email"]] = u["id"]
    selected_ai_user = st.selectbox("筛选用户", options=list(ai_user_options.keys()), key="ai_user_filter")

    conversations = get_ai_conversations(limit=200, user_id=ai_user_options[selected_ai_user])

    if conversations:
        i = 0
        while i < len(conversations):
            log = conversations[i]
            if log["action"] == "ai_question":
                with st.container(border=True):
                    st.caption(f"👤 **{log['email']}** — {log['created_at']}")
                    st.markdown(f"**问题：** {log['detail']}")
                    if i + 1 < len(conversations) and conversations[i + 1]["action"] == "ai_answer" and conversations[i + 1]["user_id"] == log["user_id"]:
                        st.markdown(f"**回答：** {conversations[i + 1]['detail']}")
                        i += 2
                    else:
                        i += 1
            else:
                i += 1
    else:
        st.info("暂无AI问答记录")
