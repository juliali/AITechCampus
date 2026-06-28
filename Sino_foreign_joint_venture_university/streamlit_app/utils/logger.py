from utils.db import get_connection


def log_action(user_id, action, detail=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO user_logs (user_id, action, detail) VALUES (?, ?, ?)",
        (user_id, action, detail),
    )
    conn.commit()
    conn.close()


def get_user_logs(limit=200, user_id=None, action=None):
    conn = get_connection()
    query = """
        SELECT l.*, u.email FROM user_logs l
        JOIN users u ON l.user_id = u.id
        WHERE 1=1
    """
    params = []
    if user_id:
        query += " AND l.user_id = ?"
        params.append(user_id)
    if action:
        query += " AND l.action = ?"
        params.append(action)
    query += " ORDER BY l.created_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def get_ai_conversations(limit=100, user_id=None):
    conn = get_connection()
    query = """
        SELECT l.*, u.email FROM user_logs l
        JOIN users u ON l.user_id = u.id
        WHERE l.action IN ('ai_question', 'ai_answer')
    """
    params = []
    if user_id:
        query += " AND l.user_id = ?"
        params.append(user_id)
    query += " ORDER BY l.created_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def get_usage_stats():
    conn = get_connection()
    stats = {}
    stats["total_users"] = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    stats["today_logins"] = conn.execute(
        "SELECT COUNT(DISTINCT user_id) FROM user_logs WHERE action='login' AND date(created_at)=date('now')"
    ).fetchone()[0]
    stats["total_questions"] = conn.execute(
        "SELECT COUNT(*) FROM user_logs WHERE action='ai_question'"
    ).fetchone()[0]
    stats["active_users_7d"] = conn.execute(
        "SELECT COUNT(DISTINCT user_id) FROM user_logs WHERE created_at >= datetime('now', '-7 days')"
    ).fetchone()[0]
    stats["daily_activity"] = conn.execute("""
        SELECT date(created_at) as day, COUNT(*) as count
        FROM user_logs
        WHERE created_at >= datetime('now', '-7 days')
        GROUP BY date(created_at)
        ORDER BY day
    """).fetchall()
    conn.close()
    return stats
