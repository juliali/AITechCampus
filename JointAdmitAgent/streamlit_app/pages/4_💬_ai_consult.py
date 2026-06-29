import streamlit as st
import json

from utils.data_loader import load_programs, programs_to_df
from utils.llm_client import get_available_backends, get_admin_selected_backend, chat_stream
from utils.filters import filter_programs, expand_query_to_filters
from utils.logger import log_action
from utils.auth import check_ai_access

check_ai_access()

st.markdown("""
<style>
.stChatMessage [data-testid="stMarkdownContainer"] {
    font-size: 14px;
    line-height: 1.6;
}
.stChatMessage [data-testid="stMarkdownContainer"] h1,
.stChatMessage [data-testid="stMarkdownContainer"] h2,
.stChatMessage [data-testid="stMarkdownContainer"] h3 {
    font-size: 16px;
    margin-top: 0.8em;
}
.sample-btn button {
    border-radius: 20px;
    border: 1px solid #ddd;
    background: #f8f9fa;
    padding: 0.3rem 1rem;
    font-size: 13px;
    transition: all 0.2s;
}
.sample-btn button:hover {
    background: #e3f2fd;
    border-color: #90caf9;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 AI 智能咨询")
st.caption("基于教育部批准项目数据，为您提供个性化中外合作办学建议")

user_api_key = st.session_state.get("user_api_key")

if user_api_key:
    backend = "智谱GLM"
else:
    backends = get_available_backends()
    if not backends:
        st.warning("⚠️ AI 咨询功能暂未配置，请联系管理员")
        st.stop()
    backend = get_admin_selected_backend() or backends[0]

if st.session_state.get("is_admin"):
    backends = get_available_backends()
    idx = backends.index(backend) if backend in backends else 0
    selected = st.sidebar.selectbox("选择 AI 模型（管理员）", backends, index=idx, key="admin_backend_select")
    if selected != get_admin_selected_backend():
        from utils.llm_client import set_admin_backend
        set_admin_backend(selected)
    backend = selected

programs = load_programs()
df = programs_to_df(programs)

SYSTEM_PROMPT = f"""你是一位资深的中外合作办学咨询师，帮助考生和家长选择最适合的中外合作办学项目。

你掌握 {len(programs)} 个经教育部批准的中外合作办学项目的数据，覆盖北京、上海、山东、安徽、江西、山西等地区。

【常用术语与俗称解释】
你必须正确理解用户查询中的以下术语和俗称:
- 长三角: 指上海市、江苏省、浙江省、安徽省（核心城市：上海、南京、苏州、杭州、合肥等）
- 珠三角: 指广东省（核心城市：广州、深圳、珠海等）
- 京津冀: 指北京市、天津市、河北省
- 92: 985和211院校的合称（取985的"9"和211的"2"）
- 双一流: 世界一流大学和一流学科建设高校
- 4+0: 四年全部在国内就读、获得中外双学位的合作办学模式
- 2+2: 国内两年+国外两年的合作办学模式
- 3+1: 国内三年+国外一年的合作办学模式
- QS: QS世界大学排名
- 双证: 同时获得国内和国外学位证书

回答规则:
1. 始终使用中文回答
2. 推荐时说明理由，对比优劣
3. 如信息不完整，主动追问考生条件
4. 对于停招或已停办项目，明确告知
5. 提醒考生关注批准有效期和最新招生简章
6. 学费、分数等数据请注明来源年份
7. 回答尽量简洁有条理，使用列表和分段
8. 当用户使用上述术语/俗称时，正确解析其含义，只推荐符合条件的项目

你可以帮助:
- 根据分数/专业/预算/地域推荐项目
- 对比项目异同
- 解答学制、学位认证、就业前景等问题
- 分析某个项目的优劣"""

if "messages" not in st.session_state:
    st.session_state.messages = []

SAMPLE_QUESTIONS = [
    "我理科600分，想学计算机，有哪些好的中外合办项目？",
    "宁波诺丁汉和西交利物浦哪个更好？",
    "预算8万/年，有哪些性价比高的项目？",
    "中外合办的学位证国内认可吗？就业会受影响吗？",
    "上海地区有哪些985高校的中外合办项目？",
    "4+0和2+2模式各有什么优缺点？",
]

if not st.session_state.messages:
    st.markdown("##### 试试这些问题：")
    cols = st.columns(3)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        with cols[i % 3]:
            st.markdown('<div class="sample-btn">', unsafe_allow_html=True)
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

needs_reply = bool(
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
    and not st.session_state.get("_cancel_generation")
)

if needs_reply:
    if st.button("⏹ 停止生成", key="stop_gen", type="secondary"):
        st.session_state._cancel_generation = True
        st.session_state.messages.pop()
        st.rerun()

if prompt := st.chat_input("输入你的问题，如：我理科580分，想学金融，预算10万/年...", disabled=needs_reply):
    st.session_state.pop("_cancel_generation", None)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if needs_reply:
    prompt = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        with st.spinner("💭 AI 思考中..."):
            query_filters = expand_query_to_filters(prompt)
            kw_filters = {"keyword": prompt[:20], "active_only": True}
            if query_filters["regions"]:
                kw_filters["regions"] = query_filters["regions"]
            if query_filters["tiers"]:
                kw_filters["chinese_tiers"] = query_filters["tiers"]

            relevant = filter_programs(df, kw_filters)
            if len(relevant) == 0:
                fallback_filters = {"active_only": True}
                if query_filters["regions"]:
                    fallback_filters["regions"] = query_filters["regions"]
                if query_filters["tiers"]:
                    fallback_filters["chinese_tiers"] = query_filters["tiers"]
                relevant = filter_programs(df, fallback_filters)
                if len(relevant) == 0:
                    relevant = df[df["状态"] == "active"].head(30)
                else:
                    relevant = relevant.head(30)
            else:
                relevant = relevant.head(25)

            context_programs = []
            for _, row in relevant.iterrows():
                prog = next((p for p in programs if p.get("project_name", p.get("name", "")) == row["项目名称"]), None)
                if prog:
                    context_programs.append({
                        "名称": prog.get("project_name", prog.get("name", "")),
                        "中方": prog.get("chinese_partner", ""),
                        "外方": prog.get("foreign_partner", ""),
                        "层次": prog.get("level_type", ""),
                        "专业": prog.get("major", ""),
                        "学费": prog.get("tuition_per_year", ""),
                        "分数": prog.get("admission_score", ""),
                        "地区": prog.get("region", ""),
                        "QS": prog.get("foreign_university_profile", {}).get("qs_2025_rank", "") if isinstance(prog.get("foreign_university_profile"), dict) else "",
                        "中方层级": prog.get("chinese_university_profile", {}).get("tier", "") if isinstance(prog.get("chinese_university_profile"), dict) else "",
                    })

            augmented_messages = list(st.session_state.messages)
            if context_programs:
                context_text = f"\n\n[系统注入-相关项目数据，共{len(context_programs)}条]\n" + json.dumps(context_programs, ensure_ascii=False, indent=1)
                augmented_messages[-1] = {
                    "role": "user",
                    "content": prompt + context_text,
                }

            stream = chat_stream(augmented_messages, backend=backend, system_prompt=SYSTEM_PROMPT, user_api_key=user_api_key)
            first_chunk = next(stream, None)

        if first_chunk is None:
            st.warning("AI 未返回内容")
            st.stop()

        try:
            def _merged_stream():
                yield first_chunk
                yield from stream

            response = st.write_stream(_merged_stream())
        except Exception as e:
            st.error(f"AI 调用失败: {e}")
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.pop("_cancel_generation", None)
    log_action(st.session_state.user_id, "ai_question", prompt)
    log_action(st.session_state.user_id, "ai_answer", response)
    st.rerun()
