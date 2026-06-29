import streamlit as st


def get_available_backends():
    backends = []
    if st.secrets.get("ANTHROPIC_API_KEY"):
        backends.append("Claude")
    if st.secrets.get("ZHIPU_API_KEY"):
        backends.append("智谱GLM")
    return backends


def get_admin_selected_backend():
    from utils.db import get_connection
    conn = get_connection()
    row = conn.execute("SELECT value FROM settings WHERE key='ai_backend'").fetchone()
    conn.close()
    if row:
        return row["value"]
    backends = get_available_backends()
    return backends[0] if backends else None


def set_admin_backend(backend: str):
    from utils.db import get_connection
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES ('ai_backend', ?)",
        (backend,),
    )
    conn.commit()
    conn.close()


def chat(messages, backend="Claude", system_prompt="", user_api_key=None):
    if backend == "Claude":
        return _chat_claude(messages, system_prompt)
    elif backend == "智谱GLM":
        api_key = user_api_key or st.secrets.get("ZHIPU_API_KEY")
        return _chat_zhipu(messages, system_prompt, api_key=api_key)
    return "未配置 LLM 后端"


def chat_stream(messages, backend="Claude", system_prompt="", user_api_key=None):
    """Streaming version of chat - yields text chunks."""
    if backend == "Claude":
        yield from _chat_claude_stream(messages, system_prompt)
    elif backend == "智谱GLM":
        api_key = user_api_key or st.secrets.get("ZHIPU_API_KEY")
        yield from _chat_zhipu_stream(messages, system_prompt, api_key=api_key)
    else:
        yield "未配置 LLM 后端"


def _chat_claude(messages, system_prompt):
    import anthropic
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    api_messages = []
    for m in messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    kwargs = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": api_messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)
    return response.content[0].text


def _chat_claude_stream(messages, system_prompt):
    import anthropic
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    api_messages = []
    for m in messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    kwargs = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": api_messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text


def _chat_zhipu(messages, system_prompt, api_key=None):
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=api_key)

    api_messages = []
    if system_prompt:
        api_messages.append({"role": "system", "content": system_prompt})
    for m in messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=api_messages,
    )
    return response.choices[0].message.content


def _chat_zhipu_stream(messages, system_prompt, api_key=None):
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=api_key)

    api_messages = []
    if system_prompt:
        api_messages.append({"role": "system", "content": system_prompt})
    for m in messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=api_messages,
        stream=True,
    )
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
