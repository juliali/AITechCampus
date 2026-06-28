import json
import logging

import anthropic
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from config import (
    ANTHROPIC_API_KEY, MODEL, MAX_TOKENS,
    PROGRAMS_ANALYZED_FILE, PROGRAMS_FILE,
)
from prompts.consultation import (
    CONSULTATION_SYSTEM, CONSULTATION_WITH_DATA, PROFILE_INTAKE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
console = Console()

MAX_CONTEXT_PROGRAMS = 25


def load_programs():
    """加载分析后的数据（优先）或原始数据"""
    if PROGRAMS_ANALYZED_FILE.exists():
        with open(PROGRAMS_ANALYZED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    if PROGRAMS_FILE.exists():
        with open(PROGRAMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def filter_programs(programs, query):
    """基于关键词预过滤，缩小候选范围"""
    keywords = query.lower().replace("，", " ").replace(",", " ").split()
    if not keywords:
        return programs[:MAX_CONTEXT_PROGRAMS]

    scored = []
    for p in programs:
        if p.get("status") == "已停办":
            continue
        searchable = json.dumps(p, ensure_ascii=False).lower()
        score = sum(1 for kw in keywords if kw in searchable)
        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:MAX_CONTEXT_PROGRAMS]]


def format_programs_for_context(programs):
    """精简项目信息用于 Claude 上下文"""
    fields = [
        "project_name", "name", "address", "chinese_partner", "foreign_partner",
        "level_type", "duration", "major", "enrollment_per_year",
        "admission_method", "region", "status", "tags", "summary",
        "strengths", "suitable_for",
    ]
    simplified = []
    for p in programs:
        item = {k: v for k, v in p.items() if k in fields and v}
        simplified.append(item)
    return json.dumps(simplified, ensure_ascii=False, indent=2)


def consult():
    """交互式咨询主循环"""
    programs = load_programs()
    if not programs:
        console.print("[red]未找到项目数据，请先运行 scrape 命令采集数据[/red]")
        return

    active_count = sum(1 for p in programs if p.get("status") != "已停办")
    console.print(Panel(
        f"已加载 [bold]{len(programs)}[/bold] 个项目数据（其中 {active_count} 个在招）\n"
        f"输入 [bold]quit[/bold] 或 [bold]exit[/bold] 退出",
        title="中外合作办学智能咨询",
        border_style="blue",
    ))
    console.print()
    console.print(Markdown(PROFILE_INTAKE))
    console.print()

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = []

    while True:
        try:
            user_input = console.input("[bold green]你: [/bold green]")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.strip().lower() in ("quit", "exit", "q", "退出"):
            console.print("[dim]再见！祝你选到理想的项目！[/dim]")
            break

        if not user_input.strip():
            continue

        matched = filter_programs(programs, user_input)
        context_json = format_programs_for_context(matched)
        user_message = CONSULTATION_WITH_DATA.format(
            programs_json=context_json, question=user_input
        )

        messages.append({"role": "user", "content": user_message})

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": CONSULTATION_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=messages,
            )

            assistant_text = response.content[0].text
            messages.append({"role": "assistant", "content": assistant_text})

            console.print()
            console.print(Panel(
                Markdown(assistant_text),
                title="咨询师",
                border_style="cyan",
            ))
            console.print()

        except Exception as e:
            console.print(f"[red]API 调用失败: {e}[/red]")
            messages.pop()
