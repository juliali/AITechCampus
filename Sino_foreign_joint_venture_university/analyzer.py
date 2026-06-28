import json
import logging

import anthropic

from config import (
    ANTHROPIC_API_KEY, MODEL, MAX_TOKENS,
    PROGRAMS_FILE, PROGRAMS_ANALYZED_FILE, DATA_DIR,
)
from prompts.analysis import ANALYSIS_SYSTEM, ANALYSIS_BATCH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 8


def load_programs():
    if not PROGRAMS_FILE.exists():
        logger.error(f"数据文件不存在: {PROGRAMS_FILE}，请先运行 scrape 命令")
        return []
    with open(PROGRAMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_analyzed():
    if PROGRAMS_ANALYZED_FILE.exists():
        with open(PROGRAMS_ANALYZED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_analyzed(programs):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRAMS_ANALYZED_FILE, "w", encoding="utf-8") as f:
        json.dump(programs, f, ensure_ascii=False, indent=2)


def prepare_batch(programs):
    """精简项目信息用于分析（只保留关键字段）"""
    fields = [
        "project_name", "name", "address", "chinese_partner", "foreign_partner",
        "level_type", "duration", "major", "enrollment_per_year",
        "admission_method", "region", "status",
    ]
    batch = []
    for p in programs:
        item = {k: v for k, v in p.items() if k in fields and v}
        if item:
            batch.append(item)
    return batch


def analyze_batch(client, batch):
    """调用 Claude 分析一批项目"""
    programs_json = json.dumps(batch, ensure_ascii=False, indent=2)
    prompt = ANALYSIS_BATCH.format(programs_json=programs_json)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{
            "type": "text",
            "text": ANALYSIS_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text

    # Extract JSON from response
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])

    logger.warning("无法解析分析结果，尝试逐行解析")
    return []


def merge_analysis(program, analysis):
    """将分析结果合并到原始项目数据"""
    result = {**program}
    result["tags"] = analysis.get("tags", [])
    result["summary"] = analysis.get("summary", "")
    result["strengths"] = analysis.get("strengths", [])
    result["suitable_for"] = analysis.get("suitable_for", "")
    return result


def analyze():
    """主分析入口"""
    programs = load_programs()
    if not programs:
        return

    active_programs = [p for p in programs if p.get("status") != "已停办"]
    logger.info(f"共 {len(programs)} 个项目，{len(active_programs)} 个活跃项目待分析")

    already_analyzed = load_analyzed()
    analyzed_urls = {p.get("detail_url") for p in already_analyzed}
    to_analyze = [p for p in active_programs if p.get("detail_url") not in analyzed_urls]

    if not to_analyze:
        logger.info("所有项目已分析完毕")
        return already_analyzed

    logger.info(f"待分析: {len(to_analyze)} 个项目")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    all_analyzed = list(already_analyzed)

    for i in range(0, len(to_analyze), BATCH_SIZE):
        batch = to_analyze[i:i + BATCH_SIZE]
        batch_data = prepare_batch(batch)
        logger.info(f"分析批次 {i//BATCH_SIZE + 1}, 项目 {i+1}-{i+len(batch)}")

        try:
            analyses = analyze_batch(client, batch_data)
            for program, analysis in zip(batch, analyses):
                merged = merge_analysis(program, analysis)
                all_analyzed.append(merged)
        except Exception as e:
            logger.error(f"批次分析失败: {e}")
            all_analyzed.extend(batch)

        save_analyzed(all_analyzed)

    logger.info(f"分析完成，共 {len(all_analyzed)} 个项目")
    return all_analyzed
