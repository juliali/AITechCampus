import json
import time
import logging
import re
from datetime import datetime
from urllib.parse import quote_plus, unquote, parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
import html2text
import anthropic

from config import (
    ANTHROPIC_API_KEY, MODEL, MAX_TOKENS,
    DATA_DIR, PROGRAMS_FILE, PROGRAMS_ANALYZED_FILE,
    REQUEST_DELAY,
)
from prompts.enrichment import (
    ENRICHMENT_SYSTEM, ENRICHMENT_EXTRACT,
    SEARCH_QUERY_TEMPLATE, SEARCH_QUERY_DETAIL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ENRICHED_FILE = DATA_DIR / "programs_enriched.json"
SEARCH_DELAY = 5.0
FETCH_TIMEOUT = 20
MAX_PAGE_CHARS = 15000

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

PRIORITY_UNIVERSITIES = [
    "北京大学", "清华大学", "复旦大学", "上海交通大学", "浙江大学",
    "南京大学", "中国科学技术大学", "哈尔滨工业大学", "西安交通大学",
    "北京理工大学", "南开大学", "天津大学", "东南大学", "武汉大学",
    "华中科技大学", "中山大学", "厦门大学", "同济大学", "北京航空航天大学",
    "华东师范大学", "中国人民大学", "北京师范大学", "华南理工大学",
    "电子科技大学", "西北工业大学", "大连理工大学", "湖南大学",
    "重庆大学", "四川大学", "山东大学", "吉林大学", "中南大学",
    "苏州大学", "南京师范大学", "华东政法大学", "上海财经大学",
    "对外经济贸易大学", "中央财经大学", "北京外国语大学",
    "上海外国语大学", "深圳大学", "南方科技大学", "宁波诺丁汉大学",
    "西交利物浦大学", "昆山杜克大学", "上海纽约大学", "香港中文大学（深圳）",
]

h2t = html2text.HTML2Text()
h2t.ignore_links = True
h2t.ignore_images = True
h2t.body_width = 0


def load_programs():
    for path in [PROGRAMS_ANALYZED_FILE, PROGRAMS_FILE]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return []


def load_enriched():
    if ENRICHED_FILE.exists():
        with open(ENRICHED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_enriched(data):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(ENRICHED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_priority(program):
    name = program.get("project_name", "") + program.get("name", "")
    chinese = program.get("chinese_partner", "")
    return any(u in name or u in chinese for u in PRIORITY_UNIVERSITIES)


def web_search_bing(query, num_results=5):
    """用 DuckDuckGo HTML 搜索获取结果（无需 API key）"""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")

        results = []
        for item in soup.select(".result"):
            title_el = item.select_one(".result__title a, .result__a")
            snippet_el = item.select_one(".result__snippet")
            url_el = item.select_one(".result__url")
            if title_el:
                href = title_el.get("href", "")
                # DDG uses redirect: //duckduckgo.com/l/?uddg=REAL_URL
                if "uddg=" in href:
                    parsed = parse_qs(urlparse(href).query)
                    href = parsed.get("uddg", [href])[0]
                elif url_el and not href.startswith("http"):
                    href = "https://" + url_el.get_text(strip=True)
                results.append({
                    "title": title_el.get_text(strip=True),
                    "url": href,
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                })
        return results[:num_results]
    except Exception as e:
        logger.warning(f"搜索失败: {query[:50]}... - {e}")
        return []


def fetch_page_content(url):
    """抓取页面并转为 markdown 文本"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        text = h2t.handle(resp.text)
        return text[:MAX_PAGE_CHARS]
    except Exception as e:
        logger.warning(f"页面抓取失败: {url} - {e}")
        return ""


def clean_partner_name(name):
    """去掉外文名称，只保留中文部分，避免搜索 query 过长"""
    # Remove parenthesized English content (both Chinese and ASCII parens)
    name = re.sub(r"[（(][^）)]*[A-Za-z][^）)]*[)）]", "", name)
    # Remove standalone English text
    name = re.sub(r"[A-Za-z,.']+(?:\s+[A-Za-z,.']+)*", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    # If nothing left, return original trimmed
    return name[:30] if name else "未知"


def build_search_content(program):
    """构造搜索 query，执行搜索，抓取页面，返回聚合文本"""
    chinese = program.get("chinese_partner", "")
    foreign = program.get("foreign_partner", "")
    major = program.get("major", "")
    name = program.get("project_name", program.get("name", ""))

    if not chinese and not foreign:
        parts = name.split("与")
        if len(parts) >= 2:
            chinese = parts[0].strip()
            foreign = parts[1].split("合作")[0].strip()

    chinese_short = clean_partner_name(chinese)
    foreign_short = clean_partner_name(foreign)

    query1 = SEARCH_QUERY_TEMPLATE.format(
        chinese_partner=chinese_short, foreign_partner=foreign_short, major=major[:20]
    )
    time.sleep(SEARCH_DELAY)
    results1 = web_search_bing(query1)

    query2 = SEARCH_QUERY_DETAIL.format(
        chinese_partner=chinese_short, foreign_partner=foreign_short
    )
    time.sleep(SEARCH_DELAY)
    results2 = web_search_bing(query2)

    all_results = results1 + results2
    seen_urls = set()
    unique_results = []
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique_results.append(r)

    content_parts = []
    content_parts.append("### 搜索结果摘要\n")
    for i, r in enumerate(unique_results[:8], 1):
        content_parts.append(f"{i}. [{r['title']}]({r['url']})")
        if r["snippet"]:
            content_parts.append(f"   {r['snippet']}")
        content_parts.append("")

    fetched_count = 0
    for r in unique_results[:4]:
        if not r["url"] or "bing.com" in r["url"]:
            continue
        time.sleep(REQUEST_DELAY)
        page_text = fetch_page_content(r["url"])
        if page_text and len(page_text) > 200:
            content_parts.append(f"\n### 页面内容: {r['title']}\n来源: {r['url']}\n")
            content_parts.append(page_text[:8000])
            content_parts.append("")
            fetched_count += 1
            if fetched_count >= 2:
                break

    sources = [r["url"] for r in unique_results[:8] if r["url"]]
    return "\n".join(content_parts), sources


def extract_with_claude(client, program, search_content):
    """调用 Claude 从搜索结果中提取结构化信息"""
    prompt = ENRICHMENT_EXTRACT.format(
        project_name=program.get("project_name", program.get("name", "")),
        chinese_partner=program.get("chinese_partner", "未知"),
        foreign_partner=program.get("foreign_partner", "未知"),
        level_type=program.get("level_type", "未知"),
        major=program.get("major", "未知"),
        duration=program.get("duration", "未知"),
        region=program.get("region", "未知"),
        search_content=search_content,
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{
            "type": "text",
            "text": ENRICHMENT_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        json_str = text[start:end]
        # Fix common JSON issues: trailing commas, unescaped newlines
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try fixing unescaped quotes in string values
            json_str = re.sub(r'(?<=: ")(.*?)(?="[,\n}])',
                              lambda m: m.group(0).replace('"', '\\"'), json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败: {e}")

    logger.warning("无法解析 Claude 提取结果")
    return {}


def enrich_program(client, program):
    """对单个项目执行完整的信息补充流程"""
    name = program.get("project_name", program.get("name", ""))
    logger.info(f"正在补充: {name[:60]}")

    search_content, sources = build_search_content(program)
    if not search_content or len(search_content) < 100:
        logger.warning(f"  搜索内容不足，跳过")
        return None

    extracted = extract_with_claude(client, program, search_content)
    if not extracted:
        return None

    enriched = {**program}
    enriched.update(extracted)
    enriched["enriched_at"] = datetime.now().isoformat()
    enriched["enrichment_sources"] = sources

    return enriched


def enrich(limit=None, priority_first=False, region_id=None):
    """主入口: 对项目批量进行深度信息补充"""
    programs = load_programs()
    if not programs:
        logger.error("未找到项目数据，请先运行 scrape 命令")
        return

    active = [p for p in programs if p.get("status") != "已停办"]

    if region_id:
        active = [p for p in active if p.get("region_id") == region_id]

    logger.info(f"共 {len(active)} 个活跃项目")

    if priority_first:
        active.sort(key=lambda p: (0 if is_priority(p) else 1))

    already = load_enriched()
    done_urls = {p.get("detail_url") for p in already}
    to_process = [p for p in active if p.get("detail_url") not in done_urls]

    if limit:
        to_process = to_process[:limit]

    if not to_process:
        logger.info("所有项目已完成信息补充")
        return already

    logger.info(f"待处理: {len(to_process)} 个项目")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    all_enriched = list(already)
    success = 0

    for i, program in enumerate(to_process, 1):
        try:
            enriched = enrich_program(client, program)
            if enriched:
                all_enriched.append(enriched)
                success += 1
            else:
                all_enriched.append({**program, "enriched_at": datetime.now().isoformat(),
                                     "enrichment_confidence": "failed"})
        except Exception as e:
            logger.error(f"  处理失败: {e}")
            all_enriched.append({**program, "enriched_at": datetime.now().isoformat(),
                                 "enrichment_confidence": "failed"})

        save_enriched(all_enriched)
        if i % 5 == 0:
            logger.info(f"进度: {i}/{len(to_process)}, 成功: {success}")

    logger.info(f"信息补充完成: {success}/{len(to_process)} 成功")
    return all_enriched
