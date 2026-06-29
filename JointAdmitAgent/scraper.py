import json
import time
import logging
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from config import (
    BASE_URL, DATA_DIR, PROGRAMS_FILE, MANUAL_IMPORT_DIR,
    REGIONS, REQUEST_DELAY, MAX_RETRIES, TIMEOUT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def create_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def fetch_page(session, url):
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"请求失败 (尝试 {attempt+1}/{MAX_RETRIES}): {url} - {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_DELAY * (attempt + 1))
    return None


def parse_listing_page(html, region_id):
    """解析区域列表页，提取项目名称和详情页链接"""
    soup = BeautifulSoup(html, "lxml")
    programs = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/aproval/detail/" in href or "/aproval/localdetail/" in href:
            name = link.get_text(strip=True)
            if not name:
                continue

            status = "active"
            if "停止招生" in name:
                status = "停止招生"
            elif "已停办" in name:
                status = "已停办"

            if href.startswith("/"):
                href = BASE_URL + href

            programs.append({
                "name": name,
                "detail_url": href,
                "region_id": region_id,
                "region": REGIONS.get(region_id, ""),
                "status": status,
            })

    return programs


def parse_detail_page(html, listing_info):
    """解析详情页，提取项目全部字段。

    页面结构: div.maincontent 下有一个4列表格 (label-value-label-value)，
    后接一个"变更记载"表格。
    """
    soup = BeautifulSoup(html, "lxml")
    program = {**listing_info, "scraped_at": datetime.now().isoformat()}

    field_map = {
        "项目名称": "project_name",
        "机构名称": "project_name",
        "办学地址": "address",
        "机构住所": "address",
        "机构属性": "institution_type",
        "法定代表人": "legal_representative",
        "办学层次和类别": "level_type",
        "学制": "duration",
        "每期招生人数": "enrollment_per_year",
        "办学规模": "enrollment_per_year",
        "招生起止年份": "enrollment_years",
        "招生方式": "admission_method",
        "开设专业或课程": "major",
        "审批机关": "approval_authority",
        "批准书编号": "approval_number",
        "许可证编号": "approval_number",
        "批准书有效期": "approval_expiry",
        "许可证有效期": "approval_expiry",
        "备注": "remarks",
    }

    main_div = soup.find("div", class_="maincontent")
    if not main_div:
        main_div = soup

    # Parse the main data table (4-column layout: label-value-label-value)
    main_table = main_div.find("table")
    if main_table:
        for row in main_table.find_all("tr"):
            cells = row.find_all("td")
            if not cells:
                continue

            # Handle 4-column rows (label-value-label-value)
            if len(cells) == 4:
                for i in (0, 2):
                    label = cells[i].get_text(strip=True)
                    value = cells[i + 1].get_text(strip=True)
                    if label in field_map and value:
                        program[field_map[label]] = value

            # Handle 2-column rows with colspan (label + wide value)
            elif len(cells) == 2:
                label = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                if label in field_map and value:
                    program[field_map[label]] = value

            # Handle rows where first cell has rowspan (中外合作办学者, 颁发证书)
            # These have varying cell counts due to rowspan
            cell_texts = [c.get_text(strip=True) for c in cells]
            full_text = " ".join(cell_texts)

            # Determine if this is a "中外合作办学者" row or "颁发证书" row
            # by checking if the row's first rowspan cell or context mentions it
            is_certificate_row = "证书" in full_text or "证" in full_text and "颁发" in full_text

            if not is_certificate_row:
                for t in cell_texts:
                    if (t.startswith("中方：") or t.startswith("中方:")) and "证" not in t:
                        program.setdefault("chinese_partner", t.split("：", 1)[-1].split(":", 1)[-1].strip())
                    elif (t.startswith("外方：") or t.startswith("外方:")) and "证" not in t:
                        program.setdefault("foreign_partner", t.split("：", 1)[-1].split(":", 1)[-1].strip())

            # Certificates (contain "证书" in the value)
            for t in cell_texts:
                if t.startswith("中方：") and "证" in t:
                    program["chinese_certificate"] = t.split("：", 1)[-1].strip()
                elif t.startswith("外方：") and "证" in t:
                    program["foreign_certificate"] = t.split("：", 1)[-1].strip()

    return program


def scrape_region(session, region_id, program_type="national"):
    """采集单个区域的所有项目"""
    if program_type == "national":
        url = f"{BASE_URL}/aproval/getbyarea/{region_id}"
    else:
        url = f"{BASE_URL}/aproval/localbyarea/{region_id}"

    logger.info(f"采集区域: {REGIONS.get(region_id, region_id)} ({program_type})")
    html = fetch_page(session, url)
    if not html:
        logger.error(f"无法获取列表页: {url}")
        return []

    listings = parse_listing_page(html, region_id)
    logger.info(f"  发现 {len(listings)} 个项目")

    programs = []
    for i, listing in enumerate(listings):
        time.sleep(REQUEST_DELAY)
        detail_html = fetch_page(session, listing["detail_url"])
        if detail_html:
            program = parse_detail_page(detail_html, listing)
            programs.append(program)
        else:
            programs.append(listing)

        if (i + 1) % 10 == 0:
            logger.info(f"  进度: {i+1}/{len(listings)}")

    return programs


def load_existing_programs():
    if PROGRAMS_FILE.exists():
        with open(PROGRAMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_programs(programs):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(programs, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存 {len(programs)} 个项目到 {PROGRAMS_FILE}")


def load_manual_imports():
    """从 manual_import 目录加载手动导入的数据"""
    programs = []
    if not MANUAL_IMPORT_DIR.exists():
        return programs
    for f in MANUAL_IMPORT_DIR.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            if isinstance(data, list):
                programs.extend(data)
            elif isinstance(data, dict):
                programs.append(data)
    if programs:
        logger.info(f"从 manual_import 加载了 {len(programs)} 个项目")
    return programs


def scrape(region_ids=None):
    """主采集入口"""
    if region_ids is None:
        region_ids = list(REGIONS.keys())

    session = create_session()
    all_programs = load_existing_programs()
    existing_urls = {p.get("detail_url") for p in all_programs}

    for region_id in region_ids:
        for ptype in ["national", "local"]:
            programs = scrape_region(session, region_id, ptype)
            new_programs = [p for p in programs if p.get("detail_url") not in existing_urls]
            if new_programs:
                all_programs.extend(new_programs)
                existing_urls.update(p.get("detail_url") for p in new_programs)
                save_programs(all_programs)

    manual = load_manual_imports()
    if manual:
        manual_new = [p for p in manual if p.get("detail_url") not in existing_urls]
        if manual_new:
            all_programs.extend(manual_new)
            save_programs(all_programs)

    logger.info(f"采集完成，共 {len(all_programs)} 个项目")
    return all_programs
