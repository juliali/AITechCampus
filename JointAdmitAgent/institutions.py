import json
import logging
from pathlib import Path

import anthropic

from config import ANTHROPIC_API_KEY, MODEL, MAX_TOKENS, DATA_DIR, PROGRAMS_FILE, PROGRAMS_ANALYZED_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHINESE_DB_FILE = DATA_DIR / "chinese_universities.json"
FOREIGN_DB_FILE = DATA_DIR / "foreign_universities.json"

# ============================================================
# 985 院校 (39所)
# ============================================================
C9 = ["北京大学", "清华大学", "复旦大学", "上海交通大学", "浙江大学",
      "南京大学", "中国科学技术大学", "哈尔滨工业大学", "西安交通大学"]

TIER_985 = [
    *C9,
    "北京理工大学", "北京航空航天大学", "北京师范大学", "中国人民大学",
    "南开大学", "天津大学", "大连理工大学", "吉林大学", "东北大学",
    "同济大学", "华东师范大学", "厦门大学", "山东大学", "中国海洋大学",
    "武汉大学", "华中科技大学", "湖南大学", "中南大学", "中山大学",
    "华南理工大学", "四川大学", "电子科技大学", "重庆大学",
    "西北工业大学", "兰州大学", "东南大学",
    "中国农业大学", "国防科技大学", "中央民族大学",
]

# ============================================================
# 211 (非985) 院校 (73所)
# ============================================================
TIER_211_ONLY = [
    "中央财经大学", "对外经济贸易大学", "上海财经大学", "中南财经政法大学", "西南财经大学",
    "北京外国语大学", "上海外国语大学", "北京语言大学",
    "中国政法大学", "华东政法大学",
    "北京交通大学", "北京邮电大学", "北京科技大学", "北京化工大学",
    "北京林业大学", "北京中医药大学", "中国传媒大学",
    "中国地质大学", "中国矿业大学", "中国石油大学",
    "华北电力大学", "河海大学", "江南大学", "南京农业大学",
    "南京理工大学", "南京航空航天大学", "南京师范大学",
    "苏州大学", "合肥工业大学", "福州大学",
    "上海大学", "华东理工大学", "东华大学",
    "武汉理工大学", "华中农业大学", "华中师范大学",
    "中南大学", "湖南师范大学",
    "暨南大学", "华南师范大学",
    "西南大学", "西南交通大学",
    "西安电子科技大学", "长安大学", "陕西师范大学", "西北大学",
    "云南大学", "广西大学", "贵州大学",
    "郑州大学", "河南大学",
    "太原理工大学", "内蒙古大学", "辽宁大学", "延边大学",
    "东北师范大学", "东北农业大学", "东北林业大学",
    "哈尔滨工程大学", "海南大学", "宁夏大学", "青海大学",
    "新疆大学", "石河子大学", "西藏大学",
    "安徽大学", "南昌大学",
    "四川农业大学",
]

# ============================================================
# 双一流但非211的新增高校
# ============================================================
TIER_DOUBLE_FIRST_CLASS_NEW = [
    "南方科技大学", "上海科技大学", "中国科学院大学",
    "宁波大学", "南京信息工程大学", "南京邮电大学", "南京林业大学", "南京医科大学",
    "河南大学", "湘潭大学", "华南农业大学", "广州医科大学",
    "成都理工大学", "西南石油大学",
    "上海海洋大学", "上海中医药大学", "上海体育大学",
    "天津工业大学", "天津中医药大学",
    "山西大学", "湖南科技大学",
]

# ============================================================
# 中外合办独立大学 (特殊类别)
# ============================================================
SINO_FOREIGN_INDEPENDENT = [
    "宁波诺丁汉大学", "西交利物浦大学", "昆山杜克大学",
    "上海纽约大学", "香港中文大学（深圳）", "北京师范大学-香港浸会大学联合国际学院",
    "深圳北理莫斯科大学", "广东以色列理工学院", "温州肯恩大学",
]


def get_chinese_tier(name):
    """根据院校名判断层次"""
    if name in TIER_985:
        return "985/211/双一流A"
    if name in TIER_211_ONLY:
        return "211/双一流"
    if name in TIER_DOUBLE_FIRST_CLASS_NEW:
        return "双一流"
    if name in SINO_FOREIGN_INDEPENDENT:
        return "中外合办独立大学"
    return None


def get_all_chinese_partners():
    """从 programs.json 提取所有不重复的中方合作院校"""
    for path in [PROGRAMS_ANALYZED_FILE, PROGRAMS_FILE]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                programs = json.load(f)
            break
    else:
        return set()

    partners = set()
    for p in programs:
        cp = p.get("chinese_partner", "")
        if cp:
            partners.add(cp)
        else:
            name = p.get("project_name", p.get("name", ""))
            if "与" in name:
                parts = name.split("与")
                partners.add(parts[0].strip())
    return partners


def get_all_foreign_partners():
    """从 programs.json 提取所有不重复的外方合作院校"""
    for path in [PROGRAMS_ANALYZED_FILE, PROGRAMS_FILE]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                programs = json.load(f)
            break
    else:
        return set()

    partners = set()
    for p in programs:
        fp = p.get("foreign_partner", "")
        if fp:
            partners.add(fp)
    return partners


def build_chinese_db():
    """构建中方院校数据库"""
    partners = get_all_chinese_partners()
    logger.info(f"发现 {len(partners)} 个不重复的中方合作院校")

    db = {}
    if CHINESE_DB_FILE.exists():
        with open(CHINESE_DB_FILE, "r", encoding="utf-8") as f:
            db = json.load(f)

    # Step 1: 匹配已知名单
    unmatched = []
    for name in sorted(partners):
        if name in db:
            continue
        tier = get_chinese_tier(name)
        if tier:
            db[name] = {"tier": tier, "type": "", "strengths": []}
        else:
            unmatched.append(name)

    logger.info(f"已知名单匹配: {len(db)} 所, 未匹配: {len(unmatched)} 所")

    # Step 2: 用 Claude 批量标注未匹配的院校
    if unmatched:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        batch_size = 30

        for i in range(0, len(unmatched), batch_size):
            batch = unmatched[i:i + batch_size]
            prompt = f"""请为以下中国高校标注信息。每所学校标注: tier(层级), type(类型), strengths(优势学科,最多3个)。

tier 分类规则:
- "省属重点/一本": 各省重点大学、老牌一本
- "普通本科/二本": 普通本科院校
- "高职/专科": 高等职业院校、专科学校
- "民办本科": 民办本科院校
- "中外合办独立大学": 中外合作办学独立设置的大学

type: 综合类/理工类/财经类/师范类/医学类/艺术类/语言类/政法类/农林类/体育类/军事类

请以JSON对象格式返回，key是校名:

学校列表:
{json.dumps(batch, ensure_ascii=False)}"""

            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    result = json.loads(text[start:end])
                    db.update(result)
                    logger.info(f"  Claude 标注批次 {i//batch_size+1}: {len(result)} 所")
            except Exception as e:
                logger.error(f"  Claude 标注失败: {e}")

            _save_json(CHINESE_DB_FILE, db)

    _save_json(CHINESE_DB_FILE, db)
    logger.info(f"中方院校数据库构建完成: {len(db)} 所")
    return db


def build_foreign_db():
    """构建外方院校数据库"""
    partners = get_all_foreign_partners()
    logger.info(f"发现 {len(partners)} 个不重复的外方合作院校")

    db = {}
    if FOREIGN_DB_FILE.exists():
        with open(FOREIGN_DB_FILE, "r", encoding="utf-8") as f:
            db = json.load(f)

    # 提取未处理的院校
    unprocessed = [name for name in sorted(partners) if name not in db]
    if not unprocessed:
        logger.info("所有外方院校已标注")
        return db

    logger.info(f"待标注外方院校: {len(unprocessed)} 所")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    batch_size = 20

    for i in range(0, len(unprocessed), batch_size):
        batch = unprocessed[i:i + batch_size]
        prompt = f"""请为以下外国大学/教育机构标注信息。每所学校标注:
- country: 所在国家
- qs_2025_rank: QS 2025世界大学排名(整数,如不在前1000名则填null)
- tier: "世界顶尖"(QS 1-50)/"世界一流"(51-150)/"知名大学"(151-500)/"普通大学"(500+或未上榜)
- strengths: 优势学科(最多3个)
- alliance: 所属联盟(如"G5","常春藤","罗素集团","八大名校"等,无则填null)

请以JSON对象格式返回，key是原始校名（保持我提供的原文不变）:

院校列表:
{json.dumps(batch, ensure_ascii=False)}"""

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                import re
                json_str = text[start:end]
                json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
                result = json.loads(json_str)
                db.update(result)
                logger.info(f"  Claude 标注批次 {i//batch_size+1}: {len(result)} 所")
        except Exception as e:
            logger.error(f"  Claude 标注失败: {e}")

        _save_json(FOREIGN_DB_FILE, db)

    _save_json(FOREIGN_DB_FILE, db)
    logger.info(f"外方院校数据库构建完成: {len(db)} 所")
    return db


def annotate_programs():
    """把院校画像标注到项目数据上"""
    chinese_db = {}
    foreign_db = {}
    if CHINESE_DB_FILE.exists():
        with open(CHINESE_DB_FILE, "r", encoding="utf-8") as f:
            chinese_db = json.load(f)
    if FOREIGN_DB_FILE.exists():
        with open(FOREIGN_DB_FILE, "r", encoding="utf-8") as f:
            foreign_db = json.load(f)

    if not chinese_db and not foreign_db:
        logger.error("院校数据库为空，请先运行 build")
        return

    # Load programs (prefer enriched > analyzed > raw)
    from config import PROGRAMS_FILE, PROGRAMS_ANALYZED_FILE
    enriched_file = DATA_DIR / "programs_enriched.json"
    for path in [enriched_file, PROGRAMS_ANALYZED_FILE, PROGRAMS_FILE]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                programs = json.load(f)
            logger.info(f"从 {path.name} 加载 {len(programs)} 个项目")
            break
    else:
        logger.error("未找到项目数据")
        return

    annotated = 0
    for p in programs:
        # Annotate Chinese partner
        cp = p.get("chinese_partner", "")
        if cp and cp in chinese_db:
            p["chinese_university_profile"] = chinese_db[cp]
            annotated += 1
        elif cp:
            # Try partial match
            for key, val in chinese_db.items():
                if key in cp or cp in key:
                    p["chinese_university_profile"] = val
                    annotated += 1
                    break

        # Annotate Foreign partner
        fp = p.get("foreign_partner", "")
        if fp and fp in foreign_db:
            p["foreign_university_profile"] = foreign_db[fp]
        elif fp:
            for key, val in foreign_db.items():
                if key in fp or fp in key:
                    p["foreign_university_profile"] = val
                    break

    # Save back
    output_file = enriched_file if enriched_file.exists() else PROGRAMS_FILE
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(programs, f, ensure_ascii=False, indent=2)

    logger.info(f"标注完成: {annotated} 个项目已添加院校画像")


def build_and_annotate(rebuild=False):
    """构建院校数据库并标注到项目"""
    if rebuild or not CHINESE_DB_FILE.exists():
        build_chinese_db()
    else:
        logger.info("中方院校数据库已存在，跳过 (使用 --rebuild 强制重建)")

    if rebuild or not FOREIGN_DB_FILE.exists():
        build_foreign_db()
    else:
        logger.info("外方院校数据库已存在，跳过 (使用 --rebuild 强制重建)")

    annotate_programs()


def _save_json(path, data):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
