import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PROGRAMS_FILE = DATA_DIR / "programs.json"
PROGRAMS_ANALYZED_FILE = DATA_DIR / "programs_analyzed.json"
PROGRAMS_ENRICHED_FILE = DATA_DIR / "programs_enriched.json"
MANUAL_IMPORT_DIR = DATA_DIR / "manual_import"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8192

REQUEST_DELAY = 2.0
MAX_RETRIES = 3
TIMEOUT = 30

BASE_URL = "https://www.crs.jsj.edu.cn"

REGIONS = {
    1: "北京", 2: "上海", 3: "天津", 4: "重庆",
    5: "江苏", 6: "浙江", 7: "广东", 8: "海南",
    9: "福建", 10: "山东", 11: "江西", 12: "四川",
    13: "安徽", 14: "河北", 15: "河南", 16: "湖北",
    17: "湖南", 18: "陕西", 19: "山西", 20: "黑龙江",
    21: "辽宁", 22: "吉林", 23: "广西", 24: "云南",
    25: "贵州", 26: "甘肃", 27: "内蒙古", 28: "宁夏",
    29: "新疆", 30: "青海", 31: "西藏",
}
