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
    5: "河北", 6: "山西", 7: "辽宁", 8: "吉林",
    9: "黑龙江", 10: "江苏", 11: "浙江", 12: "安徽",
    13: "福建", 14: "江西", 15: "山东", 16: "河南",
    17: "湖北", 18: "湖南", 19: "广东", 20: "海南",
    21: "四川", 22: "贵州", 23: "云南", 24: "陕西",
    25: "甘肃", 26: "青海", 27: "广西", 28: "西藏",
    29: "宁夏", 30: "新疆", 31: "内蒙古",
}
