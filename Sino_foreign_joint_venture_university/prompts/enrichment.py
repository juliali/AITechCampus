ENRICHMENT_SYSTEM = """你是一位中国高等教育研究员，专注于中外合作办学领域。
你的任务是从提供的网页搜索结果和页面内容中，提取该中外合作办学项目的详细信息。

提取规则:
1. 只提取有明确来源的信息，无法确定的字段填 null
2. 不要猜测或推断未明确提及的信息
3. 金额统一换算为人民币/年
4. QS排名以最新公开的年份为准，标注年份
5. 区分"项目官方声明"和"第三方报道"的信息"""

ENRICHMENT_EXTRACT = """请从以下搜索结果和网页内容中，提取该中外合作办学项目的详细信息。

## 项目基础信息
- 项目名称: {project_name}
- 中方院校: {chinese_partner}
- 外方院校: {foreign_partner}
- 办学层次: {level_type}
- 专业: {major}
- 学制: {duration}
- 地区: {region}

## 搜索结果与网页内容
{search_content}

---

请以严格JSON格式返回以下字段（无法确定的填null）:

{{
    "tuition_total": "总学费(人民币)，如'16万'",
    "tuition_per_year": "年学费(人民币)，如'4万/年'",
    "admission_score": "录取分数线/要求，如'2024年理科最低分580分'或'一本线上30分'或'雅思6.5+GPA3.0'",
    "admission_score_detail": "各年份或各省份的详细录取分数(如有多条数据)",
    "housing_provided": true/false/null,
    "housing_conditions": "住宿条件描述，如'4人间带独卫空调'",
    "campus_location": "具体授课地点，如'上海市闵行区交大校区'",

    "foreign_university_full_name": "外方院校英文全称",
    "foreign_university_country": "国家",
    "foreign_university_qs_ranking": "QS排名，如'QS2025第45名'",
    "foreign_university_strengths": ["优势学科1", "优势学科2"],

    "faculty_total": "教师总数，如'60人'",
    "faculty_breakdown": "师资构成，如'教授15人、副教授20人、讲师25人'",
    "foreign_faculty_ratio": "外方师资比例，如'40%课程由外方教师授课'",
    "foreign_faculty_arrangement": "外方教师授课形式，如'每学期驻场2个月'",

    "overseas_exchange": true/false/null,
    "overseas_duration": "交换时长，如'大三一学年'或'暑期2个月'",
    "overseas_destination": "交换目的地，如'英国利兹大学本部'",
    "masters_pathway": "硕士直升/升学通道说明，如'GPA3.2以上可直升外方硕士'",

    "internship_arranged": true/false/null,
    "internship_details": "实习安排，如'大四上学期安排企业实习3个月'",
    "vacation_length": "假期安排，如'寒假4周、暑假8周'",
    "extra_curricular_teaching": "课外教学形式，如'企业参访、学术讲座、国际竞赛'",

    "career_guidance": true/false/null,
    "career_guidance_details": "就业指导说明",
    "employment_rate": "就业率或去向数据",

    "director_name": "项目负责人姓名",
    "director_background": "负责人学术背景简述",

    "enrichment_confidence": "high/medium/low",
    "notes": "其他值得注意的信息"
}}"""

SEARCH_QUERY_TEMPLATE = "{chinese_partner} {foreign_partner} 中外合作办学 {major} 学费 录取分数线 2024 2025"

SEARCH_QUERY_DETAIL = "{chinese_partner} {foreign_partner} 合作办学 招生简章 收费标准 住宿 海外交换 就业"
