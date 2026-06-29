import pandas as pd

REGION_ALIASES = {
    "长三角": ["上海", "江苏", "浙江", "安徽"],
    "珠三角": ["广东"],
    "京津冀": ["北京", "天津", "河北"],
}

TIER_ALIASES = {
    "92": ["985", "211"],
}


def expand_query_to_filters(query):
    """从用户自然语言查询中提取地区和层级过滤条件（展开俗称）"""
    regions = []
    for alias, region_list in REGION_ALIASES.items():
        if alias in query:
            regions.extend(region_list)

    for region in ["北京", "上海", "天津", "重庆", "广东", "江苏", "浙江",
                   "山东", "安徽", "福建", "湖北", "湖南", "四川", "河南",
                   "河北", "江西", "山西", "辽宁", "吉林", "黑龙江"]:
        if region in query and region not in regions:
            regions.append(region)

    tiers = []
    for alias, tier_list in TIER_ALIASES.items():
        if alias in query:
            tiers.extend(tier_list)
    for tier in ["985", "211", "双一流", "一本"]:
        if tier in query and tier not in tiers:
            tiers.append(tier)

    return {"regions": regions, "tiers": tiers}


def filter_programs(df, filters):
    """根据筛选条件过滤 DataFrame"""
    result = df.copy()

    if filters.get("regions"):
        result = result[result["地区"].isin(filters["regions"])]

    if filters.get("levels"):
        level_mask = result["层次"].apply(
            lambda x: any(lv in str(x) for lv in filters["levels"])
        )
        result = result[level_mask]

    if filters.get("chinese_tiers"):
        result = result[result["中方层级"].isin(filters["chinese_tiers"])]

    if filters.get("qs_max"):
        qs_mask = result["外方QS排名"].notna() & (result["外方QS排名"] <= filters["qs_max"])
        result = result[qs_mask]

    if filters.get("keyword"):
        kw = filters["keyword"].lower()
        kw_mask = result["专业"].str.lower().str.contains(kw, na=False) | \
                  result["项目名称"].str.lower().str.contains(kw, na=False)
        result = result[kw_mask]

    if filters.get("overseas_only"):
        result = result[result["海外交换"] == "是"]

    if filters.get("active_only"):
        result = result[result["状态"] == "active"]

    return result
