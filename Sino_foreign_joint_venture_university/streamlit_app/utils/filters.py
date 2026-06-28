import pandas as pd


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
        qs_mask = result["QS排名"].notna() & (result["QS排名"] <= filters["qs_max"])
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
