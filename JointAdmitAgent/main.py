import argparse
import json
import sys
from collections import Counter

from rich.console import Console
from rich.table import Table

from config import PROGRAMS_FILE, PROGRAMS_ANALYZED_FILE, REGIONS

console = Console()


def cmd_scrape(args):
    from scraper import scrape
    region_ids = [args.region] if args.region else None
    scrape(region_ids)


def cmd_analyze(args):
    from analyzer import analyze
    analyze()


def cmd_consult(args):
    from agent import consult
    consult()


def cmd_enrich(args):
    from enricher import enrich
    enrich(limit=args.limit, priority_first=args.priority, region_id=args.region)


def cmd_institutions(args):
    from institutions import build_and_annotate
    build_and_annotate(rebuild=args.rebuild)


def cmd_stats(args):
    """展示数据统计概览"""
    data_file = PROGRAMS_ANALYZED_FILE if PROGRAMS_ANALYZED_FILE.exists() else PROGRAMS_FILE
    if not data_file.exists():
        console.print("[red]未找到数据文件，请先运行 scrape 命令[/red]")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        programs = json.load(f)

    console.print(f"\n[bold]数据文件:[/bold] {data_file}")
    console.print(f"[bold]项目总数:[/bold] {len(programs)}\n")

    # 按状态统计
    status_counter = Counter(p.get("status", "unknown") for p in programs)
    table = Table(title="按状态分布")
    table.add_column("状态", style="cyan")
    table.add_column("数量", style="green", justify="right")
    for status, count in status_counter.most_common():
        table.add_row(status, str(count))
    console.print(table)
    console.print()

    # 按区域统计
    region_counter = Counter(p.get("region", "未知") for p in programs)
    table = Table(title="按区域分布 (前15)")
    table.add_column("区域", style="cyan")
    table.add_column("数量", style="green", justify="right")
    for region, count in region_counter.most_common(15):
        table.add_row(region, str(count))
    console.print(table)
    console.print()

    # 按层次统计
    level_counter = Counter(p.get("level_type", "未知") for p in programs)
    table = Table(title="按办学层次分布")
    table.add_column("层次", style="cyan")
    table.add_column("数量", style="green", justify="right")
    for level, count in level_counter.most_common():
        table.add_row(level, str(count))
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="中外合作办学信息采集与咨询系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py scrape --region 2     采集上海地区项目
  python main.py scrape                采集所有地区
  python main.py enrich --limit 5      补充前5个项目的深度信息
  python main.py enrich --priority     优先补充重点院校项目
  python main.py analyze               AI 分析项目特色
  python main.py consult               启动交互式咨询
  python main.py stats                 查看数据统计
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    sp_scrape = subparsers.add_parser("scrape", help="采集教育部平台数据")
    sp_scrape.add_argument("--region", type=int, help="指定区域ID (1-31)")
    sp_scrape.set_defaults(func=cmd_scrape)

    sp_enrich = subparsers.add_parser("enrich", help="补充深度信息(学费/师资/住宿等)")
    sp_enrich.add_argument("--limit", type=int, help="只处理前N个项目(测试用)")
    sp_enrich.add_argument("--region", type=int, help="指定区域ID (1-31)")
    sp_enrich.add_argument("--priority", action="store_true", help="优先处理重点院校")
    sp_enrich.set_defaults(func=cmd_enrich)

    sp_inst = subparsers.add_parser("institutions", help="构建院校成色数据库并标注")
    sp_inst.add_argument("--rebuild", action="store_true", help="强制重建院校数据库")
    sp_inst.set_defaults(func=cmd_institutions)

    sp_analyze = subparsers.add_parser("analyze", help="AI 分析项目特色")
    sp_analyze.set_defaults(func=cmd_analyze)

    sp_consult = subparsers.add_parser("consult", help="启动交互式咨询")
    sp_consult.set_defaults(func=cmd_consult)

    sp_stats = subparsers.add_parser("stats", help="查看数据统计")
    sp_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
