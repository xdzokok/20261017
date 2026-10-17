"""Reproduce the order-level inventory fulfillment diagnosis from the official CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import pandas as pd


COLUMNS = (
    "order_id", "order_time", "last_sync_time", "warehouse_region",
    "category", "stock_level", "daily_traffic", "fulfilled",
)
ORDER_COLUMNS = COLUMNS[:-1]
FAILURE_COST = 120
ALERT_COST = 20
THRESHOLD_MINUTES = 20


def load_orders(path: Path, *, require_outcome: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(path)
    required = COLUMNS if require_outcome else ORDER_COLUMNS
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"缺少必要字段: {', '.join(missing)}")
    if raw.empty:
        raise ValueError("CSV 没有订单数据")
    if raw[list(ORDER_COLUMNS)].isna().any().any():
        raise ValueError("预测字段存在缺失，请核对源数据")

    if not require_outcome:
        # If an outcome column is supplied accidentally, ignore it entirely.
        raw = raw.drop(columns="fulfilled", errors="ignore")
    orders = raw.drop_duplicates().copy()
    if orders.order_id.duplicated().any():
        raise ValueError("同一 order_id 存在不同记录，不能安全去重")
    if require_outcome and not orders.fulfilled.dropna().isin([0, 1]).all():
        raise ValueError("fulfilled 必须为 0、1 或缺失")
    for field in ("stock_level", "daily_traffic"):
        orders[field] = pd.to_numeric(orders[field], errors="raise")
        if (orders[field] < 0).any():
            raise ValueError(f"{field} 存在负值")
    for field in ("order_time", "last_sync_time"):
        orders[field + "_utc"] = pd.to_datetime(orders[field], utc=True, errors="raise")
    orders["sync_lag_min"] = (
        orders.order_time_utc - orders.last_sync_time_utc
    ).dt.total_seconds() / 60
    if (orders.sync_lag_min < 0).any():
        raise ValueError("存在库存同步时间晚于下单时间的订单")

    quality = {
        "raw_rows": len(raw),
        "duplicate_rows": len(raw) - len(orders),
        "unique_orders": len(orders),
        "unknown_results": int(orders.fulfilled.isna().sum()) if require_outcome else 0,
        "zero_stock_successes": int(((orders.stock_level == 0) & (orders.fulfilled == 1)).sum()) if require_outcome else 0,
    }
    return orders, quality


def grouped(known: pd.DataFrame, column: str) -> pd.DataFrame:
    result = known.groupby(column, observed=True).failed.agg(
        orders="size", failures="sum", failure_rate="mean"
    )
    result["failures"] = result.failures.astype(int)
    result["failure_rate"] = (result.failure_rate * 100).round(2)
    return result


def rule_metrics(known: pd.DataFrame, alerts: pd.Series) -> dict[str, float | int]:
    count = int(alerts.sum())
    hits = int(known.loc[alerts, "failed"].sum())
    failures = int(known.failed.sum())
    return {
        "预警数": count,
        "命中失败": hits,
        "未命中预警": count - hits,
        "命中率%": round(100 * hits / count, 2) if count else 0.0,
        "失败覆盖率%": round(100 * hits / failures, 2) if failures else 0.0,
        "净值上限_元": FAILURE_COST * hits - ALERT_COST * count,
    }


def rule_masks(known: pd.DataFrame) -> dict[str, pd.Series]:
    delayed = known.sync_lag_min >= THRESHOLD_MINUTES
    return {
        "A 延迟≥20分钟": delayed,
        "B 延迟≥20分钟且非华北分仓": delayed & (known.warehouse_region != "华北分仓"),
    }


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(table.to_string())


def analyze(path: Path) -> None:
    orders, quality = load_orders(path, require_outcome=True)
    known = orders.loc[orders.fulfilled.notna()].copy()
    known["failed"] = (known.fulfilled == 0).astype(int)
    success_count = int((known.fulfilled == 1).sum())
    failure_count = int(known.failed.sum())

    print("库存履约风险诊断（仅使用官方 CSV；订单级去重）")
    print(f"原始行数 {quality['raw_rows']}；完全重复行 {quality['duplicate_rows']}；唯一订单 {quality['unique_orders']}")
    print(f"结果缺失 {quality['unknown_results']}（不纳入失败率和回测）；已知结果 {len(known)}")
    print(f"成功 {success_count}；缺货失败 {failure_count}；失败率 {failure_count / len(known):.2%}")
    print(f"历史失败理论损失 {failure_count * FAILURE_COST} 元")
    print(f"账面库存为 0 但成功履约 {quality['zero_stock_successes']} 单")
    print(
        "同步延迟（分钟）：最小 {:.0f}，中位 {:.0f}，平均 {:.2f}，最大 {:.0f}".format(
            orders.sync_lag_min.min(), orders.sync_lag_min.median(),
            orders.sync_lag_min.mean(), orders.sync_lag_min.max(),
        )
    )

    known["延迟区间_分钟"] = pd.cut(
        known.sync_lag_min, [-float("inf"), 20, 30, float("inf")],
        labels=["<20", "20–29", "≥30"], right=False,
    )
    known["账面库存区间"] = pd.cut(
        known.stock_level, [-float("inf"), 2, 5, 10, 20, float("inf")],
        labels=["0–2", "3–5", "6–10", "11–20", "21–30"],
    )
    known["访问热度区间"] = pd.cut(
        known.daily_traffic, [-float("inf"), 42, 63, float("inf")],
        labels=["≤42", "43–63", ">63"],
    )
    for field in ("延迟区间_分钟", "账面库存区间", "访问热度区间", "warehouse_region", "category"):
        show(f"{field}：订单数 / 失败数 / 失败率%", grouped(known, field))

    low_stock_hot = (known.stock_level <= 5) & (known.daily_traffic > 63)
    show("低库存≤5且高访问热度>63（与延迟因素有重叠）", pd.DataFrame(
        [rule_metrics(known, low_stock_hot)], index=["组合"]
    ))
    show("低库存且高访问热度，按同步延迟分层", grouped(
        known.loc[low_stock_hot].assign(
            延迟分层=lambda frame: frame.sync_lag_min.ge(THRESHOLD_MINUTES).map(
                {False: "<20分钟", True: "≥20分钟"}
            )
        ), "延迟分层"
    ))
    show("同步延迟≥20分钟时的仓区差异", grouped(
        known.loc[known.sync_lag_min >= THRESHOLD_MINUTES], "warehouse_region"
    ))

    masks = rule_masks(known)
    show("全样本规则回看（100% 有效率时的净值上限）", pd.DataFrame.from_dict(
        {name: rule_metrics(known, mask) for name, mask in masks.items()}, orient="index"
    ))

    sensitivity = []
    for name, mask in masks.items():
        metrics = rule_metrics(known, mask)
        for effectiveness in (0.25, 0.50, 0.75, 1.00):
            sensitivity.append({
                "方案": name,
                "有效率%": int(effectiveness * 100),
                "假设净值_元": int(
                    FAILURE_COST * metrics["命中失败"] * effectiveness
                    - ALERT_COST * metrics["预警数"]
                ),
            })
    show("有效率敏感性（预警本身不等于成功避免失败）",
         pd.DataFrame(sensitivity).set_index(["方案", "有效率%"]))

    known["本地日期"] = known.order_time_utc.dt.tz_convert("Asia/Shanghai").dt.date
    dates = sorted(known["本地日期"].unique())
    if len(dates) != 7:
        print(f"\n时间切分：发现 {len(dates)} 个日期，跳过本任务约定的前5天/后2天验证")
    else:
        rows = []
        for label, selected_dates in (("前5天", set(dates[:5])), ("后2天", set(dates[5:]))):
            subset = known.loc[known["本地日期"].isin(selected_dates)]
            for name, mask in rule_masks(subset).items():
                rows.append({"时段": label, "方案": name, "已知结果订单": len(subset),
                             "失败数": int(subset.failed.sum()), **rule_metrics(subset, mask)})
        show("时间切分验证（方案和阈值按全样本探索确定；仅是稳定性检查）",
             pd.DataFrame(rows).set_index(["时段", "方案"]))
    print("\n说明：净值是假设性历史模拟；当前数据没有预警干预效果，不能证明真实收益或技术根因。")


def alert_new_orders(path: Path, *, compare_b: bool) -> None:
    orders, _ = load_orders(path, require_outcome=False)
    writer = csv.writer(sys.stdout)
    header = ["order_id", "sync_lag_min", "alert", "reason", "suggested_action"]
    if compare_b:
        header.append("plan_b_alert_reference")
    writer.writerow(header)
    for row in orders.itertuples(index=False):
        triggered = row.sync_lag_min >= THRESHOLD_MINUTES
        output = [
            row.order_id,
            f"{row.sync_lag_min:.2f}",
            int(triggered),
            f"库存同步延迟≥{THRESHOLD_MINUTES}分钟" if triggered else f"库存同步延迟<{THRESHOLD_MINUTES}分钟",
            "先复核实时库存与预占状态，再决定调配或下架" if triggered else "按常规履约流程处理",
        ]
        if compare_b:
            output.append(int(triggered and row.warehouse_region != "华北分仓"))
        writer.writerow(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("history", "alert"), default="history")
    parser.add_argument("--csv", type=Path, help="输入 CSV；历史模式默认为 data/raw/inventory_orders.csv")
    parser.add_argument("--compare-b", action="store_true", help="预警模式附加方案 B 对照列")
    args = parser.parse_args()
    if args.mode == "alert":
        if args.csv is None:
            parser.error("预警模式须指定 --csv 新订单文件")
        alert_new_orders(args.csv, compare_b=args.compare_b)
    else:
        if args.compare_b:
            parser.error("--compare-b 仅用于预警模式")
        analyze(args.csv or Path("data/raw/inventory_orders.csv"))
