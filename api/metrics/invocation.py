"""
Track general invocation metrics in Prometheus.
"""

from prometheus_client import Counter


usage_usd = Counter(
    "usage_usd_total",
    "Total USD usage charged to users (includes paygo equivalent for subs)",
    ["chute_id"],
)
compute_seconds = Counter(
    "compute_seconds_total",
    "Total compute seconds across all invocations",
    ["chute_id"],
)


def track_invocation_usage(
    chute_id: str, balance_used: float, compute_time: float, paygo_amount: float = 0.0
):
    """
    Track USD usage and compute seconds per chute for miner metrics.
    """
    # Use paygo_amount (includes subscription equivalent) for revenue tracking.
    # Falls back to balance_used for non-subscription invocations.
    revenue = paygo_amount if paygo_amount > 0 else balance_used
    if revenue > 0:
        usage_usd.labels(chute_id=chute_id).inc(revenue)
    if compute_time > 0:
        compute_seconds.labels(chute_id=chute_id).inc(compute_time)
