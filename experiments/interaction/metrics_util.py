"""Per-episode event rate helpers for interaction training logs."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple


def episode_event_rates(
    infos: Iterable[dict],
    keys: Tuple[str, ...] = ("collision", "success", "row_violation"),
) -> Tuple[Dict[str, float], int]:
    """Fraction of episodes where each event was True at least once.

    Episodes are split when ``info['step']`` does not increase (env reset).
    """
    prev_step = None
    ep_flags = {k: False for k in keys}
    totals = {k: 0 for k in keys}
    n_eps = 0

    for info in infos:
        st = int(info.get("step", 0))
        if prev_step is not None and st <= prev_step:
            n_eps += 1
            for k in keys:
                totals[k] += int(ep_flags[k])
            ep_flags = {k: False for k in keys}

        ev = info.get("events", {}) or {}
        for k in keys:
            if ev.get(k):
                ep_flags[k] = True
        prev_step = st

    n_eps += 1
    for k in keys:
        totals[k] += int(ep_flags[k])

    denom = max(n_eps, 1)
    rates = {k: totals[k] / denom for k in keys}
    return rates, n_eps
