"""Self-check for the geo-distance taskset.

    uv run python environments/geo_distance/test_geo_distance.py

Covers the four things that can silently corrupt the reward — the geometry, the answer
parser, the score curve, the wiring — plus the one that decides whether this is worth
training on: how much reward a cheap non-solution collects.
"""

import asyncio
import math
import random
from collections import Counter

import verifiers.v1 as vf
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import AgentInfo
from verifiers.v1.types import AssistantMessage

from geo_distance.taskset import (
    DEG_KM,
    R_KM,
    GeoDistanceConfig,
    GeoDistanceTaskset,
    destination,
    haversine_km,
    parse_km,
    score,
)


def test_geometry():
    # Placing a point at a known distance and measuring it back must round-trip.
    rng = random.Random(0)
    worst = 0.0
    for _ in range(5000):
        lat = math.degrees(math.asin(2 * rng.random() - 1))
        lon = rng.uniform(-180, 180)
        target = math.exp(rng.uniform(math.log(10), math.log(19_000)))
        lat2, lon2 = destination(lat, lon, rng.uniform(0, 360), target)
        worst = max(worst, abs(haversine_km(lat, lon, lat2, lon2) - target) / target)
    assert worst < 1e-9, f"round-trip drift {worst}"

    assert haversine_km(10, 20, 10, 20) == 0.0
    assert math.isclose(haversine_km(0, 0, 0, 180), math.pi * R_KM, rel_tol=1e-12)
    assert math.isclose(haversine_km(90, 0, -90, 0), math.pi * R_KM, rel_tol=1e-12)
    # Crossing the dateline is 1 degree apart, not 359.
    assert math.isclose(haversine_km(0, 179.5, 0, -179.5), math.radians(1) * R_KM, rel_tol=1e-9)
    # A same-latitude pair is SHORTER than the parallel arc (the `parallel` tier trap).
    assert haversine_km(60, 0, 60, 90) < math.radians(90) * R_KM * math.cos(math.radians(60))


def test_parser():
    cases = {
        r"the answer is \boxed{1234.5} km": 1234.5,
        r"\boxed{12 345}": 12345.0,
        r"\boxed{1,234.5}": 1234.5,
        r"\boxed{1{,}234.5\text{ km}}": 1234.5,
        r"\boxed{1.2345e3}": 1234.5,
        r"\boxed{\approx 608}": 608.0,
        r"\boxed{6.08\times10^2}": 608.0,
        r"\boxed{ 5539 }": 5539.0,
        "no box, just 1,234.5 km": 1234.5,
        "first 100 then finally 987.6": 987.6,  # unboxed: the LAST number wins
        r"\boxed{1e999}": None,  # never return a non-finite guess: it would NaN the reward
        "no number here": None,
        "": None,
    }
    for reply, want in cases.items():
        got = parse_km(reply)
        assert got == want or (got is not None and math.isclose(got, want)), (reply, got, want)


def test_score_curve():
    full, zero = 0.01, 0.5
    assert score(0.0, full, zero) == 1.0
    assert score(full, full, zero) == 1.0
    assert score(zero, full, zero) == 0.0
    assert score(5.0, full, zero) == 0.0
    # Strictly decreasing in between, so an all-wrong rollout group still has a gradient.
    mid = [score(e, full, zero) for e in (0.02, 0.05, 0.1, 0.3)]
    assert all(a > b for a, b in zip(mid, mid[1:])), mid
    assert all(0.0 < v < 1.0 for v in mid), mid
    assert 0.3 < mid[1] < 0.7, mid  # 5% off lands mid-scale, not pinned to an end


def test_taskset():
    tasks = GeoDistanceTaskset(GeoDistanceConfig(num_tasks=40, seed=0)).load()
    assert len(tasks) == 40
    # Tiers are round-robined, so any prefix stays difficulty-balanced.
    assert len(set(Counter(t.data.tier for t in tasks).values())) == 1

    for task in tasks:
        row = task.data
        # The reference answer must match the coordinates the prompt actually shows.
        assert math.isclose(
            row.answer_km, haversine_km(row.lat1, row.lon1, row.lat2, row.lon2), rel_tol=1e-12
        )
        assert row.answer_km > 0
        # The prompt shows the coordinates at the precision they were rounded to.
        for value in (row.lat1, row.lon1, row.lat2, row.lon2):
            assert f"{value:.6f}" in str(row.prompt)

    # Same seed, same tasks; different seed, different tasks (train/eval must diverge).
    again = GeoDistanceTaskset(GeoDistanceConfig(num_tasks=40, seed=0)).load()
    assert [t.data.answer_km for t in again] == [t.data.answer_km for t in tasks]
    other = GeoDistanceTaskset(GeoDistanceConfig(num_tasks=40, seed=1)).load()
    assert [t.data.answer_km for t in other] != [t.data.answer_km for t in tasks]

    answers = sorted(t.data.answer_km for t in tasks)
    assert answers[0] < 500 < 5000 < answers[-1]


def reward_for(task, reply: str) -> float:
    """Score a real Trace through `Task.score`, so the decorator wiring is exercised."""
    trace = vf.Trace(
        task=vf.TraceTask(type="GeoDistanceTask", data=task.data, key=task.key, hash=task.hash),
        agent=AgentInfo(config=AgentConfig()),
        nodes=[MessageNode(message=AssistantMessage(content=reply), sampled=True)],
        ok=True,
    )
    asyncio.run(task.score(trace))
    assert set(trace.rewards) == {"accuracy"}, trace.rewards
    assert trace.info["tier"] == task.data.tier
    return trace.reward


def test_reward():
    task = GeoDistanceTaskset(GeoDistanceConfig(num_tasks=4, seed=0)).load()[0]
    truth = task.data.answer_km
    assert reward_for(task, rf"\boxed{{{truth:.6f}}}") == 1.0
    assert reward_for(task, rf"\boxed{{{truth * 1.005:.6f}}}") == 1.0
    assert reward_for(task, rf"\boxed{{{truth * 3:.6f}}}") == 0.0
    assert reward_for(task, "I don't know") == 0.0
    assert reward_for(task, "") == 0.0
    assert 0.0 < reward_for(task, rf"\boxed{{{truth * 1.05:.6f}}}") < 1.0


def baselines(num_tasks: int = 2000) -> dict[str, float]:
    """Mean reward of cheap non-solutions — the anti-hacking measurement.

    The constant baseline is taken per tier, not globally: the policy sees the
    coordinates, so it can tell the tiers apart and specialise its guess.
    """
    config = GeoDistanceConfig(num_tasks=num_tasks)
    rows = [t.data for t in GeoDistanceTaskset(config).load()]
    full, zero = config.task.full_credit_err, config.task.zero_credit_err

    def mean(guess, tier=None) -> float:
        picked = [r for r in rows if tier is None or r.tier == tier]
        return sum(
            score(abs(guess(r) - r.answer_km) / r.answer_km, full, zero) for r in picked
        ) / len(picked)

    def flat_earth(row) -> float:  # equirectangular: exact nearby, breaks down far away
        mid = math.radians((row.lat1 + row.lat2) / 2)
        dlon = (row.lon2 - row.lon1 + 540) % 360 - 180
        return math.hypot(row.lat2 - row.lat1, dlon * math.cos(mid)) * DEG_KM

    grid = range(int(12 * math.log(10)), int(12 * math.log(20_000)))
    constants = [math.exp(x / 12) for x in grid]
    out = {"oracle": mean(lambda r: r.answer_km)}
    for tier in ("axis", "near", "general", "parallel"):
        out[f"constant@{tier}"] = max(mean(lambda r, c=c: c, tier) for c in constants)
        out[f"flat_earth@{tier}"] = mean(flat_earth, tier)
    return out


def test_not_farmable():
    b = baselines()
    assert b["oracle"] == 1.0, b
    # No fixed number earns a living, even one specialised to a single tier.
    assert max(v for k, v in b.items() if k.startswith("constant@")) < 0.25, b
    assert b["flat_earth@near"] > 0.99, b  # the bootstrap rung is reachable...
    assert b["flat_earth@general"] < 0.75, b  # ...and leaves real headroom above it
    assert b["flat_earth@parallel"] < 0.60, b  # the same-latitude trap bites hardest


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print()
    for name, value in baselines().items():
        print(f"  {name:22s} {value:.3f}")
    print("\nall checks passed")
