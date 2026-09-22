"""geo-distance: great-circle distance between two points, graded on log relative error.

Signal design: one reward in [0, 1] that degrades smoothly with relative error, so a
group of all-wrong rollouts still ranks "off by 3%" above "off by 300%" and produces
advantage. Tiers are round-robined so any prefix (`-n 8`) is difficulty-balanced and a
weak policy has a rung to bootstrap on. Measured mean reward per rung (defaults, see
test_geo_distance.py): the best constant guess scores 0.04/0.10/0.16/0.14 per tier, a
flat-earth approximation 1.00/1.00/0.67/0.49, exact haversine 1.00 everywhere.
"""

import math
import random
import re
from typing import Literal, Self, get_args

from pydantic import Field, model_validator

import verifiers.v1 as vf

R_KM = 6371.0
"""Sphere radius quoted in the prompt; the reference answer uses the same one."""
DEG_KM = R_KM * math.pi / 180.0

Tier = Literal["axis", "near", "general", "parallel"]
TIERS: tuple[Tier, ...] = get_args(Tier)

PROMPT = """Points A and B lie on a sphere of radius {r} km.
A: latitude {lat1:.6f}, longitude {lon1:.6f}
B: latitude {lat2:.6f}, longitude {lon2:.6f}
How far apart are A and B along the surface (great-circle distance)?
Answer in kilometres, as a single number in \\boxed{{}}."""


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_KM * math.asin(min(1.0, math.sqrt(a)))


def destination(lat: float, lon: float, bearing: float, dist_km: float) -> tuple[float, float]:
    """The point `dist_km` from (lat, lon) along `bearing` degrees."""
    p1, l1 = math.radians(lat), math.radians(lon)
    th, d = math.radians(bearing), dist_km / R_KM
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(th))
    l2 = l1 + math.atan2(
        math.sin(th) * math.sin(d) * math.cos(p1),
        math.cos(d) - math.sin(p1) * math.sin(p2),
    )
    return math.degrees(p2), _wrap(math.degrees(l2))


def _wrap(lon: float) -> float:
    return (lon + 540.0) % 360.0 - 180.0


_NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")
# Digit separators and unit/markup noise that shows up inside \boxed{}.
_NOISE = re.compile(
    r"\\text\{[^}]*\}|\\mathrm\{[^}]*\}|[,\s{}]|\\[,;:!]|\\approx|~"
    r"|kilometers?|kilometres?|km",
    re.IGNORECASE,
)
# LaTeX scientific notation, e.g. 1.2 \times 10^{4}.
_SCI = re.compile(r"^([-+]?\d*\.?\d+)\\?(?:times|cdot)?\*?10\^?\{?([-+]?\d+)\}?$")


def _to_float(text: str) -> float | None:
    try:
        value = float(text.replace(",", ""))
    except (ValueError, OverflowError):
        return None
    # A non-finite guess would make the relative error NaN and poison the reward.
    return value if math.isfinite(value) else None


def parse_km(reply: str) -> float | None:
    """The boxed answer if there is one, else the last number in the reply.

    The fallback is deliberate: a format miss is reward *noise*, not signal, and the
    number still has to be right, so leniency here cannot be farmed.
    """
    boxed = _NOISE.sub("", vf.extract_boxed_answer(reply, strict=True))
    if boxed:
        if sci := _SCI.match(boxed):
            return _to_float(f"{sci.group(1)}e{sci.group(2)}")
        if num := _NUM.search(boxed):
            return _to_float(num.group())
    numbers = _NUM.findall(reply)
    return _to_float(numbers[-1]) if numbers else None


def score(rel_error: float, full: float, zero: float) -> float:
    """1.0 at/below `full` relative error, 0.0 at/above `zero`, log-linear between."""
    if rel_error <= full:
        return 1.0
    if rel_error >= zero:
        return 0.0
    return math.log(zero / rel_error) / math.log(zero / full)


class GeoDistanceData(vf.TaskData):
    tier: Tier
    lat1: float = Field(ge=-90.0, le=90.0)
    lon1: float = Field(ge=-180.0, le=180.0)
    lat2: float = Field(ge=-90.0, le=90.0)
    lon2: float = Field(ge=-180.0, le=180.0)
    answer_km: float = Field(gt=0.0)
    """Haversine distance over the *rounded* coordinates the prompt shows."""


class GeoDistanceTaskConfig(vf.TaskConfig):
    full_credit_err: float = Field(0.01, gt=0.0, lt=1.0)
    """Relative error at or below which an answer scores 1.0."""
    zero_credit_err: float = Field(0.5, gt=0.0)
    """Relative error at or above which an answer scores 0.0."""

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.zero_credit_err <= self.full_credit_err:
            # Swapped thresholds would silently invert the reward.
            raise ValueError("zero_credit_err must exceed full_credit_err")
        return self


class GeoDistanceTask(vf.Task[GeoDistanceData, vf.State, GeoDistanceTaskConfig]):
    @vf.reward(weight=1.0)
    async def accuracy(self, trace: vf.Trace) -> float:
        guess = parse_km(trace.last_reply)
        trace.info["tier"] = self.data.tier
        trace.info["guess_km"] = guess
        trace.record_metrics(
            {"answered": float(guess is not None), "truncated": float(trace.is_truncated)}
        )
        if guess is None:
            return 0.0
        rel = abs(guess - self.data.answer_km) / self.data.answer_km
        reward = score(rel, self.config.full_credit_err, self.config.zero_credit_err)
        trace.record_metrics({"rel_error": min(rel, 10.0), "solved": float(reward == 1.0)})
        return reward


class GeoDistanceConfig(vf.TasksetConfig):
    num_tasks: int = Field(400, ge=1)
    seed: int = 0
    """Generator seed — give train and eval different ones."""
    min_km: float = Field(10.0, gt=0.0)
    """Shortest distance generated; below ~5 km the 6-decimal coordinates quantize the
    answer by more than `full_credit_err`."""
    near_km: float = Field(200.0, gt=0.0)
    """Boundary between the `near` tier and the `general` tier."""
    max_km: float = Field(19_500.0, gt=0.0)
    task: GeoDistanceTaskConfig = GeoDistanceTaskConfig()


class GeoDistanceTaskset(vf.Taskset[GeoDistanceTask, GeoDistanceConfig]):
    def load(self) -> list[GeoDistanceTask]:
        rng = random.Random(self.config.seed)
        tasks = []
        for idx in range(self.config.num_tasks):
            tier = TIERS[idx % len(TIERS)]
            # Grade the coordinates the model is shown, not the ones used to place B.
            lat1, lon1, lat2, lon2 = (round(v, 6) for v in self._pair(tier, rng))
            tasks.append(
                GeoDistanceTask(
                    GeoDistanceData(
                        idx=idx,
                        name=f"geo:{tier}#{idx}",
                        prompt=PROMPT.format(
                            r=R_KM, lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2
                        ),
                        tier=tier,
                        lat1=lat1,
                        lon1=lon1,
                        lat2=lat2,
                        lon2=lon2,
                        answer_km=haversine_km(lat1, lon1, lat2, lon2),
                    ),
                    self.config.task,
                )
            )
        return tasks

    def _pair(self, tier: Tier, rng: random.Random) -> tuple[float, float, float, float]:
        """One point pair per difficulty rung. See the module docstring for the ladder."""
        config = self.config
        if tier == "axis":  # one coordinate varies: distance is a single multiplication
            span = _log_uniform(rng, config.min_km, 20_000.0) / DEG_KM
            if rng.random() < 0.5:  # same meridian
                lat1, lon = rng.uniform(-90.0, 90.0 - span), rng.uniform(-180.0, 180.0)
                return lat1, lon, lat1 + span, lon
            lon1 = rng.uniform(-180.0, 180.0)  # both on the equator
            return 0.0, lon1, 0.0, _wrap(lon1 + span)
        if tier == "parallel":  # same latitude, far apart: the parallel arc is NOT the answer
            lat = rng.choice((-1.0, 1.0)) * rng.uniform(25.0, 80.0)
            lon1 = rng.uniform(-180.0, 180.0)
            return lat, lon1, lat, _wrap(lon1 + rng.choice((-1.0, 1.0)) * rng.uniform(20.0, 180.0))
        dist = (
            # Short and log-uniform: flat-earth is exact here, so this rung is reachable.
            _log_uniform(rng, config.min_km, config.near_km)
            if tier == "near"
            # Uniform over the whole range: mass sits where flat-earth breaks down, and a
            # constant guess still averages only 0.16 (see test_geo_distance.py).
            else rng.uniform(config.near_km, config.max_km)
        )
        lat1 = math.degrees(math.asin(2 * rng.random() - 1))  # uniform on the sphere
        lon1 = rng.uniform(-180.0, 180.0)
        return (lat1, lon1, *destination(lat1, lon1, rng.uniform(0.0, 360.0), dist))


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


__all__ = ["GeoDistanceTaskset"]
