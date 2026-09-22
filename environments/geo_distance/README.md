# geo-distance

### Overview
- **Environment ID**: `geo-distance`
- **Short description**: Given two latitude/longitude points on a sphere, answer the great-circle distance in kilometres. Single turn, no tools, graded on relative error.
- **Tags**: geometry, math, train, eval

### Datasets
Generated, not downloaded — `random.Random(seed)` makes a run reproducible and
`seed` is the only thing separating a train split from an eval split.

Point B is *placed* at a log-uniform distance from point A along a random bearing, so
separations span 10 km to ~19,500 km evenly in log space. That matters: two points
sampled independently on a sphere are ~10,000 km apart on average, which would let a
policy score by always answering "10000". Here no constant does better than 0.16
(see `test_not_farmable`).

Every row belongs to one of four tiers, round-robined so any prefix (`-n 8`) stays
difficulty-balanced:

| Tier | What it is | Why |
| --- | --- | --- |
| `axis` | Both points share a meridian, or both sit on the equator | Distance is one multiplication — the rung a weak policy can reach |
| `near` | Separation below `near_km` (200 km) | The flat-earth approximation is still exact enough to score 1.0 |
| `general` | Arbitrary pair, `near_km` to `max_km` | Needs real haversine; flat-earth scores 0.67 |
| `parallel` | Both points share a latitude | Trap: the arc *along* the parallel is not the great-circle distance, and answering it scores 0.49 |

### Task
- **Type**: single-turn
- **Output format**: a single number in kilometres inside `\boxed{}`. An unboxed reply
  falls back to the last number in the text — a format miss is reward noise, not
  signal, and the number still has to be right.
- **Rubric**: one reward, `accuracy`.

The package exports `NullHarness`, which makes the default harness a single plain
model turn. Without it the loader falls back to `bash`, which both provisions a billed
container and hands the model a shell it could just compute the answer in.

### Quickstart

```bash
uv pip install -e environments/geo_distance
uv run python environments/geo_distance/test_geo_distance.py   # self-check, no network
uv run eval @ configs/eval/geo-distance.toml
```

The config pins `env.agent.runtime.type = "subprocess"`. The default (`prime`)
provisions a sandbox per task, which this environment has no use for.

### Taskset Config

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `num_tasks` | int | `400` | How many rows to build |
| `seed` | int | `0` | Generator seed; use a different one for eval than for train |
| `min_km` | float | `10.0` | Shortest separation generated |
| `near_km` | float | `200.0` | Boundary between the `near` and `general` tiers |
| `max_km` | float | `19500.0` | Longest separation generated |
| `task.full_credit_err` | float | `0.01` | Relative error at or below which the answer scores 1.0 |
| `task.zero_credit_err` | float | `0.5` | Relative error at or above which it scores 0.0 |

Between the two thresholds the score is log-linear, so a group of all-wrong rollouts
still ranks "off by 3%" above "off by 300%" and produces a usable advantage. Widening
`zero_credit_err` gives a denser gradient; narrowing it toward `full_credit_err`
approaches pass/fail.

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | The weighted sum — here just `accuracy` |
| `accuracy` | 1.0 within `full_credit_err`, 0.0 beyond `zero_credit_err`, log-linear between |
| `solved` | Fraction scoring exactly 1.0 — the strict accuracy to report |
| `rel_error` | Relative error, clipped at 10 so one wild guess cannot dominate the mean |
| `answered` | Whether a number could be parsed at all |
| `truncated` | Whether the reply hit the token limit (a low `answered` with a high `truncated` means raise `max_tokens`, not that the policy failed) |

`trace.info` also carries `tier` and `guess_km`, so scores can be broken down per tier.
