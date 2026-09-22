# RL_Training

RL environments and eval configs built on [`verifiers`](https://github.com/willccbb/verifiers).

## Environments

### [`geo-distance`](environments/geo_distance/)

Great-circle distance between two lat/lon points. Single turn, no tools, graded on
relative error. Built to be trained on, not just evaluated against:

- **Not farmable.** Points are placed at a log-uniform separation along a random
  bearing, so no constant answer scores well — `test_not_farmable` asserts none beats
  `0.16`. (Sampling two points independently on a sphere averages ~10,000 km apart,
  which a constant would exploit.)
- **Tiered difficulty**, round-robined so any prefix (`-n 8`) stays balanced. The
  `parallel` tier is a deliberate trap: the arc *along* a parallel is not the
  great-circle distance, and answering it scores `0.49`.
- **Log-linear partial credit** between 1% and 50% relative error, so a group of
  all-wrong rollouts still produces a usable advantage under GRPO.

Full design notes in [`environments/geo_distance/README.md`](environments/geo_distance/README.md).

## Layout

```
environments/     verifiers tasksets
configs/eval/     evaluation sweeps
configs/gepa/     GEPA prompt optimisation
configs/rl/       prime-rl training
```

Configs cover gpt-oss, llama-3, nemotron-3, qwen-3-5 and qwen-3-5-moe.

## Quickstart

```bash
uv pip install -e environments/geo_distance
uv run python environments/geo_distance/test_geo_distance.py   # self-check, no network
uv run eval @ configs/eval/geo-distance.toml
```
