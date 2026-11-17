# Four-Agent Price-Fitting Simulation (v2)

A daily simulation over a 40-firm AI-infrastructure supply chain, asking a narrow question:
**does channel information and network position add explanatory power beyond public information alone?**

This directory is the code, the pre-registered protocol, the input data, and the output table behind
the simulation described in my statement of purpose. It is released so that the claims can be checked
rather than taken on trust.

---

## What this is

Four agent types observe the same day's information under different access rules and each posts a
quote; the four quotes are averaged into a **fitted price**; only then is the day's realized price
revealed. Parameters are calibrated on the first half of 2023-2025 and frozen for the second half.

| Agent | What it observes | Quote rule |
|---|---|---|
| `PublicAgent` | public events only | yesterday's close nudged by public signal, clipped ±5% |
| `ChannelAgent` | public + channel messages along supply edges | same rule, channel-weighted |
| `StructureAgent` | public + position bottleneck score | `α·z(bottleneck) + β·msg_sign` |
| `LearningAgent` | public, plus realized prices in the training window | rule re-fit on training window only |

## What this is not

- **It is not an order-driven market.** The realized price is exogenous, taken from the market panel;
  no limit order book, no order matching, no cash or position constraints are simulated here.
  An order-layer module is specified in the design handbook but has **not been run**, and no result in
  this repository depends on it.
- **The fitted price is not a traded price.** It is the equal-weighted average of four quotes.
- **The value surface is a supply-side proxy**, not measured productivity: downstream coverage ×
  low substitutability × edge scarcity.

## Protocol (`prereg.md`)

1. Directed supply graph, upstream `power_cooling` → downstream `cloud_buyers`.
2. Price surface = realized closes. The fitted price never overwrites it.
3. 2023-2025 trading days split in half: calibration half, test half. 2026 held out, excluded from the main table.
4. Daily loop: publish visible packets → each agent quotes → equal-weight fitted price → reveal exogenous true price → training-half agents may update.
5. Test-half parameters frozen; no same-day peeking.
6. Four blind-protocol arms: `ma7`, `public_only`, `lgbm_lag`, `four_agent_fit`.
7. Metric: test-period MAPE between fitted and realized price.

## Result (`outputs/metrics_summary.csv`)

| Model | MAPE % | n |
|---|---|---|
| `ma7` (7-day rolling mean) | 4.160 | 14,700 |
| `public_only` | **2.316** | 14,700 |
| `lgbm_lag` (lagged price/volume, LightGBM) | 2.331 | 14,702 |
| `four_agent_fit` | 2.323 | 14,700 |

Reading it plainly: the four-agent combination beats the rolling mean and the gradient-boosted
baseline on lagged prices, and **does not beat a public-information-only model**. Adding channel
and structural information did not buy accuracy in this test. That null is the most useful thing
this experiment produced — it says the extra information may be redundant with public news, or
used by rules that are too weak to exploit it, and that the contribution of information content,
network structure and agent response has to be separated before any of it is believed.

## Run

```bash
pip install -r requirements.txt
python run_sim.py
```

Writes `outputs/run_metadata.json`, `outputs/metrics_summary.csv`, `outputs/baseline_comparison.png`.
Runtime is a few seconds; the daily loop is explicit rather than vectorized so that the timing of
information release stays auditable.

## Files

```
run_sim.py              daily loop, six steps, prints each stage
prereg.md               protocol fixed before running
src/config.py           windows, seeds, paths
src/graph.py            supply graph construction
src/surfaces.py         price and value surfaces
src/info_packets.py     PIT-safe event → per-day packets
src/agents/             public, channel, structure, learning, base
src/baselines.py        ma7, public-only, LightGBM on lagged price/volume
src/metrics.py          MAPE
src/simulation.py       orchestration
data/                   graph, price surface, value surface, sample packets
outputs/                metrics table, baseline comparison figure
NOTES.md                working notes (plain-language walkthrough)
```

## Data note

`data/price_surface.csv` is a derived panel built from public market data for research use; it is
included only so the protocol can be re-run end to end. The event packets shipped here are a
**sample** (`info_packets_sample.csv`); the full packet set is not redistributed.
