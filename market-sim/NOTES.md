# Four-Agent Daily Price-Fitting Simulation (v2) — Working Notes

Six steps in plain language (run in order; all outputs land in `data/` / `outputs/`):

## 1. Graph structure
Build the directed upstream-to-downstream supply graph from companies + relations
(`power_cooling` at the top, `cloud_buyers` at the bottom).
Exports `data/graph_nodes.csv`, `data/graph_edges.csv`, `outputs/graph_overview.png`.

## 2. The four agent types (four genuinely separate classes in the code)
Same speed, unlimited capital:
1. **PublicAgent** — sees only the public tape / public events
2. **ChannelAgent** — public + channel messages from neighbours along supply edges
3. **StructureAgent** — public + position bottleneck score; quotes `alpha*z(bottleneck) + beta*msg_sign`
4. **LearningAgent** — may fit parameters on realized prices in the first half; frozen in the second half

Each agent: `observe(info) -> belief`; `quote(prev_price)`; in the training half it may
`update_after_reveal(true_price)`.
Quote rule B: nudge from yesterday's close, clipped to +/-5%.

## 3. Price surface + value surface
- **Price surface** = the realized traded-price panel (`data/price_surface.csv`); **never
  overwritten by fitted values**
- **Value surface** = equal-weighted position bottleneck score (downstream coverage x low
  substitutability x edge scarcity); a research supply proxy, **not official productivity**

## 4. Input information shaping
events -> per-day, per-company `public_packet` / `channel_packet`; parse failures -> 0;
strict PIT (`usable_from_trade_date <= current day`).

## 5. Daily simulation (no whole-panel one-shot OLS)
`formal_2023_2025` trading days split in half; warmup supplies lags only; `holdout_2026`
is excluded from the main table.

**Each trading day:**
1. Publish the day's visible information packets to the four agent types
2. Each agent quotes from yesterday's price + visible information
3. **Equal-weight average of the four quotes -> `fitted_price` (a fitted price, not a new
   traded price)**
4. Only then is the day's exogenous **`true_price`** revealed
5. Training half: update strategies on the realized price; test half: parameters frozen,
   no looking at the realized price and re-tuning

## 6. Arms and metric
Same blind protocol: `ma7` | `public_only` | `lgbm_lag` | `four_agent_fit`
Primary metric: test-period **MAPE%** of fitted vs realized.

## Run

```bash
cd /workspace/agent_market_sim_v2
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python run_sim.py
```

Should take visibly more than a few seconds (daily loop). `elapsed` is written to
`outputs/run_metadata.json`.

## Pinned definitions
- `fitted_price` = the equal-weighted composite of the four quotes — a **fitted price**
- `true_price` = the exogenous realized traded price, used after revelation only for
  training-half updates / test-half MAPE
- Never write quotes back as a new traded-price surface
