# Pre-registration / Fixed Protocol (agent_market_sim_v2)

1. Supply-chain graph: `power_cooling` at the top, `cloud_buyers` at the bottom; only supply-type
   relations become directed edges.
2. Value surface = equal-weighted position bottleneck score (downstream coverage x low
   substitutability x edge scarcity); a research supply proxy, not official productivity.
3. Price surface = realized traded prices; **the fitted price must never overwrite the realized
   price panel**.
4. Four independent agent types, same speed: Public / Channel / Structure / Learning.
5. Quote rule B: `prev_close * (1 + clip(nudge, -0.05, 0.05))`; Structure:
   `alpha * z(bottleneck) + beta * msg_sign`.
6. Windows: only `formal_2023_2025`, split in half; warmup supplies lags only; `holdout_2026`
   is excluded from the main table.
7. Daily loop: publish visible information -> quotes -> equal-weight `fitted_price` -> reveal
   exogenous `true_price` -> updates in the training half only.
8. Parameters frozen in the test half; no tuning on same-day realized prices.
9. Arms: `ma7` | `public_only` | `lgbm_lag` (lagged price/volume only) | `four_agent_fit`.
10. Primary metric: test-period MAPE% (fitted vs realized).
