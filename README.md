# Lanju Wei — Interactive Research Portfolio

Code and interactive pages supporting my PhD applications. My research question is narrow and
repeated across all of it: **when market participants hold different information and are connected
in different ways, how does that information enter their decisions, and what does the aggregate
look like?**

## Contents

| Item | What it is |
|---|---|
| [Asset Universe (interactive)](asset_universe_en.html) | Interactive observatory of a 33-firm AI-infrastructure market: layered and spherical views of price, value and information state over 948 trading days. Open in a browser; no install. |
| [`market-sim/`](market-sim/) | Code, pre-registered protocol, input data and results for the four-agent price-fitting experiment. Includes the null result and the limitations. |
| [`empirical/`](empirical/) | Supply-chain information diffusion: the merged 14-module code, the unified result table, and the full reasoning notes. Derived panel not redistributed. |
| [`docs/`](docs/) | Design handbook v1.2.1 (English) — the honest record of what the system does and does not claim to do, including the specified-but-not-run matching engine. |
| [`WTI_stable_offline_dashboard_.html`](WTI_stable_offline_dashboard_.html) | Dashboard for my master's thesis: a neural heterogeneous-agent model of WTI, 986 weekly observations. |
| [`personal_knowledge_graph_structure.html`](personal_knowledge_graph_structure.html) | A map of what I studied outside my formal curricula — the reading that took me from public administration to economics. |

## Two experiments, and how they differ

**Empirical (supply-chain information diffusion, `empirical/`).** Event study and lead-lag panel on
the same 40-firm panel: do economically linked firms react to each other's disclosures, in what
order, and does the reaction survive market and industry controls plus network nulls. The merged
code, the unified result table and the full reasoning notes are included; the derived panel is not
redistributed (licensed source data), and the method is documented so it can be rebuilt.

**Simulation (four-agent price fitting, `market-sim/`).** Four agent types with different
information access post daily quotes; the quotes are averaged into a fitted price and compared
against three blind-protocol baselines. Result: the four-agent combination beats a rolling mean and
a gradient-boosted baseline on lagged prices, and **does not beat a public-information-only model**.
That null is reported as it came out, because the point of the exercise was to find out whether
channel information and network position add anything once public information is accounted for.

## What is deliberately absent

No order book, no matching engine, no cash-and-position constraints. The realized price in
`market-sim/` is exogenous. An order-layer module is specified but **not implemented, not run, and
no claim here depends on it**. Simulating a traded price from daily data and calling it a matching
outcome would be a claim the data cannot support.

## Contact

Lanju Wei — lanjuwei47@gmail.com
