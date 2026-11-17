# Empirical Track — Supply-Chain Information Diffusion

Event-study and lead-lag evidence on the same 40-firm AI-infrastructure panel behind
[`market-sim/`](../market-sim/). Where the simulation asks *what extra information can add*,
this track asks *what the realized data actually show*: whether economically linked firms react
to each other's disclosures, in what order, and whether that reaction survives market and
industry controls and network nulls.

## What's here

| File | What it is |
|---|---|
| `supply_chain_diffusion_all.py` | The 14 modules merged into one readable file: event-signal mapping, stylized facts, lead-lag panel, earnings event study, three diffusion sub-questions, supplier→customer robustness, out-of-sample test, minimal ABM, an exact Lux-Marchesi (2000) replication, and the supply-chain ABM with ablation. Each module prints its own results and can be read independently. |
| `results_summary_2026-11-07_en.md` | The unified result table: every reported number on one page — the only figures allowed in outward-facing material. |
| `full_reasoning_notes_v2_en.pdf` | The full working notes (English, v2): why each test was designed the way it was, including the checks that came back null. |

## Results in one line

Disclosure-linked reactions exist and cluster along disclosed relations; the effects are small,
survive the strictest controls only partially, and are reported as measured — including the nulls
(position × speed; several robustness checks).

## Data note

The panel is derived from licensed market data and is **not redistributed** here. The code
documents the full method; the reasoning notes record every design decision, so the analysis can
be followed and re-built from public sources. If you want the extracted panel or the per-module
scripts, email me.

Working notes and the result table are in English. Every module header and variable in the code
is in English.
