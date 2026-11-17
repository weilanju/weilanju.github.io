# Unified Reporting Numbers (2026-11-07)

> Purpose: every outward-facing document (RP / writing sample / SOP / cold email) must use only the numbers in this file.
> Any older figure that disagrees with this table (+0.14%, +1.98%, +6.50%, +5.01%, p=0.002 "beyond all 500 draws") is withdrawn, or kept only as a definitional note.
> Database: `ai_infra_2022_2026_v3_data.db`; scripts: `Market Wave/analysis/exp1_bh_fdr.py`, `exp2_liquidity.py` (corrected), `exp3_network_null.py`.

## 0. Sample definition (align on this first)

- Nodes: 53 firms; supply-side edges: 19 undirected, de-duplicated (supply 16 + foundry 3);
  **12 of the 19 have U.S.-listed prices at both ends** (the event study can only use these 12).
- Analyst events (2023-01-03 to 2025-12-31, directional): 159 events (X, date, d), 332 event–neighbour pairs (217 upgrades / 115 downgrades; upgrades 65%).
- Abnormal return: stock return minus the same-day equal-weighted mean of the 40 AI stocks (not SPY).

## 1. Main effect (canonical)

| Measure | Value | Test | Verdict |
|---|---|---|---|
| Event-level signed AR (main estimate) | **+1.21%** | t=2.55 (unclustered) / **t=2.50 (day-clustered)**, p=0.011 | Significant |
| BH-FDR across 5 test paths | p=0.0107 → **q=0.027** | Passes together with short volume (p=3.7e-5) | Significant |
| Network null (500 degree-preserving randomisations, signed) | Real **+1.56%** vs null mean **+0.88%** (95th pct +1.45%, max +1.58%) | **Permutation p=0.004** (below 499/500) | Structural effect, not a density effect |
| Directional asymmetry (m+ − m−, new measure) | Real **+1.92%** vs null **+0.98%** | **Permutation p=0.002** | Directionality is edge-specific |

## 2. By direction (the single most important mechanism finding)

| | Neighbour T+1 after upgrade | Neighbour T+1 after downgrade |
|---|---|---|
| Real supply edges | **+2.92%** (N=217) | +1.00% (N=115) |
| Degree-matched random networks | +1.77% | +0.80% |
| **Net increment of the edge** | **+1.15pp** | **+0.21pp (≈0)** |

**Conclusion: diffusion exists only in the upgrade direction; downgrades do not diffuse.** This matches the pre-registered hypothesis of RP project 1 ("test upgrade asymmetry") — in writing, frame it as "the pre-registered hypothesis receives preliminary support".

## 3. Corrected liquidity / size splits (replaces the old numbers)

Specification fix: the old version regressed the *signed* AR on direction d again (double-signing); the coefficient was effectively (m+ + m−)/2, inflated about twofold with inflated t. Corrected to y = neighbour raw AR, X = [1, d01], event clustering.

| Split | Direction gap (m+ − m−) | t (event-clustered) | Neighbour firms involved |
|---|---|---|---|
| Low liquidity (high Amihud) | **+4.54%** | 1.27 | ASML/COHR/CRWV/INTC/LITE/MRVL (6) |
| High liquidity | +0.37% | 1.19 | AMD/AMZN/META/NVDA (4) |
| Small cap | **+4.23%** | 1.54 | COHR/CRWV/INTC/LITE/MRVL/MU (6) |
| Large cap | +0.51% | 1.45 | AMZN/AVGO/NVDA (3) |

**How to write it: the direction matches the prior hypothesis (the effect concentrates in small-cap / low-liquidity names), but the subsamples lack power (4–6 firms per cell). Write "suggestive / consistent with", never "fully confirmed".**

## 4. Known limitations (interview must-answers)

1. Analyst events carry a sector-timing component: degree-matched random networks also show a +0.88% signed effect — the increment on real edges (+0.7pp) is the information diffusion; the rest is event timing.
2. The non-winsorised mean (1.21%) is pulled by a few low-liquidity firms; the winsorised baseline is +0.137% (t=2.32) — report both; the gap between them is itself evidence that the effect concentrates in the tail / small caps.
3. 12 tradeable edges, 159 events: all heterogeneity conclusions are suggestive.
4. 500 permutations set a resolution floor of p≈0.002; run 5,000 for a stronger claim.

## 5. Document sync (status, updated 2026-11-17)

- [x] Rebuild the reasoning notes with the 19/12-edge definition — done in `full_reasoning_notes_v2_en.html/.md`; all remaining "45" instances are re-scoped as *45 relation rows, 19 flow edges*.
- [x] Section 8.2 main estimate set to +1.21% (event-level); +0.137% labelled as the winsorised baseline.
- [x] Permutation test (p=0.004) and directional asymmetry (p=0.002) added to the notes.
- [ ] Follow-up email should use the corrected numbers in Section 3.

---

## 6. Complex-systems diagnostics (VI spectral quantities and asynchrony; added 2026-11-08)

> The diffusion update is written as a quadratic-game pseudo-gradient F(x) = (I+βL)x − b, L = I−W (row-normalised W). All spectral quantities are computed analytically — no tuning.

### 6.1 Spectral diagnostics

| Quantity | Value | Meaning |
|---|---|---|
| Spectral gap λ₂ | **0.1842** | Spectral gap of the largest connected component (11 nodes) |
| λ_max | **2.0000** | Bipartite structure (NVDA/TSM on one side, the rest on the other) |
| λ_max_sym | **2.1424** | Largest eigenvalue of (L+Lᵀ)/2 (non-normality, >2) |
| μ | 1 + 0.184β | **Always positive** → no instability threshold, only slowdown |
| κ | (1+βλ_max)/(1+βλ₂) | Asymptote **10.86** |

**Conclusion**: row normalisation → λ_min(I−W)=0 identically → μ always positive. The real, sparse industry graph is structurally incapable of going unstable — it can only slow down (a counterexample to "strong coupling → collapse"). Note: μ here uses λ₂ as a proxy for the strong-monotonicity modulus; the strict modulus is the smallest eigenvalue of the symmetric part, and with L non-symmetric this proxy bound is loose.

### 6.2 Algorithmic sensitivity (β × rounds to convergence, relative error 1e-6)

| β | best-response | GD conservative | GD spectral |
|---|---|---|---|
| 0.5 | 13 | 21 | 20 |
| 1 | 20 | 37 | 35 |
| 2 | 35 | 66 | 62 |
| 3 | 49 | 96 | 90 |
| 5 | 76 | 155 | 146 |

The best-response error iteration matrix is βW/(1+β); the bipartite factor is **−β/(1+β)** (not (1−β)/(1+β)). The factor is negative for *any* β>0, so the iterate flips sign at every step; β=1 is only where the amplitude is exactly 1/2, not where oscillation begins. GD is about twice as slow as best-response — a signature of non-normality (the spectral bound is not tight).

### 6.3 Asynchrony (projected gradient, β=1)

| Schedule | Updates per round | Ratio T/T_sync |
|---|---|---|
| Synchronous | 11 | 1.00 |
| Round-robin (forward / reverse / random) | 1 | 9.28 / 9.38 / 9.75 |
| Mini-batch | 3 | **3.77 (≈11/3)** |

- Sequential-discrete 4.9% < 10% → ordering is not a dominant variable; the equal-frequency conclusions are robust.
- **Δ ≤ 2τ is the triangle inequality**: μ>0 → unique equilibrium → any two τ-convergent algorithms are within 2τ of each other. "Asynchrony does not change the endpoint" is a **theorem**, not a finding. Breaking it requires μ→0 or active constraints (future work).
