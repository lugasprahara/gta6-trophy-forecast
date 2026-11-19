# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lugasprahara/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lugasprahara/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for every Grand Theft Auto VI PlayStation 5 trophy's earn rate
on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy, for the
make.fun competition `gta6-trophy-rarity-forecast-01`.

## How it works

The program thinks in counts, not just rates: `rate = earned / players_tracked`.
Two forces move a rate — players earning trophies (up) and new players syncing
and diluting the tracked pool at each Exophase refresh (down) — so it models
both, per trophy, from the case's own history:

- It collapses the history to its distinct page states (Exophase only refreshes
  every few hours, so most reads are frozen repeats) and fits players-tracked
  growth as a log-log curve in time since release.
- Per trophy, a joint two-parameter regression over the refresh intervals
  splits each interval's earned-count growth into an earning hazard on the
  not-yet-earners and the arriving cohort's earn rate, so the same earning is
  never counted twice. Hazards are shrunk toward their trophy kind's median
  (story / grind / other, from the trophy description).
- It then steps hour by hour from the checkpoint to the target, growing
  players-tracked along its fitted curve and earned counts by hazard plus the
  arriving cohort, and reads the rate off at the target minus half a median
  refresh gap (the answer is the page as of the last refresh before the
  target, so the displayed state lags the true state).
- With only one distinct read in the history (launch night, before the first
  refresh), no hazard or cohort rate can be estimated, so a prior takes over:
  a refresh is treated as overdue, N jumps ~4x if it comes, and the new
  cohort is assumed to hold each trophy at ~10% of the old rate (99.5% for a
  trophy everyone already holds).
- The spread is per-trophy on the logit scale: a random-walk term, a
  hazard/extrapolation term (wider for grind trophies), binomial count noise
  for rare trophies, and a refresh-timing term — calibrated on a
  refresh-simulation backtest (see `VALIDATION.md` in the work dir, not
  shipped here).

## Run it

Node 22, standard library only, no network:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

It reads every `*.in` case in the input directory and writes one
`<id>.out` forecast per case (`{"trophies": [{"id": n, "q": [q05, q25, q50,
q75, q95]}]}`). Deterministic: two runs on the same case are byte-identical.
A run on the kit's practice cases takes ~0.2 s for both.

## Files

- `main.ts` — the forecaster (the only entrypoint)
- `README.md` — this file
- `LICENSE` — MIT

---

<sub>Entered on [make.fun](https://make.fun/?ref=aes5np1n&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
