// gta6-trophy-rarity-forecast-01: a counts-based forecaster for GTA VI PS5 trophy
// earn rates on Exophase at the case's target.
//
// Model (written from scratch for this entry; the house baseline is persistence
// with a random-walk spread, and this program deliberately does something else):
//
//   rate_i(t) = E_i(t) / N(t),  earned counts over players tracked.
//
// Two forces move a rate, and they pull in opposite directions:
//   (a) EARNING - players progressively earn trophies, so E_i rises. A trophy's
//       earning hazard is estimated from its own history (earned-count growth
//       over the last 24 h of distinct reads, shrunk toward its kind's median
//       hazard), and it decays like 1/(1 + hours since release): early story
//       trophies are earned fast by fresh players, grind trophies slowly.
//   (b) DILUTION - N(t) grows as new players sync and Exophase refreshes, and
//       the new cohort holds each trophy at a rate p_i (estimated from the
//       jumps in E_i across the refreshes visible in the history: p_i =
//       dE_i/dN across each jump). New players pull every rate toward the new
//       cohort's (earlier-progress) profile.
//
// N(t) itself is extrapolated by a least-squares log-log fit (log N against
// log(1 + hours since release)) over the distinct reads, blended with the
// most recent segment's growth. The forecast then steps hour by hour from
// the checkpoint to the target: N follows its fitted curve, E_i grows by
// hazard on the not-yet-earners plus the arriving cohort's p_i share.
//
// Exophase refreshes on its own schedule, so the displayed page is stale
// between refreshes: the answer is the page as of the last refresh before
// the target, so the forecast is evaluated at target minus half the median
// refresh gap (capped at 12 h) when refreshes are visible in the history.
//
// Degenerate history (a single distinct read, e.g. launch night before the
// first refresh): no hazard and no cohort rate can be estimated from data,
// so a prior takes over, grounded in what the kit documents about the one
// observed launch-night refresh (page stale for hours, then N 638 -> 3,868
// while the new cohort had earned almost nothing beyond the earliest
// trophy): a refresh is treated as overdue (probability 0.65 before the
// target), a jump of ~4x N if it comes, and a new-cohort earn rate of ~10%
// of the old rate per trophy (99.5% for a trophy everyone already holds).
// The spread is widened a lot in this case, because the outcome is bimodal.
//
// Spread: per-trophy logit sigma combining a random-walk term, a hazard/
// extrapolation term (wider for grind trophies), binomial count noise
// (1/sqrt(E), rare trophies are noisier) and a refresh-timing term. The
// 90% band is much wider than the house baseline's over a week horizon,
// as it must be: a refresh near the target can move every rate at once.
//
// Only the standard library is used. No network, no randomness, no clock:
// two runs on the same case are byte-identical.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Release day begins 2026-11-19T00:00:00Z; log-time is measured from here. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const clampPct = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-clamp(l, -12, 12)));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const medianOf = (xs: number[]): number => {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};
/** Hours from the release-day start to an ISO timestamp. */
const hoursSinceRelease = (iso: string): number => (Date.parse(iso) - RELEASE_MS) / 3600000;
/** Log-time coordinate used by the fits (0 at release, grows slowly after). */
const logTime = (h: number): number => Math.log(1 + Math.max(0, h));

type Kind = 'story' | 'grind' | 'other';

/** Classify a trophy by what it asks the player to do (drives hazard priors). */
function kindOf(t: Trophy): Kind {
  const d = `${t.name} ${t.description}`.toLowerCase();
  if (t.tier === 'platinum') return 'grind';
  if (/250 stunts|100% completion|50 challenges|100 locations|100 vehicles|3 collectible card sets|gold medal in all|20 different races|every spot|every location|purchase all|explore all of leonida|8 mission strands/.test(d)) return 'grind';
  if (/chapter|final mission|complete .* mission|complete breakfast run|complete .* heist|complete child's play|complete headliners|operation savage glory/.test(d)) return 'story';
  return 'other';
}

/** Least-squares line y ~ a + b*x; returns [a, b] (or null when degenerate). */
function linfit(xs: number[], ys: number[]): [number, number] | null {
  const n = xs.length;
  if (n < 2) return null;
  let mx = 0;
  let my = 0;
  for (let k = 0; k < n; k++) { mx += xs[k]; my += ys[k]; }
  mx /= n; my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let k = 0; k < n; k++) { sxx += (xs[k] - mx) ** 2; sxy += (xs[k] - mx) * (ys[k] - my); }
  if (sxx <= 1e-12) return null;
  const b = sxy / sxx;
  return [my - b * mx, b];
}

/** One distinct page state: consecutive identical reads collapse to this. */
interface Snap {
  t: number; // hours since release
  n: number; // players tracked
  rates: number[]; // per trophy, last non-null rate seen at or before this snap
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const nT = c.trophies.length;
  const kinds = c.trophies.map(kindOf);
  const horizonDays = c.horizon_minutes / 1440;
  const sqrtD = Math.sqrt(Math.max(0, horizonDays));
  const tCheckpoint = hoursSinceRelease(c.at);
  const tTarget = hoursSinceRelease(c.target);

  // ---- Build distinct snapshots from the history (dedup frozen reads). ----
  const snaps: Snap[] = [];
  let lastRates: Array<number | null> = new Array(nT).fill(null);
  for (const r of c.history) {
    if (!r || typeof r.players_tracked !== 'number' || !Array.isArray(r.rates)) continue;
    const rates: number[] = [];
    for (let i = 0; i < nT; i++) {
      const v = r.rates[i];
      if (typeof v === 'number' && Number.isFinite(v)) lastRates[i] = v;
      rates.push(lastRates[i] === null ? NaN : lastRates[i] as number);
    }
    const prev = snaps[snaps.length - 1];
    const sameAsPrev = prev !== undefined && prev.n === r.players_tracked &&
      prev.rates.every((v, i) => (Number.isNaN(v) && Number.isNaN(rates[i])) || v === rates[i]);
    if (!sameAsPrev) snaps.push({ t: hoursSinceRelease(r.at), n: r.players_tracked, rates });
  }
  if (snaps.length === 0) {
    // No usable history at all: flat 1% medians with a very wide spread.
    return c.trophies.map((t) => ({ id: t.id, q: quantilesFrom(1, 2.5) }));
  }
  const last = snaps[snaps.length - 1];
  const flat = snaps.length <= 1;

  // ---- N(t): log-log fit over distinct snaps, blended with the last segment. ----
  const xs = snaps.map((s) => logTime(s.t));
  const ysN = snaps.map((s) => Math.log(Math.max(1, s.n)));
  const fitN = linfit(xs, ysN);
  let logNAt = (t: number): number => {
    if (fitN) {
      const b = clamp(fitN[1], 0, 3);
      return fitN[0] + b * logTime(t);
    }
    return Math.log(Math.max(1, last.n));
  };
  if (snaps.length >= 2) {
    // Recent-segment growth rate (last two distinct snaps), blended 50/50 in
    // log space with the full-fit extrapolation at the target.
    const a = snaps[snaps.length - 2];
    const b = last;
    const dx = logTime(b.t) - logTime(a.t);
    if (dx > 1e-9) {
      const bRecent = clamp((Math.log(Math.max(1, b.n)) - Math.log(Math.max(1, a.n))) / dx, 0, 3);
      const baseLogNAt = logNAt;
      logNAt = (t: number): number => {
        const fromFit = baseLogNAt(t);
        const fromRecent = Math.log(Math.max(1, b.n)) + bRecent * (logTime(t) - logTime(b.t));
        // Blend only when extrapolating past the last snap; at/below it, the fit.
        if (t <= b.t) return fromFit;
        return 0.5 * fromFit + 0.5 * fromRecent;
      };
    }
  }

  // ---- Refresh gaps (for the display-lag and the flat-history prior). ----
  const gaps: number[] = [];
  for (let k = 1; k < snaps.length; k++) gaps.push(snaps[k].t - snaps[k - 1].t);
  const medGap = gaps.length > 0 ? medianOf(gaps) : 0;

  // Effective target: the answer is the page as of the last refresh before
  // the target, so aim half a median refresh gap before the target.
  const lagH = flat ? 0 : clamp(medGap / 2, 0, 12);
  const tEff = Math.max(tCheckpoint, tTarget - lagH);

  // ---- Per-trophy: last earned counts, hazard, new-cohort rate. ----
  const eLast: number[] = [];
  const rLast: number[] = [];
  for (let i = 0; i < nT; i++) {
    const r = Number.isNaN(last.rates[i]) ? 1 : last.rates[i];
    rLast.push(clampPct(r));
    eLast.push((clampPct(r) / 100) * last.n);
  }

  // Joint regression per trophy over the distinct-snapshot intervals:
  //   dE_i = h_i * (mean free pool * dt) + q_i * dN
  // i.e. each interval's earned-count growth is split between (a) earning by
  // players already tracked, at hazard h_i on the not-yet-earners, and (b)
  // the arriving cohort, of whom a fraction q_i already holds the trophy
  // when they sync. Estimating the two together matters: attributing all of
  // dE to the hazard and all of it to the cohort at once double-counts the
  // same earning. Least squares with no intercept, h >= 0, q in [0, 1],
  // shrunk toward kind/prior values when intervals are few.
  const earnedAt = (s: Snap, i: number): number =>
    Number.isNaN(s.rates[i]) ? 0 : (clampPct(s.rates[i]) / 100) * s.n;
  interface Interval { dt: number; dN: number; freeDt: number[]; dE: number[] }
  const intervals: Interval[] = [];
  for (let k = 1; k < snaps.length; k++) {
    const a = snaps[k - 1];
    const b = snaps[k];
    const dt = b.t - a.t;
    if (dt <= 0) continue;
    const dN = Math.max(0, b.n - a.n);
    const freeDt: number[] = [];
    const dE: number[] = [];
    for (let i = 0; i < nT; i++) {
      const freeA = Math.max(1, a.n - earnedAt(a, i));
      const freeB = Math.max(1, b.n - earnedAt(b, i));
      freeDt.push(0.5 * (freeA + freeB) * dt);
      dE.push(Math.max(0, earnedAt(b, i) - earnedAt(a, i)));
    }
    intervals.push({ dt, dN, freeDt, dE });
  }
  const nInt = intervals.length;
  const hazRaw: number[] = [];
  const qRaw: number[] = [];
  for (let i = 0; i < nT; i++) {
    let sxx = 0; // (freeDt)^2
    let sxz = 0; // freeDt * dN
    let szz = 0; // dN^2
    let sxy = 0; // freeDt * dE
    let szy = 0; // dN * dE
    for (const iv of intervals) {
      const x = iv.freeDt[i];
      const z = iv.dN;
      const y = iv.dE[i];
      sxx += x * x; sxz += x * z; szz += z * z; sxy += x * y; szy += z * y;
    }
    let h = 0;
    let q = 0;
    const det = sxx * szz - sxz * sxz;
    if (nInt >= 2 && Math.abs(det) > 1e-9) {
      h = (sxy * szz - szy * sxz) / det;
      q = (szy * sxx - sxy * sxz) / det;
    } else if (szz > 0) {
      q = szy / szz;
    } else if (sxx > 0) {
      h = sxy / sxx;
    }
    // Enforce h >= 0, q in [0, 1], refitting the other parameter when one binds.
    if (h < 0) {
      h = 0;
      q = szz > 0 ? szy / szz : 0;
    }
    if (q < 0) {
      q = 0;
      h = sxx > 0 ? sxy / sxx : 0;
      if (h < 0) h = 0;
    }
    if (q > 1) {
      q = 1;
      h = sxx > 0 ? (sxy - sxz) / sxx : 0;
      if (h < 0) h = 0;
    }
    hazRaw.push(Math.max(0, h));
    qRaw.push(clamp(q, 0, 1));
  }
  // Shrink each hazard toward its kind's median (kinds with no signal fall
  // back to the global median hazard); shrink each cohort rate toward half
  // the trophy's current rate (new syncers are earlier in their progress).
  const kindMed = new Map<Kind, number>();
  const globalMed = medianOf(hazRaw.filter((h) => h > 0));
  for (const k of ['story', 'grind', 'other'] as Kind[]) {
    const hs = hazRaw.filter((h, i) => kinds[i] === k && h > 0);
    kindMed.set(k, hs.length > 0 ? medianOf(hs) : (Number.isFinite(globalMed) ? globalMed : 0));
  }
  const wShrink = nInt / (nInt + 4);
  const hazard: number[] = hazRaw.map((h, i) => {
    const prior = kindMed.get(kinds[i]) ?? 0;
    return wShrink * h + (1 - wShrink) * prior;
  });
  const pNew: number[] = qRaw.map((q, i) => {
    if (flat) {
      // Flat-history prior (see header): the new cohort is far behind
      // except on a trophy essentially everyone already holds.
      return rLast[i] >= 99.9 ? 0.995 : clamp(0.1 * (rLast[i] / 100), 0.00005, 0.99);
    }
    const prior = 0.5 * (rLast[i] / 100);
    return wShrink * q + (1 - wShrink) * prior;
  });
  // Flat-history N prior: a refresh is overdue; expect a jump.
  let nTarget = Math.exp(logNAt(tEff));
  if (flat) {
    const pRefresh = 1 - Math.exp(-c.horizon_minutes / 60 / 5);
    const jumpMult = 4;
    // Overdue-refresh stance: blend the memoryless refresh probability with
    // a flat 0.65 (page stale for hours), jump ~4x if it comes.
    const pR = 0.35 * pRefresh + 0.65 * Math.min(1, pRefresh + 0.4);
    nTarget = last.n * (1 + pR * (jumpMult - 1));
  }
  nTarget = clamp(nTarget, last.n, last.n * 40);

  // ---- Hour-by-hour simulation to the effective target. ----
  const out: Array<{ id: number; q: number[] }> = [];
  const t0 = last.t;
  const horizonH = Math.max(0, tEff - t0);
  for (let i = 0; i < nT; i++) {
    let e = eLast[i];
    let nCur = last.n;
    // Step in 1-hour increments (fractional final step).
    let tCur = t0;
    while (tCur < tEff - 1e-9) {
      const dt = Math.min(1, tEff - tCur);
      const tNext = tCur + dt;
      let nNext: number;
      if (flat) {
        // Linear-in-time interpolation toward the prior N target.
        const frac = horizonH > 0 ? (tNext - t0) / horizonH : 1;
        nNext = last.n + (nTarget - last.n) * frac;
      } else {
        nNext = Math.exp(logNAt(tNext));
      }
      nNext = clamp(nNext, nCur, last.n * 40);
      const dN = Math.max(0, nNext - nCur);
      // Hazard decays like 1/(1 + hours since release).
      const h = hazard[i] * ((1 + Math.max(0, t0)) / (1 + Math.max(0, tCur)));
      e += h * Math.max(0, nCur - e) * dt + dN * pNew[i];
      e = clamp(e, eLast[i] * 0.999, 0.999 * nNext);
      nCur = nNext;
      tCur = tNext;
    }
    const nEnd = flat ? nTarget : Math.exp(logNAt(tEff));
    const med = clampPct((e / Math.max(1, nEnd)) * 100);

    // ---- Spread on the logit scale. ----
    // Calibrated on the refresh-simulation backtest (see VALIDATION.md in
    // the work dir): the simulation-optimal scale was ~0.5x these base
    // coefficients, but the simulation shares the forecaster's structure, so
    // the real world is noisier than the simulation; 0.6x of the original
    // 0.45/0.30/0.30 coefficients is kept as a hedge against model error.
    const SIGMA_RW = 0.27; // random-walk term per sqrt(day)
    const SIGMA_HAZ = 0.18; // hazard/extrapolation term per sqrt(day)
    const SIGMA_REFRESH = 0.18; // refresh timing/size term
    const grindF = kinds[i] === 'grind' ? 1.5 : kinds[i] === 'story' ? 1.1 : 1.0;
    const countNoise = Math.min(0.7, 1 / Math.sqrt(Math.max(1, eLast[i])));
    let sigma = Math.sqrt(
      (SIGMA_RW * sqrtD) ** 2 +
      (SIGMA_HAZ * sqrtD * grindF) ** 2 +
      countNoise ** 2 +
      SIGMA_REFRESH ** 2,
    );
    if (flat) sigma = Math.sqrt(sigma * sigma + 0.6 ** 2);
    out.push({ id: c.trophies[i].id, q: quantilesFrom(med, sigma) });
  }
  return out;
}

/** Five quantiles around a median percent, spread sigma on the logit scale. */
function quantilesFrom(medPct: number, sigma: number): number[] {
  const l = logit(medPct);
  const qs = Z.map((z) => round4(clampPct(invLogitPct(l + z * sigma))));
  for (let k = 1; k < qs.length; k++) qs[k] = Math.max(qs[k], qs[k - 1]);
  return qs;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  let rows: Array<{ id: number; q: number[] }>;
  try {
    rows = forecastCase(c);
  } catch {
    // Never crash a case: fall back to last-rate medians, wide spread.
    const last = c.history[c.history.length - 1];
    rows = c.trophies.map((t, i) => {
      const r = last && typeof last.rates[i] === 'number' ? last.rates[i] as number : 1;
      return { id: t.id, q: quantilesFrom(clampPct(r), 1.5) };
    });
  }
  rows.sort((a, b) => a.id - b.id);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
}
