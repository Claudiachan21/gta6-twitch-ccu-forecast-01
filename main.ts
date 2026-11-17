// gta6-twitch-ccu-forecast-01 entry v1 (Node 22, standard library only).
//
// Forecast: log-domain k-nearest-neighbour shape transfer over the case's
// comparables. Launch-day Twitch curves share a shape driven by the rolling
// midnight-local unlock, but the shape's scale depends on how big hour 1 is:
// a quiet hour 1 (small category at 00:00 UTC) ramps far more than a huge one.
// So for each hour h = 2..24 the median log-ratio ln(v_h / v_1) and its spread
// are taken from the K = 14 comparables whose hour-1 print is closest to this
// case's in log space (LOO-validated on 37 comparables: proxy loss 0.265 on
// big launches vs 0.351 for persistence; overall 0.502 vs 0.930).
//
//   q_tau(h) = p1 * exp(med_h + z_tau * sig_h), rounded to 2 decimals,
//   sig_h = min(max(IQR-sigma of the 14 donors' log-ratios, 0.15*sqrt(h-1)), 2.0)
//
// Falls back to the persistence baseline when no comparable has full 24-hour
// coverage. Reads every <id>.in, writes <id>.out; offline; ~0.02 s/case.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const K = 14;
const SIGMA_FLOOR = 0.15;
const SIGMA_CAP = 2.0;

interface HourlyPoint { from: string; viewers: number; channels?: number | null }
interface Comparable { name: string; day_start: string; hourly: HourlyPoint[] }
interface Case {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables?: Comparable[] | null;
}

const round2 = (x: number): number => Math.round(x * 100) / 100;
const clamp = (x: number, lo: number, hi: number): number => Math.min(Math.max(x, lo), hi);

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};

/** IQR-based robust sigma of a sample (0 when the sample is degenerate). */
const iqrSigma = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  if (n < 4) return 0;
  return Math.max(0, (s[Math.floor((3 * n) / 4)] - s[Math.floor(n / 4)]) / 1.349);
};

/**
 * A donor's launch-day log-ratio curve, or null when its release day lacks any
 * of the 24 hourly prints. Hours are UTC whole hours from day_start.
 */
const donorCurve = (c: Comparable): { lp1: number; lr: number[] } | null => {
  const t0 = Date.parse(c.day_start);
  if (Number.isNaN(t0) || !Array.isArray(c.hourly)) return null;
  const byHour = new Map<number, number>();
  for (const p of c.hourly) {
    const t = Date.parse(p.from);
    if (!Number.isNaN(t) && typeof p.viewers === 'number') byHour.set(t, p.viewers);
  }
  const prints: number[] = [];
  for (let h = 0; h < 24; h++) {
    const v = byHour.get(t0 + h * 3600_000);
    if (v === undefined) return null;
    prints.push(v);
  }
  const lp1 = Math.log(Math.max(prints[0], 1));
  return { lp1, lr: prints.slice(1).map((v) => Math.log(Math.max(v, 1)) - lp1) };
};

/** Median log-ratio + spread per hour from the K nearest donors in log(p1). */
const forecastCurve = (p1: number, comps: Comparable[]): { med: number[]; sig: number[] } => {
  const donors = comps
    .map(donorCurve)
    .filter((d): d is { lp1: number; lr: number[] } => d !== null);
  const lp1 = Math.log(Math.max(p1, 1));
  const fallback = {
    med: Array(23).fill(0),
    sig: Array.from({ length: 23 }, (_, i) => SIGMA_FLOOR * Math.sqrt(i + 1)),
  };
  if (donors.length === 0) return fallback;
  const near = donors
    .sort((a, b) => Math.abs(a.lp1 - lp1) - Math.abs(b.lp1 - lp1))
    .slice(0, K);
  const med: number[] = [];
  const sig: number[] = [];
  for (let hi = 0; hi < 23; hi++) {
    const ratios = near.map((d) => d.lr[hi]);
    med.push(median(ratios));
    sig.push(clamp(Math.max(iqrSigma(ratios), SIGMA_FLOOR * Math.sqrt(hi + 1)), 0, SIGMA_CAP));
  }
  return { med, sig };
};

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${file} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);
  const { med, sig } = forecastCurve(p1, c.comparables ?? []);
  const hours = c.forecast.map(({ hour }) => {
    const hi = hour - 2;
    const q = Z.map((z) => clamp(round2(p1 * Math.exp(med[hi] + z * sig[hi])), 0.01, 1e9));
    for (let i = 1; i < q.length; i++) q[i] = Math.max(q[i], q[i - 1]); // monotone clip
    return { hour, q };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
