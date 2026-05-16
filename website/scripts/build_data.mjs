// Reads CSVs in ../results/tables/ and ../results/metrics.json, writes compact JSON
// to public/data/*.json so the runtime bundle stays small.
// Run via `npm run data`, or automatically before dev/build.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'csv-parse/sync';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const RESULTS = path.resolve(ROOT, '..', 'results');
const TABLES = path.resolve(RESULTS, 'tables');
const OUT = path.resolve(ROOT, 'public', 'data');

fs.mkdirSync(OUT, { recursive: true });

function readCsv(name) {
  const file = path.join(TABLES, name);
  if (!fs.existsSync(file)) {
    console.warn(`[build_data] missing CSV: ${file} — skipping`);
    return null;
  }
  const txt = fs.readFileSync(file, 'utf8');
  return parse(txt, { columns: true, skip_empty_lines: true, cast: true });
}

function writeJson(name, data) {
  const file = path.join(OUT, name);
  fs.writeFileSync(file, JSON.stringify(data));
  const kb = (fs.statSync(file).size / 1024).toFixed(1);
  console.log(`  wrote public/data/${name}  (${kb} kB)`);
}

console.log('[build_data] starting');

// 1. metrics.json — pass through, with a touch of normalisation
const metricsSrc = path.join(RESULTS, 'metrics.json');
if (fs.existsSync(metricsSrc)) {
  const m = JSON.parse(fs.readFileSync(metricsSrc, 'utf8'));
  writeJson('metrics.json', m);
} else {
  console.warn('[build_data] WARN missing results/metrics.json');
}

// 2. scenarios catalogue — also acts as the authoritative {scenario -> growth} map.
// table2_lol_statistics.csv does NOT have a load_growth_pct_yr column; without
// this join, every row in lol_stats.json gets growth=0 and the gr045 scenarios
// leak into the no-growth panel of Fig. 3 (and break Fig. 6, the explorer, and
// the scenarios table).
const scenarios = readCsv('table1_scenario_definitions.csv');
if (scenarios) writeJson('scenarios.json', scenarios);
const growthByScenario = new Map(
  (scenarios || []).map((s) => [s.scenario, Number(s.load_growth_pct_yr)]),
);

// 3. LoL statistics — drives Fig 3, 6, 7 and the scenario explorer
const lol = readCsv('table2_lol_statistics.csv');
if (lol) {
  const tidy = lol.map((r) => {
    const g = growthByScenario.get(r.scenario_name);
    if (g === undefined) {
      console.warn(`[build_data] WARN no growth in table1 for "${r.scenario_name}", defaulting to 0`);
    }
    // `grid` distinguishes the 30 formal Cartesian-product scenarios
    // (pv###_dt##_gr###) from the 11 named scenarios that may share
    // (pv, dT, growth) coordinates but encode different physics (EV load,
    // BESS dispatch, etc.). Chart components that plot the PV × dT × growth
    // grid use only `grid: true` rows; EV/BESS scenarios live in Fig 7.
    const isGrid = /^pv\d{3}_dt\d{2}_gr\d{3}$/.test(r.scenario_name);
    return {
      scenario: r.scenario_name,
      pv: Number(r.pv_penetration),
      dT: Number(r.ambient_delta_c),
      growth: g ?? 0,
      grid: isGrid,
      n: Number(r.n_runs),
      lol_mean: Number(r.lol_pct_mean),
      lol_std: Number(r.lol_pct_std),
      lol_p5: Number(r.lol_pct_p5),
      lol_p95: Number(r.lol_pct_p95),
      peak_ths: Number(r.peak_ths_mean),
      reverse_flow_h: Number(r.reverse_flow_h_mean),
    };
  });
  writeJson('lol_stats.json', tidy);
}

// 4. surrogate performance summary
const sur = readCsv('table3_surrogate_performance.csv');
if (sur) writeJson('surrogate_metrics.json', sur);

// 5. dispatch comparison — LP vs threshold, downsample to what charts need
const disp = readCsv('dispatch_comparison.csv');
if (disp) {
  const groups = new Map();
  for (const r of disp) {
    const key = `${r.scenario_name}__${r.dispatch_method}`;
    if (!groups.has(key)) {
      groups.set(key, {
        scenario: r.scenario_name,
        method: r.dispatch_method,
        lol_pct: [],
        peak_ths: [],
        battery_lifetime: [],
        reverse_flow_h: [],
      });
    }
    const g = groups.get(key);
    g.lol_pct.push(Number(r.lol_percent_annual));
    g.peak_ths.push(Number(r.peak_theta_hs_c));
    g.battery_lifetime.push(Number(r.battery_lifetime_fraction));
    g.reverse_flow_h.push(Number(r.reverse_flow_hours));
  }
  writeJson('dispatch.json', Array.from(groups.values()));
}

// 6. surrogate holdout (only if the user ran 06_export_holdout_predictions.py)
const holdoutCsv = path.join(TABLES, 'surrogate_holdout.csv');
if (fs.existsSync(holdoutCsv)) {
  const rows = parse(fs.readFileSync(holdoutCsv, 'utf8'), {
    columns: true, skip_empty_lines: true, cast: true,
  });
  const tidy = rows.map((r) => ({
    truth: Number(r.theta_hs_true),
    pred: Number(r.theta_hs_pred),
    err: Number(r.abs_error),
  }));
  writeJson('surrogate_holdout.json', tidy);
} else {
  console.log('  (no surrogate_holdout.csv — Fig.2 will fall back to summary metrics)');
}

console.log('[build_data] done');
