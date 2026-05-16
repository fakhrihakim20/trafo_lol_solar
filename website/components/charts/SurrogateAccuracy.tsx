'use client';

import { useEffect, useMemo, useState } from 'react';
import PlotlyChart, { palette, useIsDark } from './PlotlyChart';
import { useData, type Metrics } from './useData';
import { HEADLINE } from '@/content/findings';

type Pair = { truth: number; pred: number; err: number };

export default function SurrogateAccuracy() {
  const metrics = useData<Metrics>('metrics.json');
  const dark = useIsDark();
  const [view, setView] = useState<'scatter' | 'cdf'>('scatter');
  const [holdout, setHoldout] = useState<Pair[] | 'unavailable'>('unavailable');

  useEffect(() => {
    fetch('/data/surrogate_holdout.json')
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((j: Pair[]) => setHoldout(j))
      .catch(() => setHoldout('unavailable'));
  }, []);

  const summary = useMemo(() => {
    if (holdout === 'unavailable') return { within: 0, max: 0 };
    const errs = holdout.map((p) => p.err).sort((a, b) => a - b);
    const within = errs.filter((e) => e <= 1.5).length / errs.length;
    const max = errs[errs.length - 1];
    return { within, max };
  }, [holdout]);

  const scatterData = useMemo(() => {
    if (holdout === 'unavailable') return [];
    const xs = holdout.map((p) => p.truth);
    const ys = holdout.map((p) => p.pred);
    const lo = Math.min(...xs, ...ys);
    const hi = Math.max(...xs, ...ys);
    return [
      {
        type: 'scatter' as const,
        mode: 'markers' as const,
        x: xs,
        y: ys,
        marker: { color: palette.signal, size: 4, opacity: 0.45 },
        name: 'Hold-out predictions',
        hovertemplate: 'Sim θ_HS: %{x:.2f} °C<br>Pred θ_HS: %{y:.2f} °C<extra></extra>',
      },
      {
        type: 'scatter' as const,
        mode: 'lines' as const,
        x: [lo, hi],
        y: [lo, hi],
        line: { color: dark ? '#D5D4D0' : '#37352F', width: 1, dash: 'dot' as const },
        name: 'y = x',
        hoverinfo: 'skip' as const,
      },
    ];
  }, [holdout, dark]);

  const cdfData = useMemo(() => {
    if (holdout === 'unavailable') return [];
    const errs = holdout.map((p) => p.err).sort((a, b) => a - b);
    const ys = errs.map((_, i) => (i + 1) / errs.length);
    return [
      {
        type: 'scatter' as const,
        mode: 'lines' as const,
        x: errs,
        y: ys,
        fill: 'tozeroy' as const,
        fillcolor: 'rgba(27,63,139,0.12)',
        line: { color: palette.signal, width: 2 },
        name: 'Absolute error CDF',
        hovertemplate: '|err| ≤ %{x:.2f} °C  →  %{y:.1%} of predictions<extra></extra>',
      },
    ];
  }, [holdout]);

  if (holdout === 'unavailable') {
    return <FallbackPanel metrics={metrics} />;
  }

  return (
    <div>
      <div className="mb-4 flex items-center justify-between gap-4 flex-wrap">
        <p className="kicker">Fig. 2 · XGBoost surrogate accuracy on the 30-trajectory hold-out</p>
        <fieldset className="inline-flex border hairline rounded-md overflow-hidden text-sm">
          <legend className="sr-only">View</legend>
          {[
            { v: 'scatter', l: 'Predicted vs. simulated' },
            { v: 'cdf', l: 'Error CDF' },
          ].map((o) => (
            <label
              key={o.v}
              className={
                'px-3.5 py-2 cursor-pointer ' +
                (view === o.v
                  ? 'bg-ink-950 text-ink-50 dark:bg-ink-50 dark:text-ink-950'
                  : 'text-ink-600 hover:bg-ink-100 dark:hover:bg-ink-900')
              }
            >
              <input
                type="radio"
                name="surrogate-view"
                className="sr-only"
                checked={view === o.v}
                onChange={() => setView(o.v as 'scatter' | 'cdf')}
              />
              {o.l}
            </label>
          ))}
        </fieldset>
      </div>

      {view === 'scatter' ? (
        <PlotlyChart
          className="h-[440px]"
          ariaLabel="Scatter plot of predicted vs simulated hot-spot temperature"
          data={scatterData}
          layout={{
            xaxis: { title: { text: 'Simulated θ_HS (°C, IEEE C57.91 ODE)', standoff: 12 } },
            yaxis: { title: { text: 'Predicted θ_HS (°C, XGBoost surrogate)', standoff: 12 } },
          }}
          fallback={null}
        />
      ) : (
        <PlotlyChart
          className="h-[440px]"
          ariaLabel="Cumulative distribution of absolute prediction error"
          data={cdfData}
          layout={{
            xaxis: { title: { text: 'Absolute error |θ_HS pred − sim| (°C)', standoff: 12 } },
            yaxis: { title: { text: 'Cumulative fraction', standoff: 12 }, tickformat: '.0%', range: [0, 1.01] },
            shapes: [
              {
                type: 'line',
                x0: 1.5,
                x1: 1.5,
                y0: 0,
                y1: 1,
                line: { color: '#956400', width: 1, dash: 'dash' },
              },
            ],
            annotations: [
              {
                x: 1.5,
                y: 0.6,
                text: '±1.5 °C engineering band',
                showarrow: false,
                xshift: 12,
                font: { color: '#956400', size: 11, family: 'var(--font-jetbrains), monospace' },
                align: 'left',
              },
            ],
          }}
          fallback={null}
        />
      )}
      <p className="mt-4 text-sm text-ink-600 dark:text-ink-300 max-w-prose">
        <span className="text-ink-700 dark:text-ink-200">How to read.</span>{' '}
        Each point compares the surrogate’s θ_HS prediction to the IEEE C57.91 ODE on a withheld
        trajectory. <span className="numera">{(summary.within * 100).toFixed(2)}%</span> of
        predictions fall within ±1.5 °C; the worst-case error is{' '}
        <span className="numera">{summary.max.toFixed(2)} °C</span>.
      </p>
    </div>
  );
}

function FallbackPanel({ metrics }: { metrics: Metrics | null }) {
  const rmse = metrics?.surrogate_rmse_c ?? HEADLINE.surrogateRmseC;
  const r2 = metrics?.surrogate_r2 ?? HEADLINE.surrogateR2;
  const lolErr = metrics?.surrogate_lol_abs_err_pct ?? 0.000657;
  return (
    <div>
      <p className="kicker mb-4">Fig. 2 · XGBoost surrogate accuracy (summary)</p>
      <div className="border hairline rounded-lg p-8 grid grid-cols-1 md:grid-cols-3 gap-y-8 gap-x-10">
        <Cell value={`${rmse.toFixed(3)} °C`} label="Hold-out RMSE on θ_HS" />
        <Cell value={r2.toFixed(6)} label="Hold-out R²" />
        <Cell value={`${(lolErr * 100).toFixed(4)} %`} label="Absolute LoL error (1 week)" />
      </div>
      <p className="mt-4 text-sm text-ink-600 dark:text-ink-300 max-w-prose">
        Interactive scatter and CDF views will appear here once{' '}
        <code className="numera bg-ink-100 dark:bg-ink-900 px-1.5 py-0.5 rounded">scripts/06_export_holdout_predictions.py</code>{' '}
        has been run and <code className="numera">surrogate_holdout.csv</code> is available.
      </p>
    </div>
  );
}

function Cell({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <p className="numera text-3xl font-medium text-ink-950 dark:text-ink-50">{value}</p>
      <p className="mt-2 text-xs leading-snug text-ink-600 dark:text-ink-300">{label}</p>
    </div>
  );
}
