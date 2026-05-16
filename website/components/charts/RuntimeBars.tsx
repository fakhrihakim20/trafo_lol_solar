'use client';

import { useMemo } from 'react';
import PlotlyChart, { palette } from './PlotlyChart';
import { useData, type Metrics } from './useData';
import { HEADLINE } from '@/content/findings';

export default function RuntimeBars() {
  const metrics = useData<Metrics>('metrics.json');

  const fig = useMemo(() => {
    const wallClockSec = metrics?.runtime_seconds ?? HEADLINE.wallClockMin * 60;
    const surrogateMin = wallClockSec / 60;
    const odeMin = surrogateMin * HEADLINE.speedupX;
    return {
      labels: ['XGBoost surrogate (this work)', 'IEEE C57.91 ODE only (projected)'],
      values: [surrogateMin, odeMin],
      colors: [palette.signal, palette.warm],
      odeMin,
      surrogateMin,
    };
  }, [metrics]);

  return (
    <div>
      <p className="kicker mb-4">Fig. 5 · Wall-clock for the full 41 000-run Monte Carlo</p>
      <PlotlyChart
        className="h-[360px]"
        ariaLabel="Bar chart comparing wall-clock runtime of the surrogate vs the IEEE ODE baseline"
        data={[
          {
            type: 'bar',
            orientation: 'h',
            x: fig.values,
            y: fig.labels,
            marker: { color: fig.colors },
            text: fig.values.map((v) =>
              v < 120 ? `${v.toFixed(1)} min` : `${(v / 60).toFixed(1)} hours`,
            ),
            textposition: 'outside',
            textfont: { family: 'var(--font-jetbrains), monospace', size: 13 },
            hovertemplate:
              '<b>%{y}</b><br>%{x:.1f} minutes  (%{customdata})<extra></extra>',
            customdata: fig.values.map((v) =>
              v < 120 ? `${v.toFixed(1)} min` : `${(v / 60).toFixed(1)} h ≈ ${(v / 60 / 24).toFixed(1)} days`,
            ),
            cliponaxis: false,
          },
        ]}
        layout={{
          xaxis: {
            title: { text: 'Minutes (log scale)', standoff: 12 },
            type: 'log',
            tickformat: '.0f',
          },
          yaxis: { automargin: true, ticks: '' },
          margin: { l: 260, r: 64, t: 12, b: 48 },
          showlegend: false,
        }}
        fallback={null}
      />
      <p className="mt-4 text-sm text-ink-600 dark:text-ink-300 max-w-prose">
        <span className="text-ink-700 dark:text-ink-200">How to read.</span>{' '}
        Same 41-scenario × 1 000-run sweep; the surrogate completes in{' '}
        <span className="numera">{fig.surrogateMin.toFixed(0)} min</span> on a laptop. The ODE
        baseline is projected by the measured per-scenario speedup of <span className="numera">474×</span>.
      </p>
    </div>
  );
}
