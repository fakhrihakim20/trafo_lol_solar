'use client';

import { useMemo } from 'react';
import PlotlyChart, { palette } from './PlotlyChart';
import { useData, type LolRow } from './useData';

const ORDER = [
  { id: 'baseline', label: 'Today, no PV' },
  { id: 'pv75_tropical_today', label: '75 % PV · today' },
  { id: 'pv50_ev30_today', label: '50 % PV + 30 % EV · today' },
  { id: 'pv75_battery5mwh_today', label: '75 % PV + 5 MWh battery · today' },
  { id: 'pv75_battery5mwh_ev30_warmer_2030', label: '75 % PV + battery + EV · 2030' },
];

export default function EvBatteryImpact() {
  const rows = useData<LolRow[]>('lol_stats.json');

  const fig = useMemo(() => {
    if (!rows) return null;
    const map = new Map(rows.map((r) => [r.scenario, r]));
    const baseline = map.get('baseline')!;
    const points = ORDER.map((o) => {
      const r = map.get(o.id);
      if (!r) return null;
      const delta = ((r.lol_mean - baseline.lol_mean) / baseline.lol_mean) * 100;
      return { ...o, lol: r.lol_mean, std: r.lol_std, delta, peak: r.peak_ths, rev: r.reverse_flow_h };
    }).filter(Boolean) as Array<{
      id: string; label: string; lol: number; std: number; delta: number; peak: number; rev: number;
    }>;
    return { baseline, points };
  }, [rows]);

  if (!fig) return <div className="h-[420px]" />;

  return (
    <div>
      <p className="kicker mb-4">Fig. 7 · How EVs, batteries, and climate reshape the PV gain</p>
      <PlotlyChart
        className="h-[460px]"
        ariaLabel="Grouped bar chart of annual loss-of-life across PV, EV, battery, and 2030 climate scenarios"
        data={[
          {
            type: 'bar',
            x: fig.points.map((p) => p.label),
            y: fig.points.map((p) => p.lol),
            marker: {
              color: fig.points.map((p) =>
                p.delta < -5
                  ? palette.signal
                  : p.delta > 5
                    ? palette.warm
                    : '#9B9A95',
              ),
            },
            error_y: {
              type: 'data',
              array: fig.points.map((p) => p.std),
              color: '#9B9A95',
              thickness: 1,
              width: 6,
            },
            text: fig.points.map((p) =>
              `${(p.delta > 0 ? '+' : '')}${p.delta.toFixed(0)}%`,
            ),
            textposition: 'outside',
            textfont: { family: 'var(--font-jetbrains), monospace', size: 12 },
            hovertemplate:
              '<b>%{x}</b><br>' +
              'LoL: <b>%{y:.4f}%</b><br>' +
              'Δ vs baseline: %{customdata[0]}<br>' +
              'Peak θ_HS: %{customdata[1]:.1f} °C<br>' +
              'Reverse-flow: %{customdata[2]:.0f} h/yr<extra></extra>',
            customdata: fig.points.map((p) => [
              `${p.delta > 0 ? '+' : ''}${p.delta.toFixed(1)}%`,
              p.peak,
              p.rev,
            ]) as any,
            cliponaxis: false,
          },
          {
            type: 'scatter',
            mode: 'lines',
            x: fig.points.map((p) => p.label),
            y: fig.points.map(() => fig.baseline.lol_mean),
            line: { dash: 'dot', color: '#56544F', width: 1 },
            name: 'No-PV baseline',
            hoverinfo: 'skip',
            showlegend: false,
          },
        ]}
        layout={{
          showlegend: false,
          xaxis: { title: { text: '', standoff: 8 }, tickangle: -18 },
          yaxis: {
            title: { text: 'Annual loss-of-life (%)', standoff: 12 },
            tickformat: '.4f',
          },
          margin: { l: 80, r: 24, t: 20, b: 110 },
        }}
        fallback={null}
      />
      <p className="mt-4 text-sm text-ink-600 dark:text-ink-300 max-w-prose">
        <span className="text-ink-700 dark:text-ink-200">How to read.</span>{' '}
        Bars are mean annual loss-of-life with ±1σ error bars; the dotted reference is today’s
        no-PV baseline. The percentage above each bar is the change versus that baseline. EV
        charging at 30 % fleet penetration cancels most of the PV benefit; the 2030 stack pushes
        well above the baseline.
      </p>
    </div>
  );
}
