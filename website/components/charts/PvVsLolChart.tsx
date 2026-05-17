'use client';

import { useMemo, useState } from 'react';
import PlotlyChart, { palette, useIsDark } from './PlotlyChart';
import { useData, type LolRow } from './useData';

export default function PvVsLolChart() {
  const rows = useData<LolRow[]>('lol_stats.json');
  const dark = useIsDark();
  const [growth, setGrowth] = useState<0 | 4.5>(0);

  const traces = useMemo(() => {
    if (!rows) return [];
    // Formal grid only (`r.grid` set by build_data.mjs from the pv###_dt##_gr###
    // naming convention). Named EV/BESS scenarios sit at the same (pv, dT, growth)
    // coordinates but encode different physics — they belong in Fig. 7.
    const grid = rows.filter((r) => r.grid && Number(r.growth) === growth);

    const dTs = [0, 1, 2];
    const colors = [palette.signal, palette.warm, palette.leaf];

    const out: any[] = [];
    dTs.forEach((dT, i) => {
      const sub = grid
        .filter((r) => r.dT === dT)
        .sort((a, b) => a.pv - b.pv);
      if (!sub.length) return;
      const x = sub.map((r) => r.pv * 100);
      const y = sub.map((r) => r.lol_mean);
      const upper = sub.map((r) => r.lol_mean + r.lol_std);
      const lower = sub.map((r) => r.lol_mean - r.lol_std);

      // ribbon (±1σ band)
      out.push({
        x: [...x, ...[...x].reverse()],
        y: [...upper, ...[...lower].reverse()],
        fill: 'toself',
        fillcolor: hexToRgba(colors[i], dark ? 0.15 : 0.1),
        line: { color: 'transparent' },
        hoverinfo: 'skip',
        showlegend: false,
        type: 'scatter',
      });
      out.push({
        x,
        y,
        mode: 'lines+markers',
        name: `ΔT = +${dT} °C`,
        line: { color: colors[i], width: 2, dash: dT === 0 ? 'solid' : dT === 1 ? 'dot' : 'dashdot' },
        marker: { size: 7, color: colors[i] },
        hovertemplate:
          '<b>%{x}% rooftop PV</b><br>' +
          'Annual LoL: <b>%{y:.4f}%</b><br>' +
          `Ambient: +${dT} °C` +
          '<extra></extra>',
        type: 'scatter',
      });
    });

    // baseline reference line
    const baseline = rows.find((r) => r.scenario === 'baseline');
    if (baseline) {
      out.push({
        x: [0, 75],
        y: [baseline.lol_mean, baseline.lol_mean],
        mode: 'lines',
        name: 'Today, no PV (baseline)',
        line: { color: dark ? '#9B9A95' : '#56544F', width: 1, dash: 'dot' },
        hovertemplate: `Baseline: <b>${baseline.lol_mean.toFixed(4)}%</b><extra></extra>`,
        type: 'scatter',
      });
    }
    return out;
  }, [rows, dark, growth]);

  return (
    <div>
      <div className="mb-4 flex items-center justify-between gap-4 flex-wrap">
        <p className="kicker">Fig. 3 · PV penetration vs annual loss-of-life</p>
        <fieldset className="inline-flex border hairline rounded-md overflow-hidden text-sm">
          <legend className="sr-only">Load growth scenario</legend>
          {[0, 4.5].map((g) => (
            <label
              key={g}
              className={
                'px-3.5 py-2 cursor-pointer transition-colors ' +
                (growth === g
                  ? 'bg-ink-950 text-ink-50 dark:bg-ink-50 dark:text-ink-950'
                  : 'text-ink-600 hover:bg-ink-100 dark:hover:bg-ink-900')
              }
            >
              <input
                type="radio"
                name="growth"
                className="sr-only"
                checked={growth === g}
                onChange={() => setGrowth(g as 0 | 4.5)}
              />
              {g === 0 ? 'No load growth' : '+4.5 %/yr growth'}
            </label>
          ))}
        </fieldset>
      </div>

      <PlotlyChart
        className="h-[480px]"
        ariaLabel="Line chart of annual transformer loss-of-life versus rooftop-PV penetration, faceted by ambient warming and load growth"
        data={traces}
        layout={{
          // Extra left margin: the italic serif y-axis title is long; the shared
          // chartBase l:64 clips it into the tick labels at this chart's scale.
          margin: { l: 100, r: 24, t: 24, b: 56 },
          xaxis: {
            title: { text: 'Rooftop-PV penetration', standoff: 14 },
            // Tick marks at the actual 5 data positions only — not the auto
            // every-10 grid that leaves empty ticks at 20 %, 40 %, 60 %.
            tickmode: 'array',
            tickvals: [0, 10, 30, 50, 75],
            ticktext: ['0 %', '10 %', '30 %', '50 %', '75 %'],
            range: [-3, 78],
          },
          yaxis: {
            // Shorter title + more standoff so the rotated serif text doesn't
            // overlap the tick values.
            title: { text: 'Annual loss-of-life (%)', standoff: 28 },
            tickformat: '.4f',
            ticksuffix: '%',
          },
          showlegend: true,
        }}
        fallback={
          <table className="text-xs numera">
            <thead className="text-ink-500">
              <tr><th className="text-left pr-4">PV</th><th className="text-left pr-4">ΔT</th><th className="text-left pr-4">Growth</th><th className="text-left">LoL ± σ</th></tr>
            </thead>
            <tbody>
              {rows?.filter((r) => r.growth === growth && [0, 0.1, 0.3, 0.5, 0.75].includes(r.pv)).map((r) => (
                <tr key={r.scenario}>
                  <td className="pr-4">{(r.pv * 100).toFixed(0)}%</td>
                  <td className="pr-4">+{r.dT.toFixed(0)}°C</td>
                  <td className="pr-4">{r.growth.toFixed(1)}%/yr</td>
                  <td>{r.lol_mean.toFixed(4)} ± {r.lol_std.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        }
      />
      <p className="mt-4 text-sm text-ink-600 dark:text-ink-300 max-w-prose">
        <span className="text-ink-700 dark:text-ink-200">How to read.</span>{' '}
        Each line is one ambient-warming scenario; shaded bands are ±1σ across 1 000 Monte-Carlo runs.
        The dotted reference is today’s no-PV baseline. Drag to zoom, click a legend entry to isolate a curve.
      </p>
    </div>
  );
}

function hexToRgba(hex: string, a: number) {
  const m = hex.replace('#', '').match(/.{1,2}/g);
  if (!m) return hex;
  const [r, g, b] = m.map((h) => parseInt(h, 16));
  return `rgba(${r},${g},${b},${a})`;
}
