'use client';

import { useMemo, useState } from 'react';
import PlotlyChart, { useIsDark } from './PlotlyChart';
import { useData, type LolRow } from './useData';

export default function SensitivityHeatmap() {
  const rows = useData<LolRow[]>('lol_stats.json');
  const dark = useIsDark();
  const [growth, setGrowth] = useState<0 | 4.5>(0);

  const fig = useMemo(() => {
    if (!rows) return null;
    const pvAxis = [0, 0.1, 0.3, 0.5, 0.75];
    const dTAxis = [0, 1, 2];

    const z: (number | null)[][] = dTAxis.map(() => pvAxis.map(() => null));
    const text: string[][] = dTAxis.map(() => pvAxis.map(() => ''));
    const hover: string[][] = dTAxis.map(() => pvAxis.map(() => ''));

    rows.forEach((r) => {
      // Formal-grid rows only — otherwise named EV/BESS scenarios at the same
      // (pv, dT, growth) overwrite the grid value via last-write-wins.
      if (!r.grid) return;
      if (r.growth !== growth) return;
      const i = dTAxis.indexOf(r.dT);
      const j = pvAxis.indexOf(r.pv);
      if (i < 0 || j < 0) return;
      z[i][j] = r.lol_mean;
      text[i][j] = r.lol_mean.toFixed(4);
      hover[i][j] =
        `<b>${(r.pv * 100).toFixed(0)}% PV · +${r.dT}°C · ${r.growth.toFixed(1)}%/yr</b><br>` +
        `LoL: <b>${r.lol_mean.toFixed(4)}% ± ${r.lol_std.toFixed(4)}%</b><br>` +
        `Peak θ_HS: ${r.peak_ths.toFixed(1)} °C<br>` +
        `Reverse-flow: ${r.reverse_flow_h.toFixed(0)} h/yr`;
    });

    // Anchor color scale at baseline 0.0342
    return {
      x: pvAxis.map((p) => `${(p * 100).toFixed(0)}%`),
      y: dTAxis.map((d) => `+${d} °C`),
      z,
      text,
      hover,
    };
  }, [rows, growth]);

  if (!fig) return <div className="h-[420px]" />;

  return (
    <div>
      <div className="mb-4 flex items-center justify-between gap-4 flex-wrap">
        <p className="kicker">Fig. 6 · Sensitivity to PV × ambient warming</p>
        <fieldset className="inline-flex border hairline rounded-md overflow-hidden text-sm">
          <legend className="sr-only">Load growth</legend>
          {[0, 4.5].map((g) => (
            <label
              key={g}
              className={
                'px-3.5 py-2 cursor-pointer ' +
                (growth === g
                  ? 'bg-ink-950 text-ink-50 dark:bg-ink-50 dark:text-ink-950'
                  : 'text-ink-600 hover:bg-ink-100 dark:hover:bg-ink-900')
              }
            >
              <input
                type="radio"
                name="hm-growth"
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
        className="h-[420px]"
        ariaLabel="Heatmap of annual loss-of-life across PV penetration and ambient temperature offsets"
        data={[
          ({
            type: 'heatmap',
            x: fig.x,
            y: fig.y,
            z: fig.z as any,
            text: fig.text as any,
            texttemplate: '%{text}%',
            textfont: {
              family: 'var(--font-jetbrains), monospace',
              size: 12,
              color: dark ? '#F7F6F3' : '#111111',
            },
            customdata: fig.hover as any,
            hovertemplate: '%{customdata}<extra></extra>',
            zmin: 0.025,
            zmid: 0.0342, // today's no-PV baseline
            zmax: 0.055,
            colorscale: [
              [0.0, '#E1F3FE'],   // cool (PV helps)
              [0.4, '#FBFBFA'],
              [0.55, '#FBF3DB'],
              [1.0, '#956400'],   // warm (degradation)
            ] as any,
            colorbar: {
              title: { text: 'LoL %/yr', side: 'right' },
              thickness: 10,
              len: 0.85,
              tickformat: '.4f',
              ticksuffix: '%',
              outlinewidth: 0,
              tickfont: { family: 'var(--font-jetbrains), monospace', size: 11 },
            },
            xgap: 1,
            ygap: 1,
          } as any),
        ]}
        layout={{
          xaxis: { title: { text: 'Rooftop-PV penetration', standoff: 12 }, type: 'category' },
          yaxis: { title: { text: 'Ambient offset', standoff: 12 }, type: 'category' },
          margin: { l: 70, r: 24, t: 12, b: 56 },
        }}
        fallback={null}
      />
      <p className="mt-4 text-sm text-ink-600 dark:text-ink-300 max-w-prose">
        <span className="text-ink-700 dark:text-ink-200">How to read.</span>{' '}
        Color is centred at today’s no-PV baseline (<span className="numera">0.0342%</span>). Blue cells
        are cooler than today; amber cells are net deterioration. Hover for the peak hot-spot and
        reverse-flow hours behind each cell.
      </p>
    </div>
  );
}
