'use client';

import { useMemo, useState } from 'react';
import PlotlyChart, { palette } from './PlotlyChart';
import { useData, type LolRow } from './useData';

const PV_OPTS = [0, 0.1, 0.3, 0.5, 0.75];
const DT_OPTS = [0, 1, 2];
const G_OPTS = [0, 4.5];

export default function ScenarioExplorer() {
  const rows = useData<LolRow[]>('lol_stats.json');
  const [pv, setPv] = useState(0.5);
  const [dT, setDT] = useState(0);
  const [growth, setGrowth] = useState(0);

  const baseline = rows?.find((r) => r.scenario === 'baseline');
  const match = useMemo(() => {
    if (!rows) return null;
    // Prefer the formal-grid row when multiple scenarios share the same coords
    // (e.g. pv050_dt00_gr000 over pv50_tropical_today over pv50_ev30_today).
    return rows.find(
      (r) =>
        r.grid &&
        Math.abs(r.pv - pv) < 1e-6 &&
        Math.abs(r.dT - dT) < 1e-6 &&
        Math.abs(r.growth - growth) < 1e-6,
    );
  }, [rows, pv, dT, growth]);

  const delta = match && baseline ? ((match.lol_mean - baseline.lol_mean) / baseline.lol_mean) * 100 : null;

  // Sparkline showing all 30 formal-grid scenarios sorted by LoL, with selected
  // highlighted. `r.grid` is set by build_data.mjs from the pv###_dt##_gr### regex.
  const spark = useMemo(() => {
    if (!rows) return [];
    const grid = rows.filter((r) => r.grid);
    grid.sort((a, b) => a.lol_mean - b.lol_mean);
    return grid;
  }, [rows]);

  const selectedIdx = match ? spark.findIndex((s) => s.scenario === match.scenario) : -1;

  return (
    <section className="border hairline rounded-xl p-6 md:p-10 bg-white/60 dark:bg-ink-900/40">
      <div className="grid md:grid-cols-12 gap-8 md:gap-12">
        <div className="md:col-span-5">
          <p className="kicker">Scenario explorer</p>
          <h3 className="font-serif text-3xl md:text-4xl mt-3 text-ink-950 dark:text-ink-50 tracking-tighter">
            Pick a future and see the loss.
          </h3>

          <Slider label="Rooftop-PV penetration" value={pv} options={PV_OPTS} format={(v) => `${(v * 100).toFixed(0)}%`} onChange={setPv} />
          <Slider label="Ambient warming offset" value={dT} options={DT_OPTS} format={(v) => `+${v.toFixed(0)} °C`} onChange={setDT} />
          <Slider label="Load growth" value={growth} options={G_OPTS} format={(v) => `${v.toFixed(1)} %/yr`} onChange={setGrowth} />

          {match ? (
            <div className="mt-8 border-t hairline pt-6">
              <p className="kicker">Annual loss-of-life</p>
              <div className="mt-2 flex items-end gap-3">
                <p className="numera text-5xl md:text-6xl text-ink-950 dark:text-ink-50 tracking-tighter">
                  {match.lol_mean.toFixed(4)}
                </p>
                <span className="text-2xl text-ink-500 mb-2">%</span>
              </div>
              <p className="mt-2 text-sm text-ink-600 dark:text-ink-300">
                ±{match.lol_std.toFixed(4)} % across {match.n.toLocaleString('en-US')} Monte-Carlo runs
              </p>

              {delta !== null && (
                <span
                  className={
                    'inline-flex items-center mt-4 numera text-sm px-2.5 py-1 rounded-md ' +
                    (delta < -2
                      ? 'bg-cool text-coolInk'
                      : delta > 2
                        ? 'bg-warm text-warmInk'
                        : 'bg-ink-100 text-ink-700 dark:bg-ink-800 dark:text-ink-200')
                  }
                >
                  {delta > 0 ? '+' : ''}{delta.toFixed(1)}% vs no-PV baseline
                </span>
              )}

              <dl className="mt-6 grid grid-cols-2 gap-y-3 gap-x-6 text-sm">
                <Pair k="Peak θ_HS" v={`${match.peak_ths.toFixed(1)} °C`} />
                <Pair k="Reverse-flow" v={`${match.reverse_flow_h.toFixed(0)} h/yr`} />
              </dl>
            </div>
          ) : (
            <p className="mt-8 text-sm text-ink-600 dark:text-ink-300">
              No matching scenario in the grid for this combination.
            </p>
          )}
        </div>

        <div className="md:col-span-7">
          <p className="kicker mb-4">All 30 grid scenarios, sorted by loss-of-life</p>
          <PlotlyChart
            className="h-[420px]"
            ariaLabel="Sparkline ranking of all grid scenarios with the currently selected scenario highlighted"
            data={[
              {
                type: 'bar',
                orientation: 'h',
                x: spark.map((s) => s.lol_mean),
                y: spark.map((s) => labelFor(s)),
                marker: {
                  color: spark.map((_, i) =>
                    i === selectedIdx ? palette.signal : '#D5D4D0',
                  ),
                },
                hovertemplate:
                  '<b>%{y}</b><br>LoL: %{x:.4f}%<extra></extra>',
                cliponaxis: false,
              },
            ]}
            layout={{
              showlegend: false,
              xaxis: {
                title: { text: 'Annual loss-of-life (%)', standoff: 12 },
                tickformat: '.4f',
              },
              yaxis: { tickfont: { size: 10 }, automargin: true },
              margin: { l: 180, r: 32, t: 6, b: 48 },
              shapes: baseline
                ? [
                    {
                      type: 'line',
                      x0: baseline.lol_mean,
                      x1: baseline.lol_mean,
                      y0: -0.5,
                      y1: spark.length - 0.5,
                      line: { color: '#56544F', width: 1, dash: 'dot' },
                    },
                  ]
                : [],
            }}
            fallback={null}
          />
        </div>
      </div>
    </section>
  );
}

function labelFor(r: LolRow) {
  return `${(r.pv * 100).toFixed(0)}% PV · +${r.dT}°C · ${r.growth.toFixed(1)}%/yr`;
}

function Slider({
  label,
  value,
  options,
  format,
  onChange,
}: {
  label: string;
  value: number;
  options: number[];
  format: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="mt-6">
      <div className="flex items-baseline justify-between mb-2">
        <p className="text-sm font-medium text-ink-800 dark:text-ink-100">{label}</p>
        <p className="numera text-sm text-signal-500 dark:text-signal-300">{format(value)}</p>
      </div>
      <div role="radiogroup" aria-label={label} className="flex flex-wrap gap-2">
        {options.map((o) => (
          <button
            key={o}
            type="button"
            role="radio"
            aria-checked={Math.abs(o - value) < 1e-6}
            onClick={() => onChange(o)}
            className={
              'numera text-sm px-3.5 py-2 rounded-md border transition-colors active:scale-[0.98] ' +
              (Math.abs(o - value) < 1e-6
                ? 'border-signal-500 bg-signal-500 text-white'
                : 'hairline text-ink-700 dark:text-ink-200 hover:bg-ink-100 dark:hover:bg-ink-900')
            }
          >
            {format(o)}
          </button>
        ))}
      </div>
    </div>
  );
}

function Pair({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt className="text-ink-600 dark:text-ink-300">{k}</dt>
      <dd className="numera text-ink-900 dark:text-ink-100 text-right">{v}</dd>
    </>
  );
}
