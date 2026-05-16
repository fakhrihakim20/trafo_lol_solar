'use client';

import { useMemo, useState } from 'react';
import PlotlyChart, { palette } from './PlotlyChart';
import { useData, type DispatchRow } from './useData';

type MetricKey = 'lol_pct' | 'battery_lifetime' | 'peak_ths';
const SCENS = [
  { id: 'pv75_battery5mwh_today', label: '75 % PV + 5 MWh battery · today' },
  { id: 'pv75_battery5mwh_ev30_warmer_2030', label: '75 % PV + battery + 30 % EV · 2030' },
];

export default function DispatchBoxplot() {
  const rows = useData<DispatchRow[]>('dispatch.json');
  const [scen, setScen] = useState<string>(SCENS[0].id);
  const [metric, setMetric] = useState<MetricKey>('lol_pct');

  const traces = useMemo(() => {
    if (!rows) return [];
    const subset = rows.filter((r) => r.scenario === scen);
    const ord = ['threshold', 'pypsa_lp'];
    return subset
      .slice()
      .sort((a, b) => ord.indexOf(a.method) - ord.indexOf(b.method))
      .map((r) => ({
        type: 'box' as const,
        y: r[metric],
        name: r.method === 'pypsa_lp' ? 'PyPSA HiGHS LP' : 'Threshold rule',
        marker: {
          color: r.method === 'pypsa_lp' ? palette.warm : palette.signal,
          size: 4,
          opacity: 0.6,
        },
        line: { color: r.method === 'pypsa_lp' ? palette.warm : palette.signal },
        boxmean: 'sd' as const,
        boxpoints: 'all' as const,
        jitter: 0.4,
        pointpos: 0,
        hovertemplate: '<b>%{fullData.name}</b><br>%{y:.5f}<extra></extra>',
      }));
  }, [rows, scen, metric]);

  const yTitle =
    metric === 'lol_pct'
      ? 'Annual loss-of-life (%)'
      : metric === 'battery_lifetime'
        ? 'Battery lifetime fraction consumed'
        : 'Peak hot-spot temperature θ_HS (°C)';

  const tickformat =
    metric === 'lol_pct' ? '.5f' : metric === 'battery_lifetime' ? '.5f' : '.1f';

  return (
    <div>
      <div className="mb-4 flex items-center justify-between gap-4 flex-wrap">
        <p className="kicker">Fig. 4 · PyPSA HiGHS LP vs threshold-rule dispatch</p>
        <div className="flex flex-wrap gap-2">
          <Selector
            label="Scenario"
            value={scen}
            onChange={setScen}
            options={SCENS.map((s) => ({ value: s.id, label: s.label }))}
          />
          <Selector
            label="Metric"
            value={metric}
            onChange={(v) => setMetric(v as MetricKey)}
            options={[
              { value: 'lol_pct', label: 'Loss-of-life (%/yr)' },
              { value: 'battery_lifetime', label: 'Battery lifetime' },
              { value: 'peak_ths', label: 'Peak θ_HS' },
            ]}
          />
        </div>
      </div>

      <PlotlyChart
        className="h-[460px]"
        ariaLabel="Paired Monte Carlo box plot comparing economic LP and threshold-rule dispatch"
        data={traces}
        layout={{
          showlegend: false,
          xaxis: { title: { text: 'Dispatch method', standoff: 12 } },
          yaxis: { title: { text: yTitle, standoff: 12 }, tickformat },
          margin: { l: 72, r: 24, t: 12, b: 48 },
        }}
        fallback={null}
      />
      <p className="mt-4 text-sm text-ink-600 dark:text-ink-300 max-w-prose">
        <span className="text-ink-700 dark:text-ink-200">How to read.</span>{' '}
        Each dot is one of 100 paired Monte-Carlo runs; the same random seed feeds both methods. The
        economic LP minimises slack-import cost; the threshold rule discharges above 0.85 pu net load.
        The cost-optimal LP is consistently above the threshold rule for loss-of-life — economic dispatch
        is not aging-optimal.
      </p>
    </div>
  );
}

function Selector({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <label className="text-xs text-ink-600 dark:text-ink-300 flex items-center gap-2">
      <span className="kicker mr-1">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-transparent border hairline rounded-md text-sm px-2 py-1 text-ink-900 dark:text-ink-50 focus:border-signal-500"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}
