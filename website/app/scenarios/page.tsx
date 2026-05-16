'use client';

import { useMemo, useState } from 'react';
import { useData, type LolRow } from '@/components/charts/useData';
import ScenarioExplorer from '@/components/charts/ScenarioExplorer';

type SortKey = 'lol_mean' | 'pv' | 'dT' | 'growth' | 'peak_ths' | 'reverse_flow_h';

export default function ScenariosPage() {
  const rows = useData<LolRow[]>('lol_stats.json');
  const [q, setQ] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('lol_mean');
  const [asc, setAsc] = useState(true);

  const filtered = useMemo(() => {
    if (!rows) return [];
    const out = rows.filter((r) =>
      q.trim() ? r.scenario.toLowerCase().includes(q.toLowerCase()) : true,
    );
    out.sort((a, b) => {
      const va = a[sortKey] as number;
      const vb = b[sortKey] as number;
      return asc ? va - vb : vb - va;
    });
    return out;
  }, [rows, q, sortKey, asc]);

  const toggleSort = (k: SortKey) => {
    if (sortKey === k) setAsc(!asc);
    else {
      setSortKey(k);
      setAsc(k === 'lol_mean');
    }
  };

  const downloadCsv = () => {
    if (!filtered.length) return;
    const header = ['scenario', 'pv', 'dT', 'growth', 'n', 'lol_mean', 'lol_std', 'lol_p5', 'lol_p95', 'peak_ths', 'reverse_flow_h'];
    const lines = [header.join(',')];
    for (const r of filtered) lines.push(header.map((k) => r[k as keyof LolRow]).join(','));
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'scenarios_filtered.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="px-6 py-20 md:py-28">
      <div className="max-w-page mx-auto">
        <p className="kicker">Data explorer</p>
        <h1 className="font-serif text-4xl md:text-6xl tracking-tighter text-ink-950 dark:text-ink-50 mt-3 max-w-[20ch]">
          All 41 Monte-Carlo scenarios.
        </h1>
        <p className="prose-body mt-6 text-ink-700 dark:text-ink-200">
          Each row is a scenario averaged over 1 000 Monte-Carlo runs. Filter by scenario name,
          sort by any column, and export the filtered slice as CSV.
        </p>

        <div className="my-12">
          <ScenarioExplorer />
        </div>

        <div className="flex flex-wrap gap-4 items-center mt-16 mb-6">
          <input
            type="text"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Filter by scenario name…"
            aria-label="Filter scenarios by name"
            className="flex-1 min-w-[240px] bg-transparent border hairline rounded-md px-3 py-2 text-sm focus:border-signal-500 outline-none"
          />
          <button
            type="button"
            onClick={downloadCsv}
            className="text-sm border hairline rounded-md px-3 py-2 hover:bg-ink-100 dark:hover:bg-ink-900 active:scale-[0.98]"
          >
            Download filtered CSV
          </button>
          <p className="text-sm text-ink-600 dark:text-ink-300 numera">
            {filtered.length} / {rows?.length ?? 0} rows
          </p>
        </div>

        <div className="overflow-x-auto border hairline rounded-lg">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-ink-100/60 dark:bg-ink-900/60 text-left">
                <Th>Scenario</Th>
                <Th onClick={() => toggleSort('pv')} active={sortKey === 'pv'} asc={asc}>PV</Th>
                <Th onClick={() => toggleSort('dT')} active={sortKey === 'dT'} asc={asc}>ΔT</Th>
                <Th onClick={() => toggleSort('growth')} active={sortKey === 'growth'} asc={asc}>Growth</Th>
                <Th onClick={() => toggleSort('lol_mean')} active={sortKey === 'lol_mean'} asc={asc}>LoL (%)</Th>
                <Th>± σ</Th>
                <Th onClick={() => toggleSort('peak_ths')} active={sortKey === 'peak_ths'} asc={asc}>Peak θ_HS</Th>
                <Th onClick={() => toggleSort('reverse_flow_h')} active={sortKey === 'reverse_flow_h'} asc={asc}>Rev. flow</Th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r, i) => (
                <tr
                  key={r.scenario}
                  className={
                    'border-t hairline numera ' +
                    (i % 2 ? 'bg-ink-50/40 dark:bg-ink-900/30' : '')
                  }
                >
                  <td className="px-4 py-2.5 font-sans text-ink-900 dark:text-ink-100">{r.scenario}</td>
                  <td className="px-4 py-2.5">{(r.pv * 100).toFixed(0)}%</td>
                  <td className="px-4 py-2.5">+{r.dT.toFixed(0)} °C</td>
                  <td className="px-4 py-2.5">{r.growth.toFixed(1)} %/yr</td>
                  <td className="px-4 py-2.5">{r.lol_mean.toFixed(4)}</td>
                  <td className="px-4 py-2.5 text-ink-500">±{r.lol_std.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{r.peak_ths.toFixed(1)} °C</td>
                  <td className="px-4 py-2.5">{r.reverse_flow_h.toFixed(0)} h</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="mt-8 text-sm">
          <a href="/" className="underline underline-offset-4 text-signal-500 dark:text-signal-300 hover:opacity-80">
            ← Back to the report
          </a>
        </p>
      </div>
    </div>
  );
}

function Th({
  children,
  onClick,
  active,
  asc,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  active?: boolean;
  asc?: boolean;
}) {
  const arrow = active ? (asc ? '↑' : '↓') : '';
  return (
    <th
      scope="col"
      aria-sort={active ? (asc ? 'ascending' : 'descending') : undefined}
      className="px-4 py-3 text-xs uppercase tracking-wider text-ink-600 dark:text-ink-300"
    >
      {onClick ? (
        <button
          type="button"
          onClick={onClick}
          className="inline-flex items-center gap-1 hover:text-ink-900 dark:hover:text-ink-50 select-none focus-visible:outline-2 focus-visible:outline-signal-500"
        >
          {children} <span className="numera">{arrow}</span>
        </button>
      ) : (
        <>{children}</>
      )}
    </th>
  );
}
