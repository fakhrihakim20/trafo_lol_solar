'use client';

import dynamic from 'next/dynamic';
import type Plotly from 'plotly.js';
type Layout = Plotly.Layout;
type Data = Plotly.Data;
type Config = Plotly.Config;
import { useEffect, useState } from 'react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export type PlotlyChartProps = {
  data: Partial<Data>[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
  ariaLabel: string;
  // Plain-text fallback table for screen readers
  fallback?: React.ReactNode;
};

const PALETTE = {
  ink: '#111111',
  inkSoft: '#56544F',
  hair: '#EAEAEA',
  bg: '#F7F6F3',
  bgDark: '#111111',
  signal: '#1B3F8B',
  warm: '#956400',
  leaf: '#346538',
  cool: '#1F6C9F',
};

export function useIsDark() {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    // Track the .dark class on <html> (set by ThemeToggle + the FOUC preload
    // script in layout.tsx). MutationObserver fires whenever the toggle flips
    // the class, so Plotly axis/grid colours follow the user's chosen theme
    // without any explicit wiring from the toggle component.
    const root = document.documentElement;
    const update = () => setDark(root.classList.contains('dark'));
    update();
    const obs = new MutationObserver(update);
    obs.observe(root, { attributes: true, attributeFilter: ['class'] });
    return () => obs.disconnect();
  }, []);
  return dark;
}

export function chartBase(dark: boolean): Partial<Layout> {
  // Axis titles in italic Newsreader serif give the charts a magazine
  // "plate caption" register that matches the editorial body type.
  const serifTitleFont = {
    family: 'var(--font-newsreader), Georgia, serif',
    size: 14,
    color: dark ? '#D5D4D0' : '#37352F',
  } as const;

  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 64, r: 24, t: 24, b: 56 },
    font: {
      family:
        'var(--font-geist-sans), ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
      color: dark ? '#D5D4D0' : '#37352F',
      size: 13,
    },
    hoverlabel: {
      bgcolor: dark ? '#1A1A1A' : '#FFFFFF',
      bordercolor: dark ? 'rgba(255,255,255,0.08)' : PALETTE.hair,
      font: {
        family: 'var(--font-geist-sans), system-ui, sans-serif',
        color: dark ? '#F7F6F3' : '#111111',
        size: 13,
      },
      align: 'left',
    },
    xaxis: ({
      gridcolor: dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
      zerolinecolor: dark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.12)',
      tickfont: { family: 'var(--font-jetbrains), monospace', size: 11 },
      // `title` font is applied here for the per-chart x-axis title text.
      // Plotly merges this with the chart-level `xaxis.title` set in each
      // chart component. Italic Newsreader gives the "plate caption" feel.
      title: { font: { ...serifTitleFont, style: 'italic' } },
      linecolor: dark ? 'rgba(255,255,255,0.18)' : 'rgba(0,0,0,0.2)',
      linewidth: 1,
      mirror: false,
      ticks: 'outside',
      ticklen: 4,
      tickcolor: dark ? 'rgba(255,255,255,0.18)' : 'rgba(0,0,0,0.2)',
    } as any),
    yaxis: ({
      gridcolor: dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
      zerolinecolor: dark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.12)',
      tickfont: { family: 'var(--font-jetbrains), monospace', size: 11 },
      title: { font: { ...serifTitleFont, style: 'italic' } },
      linecolor: dark ? 'rgba(255,255,255,0.18)' : 'rgba(0,0,0,0.2)',
      linewidth: 1,
      ticks: 'outside',
      ticklen: 4,
      tickcolor: dark ? 'rgba(255,255,255,0.18)' : 'rgba(0,0,0,0.2)',
    } as any),
    legend: {
      orientation: 'h',
      yanchor: 'bottom',
      y: 1.02,
      xanchor: 'right',
      x: 1,
      bgcolor: 'rgba(0,0,0,0)',
      font: { size: 12 },
    },
  };
}

export const palette = PALETTE;

export default function PlotlyChart({
  data,
  layout,
  config,
  className,
  ariaLabel,
  fallback,
}: PlotlyChartProps) {
  const dark = useIsDark();
  const merged: Partial<Layout> = { ...chartBase(dark), ...layout };
  const cfg: Partial<Config> = {
    displaylogo: false,
    responsive: true,
    toImageButtonOptions: { format: 'png', scale: 2 },
    modeBarButtonsToRemove: [
      'lasso2d',
      'select2d',
      'autoScale2d',
      'hoverClosestCartesian',
      'hoverCompareCartesian',
    ],
    ...config,
  };
  return (
    <figure
      className={className}
      role="img"
      aria-label={ariaLabel}
    >
      <Plot
        data={data as any}
        layout={merged as any}
        config={cfg as any}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
      {fallback ? (
        <details className="mt-3 text-sm text-ink-600 dark:text-ink-300">
          <summary className="cursor-pointer underline underline-offset-4">
            View underlying data
          </summary>
          <div className="mt-3">{fallback}</div>
        </details>
      ) : null}
    </figure>
  );
}
