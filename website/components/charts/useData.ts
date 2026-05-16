'use client';

import { useEffect, useState } from 'react';

// When the site is built for a GitHub Pages subpath (NEXT_PUBLIC_BASE_PATH set
// at build time), all fetch URLs need the same prefix or they 404 on Pages.
// Empty string for local dev and root-served deployments.
export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || '';

export function dataUrl(file: string): string {
  return `${BASE_PATH}/data/${file}`;
}

export function useData<T = unknown>(file: string): T | null {
  const [data, setData] = useState<T | null>(null);
  useEffect(() => {
    let alive = true;
    fetch(dataUrl(file))
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then((j) => alive && setData(j))
      .catch((e) => console.error(`[useData] ${file}:`, e));
    return () => {
      alive = false;
    };
  }, [file]);
  return data;
}

export type LolRow = {
  scenario: string;
  pv: number;
  dT: number;
  growth: number;
  /** true for the 30 formal pv###_dt##_gr### grid scenarios; false for named aliases (baseline, *_tropical_today) and named non-grid scenarios (EV, BESS, warmer_2030). */
  grid: boolean;
  n: number;
  lol_mean: number;
  lol_std: number;
  lol_p5: number;
  lol_p95: number;
  peak_ths: number;
  reverse_flow_h: number;
};

export type DispatchRow = {
  scenario: string;
  method: 'threshold' | 'pypsa_lp' | string;
  lol_pct: number[];
  peak_ths: number[];
  battery_lifetime: number[];
  reverse_flow_h: number[];
};

export type Metrics = Record<string, number>;
