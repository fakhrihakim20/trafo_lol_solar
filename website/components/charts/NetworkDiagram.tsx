'use client';

import { useState } from 'react';

const NODES = [
  {
    id: 'slack',
    x: 80,
    y: 220,
    label: 'Java–Bali 150 kV ring',
    sub: 'Slack generator',
    detail: 'Models the upstream HV system as an infinite bus at the substation’s HV terminal.',
  },
  {
    id: 'hvbus',
    x: 280,
    y: 220,
    label: 'HV bus',
    sub: '150 kV',
    detail: 'High-voltage side of the substation. Series impedance to the slack is negligible at this level.',
  },
  {
    id: 'tx',
    x: 480,
    y: 220,
    label: 'tfr_60mva',
    sub: '60 MVA · ONAF',
    detail:
      'IEEE C57.91-2011 thermal model: ΔΘ_TO,R = 55 °C, ΔΘ_HS,R = 25 °C, τ_TO = 150 min, τ_W = 7 min, R = 5, n = m = 0.8. s_max_pu = 2.0.',
  },
  {
    id: 'mvbus',
    x: 680,
    y: 220,
    label: 'MV bus',
    sub: '20 kV',
    detail: 'Medium-voltage bus where rooftop PV, the inelastic load, and the optional battery are coupled.',
  },
  {
    id: 'load',
    x: 820,
    y: 100,
    label: 'Load',
    sub: 'Residential + EV',
    detail: 'East-Java synthetic load profile with evening peak at 21:00 and weekend reduction × 0.92.',
  },
  {
    id: 'pv',
    x: 820,
    y: 220,
    label: 'Rooftop PV',
    sub: '0–75 % penetration',
    detail: 'Haurwitz clear-sky × Markov cloud chain at lat −8.17°, lon 113.70°. Falls back to ERA5 if available.',
  },
  {
    id: 'bess',
    x: 820,
    y: 340,
    label: 'BESS',
    sub: '5 MWh · 2.0 MW',
    detail:
      'Optional storage with split efficiency η_c · η_d = 0.95 · 0.95 = 0.9025. Dispatched by threshold rule or HiGHS LP.',
  },
];

const EDGES: Array<[string, string]> = [
  ['slack', 'hvbus'],
  ['hvbus', 'tx'],
  ['tx', 'mvbus'],
  ['mvbus', 'load'],
  ['mvbus', 'pv'],
  ['mvbus', 'bess'],
];

export default function NetworkDiagram() {
  const [active, setActive] = useState<string | null>(null);
  const node = NODES.find((n) => n.id === active);

  return (
    <div>
      <p className="kicker mb-4">Fig. 1 · Two-bus PyPSA network of GI Jember</p>
      <div className="grid md:grid-cols-12 gap-6 items-start">
        <div className="md:col-span-8 border hairline rounded-lg p-2 bg-ink-50 dark:bg-ink-900/40">
          <svg
            viewBox="0 0 920 440"
            role="img"
            aria-label="Single-line diagram of the GI Jember two-bus network"
            className="w-full h-auto"
          >
            <defs>
              <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#56544F" />
              </marker>
            </defs>
            {EDGES.map(([a, b]) => {
              const na = NODES.find((n) => n.id === a)!;
              const nb = NODES.find((n) => n.id === b)!;
              const isActive = active === a || active === b;
              return (
                <line
                  key={`${a}-${b}`}
                  x1={na.x}
                  y1={na.y}
                  x2={nb.x}
                  y2={nb.y}
                  stroke={isActive ? '#1B3F8B' : '#9B9A95'}
                  strokeWidth={isActive ? 2.5 : 1.5}
                />
              );
            })}

            {/* Transformer special glyph */}
            {(() => {
              const t = NODES.find((n) => n.id === 'tx')!;
              return (
                <g key="tx-glyph" onMouseEnter={() => setActive('tx')} onMouseLeave={() => setActive(null)}>
                  <circle cx={t.x - 16} cy={t.y} r="22" fill="none" stroke="#1B3F8B" strokeWidth="2" />
                  <circle cx={t.x + 16} cy={t.y} r="22" fill="none" stroke="#1B3F8B" strokeWidth="2" />
                </g>
              );
            })()}

            {NODES.map((n) => (
              <g
                key={n.id}
                transform={`translate(${n.x},${n.y})`}
                onMouseEnter={() => setActive(n.id)}
                onMouseLeave={() => setActive(null)}
                onFocus={() => setActive(n.id)}
                onBlur={() => setActive(null)}
                tabIndex={0}
                role="button"
                aria-label={`${n.label}, ${n.sub}`}
                className="cursor-pointer focus:outline-none"
              >
                {n.id !== 'tx' && (
                  <rect
                    x={-44}
                    y={-18}
                    width={88}
                    height={36}
                    rx={4}
                    fill={active === n.id ? '#1B3F8B' : 'white'}
                    stroke={active === n.id ? '#1B3F8B' : '#37352F'}
                    strokeWidth={1.5}
                  />
                )}
                <text
                  textAnchor="middle"
                  y={n.id === 'tx' ? 48 : 4}
                  fontFamily="var(--font-geist-sans), system-ui"
                  fontSize="12"
                  fontWeight={600}
                  fill={active === n.id && n.id !== 'tx' ? '#FFFFFF' : '#111111'}
                >
                  {n.label}
                </text>
                <text
                  textAnchor="middle"
                  y={n.id === 'tx' ? 62 : 16}
                  fontFamily="var(--font-jetbrains), monospace"
                  fontSize="10"
                  fill={active === n.id && n.id !== 'tx' ? '#DDE5F2' : '#56544F'}
                >
                  {n.sub}
                </text>
              </g>
            ))}
          </svg>
        </div>
        <aside className="md:col-span-4">
          <div className="border hairline rounded-lg p-5 min-h-[200px]">
            {node ? (
              <>
                <p className="kicker">{node.sub}</p>
                <h3 className="font-serif text-xl mt-2 text-ink-950 dark:text-ink-50">
                  {node.label}
                </h3>
                <p className="mt-3 text-sm text-ink-600 dark:text-ink-300 leading-relaxed">
                  {node.detail}
                </p>
              </>
            ) : (
              <p className="text-sm text-ink-600 dark:text-ink-300">
                Hover a component to inspect its parameters. The 60 MVA ONAF transformer is the
                element whose hot-spot temperature this study tracks.
              </p>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
