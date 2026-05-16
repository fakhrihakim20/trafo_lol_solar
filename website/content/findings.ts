// Single source of truth for the five headline findings + supporting prose.
// Numbers must match results/metrics.json exactly — if they ever diverge,
// the paper is authoritative.

export const HEADLINE = {
  pvCoolingPct: 19,             // -19% LoL at 75% PV today
  climateUpliftPct: 43,         // +43% under +1C / +4.5%/yr
  evUpliftPct: 42,              // EV at 30% raises LoL by +42% vs same scenario without EVs
  batteryDeltaPct: 1.5,         // -1.5% further reduction from 5 MWh battery
  dispatchGapPct: 1.2,          // LP is +1.2% worse for LoL than threshold rule
  surrogateRmseC: 0.230,
  surrogateR2: 0.9998,
  speedupX: 474,
  wallClockMin: 53,
  mcRuns: 41_000,
  nScenarios: 41,
};

export const SUBSTATION = {
  name: 'GI Jember',
  province: 'East Java, Indonesia',
  ratingMVA: 60,
  cooling: 'ONAF',
  hvKv: 150,
  mvKv: 20,
  meanAmbientC: 26.5,
  diurnalHalfAmpC: 5.0,
  seasonalHalfAmpC: 1.2,
  operator: 'PT PLN (Persero)',
};

export type Finding = {
  id: string;
  kicker: string;
  title: string;
  oneLiner: string;
  body: string;
  callouts: { value: string; label: string }[];
  chart: string; // component key
};

export const FINDINGS: Finding[] = [
  {
    id: 'surrogate',
    kicker: '01 — Faster physics',
    title: 'An XGBoost surrogate reproduces 41 000 thermal trajectories 474× faster than the IEEE ODE.',
    oneLiner: 'Same physics, fraction of the runtime — and the surrogate is honest about its errors.',
    body:
      'We train an XGBoost regressor on 200 IEEE C57.91 ODE trajectories (33 000 supervised rows) using teacher forcing, ' +
      'then roll it out autoregressively at inference. On a 30-trajectory hold-out, the surrogate predicts hot-spot ' +
      'temperature θ_HS to within 0.230 °C RMSE (R² = 0.9998); 99.78 % of predictions fall inside the engineering-relevant ' +
      '±1.5 °C band. That makes the full 41-scenario × 1 000-run Monte Carlo tractable on a single laptop in 53 minutes — ' +
      'a 474× wall-clock speedup at 1 000 runs versus the ODE baseline.',
    callouts: [
      { value: '0.230 °C', label: 'Hold-out RMSE' },
      { value: '0.9998', label: 'R²' },
      { value: '474×', label: 'Speedup' },
      { value: '53 min', label: '41 000-run wall-clock' },
    ],
    chart: 'SurrogateAccuracy',
  },
  {
    id: 'pv-cooling',
    kicker: '02 — Today, more PV cools the transformer',
    title: 'Rooftop PV monotonically reduces annual loss-of-life — up to 19 % at 75 % penetration.',
    oneLiner:
      'In today’s climate, midday PV generation strips load from the transformer at its hottest hour. ' +
      'Every additional percentage point of penetration buys aging margin.',
    body:
      'At GI Jember’s 26.5 °C mean ambient, baseline annual loss-of-life is 0.0342 % (≈ 62 hours of equivalent life ' +
      'lost per year against the IEEE 180 000-hour reference). Adding rooftop PV monotonically cools the daytime hot-spot: ' +
      'LoL falls to 0.0303 % at 30 % PV, 0.0285 % at 50 %, and 0.0277 % at 75 %. At the highest penetration, the substation ' +
      'exports for ≈ 643 hours per year — 7 % of the year in reverse-flow operation. The ±1σ Monte Carlo bands are tight ' +
      '(σ ≈ 0.0002 %), so even sub-percent scenario differences are statistically resolvable.',
    callouts: [
      { value: '−19 %', label: 'LoL at 75 % PV vs baseline' },
      { value: '0.0342 %', label: 'Baseline annual LoL' },
      { value: '643 h', label: 'Reverse-flow hours at 75 % PV' },
      { value: '±0.0002 %', label: 'Monte Carlo σ' },
    ],
    chart: 'PvVsLolChart',
  },
  {
    id: 'climate',
    kicker: '03 — But climate eats the gains',
    title: 'A +1 °C warmer ambient and +4.5 %/yr load growth raise LoL by 43 % — more than PV can recover.',
    oneLiner:
      'Even 75 % rooftop PV cannot fully restore today’s no-PV baseline under projected 2030 conditions.',
    body:
      'Decomposing the stressors: +1 °C ambient alone raises LoL to 0.0367 % (+7 %); +4.5 %/yr load growth alone raises ' +
      'it to 0.0454 % (+33 %); the combined 2030 stress reaches 0.0488 % (+43 %). At 75 % PV under that same combined stress, ' +
      'LoL is 0.0376 % — still 10 % above today’s no-PV baseline. The takeaway for fleet planners: rooftop PV is a real ' +
      'aging benefit today, but it is not by itself sufficient to absorb the climate × growth headwind across the 2030 horizon. ' +
      'Other levers — cooling-system upgrades, transformer up-rating, demand-side management — must run in parallel.',
    callouts: [
      { value: '+7 %', label: 'From +1 °C alone' },
      { value: '+33 %', label: 'From +4.5 %/yr growth alone' },
      { value: '+43 %', label: 'Combined 2030 stress' },
      { value: '+10 %', label: '75 % PV under 2030 stress vs today' },
    ],
    chart: 'SensitivityHeatmap',
  },
  {
    id: 'dispatch',
    kicker: '04 — Economic LP is not aging-optimal',
    title: 'A 5 MWh battery only saves another 1.5 %, and the HiGHS economic LP is 1.2 % worse for LoL than a simple rule.',
    oneLiner:
      'Minimising grid-import cost and minimising insulation aging are not the same problem — the LP optimises the wrong objective.',
    body:
      'Adding a 5 MWh / 2.0 MW battery on top of 75 % PV brings LoL from 0.0277 % down to 0.0273 % — a further −1.5 %. ' +
      'When we let the PyPSA HiGHS linear program dispatch the same battery to minimise slack-import cost, LoL rises ' +
      'to 0.02760 % — +1.2 % worse than the simple threshold rule at 0.02728 %. The economic LP disperses battery action ' +
      'across all hours with positive marginal value; the threshold rule only discharges above 0.85 pu, which happens to ' +
      'target precisely the hours where the Arrhenius integral is super-linear. Until aging-aware constraints enter the LP, ' +
      'practitioners should not assume that economically-optimal dispatch protects transformer life.',
    callouts: [
      { value: '−1.5 %', label: 'Battery LoL benefit (threshold)' },
      { value: '+1.2 %', label: 'Economic LP penalty vs threshold' },
      { value: '0.0273 %', label: 'Best dispatch (threshold)' },
      { value: '0.0276 %', label: 'HiGHS LP (cost-optimal)' },
    ],
    chart: 'DispatchBoxplot',
  },
  {
    id: 'ev',
    kicker: '05 — EVs reverse the rooftop-PV gain',
    title: 'EV charging at 30 % fleet penetration adds back +42 % of LoL — pushing the substation above the no-PV baseline.',
    oneLiner:
      'Evening charging coincides exactly with the existing residential peak. Coordinated tariffs or smart charging are not optional.',
    body:
      'A scenario with 50 % rooftop PV and 30 % EV-fleet penetration (`pv50_ev30_today`) yields LoL = 0.0405 %, ' +
      'which is +42 % above the same PV scenario without EVs and +18 % above today’s no-PV baseline. The mechanism ' +
      'is straightforward: the EV charging curve concentrates around the 18:00–21:00 window, which is also the existing ' +
      'residential evening peak and the daily hot-spot maximum. Without time-of-use signalling or directly controlled ' +
      'charging, EV adoption converts a rooftop-PV "aging gift" into a net deterioration.',
    callouts: [
      { value: '+42 %', label: 'EV uplift vs same PV scenario' },
      { value: '+18 %', label: 'Net vs no-PV baseline' },
      { value: '0.0405 %', label: 'LoL at 50 % PV + 30 % EV' },
      { value: '18:00–21:00', label: 'EV/peak overlap window' },
    ],
    chart: 'EvBatteryImpact',
  },
];
