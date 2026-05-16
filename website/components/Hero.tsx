import { HEADLINE, SUBSTATION } from '@/content/findings';
import Byline from './Byline';

// Full-bleed magazine-cover hero. Byline → display headline (full width,
// two-tone) → dek (drop-cap, justified, narrow measure) → colophon row
// (four tabular stats, hairline rules) → link-style CTAs.

export default function Hero() {
  return (
    <section
      id="top"
      className="relative min-h-[100dvh] flex flex-col"
    >
      <Byline />

      <div className="flex-1 flex flex-col justify-center px-6 py-16 md:py-24">
        <div className="max-w-page mx-auto w-full">
          <p className="kicker mb-8">IEEE ICT-PEP 2026 · Companion report</p>

          <h1 className="display text-[3.5rem] md:text-[6.5rem] lg:text-[8.5rem] xl:text-[9.5rem] text-ink-950 dark:text-ink-50">
            Rooftop solar buys our&nbsp;transformers years of life
            <span className="block italic font-light text-ink-500 dark:text-ink-400">
              — until the climate eats the&nbsp;gain.
            </span>
          </h1>

          <div className="mt-16 md:mt-20 grid md:grid-cols-12 gap-10">
            <div className="md:col-span-7 md:col-start-1 drop-cap">
              <p className="font-serif text-xl md:text-2xl leading-relaxed text-ink-800 dark:text-ink-100 max-w-[44ch]">
                We ran{' '}
                <span className="numera">
                  {HEADLINE.mcRuns.toLocaleString('en-US')}
                </span>{' '}
                Monte-Carlo simulations of the {SUBSTATION.ratingMVA}&nbsp;MVA
                distribution transformer at {SUBSTATION.name} — a&nbsp;
                {SUBSTATION.hvKv}/{SUBSTATION.mvKv}&nbsp;kV substation in{' '}
                {SUBSTATION.province} — under combinations of rooftop-PV
                penetration, ambient warming, load growth, EV charging, and
                battery storage. The physics is IEEE&nbsp;C57.91-2011. The
                acceleration is XGBoost. The findings are sobering.
              </p>
            </div>

            <aside className="md:col-span-4 md:col-start-9 self-end text-sm">
              <dl className="space-y-2 text-ink-600 dark:text-ink-300">
                <Row k="Substation" v={`${SUBSTATION.name} · 150/20 kV`} />
                <Row k="Transformer" v={`${SUBSTATION.ratingMVA} MVA · ${SUBSTATION.cooling}`} />
                <Row k="Climate" v={`${SUBSTATION.meanAmbientC.toFixed(1)} °C mean ambient`} />
                <Row k="Operator" v={SUBSTATION.operator} />
              </dl>
            </aside>
          </div>
        </div>
      </div>

      <footer className="px-6 pb-16 md:pb-20">
        <div className="max-w-page mx-auto">
          <div className="colophon-row">
            <Figure value={`−${HEADLINE.pvCoolingPct}%`} label="LoL at 75 % PV vs no-PV baseline today" />
            <Figure value={`+${HEADLINE.climateUpliftPct}%`} label="From +1 °C ambient and 4.5 %/yr growth" />
            <Figure value={`${HEADLINE.speedupX}×`} label="Surrogate speedup vs IEEE ODE" />
            <Figure value={`${HEADLINE.wallClockMin} min`} label={`${HEADLINE.mcRuns.toLocaleString('en-US')}-run Monte Carlo wall-clock`} />
          </div>

          <p className="mt-10 text-sm text-ink-600 dark:text-ink-300 max-w-[60ch]">
            <a
              href="#findings"
              className="underline underline-offset-4 decoration-1 text-ink-900 dark:text-ink-50 hover:text-signal-500 dark:hover:text-signal-300 mr-6"
            >
              Read the five findings ↓
            </a>
            <a
              href="/scenarios/"
              className="underline underline-offset-4 decoration-1 hover:text-signal-500 dark:hover:text-signal-300"
            >
              Explore the 41-scenario dataset ↗
            </a>
          </p>
        </div>
      </footer>
    </section>
  );
}

function Figure({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <p className="numera text-3xl md:text-5xl font-medium tracking-tight text-ink-950 dark:text-ink-50">
        {value}
      </p>
      <p className="mt-3 text-xs leading-snug uppercase tracking-wider text-ink-600 dark:text-ink-400 max-w-[22ch]">
        {label}
      </p>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-6 border-b hairline pb-2">
      <dt className="text-ink-500 dark:text-ink-400">{k}</dt>
      <dd className="numera text-ink-900 dark:text-ink-100 text-right">{v}</dd>
    </div>
  );
}
