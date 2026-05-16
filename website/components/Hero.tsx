import { HEADLINE, SUBSTATION } from '@/content/findings';

export default function Hero() {
  return (
    <section
      id="top"
      className="relative pt-28 pb-24 md:pt-36 md:pb-32 px-6 min-h-[100dvh] flex flex-col justify-between"
    >
      <div className="max-w-page mx-auto w-full grid md:grid-cols-12 gap-10 md:gap-14">
        <div className="md:col-span-7">
          <p className="kicker mb-6">
            IEEE ICT-PEP 2026 · Companion report
          </p>
          <h1 className="display text-5xl md:text-7xl lg:text-[5.5rem] text-ink-950 dark:text-ink-50">
            Rooftop solar buys our transformers years of life
            <span className="italic font-light"> — </span>
            <span className="text-ink-500">until the climate eats the gain.</span>
          </h1>
          <p className="prose-body mt-10 text-ink-700 dark:text-ink-200">
            We ran <span className="numera">{HEADLINE.mcRuns.toLocaleString('en-US')}</span> Monte-Carlo simulations of
            the {SUBSTATION.ratingMVA} MVA distribution transformer at {SUBSTATION.name} —
            a {SUBSTATION.hvKv}/{SUBSTATION.mvKv} kV substation in {SUBSTATION.province} — under combinations of
            rooftop-PV penetration, ambient warming, load growth, EV charging, and battery storage.
            The physics is IEEE C57.91-2011. The acceleration is XGBoost. The findings are sobering.
          </p>
          <div className="mt-10 flex flex-wrap gap-3">
            <a
              href="#findings"
              className="inline-flex items-center gap-2 text-sm font-medium bg-ink-950 text-ink-50 dark:bg-ink-50 dark:text-ink-950 rounded-md px-4 py-2.5 transition-transform active:scale-[0.98]"
            >
              Read the five findings
            </a>
            <a
              href="/scenarios/"
              className="inline-flex items-center gap-2 text-sm font-medium border hairline text-ink-900 dark:text-ink-100 rounded-md px-4 py-2.5 hover:bg-ink-100 dark:hover:bg-ink-900 transition-colors"
            >
              Explore the 41-scenario dataset
            </a>
          </div>
        </div>

        <aside className="md:col-span-5 self-end">
          <div className="border-t hairline pt-8 md:pt-12 grid grid-cols-2 gap-x-8 gap-y-10">
            <Stat value={`−${HEADLINE.pvCoolingPct}%`} label="Annual loss-of-life at 75 % PV vs no-PV baseline (today)" tone="cool" />
            <Stat value={`+${HEADLINE.climateUpliftPct}%`} label="Loss-of-life uplift from +1 °C ambient + 4.5 %/yr load growth" tone="warm" />
            <Stat value={`${HEADLINE.speedupX}×`} label="Wall-clock speedup of the XGBoost surrogate vs the IEEE ODE" />
            <Stat value={`${HEADLINE.wallClockMin} min`} label={`Full ${HEADLINE.mcRuns.toLocaleString('en-US')}-run Monte Carlo on a laptop`} />
          </div>
        </aside>
      </div>

      <div className="max-w-page mx-auto w-full mt-20 md:mt-28">
        <div className="flex items-end justify-between border-t hairline pt-6">
          <p className="text-xs kicker">Scroll for the full report</p>
          <p className="text-xs numera text-ink-500">
            {SUBSTATION.meanAmbientC.toFixed(1)} °C mean ambient · {SUBSTATION.cooling} cooling
          </p>
        </div>
      </div>
    </section>
  );
}

function Stat({ value, label, tone }: { value: string; label: string; tone?: 'cool' | 'warm' }) {
  const accent =
    tone === 'cool'
      ? 'text-signal-500 dark:text-signal-300'
      : tone === 'warm'
        ? 'text-warmInk dark:text-warm'
        : 'text-ink-950 dark:text-ink-50';
  return (
    <div>
      <p className={`numera text-3xl md:text-4xl font-medium tracking-tight ${accent}`}>{value}</p>
      <p className="mt-2 text-sm text-ink-600 dark:text-ink-300 leading-snug">{label}</p>
    </div>
  );
}
