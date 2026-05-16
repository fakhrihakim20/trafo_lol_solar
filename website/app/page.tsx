import Hero from '@/components/Hero';
import Equation from '@/components/Equation';
import StatCallouts from '@/components/StatCallout';
import PullQuote from '@/components/PullQuote';
import NetworkDiagram from '@/components/charts/NetworkDiagram';
import SurrogateAccuracy from '@/components/charts/SurrogateAccuracy';
import PvVsLolChart from '@/components/charts/PvVsLolChart';
import DispatchBoxplot from '@/components/charts/DispatchBoxplot';
import RuntimeBars from '@/components/charts/RuntimeBars';
import SensitivityHeatmap from '@/components/charts/SensitivityHeatmap';
import EvBatteryImpact from '@/components/charts/EvBatteryImpact';
import ScenarioExplorer from '@/components/charts/ScenarioExplorer';
import { FINDINGS, HEADLINE, SUBSTATION } from '@/content/findings';

const chartMap: Record<string, React.ComponentType> = {
  SurrogateAccuracy,
  PvVsLolChart,
  SensitivityHeatmap,
  DispatchBoxplot,
  EvBatteryImpact,
};

const FINDING_NUMERALS = ['I', 'II', 'III', 'IV', 'V'];

export default function Page() {
  return (
    <>
      <Hero />

      {/* I. Why it matters */}
      <Section id="problem" numeral="I" kicker="Why it matters">
        <SectionHeader>
          A 60 MVA transformer is a million-dollar mistake waiting for a hot afternoon.
        </SectionHeader>
        <div className="grid md:grid-cols-12 gap-10 md:gap-16">
          <div className="md:col-span-7 prose-body text-ink-700 dark:text-ink-200 drop-cap">
            <p>
              PT PLN operates more than 5 000 distribution-class power transformers across the
              Java–Bali 150/20 kV network. Each one ages by Arrhenius kinetics: every 6 °C above
              110 °C hot-spot temperature doubles the rate at which paper insulation degrades. A
              single severe thermal excursion can consume months of equivalent service life.
            </p>
            <p>
              A replacement 60 MVA unit costs roughly USD 800 000 and takes six to twelve months to
              ship into remote substations. Predicting how the rooftop-PV transition reshapes
              individual asset lifetimes therefore has direct, large-magnitude economic
              consequences — and the answer is not the obvious one.
            </p>
            <p>
              We rebuilt the IEEE C57.91-2011 thermal model end-to-end, coupled it to a fast
              learned surrogate, and ran 41 000 Monte-Carlo scenarios at {SUBSTATION.name}. The
              charts below let you interrogate any combination of rooftop-PV penetration, ambient
              warming, load growth, EV charging, and battery dispatch.
            </p>
          </div>
          <aside className="md:col-span-5 md:pl-8 md:border-l hairline">
            <p className="kicker mb-6">The site</p>
            <dl className="grid grid-cols-1 gap-y-4 text-sm">
              <SitePair k="Substation" v={`${SUBSTATION.name} · 150/20 kV`} />
              <SitePair k="Transformer" v={`${SUBSTATION.ratingMVA} MVA · ${SUBSTATION.cooling}`} />
              <SitePair k="Climate" v={`${SUBSTATION.meanAmbientC} °C mean · ±${SUBSTATION.diurnalHalfAmpC} °C diurnal`} />
              <SitePair k="Operator" v={SUBSTATION.operator} />
              <SitePair k="Coordinates" v="−8.17°, 113.70°" />
              <SitePair k="Sources" v="ERA5 / atlite · BMKG Jember · PLN East-Java SCADA" />
            </dl>
          </aside>
        </div>
      </Section>

      {/* II. Methodology */}
      <Section id="methodology" numeral="II" kicker="Methodology">
        <SectionHeader>
          IEEE C57.91 physics, accelerated by XGBoost, dispatched by PyPSA.
        </SectionHeader>

        <div className="prose-body text-ink-700 dark:text-ink-200 drop-cap">
          <p>
            The substation is modelled as a two-bus PyPSA network. The high-voltage bus connects to
            a slack generator representing the Java–Bali 150 kV ring; the 60 MVA ONAF transformer
            couples to the medium-voltage bus where the rooftop PV, the inelastic load, and the
            optional battery are connected. Inspect each component below.
          </p>
        </div>

        <div className="my-12">
          <NetworkDiagram />
        </div>

        <div className="grid md:grid-cols-12 gap-10">
          <div className="md:col-span-7 prose-body text-ink-700 dark:text-ink-200">
            <p>
              Top-oil and hot-spot temperature rises above oil follow the IEEE C57.91-2011 first-order
              differential equations. The hot-spot temperature is the ambient plus the top-oil rise
              plus the hot-spot-over-oil rise. We solve the system with SciPy’s RK45 integrator at
              one-hour resolution and verify against the standard’s Annex G two-phase fixture.
            </p>
            <details className="my-6 border hairline rounded-lg px-5 py-4 bg-white/60 dark:bg-ink-900/40">
              <summary className="cursor-pointer text-sm font-medium text-ink-900 dark:text-ink-50">
                Show the differential equations
              </summary>
              <div className="mt-4 text-sm">
                <Equation
                  display
                  tex="\tau_{TO}\,\frac{d\Delta\theta_{TO}}{dt}=\Delta\theta_{TO,U}-\Delta\theta_{TO},\quad \tau_{W}\,\frac{d\Delta\theta_{HS}}{dt}=\Delta\theta_{HS,U}-\Delta\theta_{HS}"
                />
                <p className="text-ink-500 mt-2">
                  Ultimate values scale with the loading factor <em>K(t)</em> = S(t) / S<sub>rated</sub>{' '}
                  through <Equation tex="(K^2R+1)/(R+1)" /> for top-oil and{' '}
                  <Equation tex="K^{2m}" /> for hot-spot-over-oil.
                </p>
              </div>
            </details>
            <p>
              The XGBoost surrogate is trained on 200 ODE trajectories (33 000 supervised rows) with
              teacher forcing, then rolled out autoregressively at inference using a 13-feature
              vector. We chose XGBoost over an attention model for CPU-only portability on PLN
              field hardware.
            </p>
            <p>
              The Arrhenius aging acceleration factor and annual loss-of-life follow Annex A of the
              standard, with <em>B</em> = 15 000 K for dry Kraft insulation.
            </p>
            <details className="my-6 border hairline rounded-lg px-5 py-4 bg-white/60 dark:bg-ink-900/40">
              <summary className="cursor-pointer text-sm font-medium text-ink-900 dark:text-ink-50">
                Aging acceleration and loss-of-life
              </summary>
              <div className="mt-4 text-sm">
                <Equation
                  display
                  tex="F_{AA}(t)=\exp\!\left[\frac{B}{\theta_{HS,R}+273}-\frac{B}{\theta_{HS}(t)+273}\right]"
                />
                <Equation
                  display
                  tex="\mathrm{LoL}_{\%}=\frac{\overline{F_{AA}}\,\cdot n_{\text{hours}}\,\Delta t}{180\,000}\times 100\,\%"
                />
              </div>
            </details>
          </div>

          <aside className="md:col-span-5 md:pl-8 md:border-l hairline">
            <p className="kicker">The 13 surrogate features</p>
            <ul className="mt-4 space-y-2.5 text-sm text-ink-600 dark:text-ink-300">
              {FEATURES.map((f) => (
                <li key={f} className="flex gap-3">
                  <span className="numera text-ink-400 select-none">·</span>
                  <span>{f}</span>
                </li>
              ))}
            </ul>
          </aside>
        </div>

        <div className="mt-16">
          <RuntimeBars />
        </div>
      </Section>

      {/* III. Findings */}
      <Section id="findings" numeral="III" kicker="Findings">
        <SectionHeader>Five things the 41 000 simulations told us.</SectionHeader>

        <div className="space-y-32 md:space-y-40">
          {FINDINGS.map((f, i) => {
            const Chart = chartMap[f.chart];
            const num = FINDING_NUMERALS[i] || String(i + 1);
            return (
              <article
                key={f.id}
                id={f.id}
                className="scroll-mt-28"
                aria-labelledby={`f-${f.id}`}
              >
                <div className="grid md:grid-cols-12 gap-8 md:gap-12 mb-12">
                  <div className="md:col-span-1 md:pt-3">
                    <p className="numera text-3xl md:text-4xl text-ink-300 dark:text-ink-700 leading-none">
                      {num}.
                    </p>
                  </div>
                  <div className="md:col-span-11">
                    <p className="kicker">{f.kicker}</p>
                    <h3
                      id={`f-${f.id}`}
                      className="font-serif text-3xl md:text-5xl mt-3 text-ink-950 dark:text-ink-50 tracking-tighter leading-[1.05]"
                    >
                      {f.title}
                    </h3>
                    <p className="prose-body mt-6 text-ink-600 dark:text-ink-300 italic">
                      {f.oneLiner}
                    </p>
                  </div>
                </div>

                <StatCallouts items={f.callouts} />

                <div className="prose-body text-ink-700 dark:text-ink-200 mb-2">
                  <p>{f.body}</p>
                </div>

                {f.pullQuote ? <PullQuote>{f.pullQuote}</PullQuote> : null}

                {Chart ? (
                  <div className="max-w-chart mt-10">
                    <Chart />
                  </div>
                ) : null}
              </article>
            );
          })}
        </div>
      </Section>

      {/* IV. Embedded explorer */}
      <Section id="scenarios" numeral="IV" kicker="Explore">
        <SectionHeader>Pick a future. See the loss.</SectionHeader>
        <p className="prose-body text-ink-700 dark:text-ink-200 mb-10 drop-cap">
          The Monte-Carlo grid covers PV penetration ∈ {`{0, 10, 30, 50, 75}`} %, ambient offsets ∈{' '}
          {`{0, +1, +2}`} °C, and load growth ∈ {`{0, +4.5}`} %/yr — thirty cells in total, plus
          eleven named scenarios for EV and battery storage. Use the controls below to read off the
          annual loss-of-life for any combination.
        </p>
        <ScenarioExplorer />
        <p className="mt-8 text-sm">
          <a
            className="underline underline-offset-4 text-signal-500 dark:text-signal-300 hover:opacity-80"
            href="/scenarios/"
          >
            Open the full data explorer ↗
          </a>
        </p>
      </Section>

      {/* V. Implications */}
      <Section id="implications" numeral="V" kicker="Implications">
        <SectionHeader>
          What PLN planners should take from this — and what is still open.
        </SectionHeader>

        <ol className="space-y-14 max-w-[68ch]">
          <Implication
            n={1}
            title="Rooftop PV is a real aging benefit today."
            body="At GI Jember’s 26.5 °C ambient, every additional percentage point of rooftop-PV penetration cools the daytime hot-spot. The 75 % PV case buys nearly a fifth of the annual loss-of-life back — quantitatively reproducible across paired Monte-Carlo seeds."
          />
          <Implication
            n={2}
            title="It will not be enough on its own."
            body="Under projected 2030 ambient warming and 4.5 %/yr load growth, even maximum rooftop-PV penetration cannot fully restore today’s baseline. Cooling-system upgrades, transformer up-rating, and explicit demand-side management must run alongside the rooftop deployment."
          />
          <Implication
            n={3}
            title="EV charging must be coordinated, not just permitted."
            body="A 30 % EV fleet, charging on its uncoordinated default schedule, cancels most of the rooftop-PV gain and pushes the substation above the no-PV baseline. Time-of-use tariffs or directly controlled charging are not optional features of the transition — they are prerequisites."
          />
          <Implication
            n={4}
            title="Economic dispatch is not aging-optimal."
            body="The HiGHS LP optimises slack-import cost; the threshold rule, almost by accident, targets the hours where Arrhenius aging is super-linear. Operators relying on cost-driven optimisation should not assume it protects insulation life — until aging constraints enter the LP objective, dispatch heuristics with explicit thermal awareness perform better."
          />
        </ol>

        <p className="prose-body mt-20 text-ink-700 dark:text-ink-200">
          The framework is portable. Re-targeting it to any 150/20 kV PLN substation requires only
          a new location configuration — site-specific ambient parameters, load profile, and PV
          coordinates. No code changes. Fleet-level loss-of-life risk mapping is therefore feasible
          with the existing tooling.
        </p>
      </Section>

      {/* VI. Reproducibility */}
      <Section id="reproducibility" numeral="VI" kicker="Reproduce">
        <SectionHeader>Run the entire pipeline on a laptop.</SectionHeader>
        <div className="grid md:grid-cols-2 gap-10">
          <div className="prose-body text-ink-700 dark:text-ink-200 drop-cap">
            <p>
              The simulation is open-source. From a clean checkout, the full pipeline — thermal
              validation, surrogate training, 41 000-run Monte Carlo, and figure generation —
              completes in roughly an hour on a 2024-class laptop.
            </p>
            <p>
              When measured BMKG, PLN, or ERA5 data files are absent, documented synthesis fallbacks
              preserve the code path so that peer reviewers can replicate the methodology end-to-end
              without proprietary inputs.
            </p>
          </div>
          <pre className="numera text-xs leading-relaxed text-ink-100 bg-ink-950 dark:bg-ink-900 border hairline rounded-lg p-6 overflow-x-auto">
{`# 1. Clone and install
git clone https://github.com/fakhrihakim20/trafo_lol_solar.git
cd trafo_lol_solar
python -m venv .venv && .venv\\Scripts\\activate
pip install -e .

# 2. Validate against IEEE C57.91 Annex G
python scripts/01_validate_thermal.py

# 3. Train the surrogate
python scripts/02_generate_training_data.py
python scripts/03_train_surrogate.py

# 4. Run the 41 000-scenario Monte Carlo
python scripts/04_run_monte_carlo.py --n_runs 1000

# 5. Regenerate paper figures
python scripts/05_generate_figures.py`}
          </pre>
        </div>
      </Section>

      {/* Colophon — replaces the old footer */}
      <Colophon />
    </>
  );
}

function Section({
  id,
  numeral,
  kicker,
  children,
}: {
  id: string;
  numeral: string;
  kicker: string;
  children: React.ReactNode;
}) {
  return (
    <section
      id={id}
      className="px-6 py-24 md:py-36 scroll-mt-24 border-t hairline first-of-type:border-t-0"
    >
      <div className="max-w-page mx-auto">
        <p className="kicker mb-4">
          <span className="text-ink-400 dark:text-ink-600 mr-2">{numeral}.</span>
          {kicker}
        </p>
        {children}
      </div>
    </section>
  );
}

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="font-serif text-4xl md:text-6xl text-ink-950 dark:text-ink-50 tracking-tighter leading-[1.02] mb-14 max-w-[22ch]">
      {children}
    </h2>
  );
}

function SitePair({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-6 border-b hairline pb-2">
      <dt className="text-ink-600 dark:text-ink-300">{k}</dt>
      <dd className="text-ink-900 dark:text-ink-100 numera text-right">{v}</dd>
    </div>
  );
}

function Implication({ n, title, body }: { n: number; title: string; body: string }) {
  return (
    <li className="grid grid-cols-[2.5rem_1fr] md:grid-cols-[3rem_1fr] gap-x-6">
      <span className="numera text-2xl md:text-3xl text-ink-300 dark:text-ink-700 leading-none pt-1.5">
        {n}.
      </span>
      <div>
        <h3 className="font-serif text-2xl md:text-3xl text-ink-950 dark:text-ink-50 tracking-tighter leading-snug">
          {title}
        </h3>
        <p className="mt-4 prose-body text-ink-600 dark:text-ink-300">{body}</p>
      </div>
    </li>
  );
}

function Colophon() {
  return (
    <footer className="border-t hairline px-6 py-16 md:py-20 mt-20 text-sm">
      <div className="max-w-page mx-auto grid md:grid-cols-12 gap-10">
        <div className="md:col-span-5">
          <p className="font-serif italic text-lg text-ink-900 dark:text-ink-100">
            Transformer Loss-of-Life under Rooftop-PV Penetration.
          </p>
          <p className="mt-3 text-ink-600 dark:text-ink-300 max-w-[40ch] leading-relaxed">
            A stakeholder companion to the IEEE&nbsp;ICT-PEP&nbsp;2026 manuscript on transformer
            aging at GI&nbsp;Jember.
          </p>
        </div>

        <dl className="md:col-span-4 grid grid-cols-[6rem_1fr] gap-y-3 gap-x-4 text-ink-600 dark:text-ink-300">
          <dt className="kicker self-center">Typeset in</dt>
          <dd>Newsreader, Geist, JetBrains Mono</dd>
          <dt className="kicker self-center">Charts</dt>
          <dd>Plotly.js, hand-built SVG</dd>
          <dt className="kicker self-center">Stack</dt>
          <dd>Next.js 14, Tailwind CSS, KaTeX</dd>
          <dt className="kicker self-center">Hosting</dt>
          <dd>GitHub Pages, static export</dd>
        </dl>

        <div className="md:col-span-3 text-ink-600 dark:text-ink-300">
          <p className="kicker mb-3">Cite</p>
          <p className="font-serif italic leading-relaxed">
            Hakim, F. (2026). Transformer Loss-of-Life Acceleration under High Rooftop-PV
            Penetration on Indonesian 150/20 kV Substations. <em>IEEE ICT-PEP</em>.
          </p>
          <p className="mt-5 kicker">Source</p>
          <p className="mt-2">
            <a
              className="underline underline-offset-4 hover:text-ink-900 dark:hover:text-ink-50"
              href="https://github.com/fakhrihakim20/trafo_lol_solar"
            >
              github.com/fakhrihakim20/trafo_lol_solar
            </a>
          </p>
        </div>
      </div>

      <p className="max-w-page mx-auto mt-14 pt-6 border-t hairline text-xs text-ink-500 dark:text-ink-400 flex flex-wrap justify-between gap-4">
        <span>Fakhri Hakim · PT PLN (Persero) · 2026</span>
        <span className="numera">{`v1.0 · ${SUBSTATION.name} calibration`}</span>
      </p>
    </footer>
  );
}

const FEATURES = [
  'Loading factor K(t) and lags K(t−1), K(t−2)',
  'Ambient temperature at t and t−1',
  'Rooftop-PV per-unit generation',
  'Hour-of-day encoded as sine and cosine',
  'θ_HS lags at t−1 and t−2',
  'Top-oil temperature at t−1',
  '1-h and 3-h load ramp magnitudes',
];
