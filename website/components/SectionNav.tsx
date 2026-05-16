'use client';

import { useEffect, useState } from 'react';

const SECTIONS = [
  { id: 'problem', numeral: 'I', label: 'Why' },
  { id: 'methodology', numeral: 'II', label: 'Methods' },
  { id: 'findings', numeral: 'III', label: 'Findings' },
  { id: 'scenarios', numeral: 'IV', label: 'Explore' },
  { id: 'implications', numeral: 'V', label: 'Implications' },
  { id: 'reproducibility', numeral: 'VI', label: 'Reproduce' },
];

export default function SectionNav() {
  const [active, setActive] = useState<string>('');
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 60);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });

    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) setActive(e.target.id);
        });
      },
      { rootMargin: '-40% 0px -55% 0px' },
    );
    SECTIONS.forEach((s) => {
      const el = document.getElementById(s.id);
      if (el) obs.observe(el);
    });
    return () => {
      window.removeEventListener('scroll', onScroll);
      obs.disconnect();
    };
  }, []);

  return (
    <nav
      aria-label="Table of contents"
      className={
        'sticky top-0 z-30 w-full transition-colors duration-200 ' +
        (scrolled
          ? 'bg-ink-50/85 dark:bg-ink-950/85 backdrop-blur-md border-b hairline'
          : 'bg-transparent border-b border-transparent')
      }
    >
      <div className="max-w-page mx-auto px-6 py-3 flex items-center justify-between gap-6">
        <a
          href="#top"
          className="font-serif italic text-sm md:text-base text-ink-900 dark:text-ink-100 tracking-tight"
        >
          Transformer Loss-of-Life
          <span className="hidden md:inline text-ink-500"> · IEEE ICT-PEP 2026</span>
        </a>

        <ol className="hidden md:flex items-center gap-x-7 text-sm" role="list">
          {SECTIONS.map((s) => (
            <li key={s.id}>
              <a
                href={`#${s.id}`}
                className={
                  'group inline-flex items-baseline gap-1.5 py-2 transition-colors ' +
                  (active === s.id
                    ? 'text-ink-950 dark:text-ink-50'
                    : 'text-ink-500 hover:text-ink-900 dark:hover:text-ink-100')
                }
              >
                <span className="numera text-[0.7rem] uppercase tracking-widest text-ink-400 group-hover:text-ink-700 dark:group-hover:text-ink-200">
                  {s.numeral}
                </span>
                <span
                  className={
                    'border-b transition-colors ' +
                    (active === s.id
                      ? 'border-ink-950 dark:border-ink-50'
                      : 'border-transparent')
                  }
                >
                  {s.label}
                </span>
              </a>
            </li>
          ))}
        </ol>

        <a
          href="/scenarios/"
          className="hidden lg:inline-flex items-baseline gap-1 text-sm text-ink-600 dark:text-ink-300 hover:text-ink-950 dark:hover:text-ink-50 transition-colors"
        >
          Data explorer
          <span aria-hidden className="numera text-xs">↗</span>
        </a>
      </div>
    </nav>
  );
}
