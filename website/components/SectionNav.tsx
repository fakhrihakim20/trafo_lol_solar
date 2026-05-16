'use client';

import { useEffect, useState } from 'react';

const SECTIONS = [
  { id: 'problem', label: 'Why it matters' },
  { id: 'methodology', label: 'How we measured' },
  { id: 'findings', label: 'Findings' },
  { id: 'scenarios', label: 'Explore' },
  { id: 'implications', label: 'Implications' },
  { id: 'reproducibility', label: 'Reproduce' },
];

export default function SectionNav() {
  const [active, setActive] = useState<string>('');
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
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
      aria-label="Section navigation"
      className={
        'sticky top-0 z-30 w-full border-b transition-colors duration-200 ' +
        (scrolled
          ? 'bg-ink-50/80 dark:bg-ink-950/80 backdrop-blur-md hairline'
          : 'bg-transparent border-transparent')
      }
    >
      <div className="max-w-page mx-auto px-6 py-3 flex items-center justify-between gap-6">
        <a href="#top" className="text-sm tracking-tight font-medium text-ink-950 dark:text-ink-50">
          GI Jember · Transformer LoL Report
        </a>
        <ul className="hidden md:flex items-center gap-1 text-sm">
          {SECTIONS.map((s) => (
            <li key={s.id}>
              <a
                href={`#${s.id}`}
                className={
                  'px-3.5 py-2 rounded-md transition-colors ' +
                  (active === s.id
                    ? 'text-ink-950 dark:text-ink-50 bg-ink-200/60 dark:bg-ink-800/60'
                    : 'text-ink-500 hover:text-ink-900 dark:hover:text-ink-100')
                }
              >
                {s.label}
              </a>
            </li>
          ))}
        </ul>
        <a
          href="/scenarios/"
          className="hidden md:inline-flex items-center gap-2 text-sm font-medium text-ink-950 dark:text-ink-50 border hairline rounded-md px-3.5 py-2 hover:bg-ink-100 dark:hover:bg-ink-900 transition-colors"
        >
          Open data explorer
          <span aria-hidden className="numera text-xs text-ink-500">→</span>
        </a>
      </div>
    </nav>
  );
}
